// CUDA form of TFSimulator's graph finite-difference loop.
//
// Every float32 operation uses an explicit round-to-nearest intrinsic and the
// file is compiled with -fmad=false, so no multiply-add is contracted. The
// operations follow TFSimulator._graph_fd_loop_impl in the same order, which
// itself follows update_temperature_estimates, so results are the same IEEE
// float32 values TensorFlow produces.
//
// Two layout choices reduce memory traffic without changing any value:
//
// * Cell classes. Every per-cell constant (oriented conductivities, CV sizes,
//   boundary flags, density, heat capacity) is one of a few distinct tuples.
//   Each cell stores a uint8 class index. Derived constants are computed per
//   class with the same operations and inputs as per cell, so they are equal.
//
// * Temporal blocking. A block loads a REGION_H x REGION_W region into shared
//   memory and runs up to K Jacobi iterations there. After iteration j the
//   valid area has shrunk by j + 1 cells on each side; the inner tile, which
//   stays valid for all K iterations, is written back. Each cell's |delta| is
//   counted only by the block that owns it.
//
// The host runs groups of K iterations. If an environment converges inside a
// group, the group is rerun from its (unchanged) input for exactly the number
// of iterations the original loop would have run.
//
// One context holds B environments sharing a floor plan and materials. Each
// has its own ambient temperature, convection coefficient, convergence test,
// and iteration count.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FLAG_EXTERIOR 1
#define FLAG_H_LEFT 2
#define FLAG_H_RIGHT 4
#define FLAG_H_TOP 8
#define FLAG_H_BOTTOM 16

#define MAX_CLASSES 256
#define REGION_W 64
#define BLOCK_Y 8
#define ROWS_PER_THREAD 8
#define REGION_H (BLOCK_Y * ROWS_PER_THREAD)
#define THREADS (REGION_W * BLOCK_Y)

// Per class constants, env independent.
struct ClassStatic {
  float k1u, k3u, k2v, k4v, uz, vz, storage;
  int flags;
};

// Per class constants for one environment and timestep. The fields the
// iteration reads fill the first three 16-byte words, so the kernel loads them
// from shared memory with three vector loads (see iterate_fused).
struct __align__(16) ClassStep {
  int exterior;
  float k1u, k3u, k2v;
  float k4v, uz, vz, denom;
  float h_left_tinf, h_right_tinf, h_above_tinf, h_below_tinf;
  float storage, pad0, pad1, pad2;
};

struct FdContext {
  int batch, rows, cols, limit, fuse_k, n_classes;
  int64_t cells;
  double threshold;
  float z, delta_t;
  uint8_t *cls;               // cells
  ClassStatic *class_static;  // n_classes
  ClassStep *class_step;      // batch * n_classes
  float *temp_minus, *input_q, *buf0, *buf1, *out;  // batch * cells
  float *temp_inf, *conv;                           // batch
  unsigned int *max_bits;                           // batch * limit
  // Per environment loop state, kept on the device: iterations for the next
  // fused launch, done flag, iterations run, and which buffer holds the
  // result (-1 temp_minus, 0 buf0, 1 buf1).
  int *group_iters, *done, *n_iter, *result_buf;  // batch
  // 1 for tiles (in launch order) whose in-grid owner CVs are all exterior.
  unsigned char *tile_ext;
  cudaStream_t stream;
  cudaEvent_t ev_start, ev_stop;
  float last_compute_ms;
};

static __device__ __forceinline__ float edge_h(int f, int bit, float c) {
  return (f & bit) ? c : 0.0f;
}

__global__ void init_class_static(int n, float z, const float *kl,
                                  const float *kr, const float *kt,
                                  const float *kb, const float *u,
                                  const float *v, const float *density,
                                  const float *heat_capacity,
                                  const uint8_t *flags, ClassStatic *out) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  ClassStatic s;
  s.uz = __fmul_rn(z, u[k]);
  s.vz = __fmul_rn(z, v[k]);
  s.k1u = __fdiv_rn(kl[k], u[k]);
  s.k3u = __fdiv_rn(kr[k], u[k]);
  s.k2v = __fdiv_rn(kb[k], v[k]);
  s.k4v = __fdiv_rn(kt[k], v[k]);
  float storage = __fmul_rn(density[k], u[k]);
  storage = __fmul_rn(storage, v[k]);
  storage = __fmul_rn(storage, heat_capacity[k]);
  s.storage = __fmul_rn(z, storage);
  s.flags = flags[k];
  out[k] = s;
}

// Terms that depend on the environment's ambient temperature and convection
// coefficient but not on the current estimate.
__global__ void prepare_class_step(FdContext c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= c.batch * c.n_classes) return;
  int b = idx / c.n_classes;
  int k = idx - b * c.n_classes;
  ClassStatic s = c.class_static[k];
  float cc = c.conv[b];
  float tinf = c.temp_inf[b];
  float h_left = edge_h(s.flags, FLAG_H_LEFT, cc);
  float h_right = edge_h(s.flags, FLAG_H_RIGHT, cc);
  float h_top = edge_h(s.flags, FLAG_H_TOP, cc);
  float h_bottom = edge_h(s.flags, FLAG_H_BOTTOM, cc);
  float dt1 = __fadd_rn(s.k1u, s.k3u);
  dt1 = __fadd_rn(dt1, h_left);
  dt1 = __fadd_rn(dt1, h_right);
  dt1 = __fmul_rn(s.vz, dt1);
  float dt2 = __fadd_rn(s.k2v, s.k4v);
  dt2 = __fadd_rn(dt2, h_bottom);
  dt2 = __fadd_rn(dt2, h_top);
  dt2 = __fmul_rn(s.uz, dt2);
  float dt3 = __fdiv_rn(s.storage, c.delta_t);
  float denom = __fadd_rn(dt1, dt2);
  denom = __fadd_rn(denom, dt3);
  denom = __fadd_rn(denom, 0.0f);
  ClassStep o;
  o.k1u = s.k1u;
  o.k3u = s.k3u;
  o.k2v = s.k2v;
  o.k4v = s.k4v;
  o.uz = s.uz;
  o.vz = s.vz;
  o.storage = s.storage;
  o.denom = denom;
  o.h_left_tinf = __fmul_rn(tinf, h_left);
  o.h_right_tinf = __fmul_rn(tinf, h_right);
  o.h_above_tinf = __fmul_rn(tinf, h_top);
  o.h_below_tinf = __fmul_rn(tinf, h_bottom);
  o.exterior = (s.flags & FLAG_EXTERIOR) != 0;
  o.pad0 = o.pad1 = o.pad2 = 0.0f;
  c.class_step[idx] = o;
}

// Runs group_iters[b] (<= K) iterations starting from `cur`, writing `next`.
// max_delta of iteration j goes to max_bits[b * limit + it0 + j].
template <int K>
__global__ void __launch_bounds__(THREADS)
    iterate_fused(FdContext c, const float *cur, float *next, int it0) {
  constexpr int TILE_W = REGION_W - 2 * K;
  constexpr int TILE_H = REGION_H - 2 * K;
  __shared__ float sa[REGION_H][REGION_W];
  __shared__ float sb[REGION_H][REGION_W];
  // Words 0-2 of each ClassStep (everything the iteration reads).
  __shared__ float4 class_w[MAX_CLASSES][3];
  __shared__ unsigned int warp_max[THREADS / 32];

  const int b = blockIdx.z;
  const int niter = c.group_iters[b];
  if (niter <= 0) return;  // Uniform across the block.

  const int tid = threadIdx.y * REGION_W + threadIdx.x;
  for (int k = tid; k < c.n_classes; k += THREADS) {
    const float4 *g =
        reinterpret_cast<const float4 *>(&c.class_step[b * c.n_classes + k]);
    class_w[k][0] = g[0];
    class_w[k][1] = g[1];
    class_w[k][2] = g[2];
  }

  if (c.tile_ext[blockIdx.y * gridDim.x + blockIdx.x]) {
    // Every in-grid owner CV is exterior, so every iteration sets it to tinf.
    // Iteration it0's delta is |tinf - t| over the input; later iterations of
    // the group have delta 0, which max_bits already holds.
    const int64_t base_e = (int64_t)b * c.cells;
    const float tinf_e = c.temp_inf[b];
    const int r0_e = blockIdx.y * TILE_H - K;
    const int gc_e = blockIdx.x * TILE_W - K + threadIdx.x;
    unsigned int local_e = 0;
    for (int m = 0; m < ROWS_PER_THREAD; m++) {
      int lr = threadIdx.y + m * BLOCK_Y;
      int gr = r0_e + lr;
      bool owner = lr >= K && lr < REGION_H - K && (int)threadIdx.x >= K &&
                   (int)threadIdx.x < REGION_W - K;
      if (owner && gr >= 0 && gr < c.rows && gc_e >= 0 && gc_e < c.cols) {
        int64_t i = base_e + (int64_t)gr * c.cols + gc_e;
        unsigned int bits = __float_as_uint(fabsf(__fsub_rn(tinf_e, cur[i])));
        local_e = bits > local_e ? bits : local_e;
        next[i] = tinf_e;
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      unsigned int other = __shfl_xor_sync(0xffffffffu, local_e, off);
      local_e = other > local_e ? other : local_e;
    }
    if ((tid & 31) == 0) warp_max[tid >> 5] = local_e;
    __syncthreads();
    if (tid == 0) {
      unsigned int bm = 0;
      for (int w = 0; w < THREADS / 32; w++) bm = warp_max[w] > bm ? warp_max[w] : bm;
      if (bm != 0) atomicMax(&c.max_bits[(int64_t)b * c.limit + it0], bm);
    }
    return;
  }

  const int64_t base = (int64_t)b * c.cells;
  const float tinf = c.temp_inf[b];
  const float delta_t = c.delta_t;
  const int r0 = blockIdx.y * TILE_H - K;
  const int c0 = blockIdx.x * TILE_W - K;
  const int lc = threadIdx.x;
  const int gc = c0 + lc;

  int cls[ROWS_PER_THREAD];
  float nt3[ROWS_PER_THREAD], q[ROWS_PER_THREAD];
  bool in_grid[ROWS_PER_THREAD];
  __syncthreads();  // classes ready
#pragma unroll
  for (int m = 0; m < ROWS_PER_THREAD; m++) {
    int lr = threadIdx.y + m * BLOCK_Y;
    int gr = r0 + lr;
    in_grid[m] = gr >= 0 && gr < c.rows && gc >= 0 && gc < c.cols;
    float val = tinf;  // Beyond the grid edge, neighbors read tinf.
    cls[m] = 0;
    nt3[m] = 0.0f;
    q[m] = 0.0f;
    if (in_grid[m]) {
      int64_t i = (int64_t)gr * c.cols + gc;
      cls[m] = c.cls[i];
      val = cur[base + i];
      nt3[m] = __fdiv_rn(
          __fmul_rn(c.class_step[b * c.n_classes + cls[m]].storage,
                    c.temp_minus[base + i]),
          delta_t);
      q[m] = c.input_q[base + i];
    }
    sa[lr][lc] = val;
    sb[lr][lc] = val;
  }
  __syncthreads();

  float(*src)[REGION_W] = sa;
  float(*dst)[REGION_W] = sb;
  for (int j = 0; j < niter; j++) {
    unsigned int local = 0;
#pragma unroll
    for (int m = 0; m < ROWS_PER_THREAD; m++) {
      int lr = threadIdx.y + m * BLOCK_Y;
      bool compute = in_grid[m] && lr >= j + 1 && lr <= REGION_H - 2 - j &&
                     lc >= j + 1 && lc <= REGION_W - 2 - j;
      if (compute) {
        // w0 = {exterior, k1u, k3u, k2v}, w1 = {k4v, uz, vz, denom},
        // w2 = {h_left_tinf, h_right_tinf, h_above_tinf, h_below_tinf}.
        const float4 *sv = class_w[cls[m]];
        const float4 w0 = sv[0];
        float t = src[lr][lc];
        // Exterior CVs are set to tinf after the update, so their arithmetic
        // does not change the result and is skipped.
        float est = tinf;
        if (!__float_as_int(w0.x)) {
          const float4 w1 = sv[1];
          const float4 w2 = sv[2];
          float t_left = src[lr][lc + 1];
          float t_right = src[lr][lc - 1];
          float t_above = src[lr - 1][lc];
          float t_below = src[lr + 1][lc];
          float nt1 =
              __fadd_rn(__fmul_rn(w0.y, t_left), __fmul_rn(w0.z, t_right));
          nt1 = __fadd_rn(nt1, w2.x);
          nt1 = __fadd_rn(nt1, w2.y);
          nt1 = __fmul_rn(w1.z, nt1);
          float nt2 =
              __fadd_rn(__fmul_rn(w0.w, t_below), __fmul_rn(w1.x, t_above));
          nt2 = __fadd_rn(nt2, w2.w);
          nt2 = __fadd_rn(nt2, w2.z);
          nt2 = __fmul_rn(w1.y, nt2);
          float numer = __fadd_rn(nt1, nt2);
          numer = __fadd_rn(numer, nt3[m]);
          numer = __fadd_rn(numer, q[m]);
          numer = __fadd_rn(numer, 0.0f);
          numer = __fadd_rn(numer, 0.0f);
          est = __fdiv_rn(numer, w1.w);
        }
        dst[lr][lc] = est;
        bool owner = lr >= K && lr < REGION_H - K && lc >= K &&
                     lc < REGION_W - K;
        if (owner) {
          // Non-negative floats order the same as their bit patterns.
          unsigned int bits = __float_as_uint(fabsf(__fsub_rn(est, t)));
          local = bits > local ? bits : local;
        }
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      unsigned int other = __shfl_xor_sync(0xffffffffu, local, off);
      local = other > local ? other : local;
    }
    if ((tid & 31) == 0) warp_max[tid >> 5] = local;
    __syncthreads();
    if (tid == 0) {
      unsigned int bm = 0;
      for (int w = 0; w < THREADS / 32; w++) bm = warp_max[w] > bm ? warp_max[w] : bm;
      if (bm != 0) atomicMax(&c.max_bits[(int64_t)b * c.limit + it0 + j], bm);
    }
    float(*tmp)[REGION_W] = src;
    src = dst;
    dst = tmp;
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < ROWS_PER_THREAD; m++) {
    int lr = threadIdx.y + m * BLOCK_Y;
    bool owner = lr >= K && lr < REGION_H - K && lc >= K && lc < REGION_W - K;
    if (in_grid[m] && owner) {
      int64_t i = (int64_t)(r0 + lr) * c.cols + gc;
      next[base + i] = src[lr][lc];
    }
  }
}

// Sets the iterations of the coming group: 0 for environments already done.
__global__ void begin_group(FdContext c, int group) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= c.batch) return;
  c.group_iters[b] = c.done[b] ? 0 : group;
}

// After a group of g iterations starting at it0: finds the first iteration
// whose max_delta meets the threshold, as the original loop's break does.
// numpy compares the float32 max_delta with the Python float threshold in
// float64. If that iteration is not the group's last, group_iters is set so
// the following launch reruns the group from the same input for exactly that
// many iterations; otherwise group_iters is 0 and that launch does nothing.
__global__ void end_group(FdContext c, int it0, int next_id) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= c.batch) return;
  int g = c.group_iters[b];
  if (g <= 0) {
    c.group_iters[b] = 0;
    return;
  }
  c.result_buf[b] = next_id;
  int redo = 0;
  for (int j = 0; j < g; j++) {
    float md = __uint_as_float(c.max_bits[(int64_t)b * c.limit + it0 + j]);
    if ((double)md <= c.threshold) {
      c.n_iter[b] = it0 + j + 1;
      c.done[b] = 1;
      if (j + 1 < g) redo = j + 1;
      break;
    }
  }
  if (!c.done[b] && it0 + g == c.limit) {
    c.n_iter[b] = c.limit;
    c.done[b] = 1;
  }
  c.group_iters[b] = redo;
}

// Copies each environment's result buffer into out, and zeroes max_bits from
// n_iter on (those iterations did not happen in the original loop).
__global__ void gather(FdContext c) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t bn = (int64_t)c.batch * c.cells;
  if (idx < bn) {
    int b = (int)(idx / c.cells);
    int r = c.result_buf[b];
    const float *src = r < 0 ? c.temp_minus : (r == 0 ? c.buf0 : c.buf1);
    c.out[idx] = src[idx];
  }
  if (idx < (int64_t)c.batch * c.limit) {
    int b = (int)(idx / c.limit);
    int j = (int)(idx - (int64_t)b * c.limit);
    if (j >= c.n_iter[b]) c.max_bits[idx] = 0u;
  }
}

static void launch_group(FdContext *c, const float *cur, float *next,
                         int it0) {
  int tile_w = REGION_W - 2 * c->fuse_k, tile_h = REGION_H - 2 * c->fuse_k;
  dim3 grid((c->cols + tile_w - 1) / tile_w, (c->rows + tile_h - 1) / tile_h,
            c->batch);
  dim3 block(REGION_W, BLOCK_Y);
  switch (c->fuse_k) {
#define CASE(K)                                                     \
  case K:                                                           \
    iterate_fused<K><<<grid, block, 0, c->stream>>>(*c, cur, next, it0); \
    break;
    CASE(1)
    CASE(2)
    CASE(4)
    CASE(6)
    CASE(8)
    CASE(12)
    CASE(16)
#undef CASE
  }
}

static int fuse_k_supported(int k) {
  return k == 1 || k == 2 || k == 4 || k == 6 || k == 8 || k == 12 || k == 16;
}

// Enqueues one timestep for every environment. Inputs (temp_minus, input_q,
// temp_inf, conv) must already be on the device; the result is left in
// c->out, n_iter and max_bits on the device.
static void fd_enqueue_device(FdContext *c) {
  cudaStream_t s = c->stream;
  int B = c->batch, K = c->fuse_k, L = c->limit;
  cudaMemsetAsync(c->max_bits, 0, (size_t)B * L * sizeof(unsigned int), s);
  cudaMemsetAsync(c->done, 0, B * sizeof(int), s);
  cudaMemsetAsync(c->n_iter, 0, B * sizeof(int), s);
  cudaMemsetAsync(c->result_buf, 0xff, B * sizeof(int), s);  // -1
  cudaEventRecord(c->ev_start, s);
  prepare_class_step<<<(B * c->n_classes + 255) / 256, 256, 0, s>>>(*c);

  // Every group is enqueued without waiting: environments that are done get
  // group_iters 0, and their blocks return at once. Buffer ids: -1 is
  // temp_minus (the first estimate), 0 is buf0, 1 is buf1.
  int small_blocks = (B + 255) / 256;
  int cur_id = -1;
  for (int it0 = 0; it0 < L; it0 += K) {
    int next_id = (cur_id == 0) ? 1 : 0;
    const float *cur =
        cur_id < 0 ? c->temp_minus : (cur_id == 0 ? c->buf0 : c->buf1);
    float *next = next_id == 0 ? c->buf0 : c->buf1;
    int group = (L - it0) < K ? (L - it0) : K;
    begin_group<<<small_blocks, 256, 0, s>>>(*c, group);
    launch_group(c, cur, next, it0);
    end_group<<<small_blocks, 256, 0, s>>>(*c, it0, next_id);
    launch_group(c, cur, next, it0);  // Rerun, only where end_group asks.
    cur_id = next_id;
  }
  int64_t bn = (int64_t)B * c->cells;
  int64_t gather_n = bn > (int64_t)B * L ? bn : (int64_t)B * L;
  gather<<<(unsigned)((gather_n + 255) / 256), 256, 0, s>>>(*c);
  cudaEventRecord(c->ev_stop, s);
}

extern "C" {

const char *fd_error_string(int code) {
  return cudaGetErrorString((cudaError_t)code);
}

void fd_destroy(void *handle) {
  FdContext *c = (FdContext *)handle;
  if (!c) return;
  cudaFree(c->cls);
  cudaFree(c->class_static);
  cudaFree(c->class_step);
  cudaFree(c->temp_minus);
  cudaFree(c->input_q);
  cudaFree(c->buf0);
  cudaFree(c->buf1);
  cudaFree(c->out);
  cudaFree(c->temp_inf);
  cudaFree(c->conv);
  cudaFree(c->max_bits);
  cudaFree(c->group_iters);
  cudaFree(c->done);
  cudaFree(c->n_iter);
  cudaFree(c->result_buf);
  cudaFree(c->tile_ext);
  if (c->stream) cudaStreamDestroy(c->stream);
  if (c->ev_start) cudaEventDestroy(c->ev_start);
  if (c->ev_stop) cudaEventDestroy(c->ev_stop);
  free(c);
}

// Returns NULL on failure (allocation, unsupported fuse_k, or class count).
//
// Classes: n_classes tuples (kl, kr, kt, kb, u, v, density, heat_capacity,
// flags) and cls, a uint8 class index for each of rows * cols cells.
void *fd_create(int batch, int rows, int cols, int limit, double threshold,
                float z, float delta_t, int fuse_k, int n_classes,
                const float *kl, const float *kr, const float *kt,
                const float *kb, const float *u, const float *v,
                const float *density, const float *heat_capacity,
                const uint8_t *flags, const uint8_t *cls) {
  if (!fuse_k_supported(fuse_k) || n_classes < 1 || n_classes > MAX_CLASSES ||
      limit < 1 || batch < 1)
    return NULL;
  FdContext *c = (FdContext *)calloc(1, sizeof(FdContext));
  if (!c) return NULL;
  c->batch = batch;
  c->rows = rows;
  c->cols = cols;
  c->limit = limit;
  c->fuse_k = fuse_k;
  c->n_classes = n_classes;
  c->cells = (int64_t)rows * cols;
  c->threshold = threshold;
  c->z = z;
  c->delta_t = delta_t;
  int64_t bn = (int64_t)batch * c->cells;
  int fail = 0;
  fail |= cudaMalloc((void **)&c->cls, c->cells) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->class_static,
                     n_classes * sizeof(ClassStatic)) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->class_step,
                     (size_t)batch * n_classes * sizeof(ClassStep)) !=
          cudaSuccess;
  fail |= cudaMalloc((void **)&c->temp_minus, bn * sizeof(float)) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->input_q, bn * sizeof(float)) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->buf0, bn * sizeof(float)) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->buf1, bn * sizeof(float)) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->out, bn * sizeof(float)) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->temp_inf, batch * sizeof(float)) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->conv, batch * sizeof(float)) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->max_bits,
                     (size_t)batch * limit * sizeof(unsigned int)) !=
          cudaSuccess;
  fail |= cudaMalloc((void **)&c->group_iters, batch * sizeof(int)) !=
          cudaSuccess;
  fail |= cudaMalloc((void **)&c->done, batch * sizeof(int)) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->n_iter, batch * sizeof(int)) != cudaSuccess;
  fail |= cudaMalloc((void **)&c->result_buf, batch * sizeof(int)) !=
          cudaSuccess;
  fail |= cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking) !=
          cudaSuccess;
  fail |= cudaEventCreate(&c->ev_start) != cudaSuccess;
  fail |= cudaEventCreate(&c->ev_stop) != cudaSuccess;

  float *d_tab = NULL;
  uint8_t *d_flags = NULL;
  if (!fail) {
    // Upload the class table, then derive the env independent constants.
    size_t fb = n_classes * sizeof(float);
    fail |= cudaMalloc((void **)&d_tab, 8 * fb) != cudaSuccess;
    fail |= cudaMalloc((void **)&d_flags, n_classes) != cudaSuccess;
  }
  if (!fail) {
    size_t fb = n_classes * sizeof(float);
    const float *src[8] = {kl, kr, kt, kb, u, v, density, heat_capacity};
    for (int k = 0; k < 8 && !fail; k++)
      fail |= cudaMemcpy(d_tab + k * n_classes, src[k], fb,
                         cudaMemcpyHostToDevice) != cudaSuccess;
    fail |= cudaMemcpy(d_flags, flags, n_classes, cudaMemcpyHostToDevice) !=
            cudaSuccess;
    fail |= cudaMemcpy(c->cls, cls, c->cells, cudaMemcpyHostToDevice) !=
            cudaSuccess;
  }
  if (!fail) {
    init_class_static<<<(n_classes + 255) / 256, 256>>>(
        n_classes, z, d_tab, d_tab + n_classes, d_tab + 2 * n_classes,
        d_tab + 3 * n_classes, d_tab + 4 * n_classes, d_tab + 5 * n_classes,
        d_tab + 6 * n_classes, d_tab + 7 * n_classes, d_flags,
        c->class_static);
    fail |= cudaDeviceSynchronize() != cudaSuccess;
  }
  if (!fail) {
    // Tile flags for this fuse_k, in the launch_group grid order.
    int tile_w = REGION_W - 2 * fuse_k, tile_h = REGION_H - 2 * fuse_k;
    int tiles_x = (cols + tile_w - 1) / tile_w;
    int tiles_y = (rows + tile_h - 1) / tile_h;
    unsigned char *h_tiles = (unsigned char *)malloc((size_t)tiles_x * tiles_y);
    fail |= h_tiles == NULL;
    for (int ty = 0; !fail && ty < tiles_y; ty++) {
      for (int tx = 0; tx < tiles_x; tx++) {
        unsigned char all_ext = 1;
        for (int r = ty * tile_h; all_ext && r < rows && r < (ty + 1) * tile_h; r++)
          for (int cc = tx * tile_w; cc < cols && cc < (tx + 1) * tile_w; cc++)
            if (!(flags[cls[(int64_t)r * cols + cc]] & FLAG_EXTERIOR)) {
              all_ext = 0;
              break;
            }
        h_tiles[ty * tiles_x + tx] = all_ext;
      }
    }
    if (!fail) {
      fail |= cudaMalloc((void **)&c->tile_ext, (size_t)tiles_x * tiles_y) != cudaSuccess;
      if (!fail)
        fail |= cudaMemcpy(c->tile_ext, h_tiles, (size_t)tiles_x * tiles_y,
                           cudaMemcpyHostToDevice) != cudaSuccess;
    }
    free(h_tiles);
  }
  cudaFree(d_tab);
  cudaFree(d_flags);
  if (fail) {
    fd_destroy(c);
    return NULL;
  }
  return c;
}

float fd_last_compute_ms(void *handle) {
  return ((FdContext *)handle)->last_compute_ms;
}

// Runs one finite-difference timestep for every environment.
//
// temp_minus and input_q: batch * cells float32. temp_inf, conv: batch.
// Outputs: out (batch * cells), n_iter (batch), max_deltas (batch * limit;
// entries at or beyond n_iter are zero). Returns a cudaError_t.
int fd_step(void *handle, const float *temp_minus, const float *input_q,
            const float *temp_inf, const float *conv, float *out, int *n_iter,
            float *max_deltas) {
  FdContext *c = (FdContext *)handle;
  cudaStream_t s = c->stream;
  int B = c->batch, K = c->fuse_k, L = c->limit;
  size_t bytes = (size_t)B * c->cells * sizeof(float);
  cudaMemcpyAsync(c->temp_minus, temp_minus, bytes, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(c->input_q, input_q, bytes, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(c->temp_inf, temp_inf, B * sizeof(float),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(c->conv, conv, B * sizeof(float), cudaMemcpyHostToDevice, s);
  fd_enqueue_device(c);

  cudaMemcpyAsync(out, c->out, bytes, cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(n_iter, c->n_iter, B * sizeof(int), cudaMemcpyDeviceToHost,
                  s);
  cudaMemcpyAsync(max_deltas, c->max_bits, (size_t)B * L * sizeof(float),
                  cudaMemcpyDeviceToHost, s);
  cudaError_t err = cudaStreamSynchronize(s);
  if (err != cudaSuccess) return (int)err;
  cudaEventElapsedTime(&c->last_compute_ms, c->ev_start, c->ev_stop);
  return (int)cudaGetLastError();
}

}  // extern "C"
