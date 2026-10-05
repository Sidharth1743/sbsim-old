// GPU-resident grids for a batch of SB1 environments (phase 3 of
// port_plan.md).
//
// Holds the temperature grid and the external heat (input_q) of every
// environment on the device, and runs the per-cell work there:
//
// * the FD timestep (fd_kernel.cu, included below),
// * zone means and the global mean with numpy's float32 summation order:
//   start at 0 and add, in sequence, the pairwise sum of each consecutive
//   8192-element chunk (P1 in port_plan.md), then divide by n,
// * thermal power to diffusers: input_q = float32(q * diffuser), which is what
//   the FD reads after the original float64 assignment,
// * convection swaps, applied in order per room. Rooms are disjoint, so
//   rooms run in parallel.
//
// Compiled with the same exactness flags as fd_kernel.cu.

#include "../simulator/fd_kernel.cu"
#include <string.h>

#define MEAN_CHUNK 8192

// numpy's pairwise_sum for float32 (loops_utils.h.src), on x[start + i]
// gathered through idx (or contiguous when idx is NULL). Leaf case n <= 128.
static __device__ float np_pairwise_leaf(const float *x, const int *idx,
                                         int64_t start, int64_t n) {
#define LOAD(i) (idx ? x[idx[start + (i)]] : x[start + (i)])
  if (n < 8) {
    float res = 0.0f;
    for (int64_t i = 0; i < n; i++) res = __fadd_rn(res, LOAD(i));
    return res;
  }
  float r[8];
  for (int j = 0; j < 8; j++) r[j] = LOAD(j);
  int64_t i;
  for (i = 8; i < n - (n % 8); i += 8)
    for (int j = 0; j < 8; j++) r[j] = __fadd_rn(r[j], LOAD(i + j));
  float res = __fadd_rn(__fadd_rn(__fadd_rn(r[0], r[1]), __fadd_rn(r[2], r[3])),
                        __fadd_rn(__fadd_rn(r[4], r[5]), __fadd_rn(r[6], r[7])));
  for (; i < n; i++) res = __fadd_rn(res, LOAD(i));
  return res;
#undef LOAD
}

// The recursion of pairwise_sum unrolled by depth, so the stack size is
// static. Each level splits n into n2 = (n / 2) rounded down to a multiple of
// 8 and n - n2, exactly like numpy.
template <int DEPTH>
static __device__ float np_pairwise_t(const float *x, const int *idx,
                                      int64_t start, int64_t n) {
  if (n <= 128) return np_pairwise_leaf(x, idx, start, n);
  int64_t n2 = n / 2;
  n2 -= n2 % 8;
  return __fadd_rn(np_pairwise_t<DEPTH - 1>(x, idx, start, n2),
                   np_pairwise_t<DEPTH - 1>(x, idx, start + n2, n - n2));
}

template <>
__device__ float np_pairwise_t<0>(const float *x, const int *idx,
                                  int64_t start, int64_t n) {
  if (n > 128) __trap();  // Deeper than any chunk of <= MEAN_CHUNK needs.
  return np_pairwise_leaf(x, idx, start, n);
}

static __device__ float np_pairwise(const float *x, const int *idx,
                                    int64_t start, int64_t n) {
  return np_pairwise_t<10>(x, idx, start, n);
}

struct Sb1Gpu {
  int batch, n_zones, n_rooms;
  int64_t cells, max_swaps;
  FdContext *fd;
  float *temp, *input_q;  // batch * cells
  // Means: segments 0..n_zones-1 are zones (cells gathered in room order),
  // segment n_zones is the whole grid. Each task is one 8192-element chunk;
  // tasks are ordered by segment, then chunk.
  int *zone_cells;
  int64_t *seg_start, *seg_len;    // n_zones + 1 (start into zone_cells)
  int *task_seg;                   // n_tasks
  int64_t *task_start, *task_len;  // n_tasks (relative to the segment)
  int *seg_task_first;             // n_zones + 2
  int n_tasks;
  float *partial;  // batch * n_tasks
  float *means;    // batch * (n_zones + 1)
  // Diffusers.
  int n_dif;
  int *dif_cell, *dif_zone;
  double *dif_val;
  double *q;  // batch * n_zones
  // Swaps.
  int *swap_a, *swap_b;  // batch * max_swaps
  int64_t *swap_off;     // batch * (n_rooms + 1)
  unsigned char *active; // batch
  // Shuffle plan (StochasticConvectionSimulator._shuffle_max_dist), set by
  // sb1_set_plan: CVs of room r are cvs[room_off[r] .. room_off[r + 1]), in
  // the Python room list order; candidates of CV c are cands[cand_off[c] ..
  // cand_off[c + 1]) in the Python candidate order.
  int64_t *room_off, *cand_off;
  int *cvs, *cands;
  double p;
  // Word sources for GPU swap generation.
  uint32_t *feed;          // batch * feed_max (validation feed)
  int64_t feed_max;
  int64_t *feed_off;       // batch * (n_rooms + 1)
  uint32_t *keys;          // batch * 2 (Philox keys)
  unsigned long long *rng_step;  // batch (Philox counter, per step)
  // Fast mode: room-local index of every candidate, and rooms by size class.
  int *cand_local;
  int n_classes_fast;
  int *class_rooms[4];
  int class_count[4];
  int class_shared[4];
  // Pinned host staging for per-step inputs and outputs, so that enqueueing
  // never waits on pageable copies. When defer_sync is set, the step entry
  // points return without synchronizing; call sb1_sync before reading means.
  int defer_sync;
  float *h_temp_inf, *h_conv, *h_means;
  unsigned char *h_active;
  uint32_t *h_keys;
  unsigned long long *h_steps;
  double *h_q;
};

// Philox4x32-10 (Salmon et al., Random123).
static __device__ void philox4x32_10(const uint32_t ctr_in[4],
                                     const uint32_t key_in[2],
                                     uint32_t out[4]) {
  uint32_t c0 = ctr_in[0], c1 = ctr_in[1], c2 = ctr_in[2], c3 = ctr_in[3];
  uint32_t k0 = key_in[0], k1 = key_in[1];
  for (int i = 0; i < 10; i++) {
    uint32_t hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
    uint32_t hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
    uint32_t n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
    c0 = n0;
    c1 = lo1;
    c2 = n2;
    c3 = lo0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  out[0] = c0;
  out[1] = c1;
  out[2] = c2;
  out[3] = c3;
}

// A stream of 32-bit words: either fed words (exact MT words recorded on the
// host) or Philox with counter (block, room, step lo, step hi).
struct WordSource {
  const uint32_t *feed;
  int64_t pos;
  int philox;
  uint32_t key[2];
  uint32_t ctr[4];
  uint32_t buf[4];
  int used;
};

static __device__ uint32_t next_word(WordSource &s) {
  if (!s.philox) return s.feed[s.pos++];
  if (s.used == 4) {
    philox4x32_10(s.ctr, s.key, s.buf);
    s.ctr[0]++;
    s.used = 0;
  }
  return s.buf[s.used++];
}

// Python random.random(): 53-bit float from two words.
static __device__ double word_random(WordSource &s) {
  uint32_t a = next_word(s) >> 5;
  uint32_t b = next_word(s) >> 6;
  return __dmul_rn(__dadd_rn(__dmul_rn((double)a, 67108864.0), (double)b),
                   1.0 / 9007199254740992.0);
}

static __device__ int bit_length64(uint64_t n) {
  int k = 0;
  while (n) {
    k++;
    n >>= 1;
  }
  return k;
}

// Python random._randbelow_with_getrandbits(n) for 0 < n < 2**32.
static __device__ uint64_t word_randbelow(WordSource &s, uint64_t n) {
  int k = bit_length64(n);
  uint64_t r = next_word(s) >> (32 - k);
  while (r >= n) r = next_word(s) >> (32 - k);
  return r;
}

// For each (env, room): build the swap list exactly as _shuffle_max_dist
// does from the word stream, shuffle it (random.shuffle), and apply it in
// order. The room's scratch is its own CV range of swap_a/swap_b, since a
// room has at most one swap per CV.
__global__ void gen_apply_swaps(Sb1Gpu g, int philox) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (r >= g.n_rooms || !g.active[b]) return;
  WordSource s;
  s.philox = philox;
  s.used = 4;
  if (philox) {
    s.feed = NULL;
    s.pos = 0;
    s.key[0] = g.keys[2 * b];
    s.key[1] = g.keys[2 * b + 1];
    unsigned long long step = g.rng_step[b];
    s.ctr[0] = 0;
    s.ctr[1] = (uint32_t)r;
    s.ctr[2] = (uint32_t)(step & 0xffffffffull);
    s.ctr[3] = (uint32_t)(step >> 32);
  } else {
    s.feed = g.feed + (int64_t)b * g.feed_max;
    s.pos = g.feed_off[(int64_t)b * (g.n_rooms + 1) + r];
  }
  int64_t base = g.room_off[r];
  int *sa = g.swap_a + (int64_t)b * g.max_swaps + base;
  int *sb = g.swap_b + (int64_t)b * g.max_swaps + base;
  int64_t n = 0;
  for (int64_t c = g.room_off[r]; c < g.room_off[r + 1]; c++) {
    if (word_random(s) > g.p) continue;
    int64_t n_cands = g.cand_off[c + 1] - g.cand_off[c];
    uint64_t j = word_randbelow(s, (uint64_t)n_cands);
    sa[n] = g.cvs[c];
    sb[n] = g.cands[g.cand_off[c] + j];
    n++;
  }
  for (int64_t i = n - 1; i >= 1; i--) {
    int64_t j = (int64_t)word_randbelow(s, (uint64_t)(i + 1));
    int ta = sa[i], tb = sb[i];
    sa[i] = sa[j];
    sb[i] = sb[j];
    sa[j] = ta;
    sb[j] = tb;
  }
  float *x = g.temp + (int64_t)b * g.cells;
  for (int64_t i = 0; i < n; i++) {
    float v = x[sa[i]];
    x[sa[i]] = x[sb[i]];
    x[sb[i]] = v;
  }
}

__global__ void philox_test_kernel(const uint32_t *ctr, const uint32_t *key,
                                   uint32_t *out) {
  philox4x32_10(ctr, key, out);
}

// ------------------------------------------------------------ fast mode
// Word mapping of sb1_fast_ref.c (the CPU reference): CV with plan index c
// uses key (k0, k1 ^ s_hi), counter (c, t, s_lo, 0); Fisher-Yates index i of
// room r uses key (k0 ^ 0x6A09E667, k1 ^ 0xBB67AE85 ^ s_hi), counter
// (i, r, t, s_lo). The block counter t sits at ctr[tidx].
struct FastStream {
  uint32_t key[2];
  uint32_t ctr[4];
  int tidx;
  uint32_t buf[4];
  int used;
};

static __device__ uint32_t fast_next(FastStream &s) {
  if (s.used == 4) {
    philox4x32_10(s.ctr, s.key, s.buf);
    s.ctr[s.tidx]++;
    s.used = 0;
  }
  return s.buf[s.used++];
}

static __device__ double fast_random(FastStream &s) {
  uint32_t a = fast_next(s) >> 5;
  uint32_t b = fast_next(s) >> 6;
  return __dmul_rn(__dadd_rn(__dmul_rn((double)a, 67108864.0), (double)b),
                   1.0 / 9007199254740992.0);
}

static __device__ uint64_t fast_randbelow(FastStream &s, uint64_t n) {
  int k = bit_length64(n);
  uint64_t r = fast_next(s) >> (32 - k);
  while (r >= n) r = fast_next(s) >> (32 - k);
  return r;
}

#define FAST_RING 1024

// One warp per (env, room). Shared memory: list[m], perm[m], ring[FAST_RING]
// (uint16; rooms have fewer than 65536 CVs).
//
// The swaps of the reference are applied to a permutation of the room's cells
// instead of to the temperatures: swapping the values of cells a and b is
// swapping perm[a] and perm[b], where the value at cell x is old[perm[x]].
// The final gather writes old[perm[x]] to every cell, which is the same grid.
__global__ void fast_swaps(Sb1Gpu g, const int *rooms, int n_class_rooms) {
  int ri = blockIdx.x;
  int b = blockIdx.y;
  if (ri >= n_class_rooms || !g.active[b]) return;
  int r = rooms[ri];
  extern __shared__ unsigned short shm[];
  const int lane = threadIdx.x;
  const int64_t base = g.room_off[r];
  const int m = (int)(g.room_off[r + 1] - base);
  unsigned short *list = shm;
  unsigned short *perm = shm + m;
  unsigned short *ring = shm + 2 * m;
  float *x = g.temp + (int64_t)b * g.cells;
  int *b_local = g.swap_b + (int64_t)b * g.max_swaps + base;
  float *old = (float *)(g.swap_a + (int64_t)b * g.max_swaps + base);
  const uint32_t k0 = g.keys[2 * b], k1 = g.keys[2 * b + 1];
  const unsigned long long step = g.rng_step[b];
  const uint32_t s_lo = (uint32_t)(step & 0xffffffffull);
  const uint32_t s_hi = (uint32_t)(step >> 32);

  // Phase A: inclusion and candidate per CV, in CV order (ballot keeps the
  // order of the reference's sequential loop).
  int n = 0;
  for (int start = 0; start < m; start += 32) {
    int i = start + lane;
    bool include = false;
    if (i < m) {
      int64_t c = base + i;
      FastStream s;
      s.key[0] = k0;
      s.key[1] = k1 ^ s_hi;
      s.ctr[0] = (uint32_t)c;
      s.ctr[1] = 0;
      s.ctr[2] = s_lo;
      s.ctr[3] = 0;
      s.tidx = 1;
      s.used = 4;
      include = !(fast_random(s) > g.p);
      if (include) {
        int64_t n_cands = g.cand_off[c + 1] - g.cand_off[c];
        uint64_t j = fast_randbelow(s, (uint64_t)n_cands);
        b_local[i] = g.cand_local[g.cand_off[c] + j];
      }
      old[i] = x[g.cvs[c]];
      perm[i] = (unsigned short)i;
    }
    unsigned int mask = __ballot_sync(0xffffffffu, include);
    if (include) list[n + __popc(mask & ((1u << lane) - 1u))] = (unsigned short)i;
    n += __popc(mask);
  }
  __syncwarp();

  // Fisher-Yates on list: j_i depends only on i, so lanes draw a chunk of
  // j's, then lane 0 performs that chunk's swaps in descending i.
  for (int hi = n - 1; hi >= 1; hi -= FAST_RING) {
    int lo = hi - FAST_RING + 1;
    if (lo < 1) lo = 1;
    for (int i = hi - lane; i >= lo; i -= 32) {
      FastStream s;
      s.key[0] = k0 ^ 0x6A09E667u;
      s.key[1] = k1 ^ 0xBB67AE85u ^ s_hi;
      s.ctr[0] = (uint32_t)i;
      s.ctr[1] = (uint32_t)r;
      s.ctr[2] = 0;
      s.ctr[3] = s_lo;
      s.tidx = 2;
      s.used = 4;
      ring[hi - i] = (unsigned short)fast_randbelow(s, (uint64_t)(i + 1));
    }
    __syncwarp();
    if (lane == 0) {
      for (int i = hi; i >= lo; i--) {
        int j = ring[hi - i];
        unsigned short t = list[i];
        list[i] = list[j];
        list[j] = t;
      }
    }
    __syncwarp();
  }

  // Apply the swaps in list order to perm; lanes prefetch the partners.
  for (int start = 0; start < n; start += FAST_RING) {
    int end = start + FAST_RING < n ? start + FAST_RING : n;
    for (int k = start + lane; k < end; k += 32)
      ring[k - start] = (unsigned short)b_local[list[k]];
    __syncwarp();
    if (lane == 0) {
      for (int k = start; k < end; k++) {
        int a = list[k], c = ring[k - start];
        unsigned short t = perm[a];
        perm[a] = perm[c];
        perm[c] = t;
      }
    }
    __syncwarp();
  }

  for (int i = lane; i < m; i += 32) x[g.cvs[base + i]] = old[perm[i]];
}

__global__ void mean_chunks(Sb1Gpu g) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (t >= g.n_tasks) return;
  const float *x = g.temp + (int64_t)b * g.cells;
  int seg = g.task_seg[t];
  float s;
  if (seg < g.n_zones)
    s = np_pairwise(x, g.zone_cells, g.seg_start[seg] + g.task_start[t],
                    g.task_len[t]);
  else
    s = np_pairwise(x, NULL, g.task_start[t], g.task_len[t]);
  g.partial[(int64_t)b * g.n_tasks + t] = s;
}

__global__ void mean_combine(Sb1Gpu g) {
  int seg = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (seg > g.n_zones) return;
  float acc = 0.0f;
  for (int t = g.seg_task_first[seg]; t < g.seg_task_first[seg + 1]; t++)
    acc = __fadd_rn(acc, g.partial[(int64_t)b * g.n_tasks + t]);
  // n < 2^24, so float32 and float64 division then rounding agree.
  g.means[(int64_t)b * (g.n_zones + 1) + seg] =
      __fdiv_rn(acc, __ll2float_rn(g.seg_len[seg]));
}

__global__ void set_thermal_power(Sb1Gpu g) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (i >= g.n_dif || !g.active[b]) return;
  double q = g.q[(int64_t)b * g.n_zones + g.dif_zone[i]];
  g.input_q[(int64_t)b * g.cells + g.dif_cell[i]] =
      __double2float_rn(__dmul_rn(q, g.dif_val[i]));
}

__global__ void apply_swaps(Sb1Gpu g) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (r >= g.n_rooms || !g.active[b]) return;
  float *x = g.temp + (int64_t)b * g.cells;
  const int64_t *off = g.swap_off + (int64_t)b * (g.n_rooms + 1);
  const int *sa = g.swap_a + (int64_t)b * g.max_swaps;
  const int *sb = g.swap_b + (int64_t)b * g.max_swaps;
  for (int64_t i = off[r]; i < off[r + 1]; i++) {
    float v = x[sa[i]];
    x[sa[i]] = x[sb[i]];
    x[sb[i]] = v;
  }
}

extern "C" {

void sb1_destroy(void *handle) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  if (!g) return;
  void *ptrs[] = {g->temp,      g->input_q,   g->zone_cells, g->seg_start,
                  g->seg_len,   g->task_seg,  g->task_start, g->task_len,
                  g->seg_task_first, g->partial, g->means,  g->dif_cell,
                  g->dif_zone,  g->dif_val,   g->q,          g->swap_a,
                  g->swap_b,    g->swap_off,  g->active,   g->room_off,
                  g->cand_off,  g->cvs,       g->cands,      g->feed,
                  g->feed_off,  g->keys,      g->rng_step,   g->cand_local,
                  g->class_rooms[0], g->class_rooms[1], g->class_rooms[2],
                  g->class_rooms[3]};
  for (size_t k = 0; k < sizeof(ptrs) / sizeof(ptrs[0]); k++) cudaFree(ptrs[k]);
  void *pinned[] = {g->h_temp_inf, g->h_conv, g->h_means, g->h_active,
                    g->h_keys, g->h_steps, g->h_q};
  for (size_t k = 0; k < sizeof(pinned) / sizeof(pinned[0]); k++)
    if (pinned[k]) cudaFreeHost(pinned[k]);
  free(g);
}

static int up(void **dst, const void *src, size_t bytes) {
  if (cudaMalloc(dst, bytes ? bytes : 1) != cudaSuccess) return 1;
  if (bytes && cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice) !=
                   cudaSuccess)
    return 1;
  return 0;
}

// fd_handle: an fd_create() context with the same batch and grid. Ownership
// stays with the caller. Returns NULL on failure.
void *sb1_create(void *fd_handle, int n_zones, const int *zone_cells,
                 const int64_t *seg_start, const int64_t *seg_len, int n_tasks,
                 const int *task_seg, const int64_t *task_start,
                 const int64_t *task_len, const int *seg_task_first, int n_dif,
                 const int *dif_cell, const int *dif_zone,
                 const double *dif_val, int n_rooms, int64_t max_swaps) {
  FdContext *fd = (FdContext *)fd_handle;
  Sb1Gpu *g = (Sb1Gpu *)calloc(1, sizeof(Sb1Gpu));
  if (!g) return NULL;
  g->fd = fd;
  g->batch = fd->batch;
  g->cells = fd->cells;
  g->n_zones = n_zones;
  g->n_tasks = n_tasks;
  g->n_dif = n_dif;
  g->n_rooms = n_rooms;
  g->max_swaps = max_swaps;
  int64_t B = g->batch, N = g->cells;
  int fail = 0;
  fail |= cudaMalloc((void **)&g->temp, B * N * sizeof(float)) != cudaSuccess;
  fail |= cudaMalloc((void **)&g->input_q, B * N * sizeof(float)) != cudaSuccess;
  if (!fail) fail |= cudaMemset(g->input_q, 0, B * N * sizeof(float)) != cudaSuccess;
  fail |= up((void **)&g->zone_cells, zone_cells,
             (size_t)(seg_start[n_zones - 1] + seg_len[n_zones - 1]) * sizeof(int));
  fail |= up((void **)&g->seg_start, seg_start, (n_zones + 1) * sizeof(int64_t));
  fail |= up((void **)&g->seg_len, seg_len, (n_zones + 1) * sizeof(int64_t));
  fail |= up((void **)&g->task_seg, task_seg, n_tasks * sizeof(int));
  fail |= up((void **)&g->task_start, task_start, n_tasks * sizeof(int64_t));
  fail |= up((void **)&g->task_len, task_len, n_tasks * sizeof(int64_t));
  fail |= up((void **)&g->seg_task_first, seg_task_first, (n_zones + 2) * sizeof(int));
  fail |= cudaMalloc((void **)&g->partial, B * n_tasks * sizeof(float)) != cudaSuccess;
  fail |= cudaMalloc((void **)&g->means, B * (n_zones + 1) * sizeof(float)) != cudaSuccess;
  fail |= up((void **)&g->dif_cell, dif_cell, n_dif * sizeof(int));
  fail |= up((void **)&g->dif_zone, dif_zone, n_dif * sizeof(int));
  fail |= up((void **)&g->dif_val, dif_val, n_dif * sizeof(double));
  fail |= cudaMalloc((void **)&g->q, B * n_zones * sizeof(double)) != cudaSuccess;
  fail |= cudaMalloc((void **)&g->swap_a, B * max_swaps * sizeof(int)) != cudaSuccess;
  fail |= cudaMalloc((void **)&g->swap_b, B * max_swaps * sizeof(int)) != cudaSuccess;
  fail |= cudaMalloc((void **)&g->swap_off, B * (n_rooms + 1) * sizeof(int64_t)) != cudaSuccess;
  fail |= cudaMalloc((void **)&g->active, B) != cudaSuccess;
  fail |= cudaMallocHost((void **)&g->h_temp_inf, B * sizeof(float)) != cudaSuccess;
  fail |= cudaMallocHost((void **)&g->h_conv, B * sizeof(float)) != cudaSuccess;
  fail |= cudaMallocHost((void **)&g->h_means, B * (n_zones + 1) * sizeof(float)) != cudaSuccess;
  fail |= cudaMallocHost((void **)&g->h_active, B) != cudaSuccess;
  fail |= cudaMallocHost((void **)&g->h_keys, B * 2 * sizeof(uint32_t)) != cudaSuccess;
  fail |= cudaMallocHost((void **)&g->h_steps, B * sizeof(unsigned long long)) != cudaSuccess;
  fail |= cudaMallocHost((void **)&g->h_q, B * n_zones * sizeof(double)) != cudaSuccess;
  if (fail) {
    sb1_destroy(g);
    return NULL;
  }
  return g;
}

// Uploads a float32 grid for environment b and zeroes its input_q.
int sb1_set_env(void *handle, int b, const float *grid) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  cudaStream_t s = g->fd->stream;
  cudaMemcpyAsync(g->temp + (int64_t)b * g->cells, grid, g->cells * sizeof(float),
                  cudaMemcpyHostToDevice, s);
  cudaMemsetAsync(g->input_q + (int64_t)b * g->cells, 0, g->cells * sizeof(float), s);
  return (int)cudaStreamSynchronize(s);
}

int sb1_get_env(void *handle, int b, float *grid) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  cudaStream_t s = g->fd->stream;
  cudaMemcpyAsync(grid, g->temp + (int64_t)b * g->cells, g->cells * sizeof(float),
                  cudaMemcpyDeviceToHost, s);
  return (int)cudaStreamSynchronize(s);
}

// FD timestep for every environment; only active ones keep the result.
int sb1_fd(void *handle, const float *temp_inf, const float *conv,
           const unsigned char *active) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  FdContext *c = g->fd;
  cudaStream_t s = c->stream;
  size_t bytes = (size_t)g->batch * g->cells * sizeof(float);
  memcpy(g->h_temp_inf, temp_inf, g->batch * sizeof(float));
  memcpy(g->h_conv, conv, g->batch * sizeof(float));
  cudaMemcpyAsync(c->temp_minus, g->temp, bytes, cudaMemcpyDeviceToDevice, s);
  cudaMemcpyAsync(c->input_q, g->input_q, bytes, cudaMemcpyDeviceToDevice, s);
  cudaMemcpyAsync(c->temp_inf, g->h_temp_inf, g->batch * sizeof(float),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(c->conv, g->h_conv, g->batch * sizeof(float),
                  cudaMemcpyHostToDevice, s);
  fd_enqueue_device(c);
  for (int b = 0; b < g->batch; b++)
    if (active[b])
      cudaMemcpyAsync(g->temp + (int64_t)b * g->cells,
                      c->out + (int64_t)b * g->cells, g->cells * sizeof(float),
                      cudaMemcpyDeviceToDevice, s);
  return g->defer_sync ? (int)cudaGetLastError() : (int)cudaStreamSynchronize(s);
}

// Applies swap lists (host arrays, layout as in Sb1Gpu) to active envs.
int sb1_swaps(void *handle, const int *swap_a, const int *swap_b,
              const int64_t *swap_off, const unsigned char *active) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  cudaStream_t s = g->fd->stream;
  int64_t B = g->batch;
  cudaMemcpyAsync(g->active, active, B, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(g->swap_off, swap_off, B * (g->n_rooms + 1) * sizeof(int64_t),
                  cudaMemcpyHostToDevice, s);
  for (int64_t b = 0; b < B; b++) {
    if (!active[b]) continue;
    int64_t n = swap_off[b * (g->n_rooms + 1) + g->n_rooms];
    cudaMemcpyAsync(g->swap_a + b * g->max_swaps, swap_a + b * g->max_swaps,
                    n * sizeof(int), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(g->swap_b + b * g->max_swaps, swap_b + b * g->max_swaps,
                    n * sizeof(int), cudaMemcpyHostToDevice, s);
  }
  dim3 grid((g->n_rooms + 63) / 64, (unsigned)B);
  apply_swaps<<<grid, 64, 0, s>>>(*g);
  return (int)cudaStreamSynchronize(s);
}

// Zone means and global mean of every env: out is batch * (n_zones + 1).
int sb1_means(void *handle, float *out) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  cudaStream_t s = g->fd->stream;
  dim3 g1((g->n_tasks + 63) / 64, g->batch);
  mean_chunks<<<g1, 64, 0, s>>>(*g);
  dim3 g2((g->n_zones + 1 + 63) / 64, g->batch);
  mean_combine<<<g2, 64, 0, s>>>(*g);
  size_t bytes = (size_t)g->batch * (g->n_zones + 1) * sizeof(float);
  cudaMemcpyAsync(g->h_means, g->means, bytes, cudaMemcpyDeviceToHost, s);
  if (g->defer_sync) return (int)cudaGetLastError();
  int err = (int)cudaStreamSynchronize(s);
  if (!err) memcpy(out, g->h_means, bytes);
  return err;
}

// Sets input_q of active envs from per-zone thermal power q (batch * zones).
int sb1_thermal_power(void *handle, const double *q, const unsigned char *active) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  cudaStream_t s = g->fd->stream;
  memcpy(g->h_active, active, g->batch);
  memcpy(g->h_q, q, (size_t)g->batch * g->n_zones * sizeof(double));
  cudaMemcpyAsync(g->active, g->h_active, g->batch, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(g->q, g->h_q, (size_t)g->batch * g->n_zones * sizeof(double),
                  cudaMemcpyHostToDevice, s);
  dim3 grid((g->n_dif + 255) / 256, g->batch);
  set_thermal_power<<<grid, 256, 0, s>>>(*g);
  return g->defer_sync ? (int)cudaGetLastError() : (int)cudaStreamSynchronize(s);
}

// Shuffle plan (see Sb1Gpu). Must be set before swap generation on the GPU.
int sb1_set_plan(void *handle, const int64_t *room_off, int64_t n_cvs,
                 const int *cvs, const int64_t *cand_off, int64_t n_cands,
                 const int *cands, double p) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  int fail = 0;
  fail |= up((void **)&g->room_off, room_off, (g->n_rooms + 1) * sizeof(int64_t));
  fail |= up((void **)&g->cvs, cvs, n_cvs * sizeof(int));
  fail |= up((void **)&g->cand_off, cand_off, (n_cvs + 1) * sizeof(int64_t));
  fail |= up((void **)&g->cands, cands, n_cands * sizeof(int));
  fail |= cudaMalloc((void **)&g->keys, g->batch * 2 * sizeof(uint32_t)) != cudaSuccess;
  fail |= cudaMalloc((void **)&g->rng_step, g->batch * sizeof(unsigned long long)) != cudaSuccess;
  g->p = p;
  return fail ? (int)cudaErrorMemoryAllocation : 0;
}

// Generates and applies swaps on the GPU from fed words (validation).
// words: batch * feed_max; feed_off: batch * (n_rooms + 1).
int sb1_swaps_feed(void *handle, const uint32_t *words, int64_t feed_max,
                   const int64_t *feed_off, const unsigned char *active) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  cudaStream_t s = g->fd->stream;
  if (g->feed_max < feed_max) {
    cudaFree(g->feed);
    cudaFree(g->feed_off);
    g->feed = NULL;
    g->feed_off = NULL;
    if (cudaMalloc((void **)&g->feed, g->batch * feed_max * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **)&g->feed_off, g->batch * (g->n_rooms + 1) * sizeof(int64_t)) != cudaSuccess)
      return (int)cudaErrorMemoryAllocation;
    g->feed_max = feed_max;
  }
  // The device rows use g->feed_max as their stride.
  for (int64_t b = 0; b < g->batch; b++) {
    if (!active[b]) continue;
    int64_t n = feed_off[b * (g->n_rooms + 1) + g->n_rooms];
    cudaMemcpyAsync(g->feed + b * g->feed_max, words + b * feed_max,
                    n * sizeof(uint32_t), cudaMemcpyHostToDevice, s);
  }
  cudaMemcpyAsync(g->feed_off, feed_off, g->batch * (g->n_rooms + 1) * sizeof(int64_t),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(g->active, active, g->batch, cudaMemcpyHostToDevice, s);
  dim3 grid((g->n_rooms + 31) / 32, g->batch);
  gen_apply_swaps<<<grid, 32, 0, s>>>(*g, 0);
  return (int)cudaStreamSynchronize(s);
}

// Generates and applies swaps on the GPU from per-env Philox streams.
// keys: batch * 2; steps: batch (a fresh counter per call and env).
int sb1_swaps_philox(void *handle, const uint32_t *keys,
                     const unsigned long long *steps,
                     const unsigned char *active) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  cudaStream_t s = g->fd->stream;
  cudaMemcpyAsync(g->keys, keys, g->batch * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(g->rng_step, steps, g->batch * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(g->active, active, g->batch, cudaMemcpyHostToDevice, s);
  dim3 grid((g->n_rooms + 31) / 32, g->batch);
  gen_apply_swaps<<<grid, 32, 0, s>>>(*g, 1);
  return (int)cudaStreamSynchronize(s);
}

int sb1_philox_test(const uint32_t *ctr, const uint32_t *key, uint32_t *out) {
  uint32_t *d;
  if (cudaMalloc((void **)&d, 10 * sizeof(uint32_t)) != cudaSuccess) return 1;
  cudaMemcpy(d, ctr, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d + 4, key, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
  philox_test_kernel<<<1, 1>>>(d, d + 4, d + 6);
  cudaMemcpy(out, d + 6, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d);
  return (int)cudaGetLastError();
}

// Fast-mode plan: room-local index of every candidate (n_cands) and up to 4
// size classes of rooms; class k has class_count[k] rooms whose sizes are at
// most class_max_cvs[k] (< 65536).
int sb1_set_fast_plan(void *handle, const int *cand_local, int64_t n_cands,
                      int n_classes, const int *class_rooms_flat,
                      const int *class_count, const int *class_max_cvs) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  if (n_classes < 1 || n_classes > 4) return (int)cudaErrorInvalidValue;
  int fail = up((void **)&g->cand_local, cand_local, n_cands * sizeof(int));
  int offset = 0, max_shared = 0;
  for (int k = 0; k < n_classes; k++) {
    if (class_max_cvs[k] >= 65536) return (int)cudaErrorInvalidValue;
    fail |= up((void **)&g->class_rooms[k], class_rooms_flat + offset,
               class_count[k] * sizeof(int));
    g->class_count[k] = class_count[k];
    g->class_shared[k] =
        (int)((2 * (int64_t)class_max_cvs[k] + FAST_RING) * sizeof(unsigned short));
    if (g->class_shared[k] > max_shared) max_shared = g->class_shared[k];
    offset += class_count[k];
  }
  g->n_classes_fast = n_classes;
  if (cudaFuncSetAttribute(fast_swaps, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           max_shared) != cudaSuccess)
    return (int)cudaErrorInvalidValue;
  return fail ? (int)cudaErrorMemoryAllocation : 0;
}

// Fast-mode swaps (shared-memory kernel), same words as sb1_fast_ref.c.
int sb1_swaps_fast(void *handle, const uint32_t *keys,
                   const unsigned long long *steps,
                   const unsigned char *active) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  cudaStream_t s = g->fd->stream;
  memcpy(g->h_keys, keys, g->batch * 2 * sizeof(uint32_t));
  memcpy(g->h_steps, steps, g->batch * sizeof(unsigned long long));
  memcpy(g->h_active, active, g->batch);
  cudaMemcpyAsync(g->keys, g->h_keys, g->batch * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(g->rng_step, g->h_steps, g->batch * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(g->active, g->h_active, g->batch, cudaMemcpyHostToDevice, s);
  for (int k = 0; k < g->n_classes_fast; k++) {
    if (g->class_count[k] == 0) continue;
    dim3 grid(g->class_count[k], g->batch);
    fast_swaps<<<grid, 32, g->class_shared[k], s>>>(*g, g->class_rooms[k],
                                                    g->class_count[k]);
  }
  return g->defer_sync ? (int)cudaGetLastError() : (int)cudaStreamSynchronize(s);
}

// Deferred synchronization for overlapping several contexts' work.
void sb1_set_defer_sync(void *handle, int defer) {
  ((Sb1Gpu *)handle)->defer_sync = defer;
}

// Waits for this context's stream; copies the last means into out (may be
// NULL).
int sb1_sync(void *handle, float *out) {
  Sb1Gpu *g = (Sb1Gpu *)handle;
  int err = (int)cudaStreamSynchronize(g->fd->stream);
  if (!err && out)
    memcpy(out, g->h_means, (size_t)g->batch * (g->n_zones + 1) * sizeof(float));
  return err;
}

}  // extern "C"
