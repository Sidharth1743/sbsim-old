// CPU reference for the fast-mode convection shuffle (port_plan.md phase 4b).
//
// Same rule as StochasticConvectionSimulator._shuffle_max_dist followed by
// random.shuffle and in-order application (see generate_swaps in
// convection_shuffle.c). Only the source of the 32-bit words differs:
//
// * CV with plan index c draws its words (uniform test, then candidate) from
//   Philox4x32-10 with key (k0, k1 ^ s_hi) and counter (c, t, s_lo, 0),
//   t = 0, 1, ...
// * Fisher-Yates draw for list index i of room r uses key
//   (k0 ^ 0x6A09E667, k1 ^ 0xBB67AE85 ^ s_hi) and counter (i, r, t, s_lo).
//
// Words of one counter are consumed in order 0..3, then t increments.
// The GPU kernel (sb1_gpu.cu fast_swaps) must match this bit for bit.

#include <stdint.h>
#include <stdlib.h>

static void philox(const uint32_t ctr_in[4], const uint32_t key_in[2],
                   uint32_t out[4]) {
  uint32_t c0 = ctr_in[0], c1 = ctr_in[1], c2 = ctr_in[2], c3 = ctr_in[3];
  uint32_t k0 = key_in[0], k1 = key_in[1];
  for (int i = 0; i < 10; i++) {
    uint64_t p0 = (uint64_t)0xD2511F53u * c0;
    uint64_t p1 = (uint64_t)0xCD9E8D57u * c2;
    uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0;
    uint32_t n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
    c0 = n0;
    c1 = (uint32_t)p1;
    c2 = n2;
    c3 = (uint32_t)p0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  out[0] = c0;
  out[1] = c1;
  out[2] = c2;
  out[3] = c3;
}

typedef struct {
  uint32_t key[2];
  uint32_t ctr[4];  // ctr[tidx] is the block counter t
  int tidx;
  uint32_t buf[4];
  int used;
} stream_t;

static uint32_t next_word(stream_t *s) {
  if (s->used == 4) {
    philox(s->ctr, s->key, s->buf);
    s->ctr[s->tidx]++;
    s->used = 0;
  }
  return s->buf[s->used++];
}

static double word_random(stream_t *s) {
  uint32_t a = next_word(s) >> 5;
  uint32_t b = next_word(s) >> 6;
  return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

static int bit_length(uint64_t n) {
  int k = 0;
  while (n) {
    k++;
    n >>= 1;
  }
  return k;
}

static uint64_t word_randbelow(stream_t *s, uint64_t n) {
  int k = bit_length(n);
  uint64_t r = next_word(s) >> (32 - k);
  while (r >= n) r = next_word(s) >> (32 - k);
  return r;
}

// Applies one fast-mode shuffle step to temp (float32 grid), in place.
// swap_a/swap_b: scratch of at least room_offsets[n_rooms] entries.
void fast_swaps_reference(uint32_t k0, uint32_t k1, uint64_t step, double p,
                          int64_t n_rooms, const int64_t *room_offsets,
                          const int64_t *cvs, const int64_t *cand_offsets,
                          const int64_t *cands, float *temp, int64_t *swap_a,
                          int64_t *swap_b) {
  uint32_t s_lo = (uint32_t)(step & 0xffffffffu);
  uint32_t s_hi = (uint32_t)(step >> 32);
  for (int64_t r = 0; r < n_rooms; r++) {
    int64_t n = 0;
    for (int64_t c = room_offsets[r]; c < room_offsets[r + 1]; c++) {
      stream_t s = {{k0, k1 ^ s_hi}, {(uint32_t)c, 0, s_lo, 0}, 1, {0}, 4};
      if (0.0 + (1.0 - 0.0) * word_random(&s) > p) continue;
      int64_t n_cands = cand_offsets[c + 1] - cand_offsets[c];
      uint64_t j = word_randbelow(&s, (uint64_t)n_cands);
      swap_a[n] = cvs[c];
      swap_b[n] = cands[cand_offsets[c] + j];
      n++;
    }
    for (int64_t i = n - 1; i >= 1; i--) {
      stream_t s = {{k0 ^ 0x6A09E667u, k1 ^ 0xBB67AE85u ^ s_hi},
                    {(uint32_t)i, (uint32_t)r, 0, s_lo}, 2, {0}, 4};
      int64_t j = (int64_t)word_randbelow(&s, (uint64_t)(i + 1));
      int64_t ta = swap_a[i], tb = swap_b[i];
      swap_a[i] = swap_a[j];
      swap_b[i] = swap_b[j];
      swap_a[j] = ta;
      swap_b[j] = tb;
    }
    for (int64_t i = 0; i < n; i++) {
      float v = temp[swap_a[i]];
      temp[swap_a[i]] = temp[swap_b[i]];
      temp[swap_b[i]] = v;
    }
  }
}
