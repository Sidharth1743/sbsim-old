// Compiled form of StochasticConvectionSimulator._shuffle_max_dist.
//
// Consumes the Mersenne Twister stream of Python's `random` module in the same
// order as the Python implementation (random.uniform, random.choice, and
// random.shuffle), so the swaps and the final generator state are identical.
// The caller passes the state from random.getstate() and writes it back with
// random.setstate().

#include <stdint.h>

#define MT_N 624
#define MT_M 397

typedef struct {
  uint32_t mt[MT_N];
  int64_t index;
} mt_state;

static void mt_twist(mt_state *s) {
  for (int i = 0; i < MT_N; i++) {
    uint32_t x = (s->mt[i] & 0x80000000U) | (s->mt[(i + 1) % MT_N] & 0x7fffffffU);
    uint32_t xa = x >> 1;
    if (x & 1U) xa ^= 0x9908b0dfU;
    s->mt[i] = s->mt[(i + MT_M) % MT_N] ^ xa;
  }
  s->index = 0;
}

static uint32_t mt_next(mt_state *s) {
  if (s->index >= MT_N) mt_twist(s);
  uint32_t y = s->mt[s->index++];
  y ^= y >> 11;
  y ^= (y << 7) & 0x9d2c5680U;
  y ^= (y << 15) & 0xefc60000U;
  y ^= y >> 18;
  return y;
}

// random.random(): 53-bit float from two 32-bit outputs.
static double mt_random(mt_state *s) {
  uint32_t a = mt_next(s) >> 5;
  uint32_t b = mt_next(s) >> 6;
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

// random._randbelow_with_getrandbits(n) for 0 < n < 2**32.
static uint64_t mt_randbelow(mt_state *s, uint64_t n) {
  int k = bit_length(n);
  uint64_t r = mt_next(s) >> (32 - k);
  while (r >= n) r = mt_next(s) >> (32 - k);
  return r;
}

// Runs _shuffle_max_dist for every room in order.
//
// mt: 624 state words, updated in place. index: state position, updated.
// room_offsets[r]..room_offsets[r+1] index the CVs of room r in cvs.
// cvs: flat grid index of each CV (row * ncols + col), in room list order.
// cand_offsets[c]..cand_offsets[c+1] index the candidates of cvs[c] in cands.
// cands: flat grid index of each candidate, in the Python candidate order.
// swap_a, swap_b: scratch of at least the largest room size.
void shuffle_rooms(uint32_t *mt, int64_t *index, double p, int64_t n_rooms,
                   const int64_t *room_offsets, const int64_t *cvs,
                   const int64_t *cand_offsets, const int64_t *cands,
                   float *temp, int64_t *swap_a, int64_t *swap_b) {
  mt_state s;
  for (int i = 0; i < MT_N; i++) s.mt[i] = mt[i];
  s.index = *index;

  for (int64_t r = 0; r < n_rooms; r++) {
    int64_t n_swaps = 0;
    for (int64_t c = room_offsets[r]; c < room_offsets[r + 1]; c++) {
      // random.uniform(0, 1) > p
      if (0.0 + (1.0 - 0.0) * mt_random(&s) > p) continue;
      int64_t n_cands = cand_offsets[c + 1] - cand_offsets[c];
      uint64_t j = mt_randbelow(&s, (uint64_t)n_cands);
      swap_a[n_swaps] = cvs[c];
      swap_b[n_swaps] = cands[cand_offsets[c] + j];
      n_swaps++;
    }
    // random.shuffle(swap_list)
    for (int64_t i = n_swaps - 1; i >= 1; i--) {
      int64_t j = (int64_t)mt_randbelow(&s, (uint64_t)(i + 1));
      int64_t ta = swap_a[i], tb = swap_b[i];
      swap_a[i] = swap_a[j];
      swap_b[i] = swap_b[j];
      swap_a[j] = ta;
      swap_b[j] = tb;
    }
    for (int64_t i = 0; i < n_swaps; i++) {
      float val = temp[swap_a[i]];
      temp[swap_a[i]] = temp[swap_b[i]];
      temp[swap_b[i]] = val;
    }
  }

  for (int i = 0; i < MT_N; i++) mt[i] = s.mt[i];
  *index = s.index;
}
