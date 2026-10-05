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

// Same random draws and order as shuffle_rooms, but writes the shuffled swap
// list instead of applying it. Swaps of room r are swap_a/swap_b[
// swap_offsets[r] .. swap_offsets[r + 1]); swap_a and swap_b hold at least
// room_offsets[n_rooms] entries. Swaps depend only on the random stream and
// the plan, so applying them in order to any grid gives shuffle_rooms' result.
void generate_swaps(uint32_t *mt, int64_t *index, double p, int64_t n_rooms,
                    const int64_t *room_offsets, const int64_t *cvs,
                    const int64_t *cand_offsets, const int64_t *cands,
                    int32_t *swap_a, int32_t *swap_b, int64_t *swap_offsets) {
  mt_state s;
  for (int i = 0; i < MT_N; i++) s.mt[i] = mt[i];
  s.index = *index;

  int64_t out = 0;
  swap_offsets[0] = 0;
  for (int64_t r = 0; r < n_rooms; r++) {
    int64_t first = out;
    for (int64_t c = room_offsets[r]; c < room_offsets[r + 1]; c++) {
      if (0.0 + (1.0 - 0.0) * mt_random(&s) > p) continue;
      int64_t n_cands = cand_offsets[c + 1] - cand_offsets[c];
      uint64_t j = mt_randbelow(&s, (uint64_t)n_cands);
      swap_a[out] = (int32_t)cvs[c];
      swap_b[out] = (int32_t)cands[cand_offsets[c] + j];
      out++;
    }
    int64_t n_swaps = out - first;
    for (int64_t i = n_swaps - 1; i >= 1; i--) {
      int64_t j = (int64_t)mt_randbelow(&s, (uint64_t)(i + 1));
      int32_t ta = swap_a[first + i], tb = swap_b[first + i];
      swap_a[first + i] = swap_a[first + j];
      swap_b[first + i] = swap_b[first + j];
      swap_a[first + j] = ta;
      swap_b[first + j] = tb;
    }
    swap_offsets[r + 1] = out;
  }

  for (int i = 0; i < MT_N; i++) mt[i] = s.mt[i];
  *index = s.index;
}

// Records the tempered 32-bit words generate_swaps consumes, per room, in
// consumption order: words of room r are words[word_offsets[r] ..
// word_offsets[r + 1]). Advances the state like generate_swaps. Returns the
// number of words, or -1 if more than max_words would be needed (the state is
// then left unspecified). Used to prove that a generator consuming these words
// reproduces generate_swaps.
static int64_t rec_n, rec_max;
static uint32_t *rec_buf;

static uint32_t rec_next(mt_state *s) {
  uint32_t w = mt_next(s);
  if (rec_n < rec_max) rec_buf[rec_n] = w;
  rec_n++;
  return w;
}

static double rec_random(mt_state *s) {
  uint32_t a = rec_next(s) >> 5;
  uint32_t b = rec_next(s) >> 6;
  return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

static uint64_t rec_randbelow(mt_state *s, uint64_t n) {
  int k = bit_length(n);
  uint64_t r = rec_next(s) >> (32 - k);
  while (r >= n) r = rec_next(s) >> (32 - k);
  return r;
}

int64_t record_swap_words(uint32_t *mt, int64_t *index, double p,
                          int64_t n_rooms, const int64_t *room_offsets,
                          const int64_t *cand_offsets, uint32_t *words,
                          int64_t max_words, int64_t *word_offsets) {
  mt_state s;
  for (int i = 0; i < MT_N; i++) s.mt[i] = mt[i];
  s.index = *index;
  rec_n = 0;
  rec_max = max_words;
  rec_buf = words;
  word_offsets[0] = 0;
  for (int64_t r = 0; r < n_rooms; r++) {
    int64_t n_swaps = 0;
    for (int64_t c = room_offsets[r]; c < room_offsets[r + 1]; c++) {
      if (0.0 + (1.0 - 0.0) * rec_random(&s) > p) continue;
      rec_randbelow(&s, (uint64_t)(cand_offsets[c + 1] - cand_offsets[c]));
      n_swaps++;
    }
    for (int64_t i = n_swaps - 1; i >= 1; i--) rec_randbelow(&s, (uint64_t)(i + 1));
    word_offsets[r + 1] = rec_n;
  }
  for (int i = 0; i < MT_N; i++) mt[i] = s.mt[i];
  *index = s.index;
  return rec_n > max_words ? -1 : rec_n;
}
