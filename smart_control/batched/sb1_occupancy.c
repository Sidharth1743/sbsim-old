// Batched RandomizedArrivalDepartureOccupancy queries (exact).
//
// Each environment has its own MT19937 state, copied from the occupancy
// model's numpy RandomState. numpy's legacy rand() is the 53-bit draw from
// two 32-bit outputs, the same as Python's random() (P5 in port_plan.md).
//
// occupancy_query runs, for every active environment and every zone in zone
// order, ZoneOccupant.peek for each occupant at one timestamp. That is what
// average_zone_occupancy does over a one-step window. It writes the number of
// occupants at WORK.

#include <stdint.h>

#define MT_N 624
#define MT_M 397

static void mt_twist(uint32_t *mt) {
  for (int i = 0; i < MT_N; i++) {
    uint32_t x = (mt[i] & 0x80000000U) | (mt[(i + 1) % MT_N] & 0x7fffffffU);
    uint32_t xa = x >> 1;
    if (x & 1U) xa ^= 0x9908b0dfU;
    mt[i] = mt[(i + MT_M) % MT_N] ^ xa;
  }
}

static uint32_t mt_next(uint32_t *mt, int64_t *index) {
  if (*index >= MT_N) {
    mt_twist(mt);
    *index = 0;
  }
  uint32_t y = mt[(*index)++];
  y ^= y >> 11;
  y ^= (y << 7) & 0x9d2c5680U;
  y ^= (y << 15) & 0xefc60000U;
  y ^= y >> 18;
  return y;
}

static double mt_random(uint32_t *mt, int64_t *index) {
  uint32_t a = mt_next(mt, index) >> 5;
  uint32_t b = mt_next(mt, index) >> 6;
  return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

// mt: batch * 624, index: batch, state: batch * zones * per_zone (1 = WORK).
// hour, work_day: local hour and work-day flag of the timestamp, per env.
// out: batch * zones, occupants at WORK (as double, like num_occupants).
void occupancy_query(uint32_t *mt, int64_t *index, uint8_t *state,
                     const uint8_t *active, const int64_t *hour,
                     const uint8_t *work_day, int64_t batch, int64_t zones,
                     int64_t per_zone, int64_t arrive_lo, int64_t arrive_hi,
                     int64_t depart_lo, double p_arrival, double p_departure,
                     double *out) {
  for (int64_t b = 0; b < batch; b++) {
    if (!active[b]) continue;
    uint32_t *m = mt + b * MT_N;
    int64_t *ix = index + b;
    for (int64_t z = 0; z < zones; z++) {
      double working = 0.0;
      for (int64_t o = 0; o < per_zone; o++) {
        uint8_t *s = state + (b * zones + z) * per_zone + o;
        if (!work_day[b]) {
          *s = 0;
        } else if (!*s) {
          if (!(hour[b] < arrive_lo || hour[b] > arrive_hi)) {
            if (mt_random(m, ix) < p_arrival) *s = 1;
          }
        } else {
          if (!(hour[b] < depart_lo)) {
            if (mt_random(m, ix) < p_departure) *s = 0;
          }
        }
        if (*s) working += 1.0;
      }
      out[b * zones + z] = working;
    }
  }
}
