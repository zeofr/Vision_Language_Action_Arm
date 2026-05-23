#pragma once
#include <Arduino.h>

// Call this when a new RPi5 command arrives
// start[]: current logical positions [J0, J1, J2, J3 gripper]
// end[]: target logical positions [J0, J1, J2, J3 gripper]
void interp_set_targets(const float* start, const float* end, uint32_t duration_us);

// Call every 20ms. Writes interpolated logical positions into out[4].
// Returns true while interpolation is in progress, false when target reached.
bool interp_get_current(float* out);
