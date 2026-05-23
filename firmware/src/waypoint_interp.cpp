#include "waypoint_interp.h"
#include <string.h>

static float interp_start[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static float interp_end[4] = {0.0f, 0.0f, 0.0f, 0.0f};
static uint32_t interp_start_us = 0;
static uint32_t interp_duration_us = 125000;  // Default to 125ms duration (8Hz command frequency)

void interp_set_targets(const float* start, const float* end, uint32_t duration_us) {
    memcpy(interp_start, start, 4 * sizeof(float));
    memcpy(interp_end,   end,   4 * sizeof(float));
    interp_start_us = micros();
    // Guard against zero-duration division
    interp_duration_us = (duration_us == 0) ? 1 : duration_us;
}

bool interp_get_current(float* out) {
    uint32_t elapsed = micros() - interp_start_us;
    if (elapsed >= interp_duration_us) {
        memcpy(out, interp_end, 4 * sizeof(float));
        return false;
    }
    float t = (float)elapsed / (float)interp_duration_us;  // Normalise to 0.0-1.0 range
    for (int i = 0; i < 4; i++) {
        out[i] = interp_start[i] + t * (interp_end[i] - interp_start[i]);
    }
    return true;
}
