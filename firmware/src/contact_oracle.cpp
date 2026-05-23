#include "contact_oracle.h"
#include "config.h"
#include <string.h>
#include <math.h>

static float gyro_buffer[CONTACT_WINDOW];  // Ring buffer of gyro magnitude samples
static int   buf_head = 0;
static int   buf_count = 0;
static float rms_sum_sq = 0.0f;           // Running sum of squares for RMS
static float threshold = 0.0f;
static bool  triggered = false;
static bool  event_flag = false;
static float current_rms = 0.0f;

void contact_oracle_init(float threshold_dps) {
    threshold = threshold_dps;
    memset(gyro_buffer, 0, sizeof(gyro_buffer));
    buf_head = 0;
    buf_count = 0;
    rms_sum_sq = 0.0f;
    triggered = false;
    event_flag = false;
    current_rms = 0.0f;
}

void contact_oracle_push(float gx, float gy, float gz) {
    if (triggered) {
        return;  // Latched state until explicitly reset (e.g. transitioning to REACH)
    }

    float mag = sqrtf(gx * gx + gy * gy + gz * gz);

    // If buffer is already full, subtract the oldest sample's squared magnitude
    if (buf_count == CONTACT_WINDOW) {
        rms_sum_sq -= gyro_buffer[buf_head] * gyro_buffer[buf_head];
    } else {
        buf_count++;
    }

    // Add new magnitude and sum its square
    gyro_buffer[buf_head] = mag;
    rms_sum_sq += mag * mag;
    buf_head = (buf_head + 1) % CONTACT_WINDOW;

    // Only compute RMS and check threshold once buffer is completely populated
    if (buf_count == CONTACT_WINDOW) {
        current_rms = sqrtf(rms_sum_sq / CONTACT_WINDOW);
        if (current_rms > threshold) {
            triggered = true;
            event_flag = true;  // Mark rising-edge event flag
        }
    }
}

bool contact_oracle_triggered() {
    return triggered;
}

bool contact_oracle_event() {
    bool e = event_flag;
    event_flag = false;  // Reset event flag on read (single-cycle pulse)
    return e;
}

float contact_oracle_rms() {
    return current_rms;
}

void contact_oracle_reset() {
    triggered = false;
    event_flag = false;
    buf_count = 0;
    buf_head = 0;
    rms_sum_sq = 0.0f;
    current_rms = 0.0f;
    memset(gyro_buffer, 0, sizeof(gyro_buffer));
}
