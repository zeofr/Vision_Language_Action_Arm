#pragma once
#include <Arduino.h>

struct ToFFrame {
    uint16_t distances_mm[64];  // 8x8 grid, row-major
    uint32_t capture_timestamp_us;
    uint8_t  valid;             // 1 if all center zones have valid status
};

void tof_init();                         // Initializes VL53L5CX in 8x8 at 15Hz
bool tof_check_ready();                  // Polls INT pin, returns true if new frame ready
ToFFrame tof_get_latest();               // Returns last captured frame
