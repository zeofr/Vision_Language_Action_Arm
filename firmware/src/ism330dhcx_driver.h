#pragma once
#include <Arduino.h>

// Raw IMU data (physical units)
struct ImuData {
    float gx, gy, gz;   // deg/s
    float ax, ay, az;   // m/s²
};

void imu_init();
bool imu_who_am_i();          // Returns true if WHO_AM_I == 0x6B
void imu_fifo_read_batch();   // Read FIFO into internal buffer, called every 20ms
ImuData imu_get_latest();     // Get most recently processed sample
uint16_t imu_fifo_depth();    // How many samples currently in FIFO
