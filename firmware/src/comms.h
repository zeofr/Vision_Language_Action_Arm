#pragma once
#include <Arduino.h>

// ESP32 → RPi5 Telemetry Packet (254 bytes, sent at 50Hz)
typedef struct __attribute__((packed)) {
    uint32_t magic;              // Always TELEMETRY_MAGIC (0xA55AA55A) — packet framing
    uint32_t timestamp_us;      // ESP32 microsecond counter (from micros())
    float    servo_pos[5];      // Servo positions in degrees [J0, J1a, J1b, J2, J3]
    float    servo_load[5];     // Normalized load 0.0–1.0 [same order]
    float    servo_speed[5];    // Speed in degrees/second [same order]
    float    servo_temp[5];     // Temperature in Celsius [same order]
    uint16_t tof_grid[64];      // VL53L5CX 8×8 zone distances in mm, row-major
    uint32_t tof_timestamp_us;  // Timestamp when this ToF frame was captured
    uint8_t  tof_resolution;    // Always 64 (8×8 mode), or 16 if fell back to 4×4
    uint8_t  tof_valid;         // 1 if this ToF frame passed validity checks, 0 if stale
    float    imu_gyro[3];       // ISM330DHCX gyro [gx, gy, gz] in deg/s
    float    imu_accel[3];      // ISM330DHCX accel [ax, ay, az] in m/s²
    uint8_t  contact_flag;      // 1 if contact oracle fired THIS cycle
    float    contact_rms;       // Current 20-sample gyro RMS (for friend's dashboard)
    uint8_t  safety_clamped;    // 1 if any joint was clamped this cycle
    uint16_t checksum;          // Simple sum of all preceding bytes, truncated to 16-bit
} ControllerTelemetry_t;

// RPi5 → ESP32 Command Packet (20 bytes, received at 8Hz)
typedef struct __attribute__((packed)) {
    float   target_arm[3];    // Target degrees for [J0, J1, J2] — arm joints only
    uint8_t skill_state;      // 0=REACH, 1=GRASP, 2=LIFT, 3=PLACE
    uint8_t execute;          // 1=execute motion, 0=hold position
    float   gripper_command;  // Gripper: 0.0=open, 1.0=closed
    uint8_t emergency_stop;   // 1=immediate stop all servos (highest priority)
    uint8_t checksum;         // Sum of all preceding bytes mod 256
} RPiCommand_t;

void comms_init();
void comms_send_telemetry(const ControllerTelemetry_t* pkt);
bool comms_receive_command(RPiCommand_t* cmd_out);
