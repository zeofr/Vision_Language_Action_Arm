#pragma once
#include <Arduino.h>

// Servo telemetry for one servo
struct ServoTelemetry {
    float pos_deg;      // position in degrees, 0-center
    float speed_dps;    // speed in deg/s (signed)
    float load_norm;    // normalized load 0.0-1.0
    float voltage_v;    // voltage in volts
    float temp_c;       // temperature in Celsius
};

void servo_bus_init();

// Write goal position to a single servo (degrees, -150 to +150 typical)
void servo_write_deg(uint8_t id, float degrees);

// Sync write: write positions to multiple servos in one bus transaction
// ids[]: servo IDs, positions_deg[]: target degrees, count: number of servos
void servo_sync_write(const uint8_t* ids, const float* positions_deg, uint8_t count);

// Read full telemetry from one servo
ServoTelemetry servo_read_telemetry(uint8_t id);

// Poll all 5 servos once per 20ms control cycle.
bool servo_poll_all(ServoTelemetry* telemetry);
