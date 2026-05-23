#include "tof_driver.h"
#include "vl53l5cx_class.h"
#include "config.h"
#include <Wire.h>

static VL53L5CX* tof_sensor = nullptr;
static ToFFrame latest_frame;

void tof_init() {
    pinMode(TOF_LPN, OUTPUT);
    pinMode(TOF_INT, INPUT_PULLUP);
    
    Wire.begin(TOF_SDA, TOF_SCL);
    Wire.setClock(400000);  // 400kHz Fast I2C mode
    
    // Construct sensor instance using Wire and LPn pin
    tof_sensor = new VL53L5CX(&Wire, TOF_LPN);
    
    // Initialize the sensor (loads 130KB+ firmware to hardware via I2C, takes ~500ms)
    int status = tof_sensor->init_sensor(TOF_I2C_ADDR);
    if (status != 0) {
        Serial.printf("VL53L5CX Initialization failed with status code: %d\n", status);
        // Halt if hardware initialization fails so the user knows there is a wiring issue
        while (true) { delay(1000); }
    }
    
    // Configure to 8x8 grid (64 active ranging zones)
    tof_sensor->vl53l5cx_set_resolution(VL53L5CX_RESOLUTION_8X8);
    
    // Configure to 15Hz update rate
    tof_sensor->vl53l5cx_set_ranging_frequency_hz(TOF_UPDATE_HZ);
    
    // Disable sharpener algorithm (provides raw metrics and increases precision for flat objects)
    tof_sensor->vl53l5cx_set_sharpener_percent(0);
    
    // Start continuous ranging mode
    tof_sensor->vl53l5cx_start_ranging();
    
    // Initialize latest frame as empty
    memset(&latest_frame, 0, sizeof(ToFFrame));
    for (int i = 0; i < 64; i++) {
        latest_frame.distances_mm[i] = 0xFFFF;
    }
    
    Serial.println("VL53L5CX Time-of-Flight successfully initialized (8x8 @ 15Hz).");
}

bool tof_check_ready() {
    // Fast hardware pin check: VL53L5CX INT pulls LOW when data is ready
    if (digitalRead(TOF_INT) == HIGH) {
        return false;
    }
    
    uint8_t is_ready = 0;
    tof_sensor->vl53l5cx_check_data_ready(&is_ready);
    if (!is_ready) {
        return false;
    }
    
    VL53L5CX_ResultsData results;
    uint8_t status = tof_sensor->vl53l5cx_get_ranging_data(&results);
    if (status != 0) {
        return false;
    }
    
    latest_frame.capture_timestamp_us = micros();
    latest_frame.valid = 1;
    
    for (int i = 0; i < 64; i++) {
        // target_status == 5 represents a valid measurement, range_sigma_mm < 35 represents low noise
        uint8_t target_status = results.target_status[i];
        if (target_status == 5 && results.range_sigma_mm[i] < 35) {
            latest_frame.distances_mm[i] = results.distance_mm[i];
        } else {
            latest_frame.distances_mm[i] = 0xFFFF;  // Invalid data sentinel
            
            // Check if this invalid zone falls in the 4 central grasp-measurement zones
            // Center 4 zones are (3,3), (3,4), (4,3), (4,4) which translate to indices:
            // 27, 28, 35, 36 in row-major representation.
            if (i == 27 || i == 28 || i == 35 || i == 36) {
                latest_frame.valid = 0;  // Latch entire frame's valid flag to false
            }
        }
    }
    
    return true;
}

ToFFrame tof_get_latest() {
    return latest_frame;
}
