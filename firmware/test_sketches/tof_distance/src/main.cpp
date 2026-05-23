#include <Arduino.h>
#include <Wire.h>
#include "vl53l5cx_class.h"

#define TOF_SDA  21
#define TOF_SCL  22
#define TOF_LPN  27   // power enable, HIGH = powered
#define TOF_INT  26   // data-ready interrupt, active LOW

VL53L5CX sensor(&Wire, TOF_LPN, TOF_INT);
bool init_ok = false;

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println("\n=== VL53L5CX Distance Test ===");

    // Power cycle the sensor
    pinMode(TOF_LPN, OUTPUT);
    pinMode(TOF_INT, INPUT_PULLUP);
    digitalWrite(TOF_LPN, LOW);
    delay(10);
    digitalWrite(TOF_LPN, HIGH);
    delay(100);

    Wire.begin(TOF_SDA, TOF_SCL);
    Wire.setClock(400000);

    Serial.println("Uploading VL53L5CX firmware (~500ms)...");
    uint8_t status = sensor.init_sensor(0x52);
    if (status != 0) {
        Serial.printf("Init FAILED: status=%d — check SDA/SCL and LPn wiring\n", status);
        return;
    }

    sensor.vl53l5cx_set_resolution(VL53L5CX_RESOLUTION_8X8);
    sensor.vl53l5cx_set_ranging_frequency_hz(15);
    sensor.vl53l5cx_set_sharpener_percent(0);
    sensor.vl53l5cx_start_ranging();

    Serial.println("Ranging started. Place a flat board ~200mm below sensor.");
    Serial.println("Center zone average will print at 15Hz.\n");
    init_ok = true;
}

void loop() {
    if (!init_ok) { delay(1000); return; }

    uint8_t ready = 0;
    sensor.vl53l5cx_check_data_ready(&ready);
    if (!ready) { delay(10); return; }

    VL53L5CX_ResultsData results;
    sensor.vl53l5cx_get_ranging_data(&results);

    // Center zones in 8x8 grid: row=3,col=3 → idx=27; row=3,col=4 → 28;
    //                             row=4,col=3 → 35; row=4,col=4 → 36
    int16_t zones[4] = {
        results.distance_mm[27], results.distance_mm[28],
        results.distance_mm[35], results.distance_mm[36]
    };
    uint8_t status_zones[4] = {
        results.target_status[27], results.target_status[28],
        results.target_status[35], results.target_status[36]
    };

    uint32_t sum = 0;
    int count = 0;
    for (int i = 0; i < 4; i++) {
        // target_status == 5 means valid ranging
        if (status_zones[i] == 5 && zones[i] > 20 && zones[i] < 3000) {
            sum += zones[i];
            count++;
        }
    }

    if (count > 0) {
        uint16_t avg = sum / count;
        bool pass = (avg > 150 && avg < 250);
        Serial.printf("Center avg: %4d mm  (%s)\n", avg,
                      pass ? "PASS — within 150-250mm" : "reading...");
    } else {
        Serial.println("No valid center zone readings — check sensor orientation (face pointing down)");
    }
}
