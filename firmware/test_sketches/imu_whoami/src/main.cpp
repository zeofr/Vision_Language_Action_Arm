#include <Arduino.h>
#include <Wire.h>

#define TOF_SDA  21
#define TOF_SCL  22
#define IMU_ADDR 0x6B
#define REG_WHO_AM_I 0x0F
#define EXPECTED_ID  0x6B

static uint8_t imu_read_reg(uint8_t reg) {
    Wire.beginTransmission(IMU_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)IMU_ADDR, (uint8_t)1);
    return Wire.available() ? Wire.read() : 0xFF;
}

void setup() {
    Serial.begin(115200);
    Wire.begin(TOF_SDA, TOF_SCL);
    Wire.setClock(400000);
    delay(100);

    Serial.println("\n=== IMU WHO_AM_I Test (I2C) ===");
    uint8_t id = imu_read_reg(REG_WHO_AM_I);
    Serial.printf("WHO_AM_I = 0x%02X  →  %s\n\n", id,
                  id == EXPECTED_ID ? "IMU OK: 0x6B — PASS"
                                    : "FAIL — expected 0x6B, check SDA/SCL wiring");
}

void loop() {
    delay(2000);
    uint8_t id = imu_read_reg(REG_WHO_AM_I);
    Serial.printf("WHO_AM_I: 0x%02X (%s)\n", id, id == EXPECTED_ID ? "OK" : "FAIL");
}
