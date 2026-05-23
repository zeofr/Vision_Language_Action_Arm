#include "ism330dhcx_driver.h"
#include "config.h"
#include "contact_oracle.h"
#include <Wire.h>

#define ISM_WHO_AM_I          0x0F
#define ISM_CTRL1_XL          0x10
#define ISM_CTRL2_G           0x11
#define ISM_CTRL3_C           0x12
#define ISM_FIFO_CTRL3        0x09
#define ISM_FIFO_CTRL4        0x0A
#define ISM_FIFO_STATUS1      0x3A
#define ISM_FIFO_STATUS2      0x3B
#define ISM_FIFO_DATA_OUT_TAG 0x78

#define GYRO_SENSITIVITY_MDPS_PER_LSB   8.75f
#define ACCEL_SENSITIVITY_MG_PER_LSB    0.061f
#define G_TO_MS2                        0.00980665f

static ImuData latest_imu = {};

static void imu_write_reg(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(IMU_I2C_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

static uint8_t imu_read_reg(uint8_t reg) {
    Wire.beginTransmission(IMU_I2C_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)IMU_I2C_ADDR, (uint8_t)1);
    return Wire.available() ? Wire.read() : 0xFF;
}

void imu_init() {
    Wire.begin(TOF_SDA, TOF_SCL);
    Wire.setClock(400000);
    while (imu_read_reg(ISM_WHO_AM_I) != 0x6B) {
        Serial.println("IMU not found on I2C — retrying...");
        delay(500);
    }

    imu_write_reg(ISM_CTRL3_C,    0x44);  // BDU=1, IF_INC=1
    imu_write_reg(ISM_CTRL1_XL,   0x60);  // 208 Hz, ±2 g
    imu_write_reg(ISM_CTRL2_G,    0x60);  // 208 Hz, ±250 dps
    imu_write_reg(ISM_FIFO_CTRL3, 0x66);  // batch accel+gyro at 208 Hz
    imu_write_reg(ISM_FIFO_CTRL4, 0x06);  // FIFO continuous mode
    delay(5);
}

bool imu_who_am_i() {
    return (imu_read_reg(ISM_WHO_AM_I) == 0x6B);
}

uint16_t imu_fifo_depth() {
    uint8_t s1 = imu_read_reg(ISM_FIFO_STATUS1);
    uint8_t s2 = imu_read_reg(ISM_FIFO_STATUS2);
    return (((uint16_t)(s2 & 0x03) << 8) | s1);
}

void imu_fifo_read_batch() {
    uint16_t depth = imu_fifo_depth();
    if (depth == 0) return;

    // 208 Hz @ 50 Hz poll ≈ 4 samples; cap at 12 to stay within Wire 128-byte buffer
    uint8_t n = (uint8_t)(depth > 12 ? 12 : depth);

    Wire.beginTransmission(IMU_I2C_ADDR);
    Wire.write(ISM_FIFO_DATA_OUT_TAG);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)IMU_I2C_ADDR, (uint8_t)(n * 7));

    for (uint8_t i = 0; i < n; i++) {
        uint8_t word[7];
        for (int b = 0; b < 7; b++)
            word[b] = Wire.available() ? Wire.read() : 0;

        uint8_t tag = word[0] >> 3;
        int16_t x = (int16_t)(word[1] | (word[2] << 8));
        int16_t y = (int16_t)(word[3] | (word[4] << 8));
        int16_t z = (int16_t)(word[5] | (word[6] << 8));

        if (tag == 0x01) {  // gyroscope sample
            latest_imu.gx = (float)x * GYRO_SENSITIVITY_MDPS_PER_LSB / 1000.0f;
            latest_imu.gy = (float)y * GYRO_SENSITIVITY_MDPS_PER_LSB / 1000.0f;
            latest_imu.gz = (float)z * GYRO_SENSITIVITY_MDPS_PER_LSB / 1000.0f;
            contact_oracle_push(latest_imu.gx, latest_imu.gy, latest_imu.gz);
        } else if (tag == 0x02) {  // accelerometer sample
            latest_imu.ax = (float)x * ACCEL_SENSITIVITY_MG_PER_LSB * G_TO_MS2;
            latest_imu.ay = (float)y * ACCEL_SENSITIVITY_MG_PER_LSB * G_TO_MS2;
            latest_imu.az = (float)z * ACCEL_SENSITIVITY_MG_PER_LSB * G_TO_MS2;
        }
    }
}

ImuData imu_get_latest() {
    return latest_imu;
}
