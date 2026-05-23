#pragma once

// ── Servo bus (UART2 on ESP32) ───────────────────────────────────────────────
#define SERVO_BAUD        1000000UL
#define SERVO_TX_PIN      17       // GPIO17 = UART2 TX → servo bus data line
#define SERVO_RX_PIN      16       // GPIO16 = UART2 RX ← servo bus data line
#define SERVO_TX_ENABLE   4        // GPIO4, HIGH=TX mode, LOW=RX mode
#define SERVO_COUNT       5
#define SERVO_ID_J0       0x01
#define SERVO_ID_J1A      0x02    // Coupled shoulder servo A
#define SERVO_ID_J1B      0x03    // Coupled shoulder servo B
#define SERVO_ID_J2       0x04
#define SERVO_ID_J3       0x05    // Gripper

// ── ISM330DHCX (shared I2C bus with ToF) ─────────────────────────────────────
#define IMU_I2C_ADDR      0x6B     // SA0/POCI=HIGH → address 0x6B

// ── VL53L5CX (Wire / I2C0 bus) ───────────────────────────────────────────────
#define TOF_SDA           21       // GPIO21
#define TOF_SCL           22       // GPIO22
#define TOF_LPN           27       // GPIO27, power enable, HIGH=powered
#define TOF_INT           26       // GPIO26, data-ready interrupt, active LOW
#define TOF_I2C_ADDR      0x52
#define TOF_UPDATE_HZ     15

// ── Control loop ─────────────────────────────────────────────────────────────
#define CONTROL_HZ        50
#define CONTROL_PERIOD_MS 20       // 20ms period for vTaskDelayUntil

// ── Packet sizes ─────────────────────────────────────────────────────────────
#define TELEMETRY_SIZE    254
#define TELEMETRY_MAGIC   0xA55AA55AUL
#define COMMAND_SIZE      20

// ── Temperature thresholds ───────────────────────────────────────────────────
#define TEMP_WARN_C       65.0f
#define TEMP_CUTOFF_C     80.0f

// ── Contact oracle ───────────────────────────────────────────────────────────
#define CONTACT_WINDOW    8       // samples at 208Hz = ~38ms
#define CONTACT_THRESHOLD 3.5f   // deg/s RMS, calibrate empirically

// ── USB Serial baud (via onboard CH340/CP2102 bridge) ────────────────────────
#define USB_BAUD          2000000UL

// ── FreeRTOS task config ─────────────────────────────────────────────────────
#define CONTROL_TASK_STACK  8192  // bytes
#define COMMS_TASK_STACK    4096  // bytes
#define CONTROL_TASK_PRIO   10    // higher than comms
#define COMMS_TASK_PRIO     5
