# ESP32-WROOM-32 Migration Design Spec

**Date:** 2026-05-20
**Status:** Approved
**Scope:** Replace Teensy 4.1 with ESP32-WROOM-32 dev module as the real-time control MCU across all firmware and project documentation.

---

## Context

The project originally specified a Teensy 4.1 (NXP IMXRT1062, Cortex-M7 @ 600MHz, bare-metal Teensyduino) as the real-time microcontroller. The hardware is unavailable. The developer has an **ESP32-WROOM-32 dev module** (Xtensa dual-core LX6 @ 240MHz, FreeRTOS, Arduino + espressif32 PlatformIO target) on hand. This spec defines all changes required to migrate cleanly.

---

## Board Confirmed

**ESP32-WROOM-32 dev board** (38-pin variant, USB via onboard CH340/CP2102 bridge)
- Xtensa LX6 dual-core @ 240MHz, hardware FPU
- 520KB SRAM, 4MB Flash
- FreeRTOS (preemptive, 1ms tick)
- 3 UARTs, 2 I2C, 2 SPI (HSPI + VSPI), all software-configurable to any GPIO
- USB via CH340/CP2102 bridge (not native USB like Teensy)
- WiFi + BT present but **disabled entirely** at runtime

---

## Architecture Decision

### Dual-Core FreeRTOS Task Split

WiFi and BT are disabled. Core 1 is fully reserved for the deterministic 50Hz control loop.

| Core | Task | Priority | Stack | Responsibilities |
|---|---|---|---|---|
| Core 1 (app_cpu) | `control_task` | 10 | 8192 B | IMU FIFO read → ToF poll → servo telemetry → contact oracle → waypoint interp → safety clamp → servo sync write → pack telemetry into shared buffer |
| Core 0 (pro_cpu) | `comms_task` | 5 | 4096 B | Read shared telemetry → send 250-byte packet to RPi5 → receive 20-byte command → write to shared command buffer |

**Timing:** `vTaskDelayUntil` at 20ms (50Hz) on Core 1. Expected jitter ±0.5–1ms with WiFi off (vs Teensy ±5µs bare-metal). Acceptable for servo PID which runs its own internal loop.

**Inter-task sync:** Two `SemaphoreHandle_t` mutexes:
- `g_telemetry_mutex` — protects `ControllerTelemetry_t g_telemetry`
- `g_command_mutex` — protects `RPiCommand_t g_last_cmd`

Both are taken with timeout=0 in `control_task` (non-blocking — skip if comms task holds) and timeout=5ms in `comms_task`.

### Communication Protocol: Unchanged

`ControllerTelemetry_t` (250 bytes, 50Hz) and `RPiCommand_t` (20 bytes, 8Hz) structs are byte-identical to the original spec. The RPi5 sees no change. Only the C type name changes: `TeensyTelemetry_t` → `ControllerTelemetry_t`.

---

## Pin Mapping

| Function | Teensy 4.1 (old) | ESP32-WROOM-32 (new) | Notes |
|---|---|---|---|
| STS3215 bus TX | TX1 / Pin 1 | GPIO17 (UART2 TX) | `Serial2` |
| STS3215 bus RX | RX1 / Pin 0 | GPIO16 (UART2 RX) | `Serial2` |
| TX_ENABLE (direction) | Pin 2 | GPIO4 | Output, HIGH=TX, LOW=RX |
| IMU MOSI | Pin 11 (SPI0) | GPIO23 (VSPI MOSI) | `SPI` |
| IMU MISO | Pin 12 (SPI0) | GPIO19 (VSPI MISO) | `SPI` |
| IMU SCK | Pin 13 (SPI0) | GPIO18 (VSPI SCK) | `SPI` |
| IMU CS | Pin 10 | GPIO5 | Output, active LOW |
| ToF SDA | Pin 17 (Wire1) | GPIO21 (Wire SDA) | `Wire` |
| ToF SCL | Pin 16 (Wire1) | GPIO22 (Wire SCL) | `Wire` |
| ToF LPn | Pin 14 | GPIO27 | Output, HIGH=powered |
| ToF INT | Pin 15 | GPIO26 | Input, active LOW |
| RPi5 serial | Native USB (USB FS) | GPIO1/3 via USB-UART bridge | `Serial` (UART0) |

**Avoided pins:** GPIO6–11 (internal flash), GPIO34/35/36/39 (input-only), GPIO0/2/12/15 (strapping pins). All assigned pins are safe for the specified use.

---

## PlatformIO Configuration Change

```ini
# OLD
[env:teensy41]
platform  = teensy
board     = teensy41
framework = arduino
build_flags = -O2 -std=c++17
lib_deps = Wire
           SPI

# NEW
[env:esp32dev]
platform    = espressif32
board       = esp32dev
framework   = arduino
build_flags = -O2 -std=c++17 -DCORE_DEBUG_LEVEL=0
monitor_speed = 2000000
lib_deps =
    Wire
    SPI
```

---

## config.h Changes

All pin defines updated. New UART pin defines added.

```c
// Servo bus — UART2 on ESP32
#define SERVO_BAUD        1000000UL
#define SERVO_TX_PIN      17        // GPIO17 = UART2 TX
#define SERVO_RX_PIN      16        // GPIO16 = UART2 RX
#define SERVO_TX_ENABLE   4         // GPIO4, HIGH=TX, LOW=RX

// ISM330DHCX — VSPI
#define IMU_SPI_CS        5         // GPIO5
#define IMU_SPI_FREQ      10000000UL

// VL53L5CX — Wire (I2C0)
#define TOF_SDA           21        // GPIO21
#define TOF_SCL           22        // GPIO22
#define TOF_LPN           27        // GPIO27
#define TOF_INT           26        // GPIO26
#define TOF_I2C_ADDR      0x52
#define TOF_UPDATE_HZ     15
```

---

## Driver API Changes

| File | Old | New |
|---|---|---|
| `servo_bus.cpp` | `Serial1.begin(SERVO_BAUD)` | `Serial2.begin(SERVO_BAUD, SERIAL_8N1, SERVO_RX_PIN, SERVO_TX_PIN)` |
| `servo_bus.cpp` | `Serial1.write/flush/available/read` | `Serial2.write/flush/available/read` |
| `tof_driver.cpp` | `Wire1.begin()` | `Wire.begin(TOF_SDA, TOF_SCL)` |
| `tof_driver.cpp` | `Wire1.setClock(400000)` | `Wire.setClock(400000)` |
| `comms.h/.cpp` | `TeensyTelemetry_t` | `ControllerTelemetry_t` |
| `main.cpp` | `setup()` + `loop()` (bare-metal) | `setup()` + `control_task` + `comms_task` (FreeRTOS) |

---

## main.cpp Architecture Change

**Old (Teensy — bare-metal busy-wait):**
```c
void loop() {
    while (micros() - last_loop_us < CONTROL_PERIOD_US) { /* busy wait */ }
    // ... all work inline
}
```

**New (ESP32 — FreeRTOS dual-core):**
```c
void setup() {
    WiFi.mode(WIFI_OFF);   // eliminate RF interference
    btStop();
    // ... init all peripherals
    xTaskCreatePinnedToCore(control_task, "ControlTask", 8192, NULL, 10, NULL, 1);
    xTaskCreatePinnedToCore(comms_task,   "CommsTask",   4096, NULL,  5, NULL, 0);
    vTaskDelete(NULL);     // delete Arduino loop() task
}

void control_task(void*) {
    TickType_t lastWakeTime = xTaskGetTickCount();
    while (true) {
        // ... all 50Hz work
        vTaskDelayUntil(&lastWakeTime, pdMS_TO_TICKS(20));
    }
}

void comms_task(void*) {
    while (true) {
        // ... serial send/receive
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}
```

---

## What Does Not Change

- `ControllerTelemetry_t` struct layout (250 bytes, identical to old `TeensyTelemetry_t`)
- `RPiCommand_t` struct layout (20 bytes)
- All 7 driver modules: logic is unchanged, only peripheral API calls updated
- HDF5 dataset format
- Calibration script format and procedure
- RPi5 inference Python codebase (friend's side) — zero changes required

---

## Known Limitations vs Teensy 4.1

| Property | Teensy 4.1 | ESP32-WROOM-32 |
|---|---|---|
| CPU | 600MHz Cortex-M7 | 240MHz Xtensa LX6 |
| Loop timing | ±5µs (bare-metal) | ±0.5–1ms (FreeRTOS, WiFi off) |
| USB | Native USB FS | Via CH340/CP2102 bridge |
| FPU | Yes (Cortex-M7) | Yes (Xtensa LX6) |
| Real-time OS | None (bare-metal) | FreeRTOS |
| 50Hz achievable | Yes | Yes (with WiFi/BT off) |

Timing jitter of ±1ms at 50Hz is acceptable. Servo position error from 1ms timing slip at typical joint velocities (<60°/s) is <0.1°, well within servo deadband.

---

## Documents Updated

1. `Team_HARDWARE_EMBEDDED_WORKPLAN.md` — all firmware sections, pin table, platformio.ini, config.h, all driver code, main.cpp
2. `VLA_Robotic_Arm_Project_Report_FINAL.md` — Section 6.6, Section 7.2, Section 12.1, Section 15.1, all inline references
