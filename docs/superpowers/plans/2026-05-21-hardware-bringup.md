# Hardware Bring-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the ESP32 firmware verified and streaming 50Hz telemetry to the RPi5, with each peripheral (servo bus, IMU, ToF) independently confirmed working before the full firmware runs.

**Architecture:** Three throwaway test sketches (each a self-contained PlatformIO project) verify one peripheral at a time. Once all three pass, the main firmware is flashed and a Python script on the RPi5 confirms the 50Hz telemetry stream. The SmartElex servo board handles UART half-duplex switching internally, so GPIO4 direction control is removed from servo_bus.cpp.

**Tech Stack:** PlatformIO + espressif32, ESP32 Arduino framework, C++17, Python 3 + pyserial + numpy on RPi5.

---

## File Map

| Action | Path |
|---|---|
| Modify | `firmware/src/servo_bus.cpp` |
| Create | `firmware/test_sketches/servo_ping/platformio.ini` |
| Create | `firmware/test_sketches/servo_ping/src/main.cpp` |
| Create | `firmware/test_sketches/imu_whoami/platformio.ini` |
| Create | `firmware/test_sketches/imu_whoami/src/main.cpp` |
| Create | `firmware/test_sketches/tof_distance/platformio.ini` |
| Create | `firmware/test_sketches/tof_distance/src/main.cpp` |
| Create | `firmware/tools/verify_telemetry.py` |

---

## Task 1: RPi5 Environment Setup

**Files:** none (setup commands only)

- [ ] **Step 1: SSH into the RPi5 and install PlatformIO**

```bash
pip3 install platformio
```

Expected: PlatformIO installs without errors. Verify with:
```bash
pio --version
```
Expected output: `PlatformIO Core, version 6.x.x`

- [ ] **Step 2: Clone or confirm the repo exists on RPi5**

```bash
cd ~
git clone <your-repo-url> vla_rob   # skip if already cloned
cd vla_rob
git pull
```

Expected: repo is at `~/vla_rob` with `firmware/` directory present.

- [ ] **Step 3: Plug ESP32 into RPi5 USB and confirm it's visible**

```bash
ls /dev/tty*
```

Expected: either `/dev/ttyUSB0` (CH340 bridge) or `/dev/ttyACM0` (CP2102 bridge) appears. If nothing appears, try:
```bash
dmesg | tail -20
```
Look for lines like `usb 1-1: ch341-uart converter now attached to ttyUSB0`. If still not visible, check the USB cable supports data (not charge-only).

- [ ] **Step 4: Add RPi5 user to dialout group (needed for serial access)**

```bash
sudo usermod -a -G dialout $USER
```

Then **log out and log back in** for the group change to take effect.

- [ ] **Step 5: Confirm PlatformIO can see the ESP32**

```bash
cd ~/vla_rob/firmware
pio device list
```

Expected: ESP32 port listed (e.g. `/dev/ttyUSB0`) with description mentioning CH340 or CP2102.

---

## Task 2: Fix servo_bus.cpp — Remove GPIO4 Direction Control

The SmartElex Serial Bus Servo Driver Board handles UART half-duplex direction switching internally. The ESP32 only needs to drive standard full-duplex UART on GPIO16/17. Remove GPIO4 toggling from `begin_tx()`, `begin_rx()`, and `servo_bus_init()`. Keep `Serial2.flush()` in `begin_rx()` — it ensures the TX buffer drains before listening for servo responses.

**Files:**
- Modify: `firmware/src/servo_bus.cpp:4-19`

- [ ] **Step 1: Apply the fix to servo_bus.cpp**

Replace lines 4–19 of `firmware/src/servo_bus.cpp`:

```cpp
static void begin_tx() {
    // Direction control handled by SmartElex board — no GPIO toggle needed
}

static void begin_rx() {
    Serial2.flush();  // ensure TX buffer is fully drained before listening
    // Direction control handled by SmartElex board — no GPIO toggle needed
}

void servo_bus_init() {
    Serial2.begin(SERVO_BAUD, SERIAL_8N1, SERVO_RX_PIN, SERVO_TX_PIN);
}
```

- [ ] **Step 2: Verify the file compiles on WSL2**

```bash
cd firmware
pio run
```

Expected: `SUCCESS` — no compile errors. If errors appear, check the edit was applied cleanly.

- [ ] **Step 3: Commit**

```bash
git add firmware/src/servo_bus.cpp
git commit -m "fix: remove GPIO4 direction control — SmartElex board handles half-duplex"
```

---

## Task 3: Servo Bus Test Sketch

A minimal PlatformIO project that pings all 5 servo IDs on the bus and prints which ones respond. Uses the same UART2 pins as the main firmware (GPIO16/17).

**Files:**
- Create: `firmware/test_sketches/servo_ping/platformio.ini`
- Create: `firmware/test_sketches/servo_ping/src/main.cpp`

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p firmware/test_sketches/servo_ping/src
```

- [ ] **Step 2: Create platformio.ini**

Create `firmware/test_sketches/servo_ping/platformio.ini`:

```ini
[env:esp32dev]
platform     = espressif32
board        = esp32dev
framework    = arduino
build_flags  = -O2 -std=c++17
monitor_speed = 115200
```

- [ ] **Step 3: Create src/main.cpp**

Create `firmware/test_sketches/servo_ping/src/main.cpp`:

```cpp
#include <Arduino.h>

#define SERVO_TX   17
#define SERVO_RX   16
#define SERVO_BAUD 1000000UL

// Sends a PING packet to servo ID and returns true if a valid response arrives.
// SCS protocol: 0xFF 0xFF [ID] [LEN=2] [0x01=PING] [CHECKSUM]
// Response:     0xFF 0xFF [ID] [2]     [ERR]       [CHECKSUM]
static bool ping_servo(uint8_t id) {
    while (Serial2.available()) Serial2.read();  // flush RX

    uint8_t pkt[6];
    pkt[0] = 0xFF;
    pkt[1] = 0xFF;
    pkt[2] = id;
    pkt[3] = 2;                                    // length
    pkt[4] = 0x01;                                 // PING instruction
    pkt[5] = ~(id + 2 + 0x01) & 0xFF;             // checksum

    Serial2.write(pkt, 6);
    Serial2.flush();  // wait for TX to drain before listening

    uint8_t resp[6];
    uint8_t idx = 0;
    uint32_t deadline = micros() + 10000;  // 10ms timeout
    while (micros() < deadline && idx < 6) {
        if (Serial2.available()) resp[idx++] = Serial2.read();
    }

    if (idx < 6) return false;
    if (resp[0] != 0xFF || resp[1] != 0xFF) return false;
    if (resp[2] != id) return false;
    return true;
}

static void scan_all() {
    const uint8_t ids[]   = {0x01, 0x02, 0x03, 0x04, 0x05};
    const char*   names[] = {"J0  Base     ", "J1a Shoulder ",
                              "J1b Shoulder ", "J2  Elbow    ", "J3  Gripper  "};
    int found = 0;
    for (int i = 0; i < 5; i++) {
        bool ok = ping_servo(ids[i]);
        Serial.printf("  ID 0x%02X (%s): %s\n", ids[i], names[i], ok ? "FOUND" : "missing");
        if (ok) found++;
        delay(50);
    }
    Serial.printf("\n  %d/5 servos found — %s\n\n", found, found == 5 ? "PASS" : "FAIL (check wiring)");
}

void setup() {
    Serial.begin(115200);
    Serial2.begin(SERVO_BAUD, SERIAL_8N1, SERVO_RX, SERVO_TX);
    delay(200);
    Serial.println("\n=== Servo Ping Test ===");
    scan_all();
}

void loop() {
    delay(3000);
    Serial.println("--- re-scan ---");
    scan_all();
}
```

- [ ] **Step 4: Commit**

```bash
git add firmware/test_sketches/servo_ping/
git commit -m "test: add servo bus ping sketch"
```

- [ ] **Step 5: On RPi5 — pull, wire, and flash**

**Wire first (3 wires between ESP32 and SmartElex board):**
- ESP32 GPIO17 → SmartElex RX
- ESP32 GPIO16 → SmartElex TX
- ESP32 GND → SmartElex GND
- 12V (from buck converter, set to 12.0V) → SmartElex DC barrel jack

**Then on RPi5:**
```bash
cd ~/vla_rob
git pull
cd firmware/test_sketches/servo_ping
pio run --target upload
pio device monitor
```

- [ ] **Step 6: Verify pass condition**

Expected serial output:
```
=== Servo Ping Test ===
  ID 0x01 (J0  Base     ): FOUND
  ID 0x02 (J1a Shoulder ): FOUND
  ID 0x03 (J1b Shoulder ): FOUND
  ID 0x04 (J2  Elbow    ): FOUND
  ID 0x05 (J3  Gripper  ): FOUND

  5/5 servos found — PASS
```

If any servo shows `missing`:
1. Check that the servo's ID was assigned correctly (Feetech SCServo software on Windows, or `servo_tool` on Linux)
2. Check that all servos are at 1Mbps baud (register 0x04 = 0x00)
3. Check the TX/RX wires are not swapped (ESP32 TX → board RX, ESP32 RX → board TX)
4. Check 12V is reaching the SmartElex board

---

## Task 4: IMU WHO_AM_I Test Sketch

Reads the ISM330DHCX WHO_AM_I register over SPI. Expected value is 0x6B. Uses VSPI on GPIO18/19/23 with CS on GPIO5.

**Files:**
- Create: `firmware/test_sketches/imu_whoami/platformio.ini`
- Create: `firmware/test_sketches/imu_whoami/src/main.cpp`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p firmware/test_sketches/imu_whoami/src
```

- [ ] **Step 2: Create platformio.ini**

Create `firmware/test_sketches/imu_whoami/platformio.ini`:

```ini
[env:esp32dev]
platform     = espressif32
board        = esp32dev
framework    = arduino
build_flags  = -O2 -std=c++17
monitor_speed = 115200
lib_deps     = SPI
```

- [ ] **Step 3: Create src/main.cpp**

Create `firmware/test_sketches/imu_whoami/src/main.cpp`:

```cpp
#include <Arduino.h>
#include <SPI.h>

#define IMU_CS    5
#define IMU_MOSI  23
#define IMU_MISO  19
#define IMU_SCK   18
#define IMU_FREQ  1000000UL   // use 1MHz for bring-up; main firmware uses 10MHz

#define REG_WHO_AM_I  0x0F
#define EXPECTED_ID   0x6B

static uint8_t imu_read_reg(uint8_t reg) {
    uint8_t val;
    digitalWrite(IMU_CS, LOW);
    SPI.beginTransaction(SPISettings(IMU_FREQ, MSBFIRST, SPI_MODE0));
    SPI.transfer(reg | 0x80);  // bit7=1 for read
    val = SPI.transfer(0x00);
    SPI.endTransaction();
    digitalWrite(IMU_CS, HIGH);
    return val;
}

void setup() {
    Serial.begin(115200);
    pinMode(IMU_CS, OUTPUT);
    digitalWrite(IMU_CS, HIGH);
    SPI.begin(IMU_SCK, IMU_MISO, IMU_MOSI, IMU_CS);
    delay(100);

    Serial.println("\n=== IMU WHO_AM_I Test ===");
    uint8_t id = imu_read_reg(REG_WHO_AM_I);
    Serial.printf("WHO_AM_I = 0x%02X  →  %s\n\n", id,
                  id == EXPECTED_ID ? "IMU OK: 0x6B — PASS"
                                    : "FAIL — expected 0x6B, check wiring");
}

void loop() {
    delay(2000);
    uint8_t id = imu_read_reg(REG_WHO_AM_I);
    Serial.printf("WHO_AM_I: 0x%02X (%s)\n", id,
                  id == EXPECTED_ID ? "OK" : "FAIL");
}
```

- [ ] **Step 4: Commit**

```bash
git add firmware/test_sketches/imu_whoami/
git commit -m "test: add IMU WHO_AM_I sketch"
```

- [ ] **Step 5: On RPi5 — pull, wire, and flash**

**Wire ISM330DHCX to ESP32 (6 wires):**
- ESP32 GPIO23 (MOSI) → IMU SDI
- ESP32 GPIO19 (MISO) → IMU SDO
- ESP32 GPIO18 (SCK)  → IMU SCL
- ESP32 GPIO5  (CS)   → IMU CS
- ESP32 3V3           → IMU VCC
- ESP32 GND           → IMU GND

**Then on RPi5:**
```bash
cd ~/vla_rob
git pull
cd firmware/test_sketches/imu_whoami
pio run --target upload
pio device monitor
```

- [ ] **Step 6: Verify pass condition**

Expected serial output:
```
=== IMU WHO_AM_I Test ===
WHO_AM_I = 0x6B  →  IMU OK: 0x6B — PASS
```

If output shows `FAIL` or `0x00` or `0xFF`:
1. Check CS is connected to GPIO5 (not GPIO15 or other boot pin)
2. Check MOSI/MISO are not swapped (MOSI=23, MISO=19)
3. Confirm IMU breakout VCC is 3.3V — ISM330DHCX is NOT 5V tolerant
4. Try a short jumper wire from ESP32 3V3 pin directly to IMU VCC to rule out breadboard contact issues

---

## Task 5: ToF Distance Test Sketch

Initialises the VL53L5CX in 8×8 mode at 15Hz and prints the average of the 4 center zones. Place a flat board directly below the sensor at approximately 200mm for the pass check.

**Files:**
- Create: `firmware/test_sketches/tof_distance/platformio.ini`
- Create: `firmware/test_sketches/tof_distance/src/main.cpp`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p firmware/test_sketches/tof_distance/src
```

- [ ] **Step 2: Create platformio.ini**

Create `firmware/test_sketches/tof_distance/platformio.ini`:

```ini
[env:esp32dev]
platform     = espressif32
board        = esp32dev
framework    = arduino
build_flags  = -O2 -std=c++17
monitor_speed = 115200
lib_deps     =
    Wire
    stm32duino/STM32duino VL53L5CX @ ^1.2.3
```

- [ ] **Step 3: Create src/main.cpp**

Create `firmware/test_sketches/tof_distance/src/main.cpp`:

```cpp
#include <Arduino.h>
#include <Wire.h>
#include "vl53l5cx_api.h"

#define TOF_SDA  21
#define TOF_SCL  22
#define TOF_LPN  27   // power enable, HIGH = powered
#define TOF_INT  26   // data-ready interrupt, active LOW

VL53L5CX_Configuration dev;
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

    dev.platform.address = 0x52;

    Serial.println("Uploading VL53L5CX firmware (~500ms)...");
    uint8_t status = vl53l5cx_init(&dev);
    if (status != 0) {
        Serial.printf("Init FAILED: status=%d — check SDA/SCL and LPn wiring\n", status);
        return;
    }

    vl53l5cx_set_resolution(&dev, VL53L5CX_RESOLUTION_8X8);
    vl53l5cx_set_ranging_frequency_hz(&dev, 15);
    vl53l5cx_set_sharpener_percent(&dev, 0);
    vl53l5cx_start_ranging(&dev);

    Serial.println("Ranging started. Place a flat board ~200mm below sensor.");
    Serial.println("Center zone average will print at 15Hz.\n");
    init_ok = true;
}

void loop() {
    if (!init_ok) { delay(1000); return; }

    uint8_t ready = 0;
    vl53l5cx_check_data_ready(&dev, &ready);
    if (!ready) { delay(10); return; }

    VL53L5CX_ResultsData results;
    vl53l5cx_get_ranging_data(&dev, &results);

    // Center zones in 8x8 grid: row=3,col=3 → idx=27; row=3,col=4 → 28;
    //                             row=4,col=3 → 35; row=4,col=4 → 36
    uint16_t zones[4] = {
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
```

- [ ] **Step 4: Commit**

```bash
git add firmware/test_sketches/tof_distance/
git commit -m "test: add VL53L5CX distance sketch"
```

- [ ] **Step 5: On RPi5 — pull, wire, and flash**

**Wire VL53L5CX to ESP32 (6 wires):**
- ESP32 GPIO21 (SDA) → ToF SDA
- ESP32 GPIO22 (SCL) → ToF SCL
- ESP32 GPIO27       → ToF LPn
- ESP32 GPIO26       → ToF INT
- ESP32 3V3          → ToF VDD
- ESP32 GND          → ToF GND

**Then on RPi5:**
```bash
cd ~/vla_rob
git pull
cd firmware/test_sketches/tof_distance
pio run --target upload
pio device monitor
```

Note: the VL53L5CX firmware upload takes ~500ms on first boot — you will see a pause before ranging starts. This is normal.

- [ ] **Step 6: Verify pass condition**

Place a flat board (e.g. A4 paper on a hard surface) exactly 200mm below the sensor face.

Expected serial output:
```
=== VL53L5CX Distance Test ===
Uploading VL53L5CX firmware (~500ms)...
Ranging started. Place a flat board ~200mm below sensor.
Center zone average will print at 15Hz.

Center avg:  198 mm  (PASS — within 150-250mm)
Center avg:  199 mm  (PASS — within 150-250mm)
Center avg:  200 mm  (PASS — within 150-250mm)
```

If output shows `Init FAILED`:
1. Check LPn (GPIO27) is HIGH — measure with multimeter
2. Check SDA/SCL wires are correct (GPIO21=SDA, GPIO22=SCL, not swapped)
3. Check VDD is 3.3V — the VL53L5CX is NOT 5V tolerant

If `No valid center zone readings`:
1. Confirm sensor face is pointing downward toward the flat board
2. Confirm object is between 20mm and 600mm from sensor face
3. The I2C address is 0x52 — run an I2C scanner sketch to confirm the device is visible

---

## Task 6: Full Firmware Flash + 50Hz Telemetry Verification

Flash the main firmware and confirm the RPi5 receives well-formed 250-byte packets at 50Hz with all sensor fields populated.

**Files:**
- Create: `firmware/tools/verify_telemetry.py`

- [ ] **Step 1: Create the tools directory and verification script on WSL2**

```bash
mkdir -p firmware/tools
```

Create `firmware/tools/verify_telemetry.py`:

```python
#!/usr/bin/env python3
"""
Run on RPi5 after flashing the full firmware.
Reads 100 telemetry packets from the ESP32, checks rate and packet integrity.

Usage:
    python3 verify_telemetry.py [/dev/ttyUSB0]
"""
import sys
import time
import serial
import numpy as np

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
BAUD = 2_000_000

TELEMETRY_DTYPE = np.dtype([
    ('timestamp_us',     np.uint32),
    ('servo_pos',        np.float32, (5,)),
    ('servo_load',       np.float32, (5,)),
    ('servo_speed',      np.float32, (5,)),
    ('servo_temp',       np.float32, (5,)),
    ('tof_grid',         np.uint16,  (64,)),
    ('tof_timestamp_us', np.uint32),
    ('tof_resolution',   np.uint8),
    ('tof_valid',        np.uint8),
    ('imu_gyro',         np.float32, (3,)),
    ('imu_accel',        np.float32, (3,)),
    ('contact_flag',     np.uint8),
    ('contact_rms',      np.float32),
    ('safety_clamped',   np.uint8),
    ('checksum',         np.uint16),
])
PACKET_SIZE = 250
assert TELEMETRY_DTYPE.itemsize == PACKET_SIZE, \
    f"dtype size mismatch: {TELEMETRY_DTYPE.itemsize} != {PACKET_SIZE}"

def verify_checksum(raw: bytes) -> bool:
    computed = sum(raw[:-2]) & 0xFFFF
    received = int.from_bytes(raw[-2:], 'little')
    return computed == received

def main():
    print(f"Opening {PORT} at {BAUD} baud...")
    ser = serial.Serial(PORT, BAUD, timeout=2.0)
    time.sleep(0.5)  # let ESP32 settle
    ser.reset_input_buffer()

    packets = []
    bad_checksums = 0
    t_start = time.monotonic()

    print("Reading 100 packets...")
    while len(packets) < 100:
        raw = ser.read(PACKET_SIZE)
        if len(raw) != PACKET_SIZE:
            print(f"  Short read: got {len(raw)} bytes (timeout?)")
            continue
        if not verify_checksum(raw):
            bad_checksums += 1
            continue
        pkt = np.frombuffer(raw, dtype=TELEMETRY_DTYPE)[0]
        packets.append(pkt)

    elapsed = time.monotonic() - t_start
    rate = len(packets) / elapsed

    print("\n=== Telemetry Verification ===")
    print(f"Packets:        {len(packets)}")
    print(f"Bad checksums:  {bad_checksums}")
    print(f"Elapsed:        {elapsed:.2f}s")
    print(f"Rate:           {rate:.1f} Hz  (target 50 Hz)  "
          f"{'PASS' if 45 < rate < 55 else 'FAIL'}")

    p = packets[-1]
    print(f"\nLatest packet fields:")
    print(f"  timestamp_us:  {p['timestamp_us']}")
    print(f"  servo_pos[5]:  {np.round(p['servo_pos'], 2)}")
    print(f"  servo_temp[5]: {np.round(p['servo_temp'], 1)}")
    print(f"  imu_gyro[3]:   {np.round(p['imu_gyro'], 3)}")
    print(f"  imu_accel[3]:  {np.round(p['imu_accel'], 3)}")
    print(f"  contact_rms:   {p['contact_rms']:.4f}")
    print(f"  tof_valid:     {p['tof_valid']}")

    # Sanity checks
    checks = [
        ("Rate 45–55 Hz",          45 < rate < 55),
        ("No bad checksums",        bad_checksums == 0),
        ("IMU accel non-zero",      np.any(np.abs(p['imu_accel']) > 0.1)),
        ("Servo temps reasonable",  np.all(p['servo_temp'] > 10) and np.all(p['servo_temp'] < 80)),
        ("timestamp_us non-zero",   p['timestamp_us'] > 0),
    ]

    print("\nChecks:")
    all_pass = True
    for label, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"  {status}  {label}")
        if not result:
            all_pass = False

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES — see above'}")
    ser.close()

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add firmware/tools/verify_telemetry.py
git commit -m "tools: add 50Hz telemetry verification script"
```

- [ ] **Step 3: On RPi5 — install dependencies**

```bash
pip3 install pyserial numpy
```

- [ ] **Step 4: On RPi5 — pull, flash main firmware, run verification**

```bash
cd ~/vla_rob
git pull

# Flash main firmware
cd firmware
pio run --target upload

# Install pyserial + numpy if not done yet
pip3 install pyserial numpy

# Detect the correct port
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null

# Run verification (replace /dev/ttyUSB0 with your actual port)
python3 tools/verify_telemetry.py /dev/ttyUSB0
```

- [ ] **Step 5: Verify pass condition**

Expected output:
```
Opening /dev/ttyUSB0 at 2000000 baud...
Reading 100 packets...

=== Telemetry Verification ===
Packets:        100
Bad checksums:  0
Elapsed:        2.01s
Rate:           49.8 Hz  (target 50 Hz)  PASS

Latest packet fields:
  timestamp_us:  4382910
  servo_pos[5]:  [  0.09  -0.18   0.09  -0.09   0.09]
  servo_temp[5]: [28.0 27.0 28.0 27.0 28.0]
  imu_gyro[3]:   [0.009 -0.018 0.004]
  imu_accel[3]:  [0.012 -0.009 9.807]
  contact_rms:   0.0182
  tof_valid:     1

Checks:
  PASS  Rate 45–55 Hz
  PASS  No bad checksums
  PASS  IMU accel non-zero
  PASS  Servo temps reasonable
  PASS  timestamp_us non-zero

Overall: ALL PASS
```

If rate is correct but `IMU accel non-zero` fails → check SPI wiring.
If rate is correct but `tof_valid: 0` → check I2C wiring.
If rate itself fails → check `CONTROL_HZ` in `config.h` is 50 and WiFi/BT are disabled in `setup()`.

---

## Summary Checklist

- [ ] Task 1: RPi5 + PlatformIO setup verified
- [ ] Task 2: GPIO4 no-op fix committed
- [ ] Task 3: All 5 servos ping — PASS
- [ ] Task 4: IMU WHO_AM_I = 0x6B — PASS
- [ ] Task 5: ToF center zone ~200mm — PASS
- [ ] Task 6: Full firmware 50Hz telemetry stream — ALL PASS

**When all 6 tasks pass:** proceed to Plan 2 (Calibration + Demo Collection).
