# Team's Workplan: Hardware Assembly, Embedded Firmware, Sensor Fusion & Simulation

**Project:** Vision–Language–Action Control for 4-DOF Robotic Manipulation  
**Role:** Hardware Engineer / Embedded Systems / Sensor Fusion  
**MCU:** ESP32-WROOM-32 dev module (Xtensa LX6 dual-core @ 240MHz, FreeRTOS, Arduino + espressif32)
**Reference Document:** `VLA_Robotic_Arm_Project_Report_FINAL.md` (read this fully before starting)  
**Collaborator:** Your friend handles everything AI/ML — training, inference pipeline, dashboard.  
**Your job ends where your friend's begins:** you deliver calibrated hardware + working firmware + raw synchronized teleoperation recordings. Your friend consumes those recordings, generates skill labels, trains models, and runs evaluation.

---

## Your Scope at a Glance

| Layer | Your Responsibility |
|---|---|
| Mechanical | Arm assembly, sensor mounting brackets, overhead camera post |
| Electrical | Power architecture, servo bus wiring, sensor wiring |
| Firmware | ESP32-WROOM-32 C++17 firmware (all drivers, FreeRTOS dual-core loop, contact oracle, safety) |
| Calibration | All 4 calibration scripts (camera intrinsic, Z_table, wrist ToF offset, T_cam_base) |
| Teleoperation | Gamepad teleoperation interface + dataset recorder |
| Simulation | Python kinematic simulator for IK/FK testing without hardware |
| Integration | Serial protocol implementation, integration testing with friend's RPi inference |

---

## Critical Integration Points (What Your Friend Needs From You)

These are hard dependencies. Deliver them on time or your friend cannot proceed.

### 1. Communication Protocol (must be agreed before either of you writes code)
Two packed binary structs over USB serial at 2Mbps. These are **locked** — any change requires coordination with your friend.

**ESP32 → RPi5 (250 bytes, sent at 50Hz):**
```c
typedef struct __attribute__((packed)) {
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
} ControllerTelemetry_t;  // 250 bytes packed — verify with sizeof() in firmware
```

**RPi5 → ESP32 (20 bytes, received at 8Hz):**
```c
typedef struct __attribute__((packed)) {
    float   target_arm[3];    // Target degrees for [J0, J1, J2] — arm joints only
    uint8_t skill_state;      // 0=REACH, 1=GRASP, 2=LIFT, 3=PLACE
    uint8_t execute;          // 1=execute motion, 0=hold position
    float   gripper_command;  // Gripper: 0.0=fully open, 1.0=fully closed
    uint8_t emergency_stop;   // 1=immediate stop all servos (highest priority)
    uint8_t checksum;         // Sum of all preceding bytes mod 256
} RPiCommand_t;  // 20 bytes packed — verify with sizeof()
```

### 2. Dataset Format (delivered end of Week 4)
Each demonstration is saved as one grouped HDF5 file in the `demos/` directory. These files are the raw synchronized recordings; your friend's AI/ML pipeline reads them, generates skill labels, and may write labeled/augmented derivatives under `dataset/processed/`.

```
demos/
  demo_001_pick_red_block.h5
  demo_002_pick_blue_block.h5
  ...
  demo_030_sort_color.h5
```

HDF5 internal structure (per file):
```
/telemetry/
    servo_pos       shape=(N, 5)  dtype=float32  units: degrees
    servo_load      shape=(N, 5)  dtype=float32  units: normalized 0-1
    servo_speed     shape=(N, 5)  dtype=float32  units: deg/s
    servo_temp      shape=(N, 5)  dtype=float32  units: Celsius
    tof_grid        shape=(N, 64) dtype=uint16   units: mm
    tof_timestamp_us shape=(N,)   dtype=uint32   units: ESP32 microseconds
    tof_resolution  shape=(N,)    dtype=uint8    value: 64 for 8×8 mode
    tof_valid       shape=(N,)    dtype=uint8
    imu_gyro        shape=(N, 3)  dtype=float32  units: deg/s
    imu_accel       shape=(N, 3)  dtype=float32  units: m/s²
    contact_flag    shape=(N,)    dtype=uint8
    contact_rms     shape=(N,)    dtype=float32
    safety_clamped  shape=(N,)    dtype=uint8
    checksum         shape=(N,)    dtype=uint16
    timestamps_us   shape=(N,)    dtype=uint32   # ESP32 timestamps
/video/
    rgb_frames      shape=(M, 480, 640, 3) dtype=uint8  # overhead camera frames
    frame_timestamps_us shape=(M,) dtype=uint64 units: microseconds in same timebase as telemetry
/metadata/
    instruction     string  e.g. "pick the red block and place it in the tray"
    task_type       string  "pick_place" | "stacking" | "sorting"
    demo_id         int
    n_telemetry     int     (N)
    n_frames        int     (M)
    telemetry_hz    float   50.0
    video_fps       float   30.0
    date_collected  string
    arm_config      string  path to arm_config.yaml snapshot
```

### 3. Calibration Files (delivered end of Week 3)
Save to `calibration/` directory. Your friend loads these at inference startup.

```
calibration/
  camera_intrinsics.yaml      # K matrix + distortion coefficients
  overhead_height.yaml        # Z_table in meters
  wrist_tof_offset.yaml       # wrist_to_sensor_offset_mm
  camera_to_base_transform.yaml  # T_cam_base 4x4 matrix
  joint_limits.yaml           # min/max degrees per joint
  servo_zero_offsets.yaml     # degree offset per servo for true zero
  arm_config.yaml             # consolidated DH params, limits, workspace, servo IDs
```

---

## Phase 1: Hardware Assembly (Week 1–2)

### Goal
A fully assembled, powered, and electrically verified arm where:
- All 5 servos respond on the serial bus with unique IDs
- ISM330DHCX returns WHO_AM_I = 0x6B over SPI
- VL53L5CX returns distance data over I2C
- Overhead camera produces a 640×480 RGB feed visible on RPi5
- No brownouts when all 5 servos move simultaneously

### 1.1 Component Checklist

Before starting, verify you have every part:

| Component | Qty | Notes |
|---|---|---|
| STS3215 servo (12V variant) | 5 | Verify 12V rated — there is also a 7.4V variant; do NOT mix |
| ESP32-WROOM-32 dev module | 1 | 38-pin variant, USB via onboard CH340 or CP2102 bridge |
| Raspberry Pi 5 (8GB) | 1 | |
| ISM330DHCX breakout board | 1 | SparkFun or Adafruit; must expose SPI pins |
| VL53L5CX breakout board | 1 | ST or Pololu; must expose I2C pins and LPn/INT |
| Pi Camera Module 3 | 1 | With CSI cable long enough to reach post |
| STS3215 UART TTL adapter | 1 | Or wire ESP32 UART2 (GPIO16/17) directly via 74HC126 half-duplex buffer |
| 12V 10A power supply | 1 | Bench supply or brick |
| 12V→5V buck converter (5A rated) | 1 | For RPi5 |
| 1000µF 16V electrolytic capacitor | 1 | Across servo power terminal |
| 22AWG silicone wire | — | 4 wires for ToF, 6 wires for IMU |
| Small cable clips / zip ties | — | Cable management along arm links |
| M3 screws, standoffs | — | For sensor mounting |
| Overhead camera post + mount | 1 | ~50cm tall, rigid, fixed to table |

### 1.2 Servo Bus Configuration

Before mechanical assembly, assign unique IDs to each servo using the Feetech SCServo software on Windows (or `servo_tool` on Linux). Do this one servo at a time with a single servo connected.

**ID Assignment:**
| Servo Physical Role | Bus ID | Notes |
|---|---|---|
| J0 — Base Yaw | 0x01 | |
| J1a — Shoulder (servo A of coupled pair) | 0x02 | |
| J1b — Shoulder (servo B of coupled pair) | 0x03 | Both receive identical commands |
| J2 — Elbow/Wrist Pitch | 0x04 | |
| J3 — Gripper | 0x05 | |

Also set each servo's baud rate to **1Mbps** (register 0x04 = 0x00 in Feetech protocol means 1Mbps).

**Verify**: After ID assignment, connect all 5 to the bus simultaneously and ping each ID. All 5 must respond.

### 1.3 Mechanical Assembly Order

1. Assemble the base with J0 servo in the yaw orientation. Secure to table base plate.
2. Mount J1a and J1b (coupled shoulder pair) on the shoulder axis. Both servo horns must be rigidly connected to the same physical link — use a dual-horn bracket or a printed coupler. The two servos must rotate as one rigid unit; any relative motion between them will damage the servos.
3. Attach the upper arm link (130mm from shoulder axis to elbow axis) to the J1 horn.
4. Mount J2 (elbow/wrist pitch servo) at the end of the upper arm link. Attach the forearm link (190mm) to J2's horn.
5. Mount J3 (gripper) at the end of the forearm. Attach gripper jaw linkage.
6. Mount the VL53L5CX bracket as close to the gripper jaws as mechanically possible, pointing directly downward. The sensor face must be parallel to the table surface when the arm is in pre-grasp hover position.
7. Mount the ISM330DHCX bracket directly on the gripper jaw bracket, as close to the jaw contact point as possible. Orientation: X-axis pointing forward (in the gripper jaw direction), Z-axis pointing down.

### 1.4 Electrical Wiring

**Power Architecture (implement exactly as specified):**

```
12V 10A Supply ─────────────────────────────────────────┐
                │                                        │
                ▼                                        ▼
          Servo Rail                              Buck Converter
          (12V, 8A min)                           (12V → 5V, 5A)
                │                                        │
    ┌───────────┤                                        │
    │    1000µF │                                        ▼
    │    16V cap│                                   Raspberry Pi 5
    └───────────┤                                  (USB-C 5V/5A)
                │                                        │
         STS3215 bus                                     │
         (all 5 servos)                                  ▼
                                                   ESP32-WROOM-32
                                                  (via USB from RPi5)
```

- **CRITICAL:** Servo rail and compute rail must share ONLY a common ground at one point (the power supply negative terminal). Do not connect servo 12V to compute 12V anywhere.
- The 1000µF cap goes directly across the servo power terminals at the JST or screw terminal block, as close to the first servo in the chain as possible.
- The ESP32 is powered from the RPi5 USB port (5V via the dev board's onboard regulator). Do NOT connect the ESP32 VIN to the 12V servo rail.

**ESP32-WROOM-32 Pin Assignments:**

> **Forbidden pins — never use for I/O:**
> - GPIO6–11: connected to internal SPI flash — wiring these will brick the board
> - GPIO34, 35, 36, 39: input-only — cannot be driven as outputs
> - GPIO0, 2, 12, 15: boot strapping pins — avoid for critical I/O

| ESP32 GPIO | Function | Connects To |
|---|---|---|
| GPIO17 (UART2 TX) | STS3215 bus TX | Servo bus data line (via half-duplex mux) |
| GPIO16 (UART2 RX) | STS3215 bus RX | Servo bus data line (via half-duplex mux) |
| GPIO4 | TX_ENABLE | Direction control for half-duplex (HIGH=TX, LOW=RX) |
| GPIO23 (VSPI MOSI) | ISM330DHCX MOSI | IMU SDI |
| GPIO19 (VSPI MISO) | ISM330DHCX MISO | IMU SDO |
| GPIO18 (VSPI SCK) | ISM330DHCX SCK | IMU SCL |
| GPIO5 | ISM330DHCX CS | IMU CS (active LOW) |
| GPIO21 (Wire SDA) | VL53L5CX SDA | ToF SDA |
| GPIO22 (Wire SCL) | VL53L5CX SCL | ToF SCL |
| GPIO27 | VL53L5CX LPn | ToF power enable (HIGH=powered) |
| GPIO26 | VL53L5CX INT | ToF data-ready interrupt (active LOW) |
| USB (via CH340/CP2102) | RPi5 communication | USB-A on RPi5 |

**Servo Bus Half-Duplex Wiring:**
The STS3215 uses a single-wire half-duplex bus. Wire GPIO17 (TX) and GPIO16 (RX) together through a 74HC126 tri-state buffer (or equivalent), with GPIO4 (TX_ENABLE) controlling direction. Alternatively, connect GPIO17 to the bus data line and GPIO16 to the bus data line through a 1kΩ resistor; when receiving, the firmware pulls GPIO4 LOW to tri-state the TX output. The ESP32 Arduino `Serial2` object is initialized with explicit pin arguments so no hardware remapping is needed.

**IMU Wiring (SPI — VSPI bus):**
| ISM330DHCX Pin | ESP32 GPIO | Wire Color (suggestion) |
|---|---|---|
| VCC | 3.3V | Red |
| GND | GND | Black |
| SDI (MOSI) | GPIO23 | Blue |
| SDO (MISO) | GPIO19 | Yellow |
| SCL (SCK) | GPIO18 | Green |
| CS | GPIO5 | Orange |

**ToF Wiring (I2C — Wire bus):**
| VL53L5CX Pin | ESP32 GPIO | Wire Color (suggestion) |
|---|---|---|
| VDD | 3.3V | Red |
| GND | GND | Black |
| SDA | GPIO21 | Blue |
| SCL | GPIO22 | Yellow |
| LPn | GPIO27 | Green |
| INT | GPIO26 | White |

Route IMU and ToF wires along the arm linkages using cable clips. Leave generous slack at each joint — wires must not go taut during full range of motion. Use silicone-insulated wire specifically because it tolerates repeated flexing.

### 1.5 Expected Outputs at End of Phase 1

- [ ] All 5 servos addressable by unique ID on the 1Mbps bus
- [ ] Servo holding positions without drift, load readings reasonable (0–30% at rest)
- [ ] ISM330DHCX WHO_AM_I register reads 0x6B over SPI (write a 5-line Arduino sketch targeting esp32dev to verify)
- [ ] VL53L5CX returns valid distance data (a flat board at 200mm should read 195–210mm)
- [ ] Pi Camera 3 visible at /dev/video0 on RPi5, streaming at 640×480
- [ ] No brownouts: run all 5 servos simultaneously through ±30° motion; RPi5 SSH session must not drop
- [ ] Overhead camera post installed and level; table surface completely visible in frame
- [ ] ESP32 visible on RPi5 as a USB serial device (`/dev/ttyUSB0` or `/dev/ttyACM0`) when connected

---

## Phase 2: ESP32-WROOM-32 Firmware (Week 2–3)

### Goal
A complete ESP32 firmware that runs a deterministic 50Hz control loop delivering:
- Full 5-servo telemetry + IMU + ToF data to RPi5 at 50Hz over USB serial
- Accurate contact detection flag from ISM330DHCX
- Smooth waypoint interpolation from 8Hz RPi5 commands to 50Hz servo writes
- Hardware joint-limit safety on every cycle

### 2.1 Project Setup

Use PlatformIO (recommended). The ESP32 Arduino core (`espressif32`) is well-supported and provides the same Arduino-compatible API surface you would expect from an Arduino framework target.

`platformio.ini`:
```ini
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

> **Note on USB baud rate:** The ESP32 communicates with the PC/RPi5 via an onboard CH340 or CP2102 USB-UART bridge. Both chips support 2Mbps. The `monitor_speed` setting above matches the firmware `USB_BAUD` define. If you see garbled output, confirm your specific bridge chip supports 2Mbps; a fallback to 921600 baud is safe for debugging (update both `monitor_speed` and `USB_BAUD` consistently).

File structure:
```
firmware/
├── src/
│   ├── main.cpp
│   ├── servo_bus.cpp / servo_bus.h
│   ├── ism330dhcx_driver.cpp / ism330dhcx_driver.h
│   ├── tof_driver.cpp / tof_driver.h
│   ├── contact_oracle.cpp / contact_oracle.h
│   ├── waypoint_interp.cpp / waypoint_interp.h
│   ├── safety_layer.cpp / safety_layer.h
│   └── comms.cpp / comms.h
├── include/
│   └── config.h
└── platformio.ini
```

### 2.2 config.h — Global Constants

```c
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

// ── ISM330DHCX (VSPI bus) ────────────────────────────────────────────────────
// VSPI defaults: MOSI=GPIO23, MISO=GPIO19, SCK=GPIO18 — no remapping needed
#define IMU_SPI_CS        5        // GPIO5
#define IMU_SPI_FREQ      10000000UL  // 10MHz

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
#define TELEMETRY_SIZE    250
#define COMMAND_SIZE      20

// ── Temperature thresholds ───────────────────────────────────────────────────
#define TEMP_WARN_C       65.0f
#define TEMP_CUTOFF_C     80.0f

// ── Contact oracle ───────────────────────────────────────────────────────────
#define CONTACT_WINDOW    20      // samples at 6667Hz = 3.0ms
#define CONTACT_THRESHOLD 3.5f   // deg/s RMS, calibrate empirically

// ── USB Serial baud (via onboard CH340/CP2102 bridge) ────────────────────────
#define USB_BAUD          2000000UL

// ── FreeRTOS task config ─────────────────────────────────────────────────────
#define CONTROL_TASK_STACK  8192  // bytes
#define COMMS_TASK_STACK    4096  // bytes
#define CONTROL_TASK_PRIO   10    // higher than comms
#define COMMS_TASK_PRIO     5
```

### 2.3 STS3215 Servo Bus Driver

The Feetech STS3215 uses the Feetech SCServo half-duplex UART protocol. All packets:
`0xFF 0xFF [ID] [LENGTH] [INSTRUCTION] [PARAMS...] [CHECKSUM]`

Checksum = `~(ID + LENGTH + INSTRUCTION + sum(PARAMS)) & 0xFF`

**Key register addresses:**

| Register | Address | Size | Description |
|---|---|---|---|
| Goal Position L | 0x2A | 1 byte | Write: target position low byte |
| Goal Position H | 0x2B | 1 byte | Write: target position high byte |
| Present Position L | 0x38 | 1 byte | Read: current position low byte |
| Present Position H | 0x39 | 1 byte | Read: current position high byte |
| Present Speed L | 0x3A | 1 byte | Read: current speed low byte |
| Present Speed H | 0x3B | 1 byte | Read: current speed high byte |
| Present Load L | 0x3C | 1 byte | Read: load low byte |
| Present Load H | 0x3D | 1 byte | Read: load high byte |
| Present Voltage | 0x3E | 1 byte | Read: voltage (× 0.1V) |
| Present Temperature | 0x3F | 1 byte | Read: temperature (°C) |

**Position encoding:**
- Range: 0 to 4095 (12-bit)
- Center / home: 2047 (servo's mechanical center)
- 1 step = 0.088°
- To convert degrees to steps: `steps = (degrees / 0.0879) + 2047`
- To convert steps to degrees: `degrees = (steps - 2047) * 0.0879`

**servo_bus.h:**
```c
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
// If timing measurements show this exceeds the bus budget, fall back to
// explicitly documented staggered polling with a lower per-servo telemetry rate.
bool servo_poll_all(ServoTelemetry* telemetry);
```

**servo_bus.cpp — key implementation details:**

```c
// Direction control: toggle TX_ENABLE before/after each bus transaction
// HIGH before writing, LOW before expecting reply, wait for reply, then done

// servo_bus.cpp must call Serial2.begin() during servo_bus_init():
//   Serial2.begin(SERVO_BAUD, SERIAL_8N1, SERVO_RX_PIN, SERVO_TX_PIN);
// ESP32 Arduino lets you specify RX/TX pins explicitly — no hardware remapping needed.

static void begin_tx() {
    digitalWrite(SERVO_TX_ENABLE, HIGH);
    delayMicroseconds(2);  // propagation delay
}

static void begin_rx() {
    Serial2.flush();                // ensure TX buffer drained
    digitalWrite(SERVO_TX_ENABLE, LOW);
    delayMicroseconds(2);
}

// Build and send a READ_DATA packet, then wait for response
// Timeout: 5ms (at 1Mbps, a 6-byte response takes ~48µs; 5ms is generous)
static bool send_read(uint8_t id, uint8_t start_addr, uint8_t data_len,
                      uint8_t* response_buf, uint8_t response_len) {
    uint8_t pkt[8];
    pkt[0] = 0xFF;
    pkt[1] = 0xFF;
    pkt[2] = id;
    pkt[3] = 4;        // length = instruction + params + checksum = 4
    pkt[4] = 0x02;     // READ_DATA instruction
    pkt[5] = start_addr;
    pkt[6] = data_len;
    pkt[7] = ~(id + 4 + 0x02 + start_addr + data_len) & 0xFF;

    begin_tx();
    Serial2.write(pkt, 8);
    begin_rx();

    // Wait for (6 + data_len) bytes: 0xFF 0xFF ID LEN ERR DATA... CHECKSUM
    uint32_t deadline = micros() + 5000;
    uint8_t idx = 0;
    while (micros() < deadline && idx < response_len) {
        if (Serial2.available()) {
            response_buf[idx++] = Serial2.read();
        }
    }
    return (idx == response_len);
}

// SYNC WRITE example for goal position to multiple servos at once
// This is the most efficient way to command all servos simultaneously
void servo_sync_write(const uint8_t* ids, const float* positions_deg, uint8_t count) {
    // Sync write packet: 0xFF 0xFF 0xFE L 0x83 start_addr data_len [ID data data]... CHKSUM
    // start_addr=0x2A, data_len=2 (2 bytes per servo for 12-bit position)
    uint8_t param_len = count * 3;  // 3 bytes per servo: ID + 2 position bytes
    uint8_t total_len = param_len + 4;  // +4: start_addr, data_len, + 2 overhead
    uint8_t pkt[3 + 4 + count * 3 + 1];  // header + length + params + checksum
    uint8_t pos = 0;

    pkt[pos++] = 0xFF;
    pkt[pos++] = 0xFF;
    pkt[pos++] = 0xFE;  // broadcast ID
    pkt[pos++] = param_len + 4;
    pkt[pos++] = 0x83;  // SYNC_WRITE
    pkt[pos++] = 0x2A;  // start address: Goal Position L
    pkt[pos++] = 0x02;  // data length per servo: 2 bytes

    uint8_t checksum = 0xFE + (param_len + 4) + 0x83 + 0x2A + 0x02;
    for (int i = 0; i < count; i++) {
        uint16_t steps = (uint16_t)((positions_deg[i] / 0.0879f) + 2047.0f);
        steps = constrain(steps, 0, 4095);
        pkt[pos++] = ids[i];
        pkt[pos++] = steps & 0xFF;
        pkt[pos++] = (steps >> 8) & 0xFF;
        checksum += ids[i] + (steps & 0xFF) + ((steps >> 8) & 0xFF);
    }
    pkt[pos++] = ~checksum & 0xFF;

    begin_tx();
    Serial2.write(pkt, pos);
    // No response expected for sync write (broadcast)
    // Wait for transmission to complete before toggling direction
    Serial2.flush();
    begin_rx();
}
```

### 2.4 ISM330DHCX SPI Driver

**ism330dhcx_driver.h:**
```c
#pragma once
#include <Arduino.h>
#include <SPI.h>

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
```

**ism330dhcx_driver.cpp — register configuration:**

```c
// ISM330DHCX register map (key registers)
#define ISM_WHO_AM_I        0x0F  // Expected value: 0x6B
#define ISM_CTRL1_XL        0x10  // Accelerometer control
#define ISM_CTRL2_G         0x11  // Gyroscope control
#define ISM_CTRL3_C         0x12  // General control
#define ISM_FIFO_CTRL1      0x07  // FIFO watermark LSB
#define ISM_FIFO_CTRL2      0x08  // FIFO watermark MSB
#define ISM_FIFO_CTRL3      0x09  // Batch data rates
#define ISM_FIFO_CTRL4      0x0A  // FIFO mode
#define ISM_FIFO_STATUS1    0x3A  // FIFO fill level LSB
#define ISM_FIFO_STATUS2    0x3B  // FIFO fill level MSB + flags
#define ISM_FIFO_DATA_OUT_TAG 0x78

// SPI helper: write one register
static void imu_write_reg(uint8_t reg, uint8_t val) {
    digitalWrite(IMU_SPI_CS, LOW);
    SPI.beginTransaction(SPISettings(IMU_SPI_FREQ, MSBFIRST, SPI_MODE0));
    SPI.transfer(reg & 0x7F);  // bit7=0 for write
    SPI.transfer(val);
    SPI.endTransaction();
    digitalWrite(IMU_SPI_CS, HIGH);
}

// SPI helper: read one register
static uint8_t imu_read_reg(uint8_t reg) {
    uint8_t val;
    digitalWrite(IMU_SPI_CS, LOW);
    SPI.beginTransaction(SPISettings(IMU_SPI_FREQ, MSBFIRST, SPI_MODE0));
    SPI.transfer(reg | 0x80);  // bit7=1 for read
    val = SPI.transfer(0x00);
    SPI.endTransaction();
    digitalWrite(IMU_SPI_CS, HIGH);
    return val;
}

void imu_init() {
    pinMode(IMU_SPI_CS, OUTPUT);
    digitalWrite(IMU_SPI_CS, HIGH);
    SPI.begin();
    delay(10);

    // Verify WHO_AM_I
    while (imu_read_reg(ISM_WHO_AM_I) != 0x6B) {
        Serial.println("IMU not found! Check wiring.");
        delay(500);
    }

    // CTRL3_C: BDU=1 (block data update), IF_INC=1 (auto address increment for burst reads)
    imu_write_reg(ISM_CTRL3_C, 0x44);

    // CTRL1_XL: ODR_XL[3:0]=1010 (6.67kHz), FS_XL[1:0]=00 (±2g), LPF2=0
    // Bits [7:4]=ODR, bits [3:2]=FS, bit1=LPF2_XL_EN, bit0=0
    // 0b 1010 00 0 0 = 0xA0
    imu_write_reg(ISM_CTRL1_XL, 0xA0);

    // CTRL2_G: ODR_G[3:0]=1010 (6.67kHz), FS_G[1:0]=00 (±250dps)
    // 0b 1010 00 0 0 = 0xA0
    imu_write_reg(ISM_CTRL2_G, 0xA0);

    // FIFO_CTRL3: BDR_GY[3:0]=1010 (6.67kHz), BDR_XL[3:0]=1010 (6.67kHz)
    // bits [7:4]=BDR_GY, bits [3:0]=BDR_XL
    // 0b 1010 1010 = 0xAA
    imu_write_reg(ISM_FIFO_CTRL3, 0xAA);

    // FIFO_CTRL4: FIFO_MODE[2:0]=110 (Continuous — overwrites oldest on full)
    // bits [2:0]=FIFO_MODE
    // 0b 0000 0110 = 0x06
    imu_write_reg(ISM_FIFO_CTRL4, 0x06);

    delay(5);  // ODR settling time
}

// Each FIFO word is 7 bytes: 1 tag byte + 6 data bytes
// Tag byte bits [7:3] identify sample type:
//   00001 = Gyroscope NC (normal compressed)
//   00010 = Accelerometer NC
// Data bytes: int16 × 3 in little-endian order

// Gyro sensitivity at ±250dps: 8.75 mdps/LSB
// Accel sensitivity at ±2g: 0.061 mg/LSB

#define GYRO_SENSITIVITY_MDPS_PER_LSB   8.75f
#define ACCEL_SENSITIVITY_MG_PER_LSB    0.061f

void imu_fifo_read_batch() {
    // Read FIFO_STATUS to get number of stored words
    uint8_t status1 = imu_read_reg(ISM_FIFO_STATUS1);
    uint8_t status2 = imu_read_reg(ISM_FIFO_STATUS2);
    uint16_t samples = ((uint16_t)(status2 & 0x03) << 8) | status1;

    // Burst read all available words (each 7 bytes)
    // Process and discard all but last gyro sample for the oracle
    for (uint16_t i = 0; i < samples; i++) {
        uint8_t word[7];
        // Multi-byte SPI read from FIFO_DATA_OUT_TAG
        digitalWrite(IMU_SPI_CS, LOW);
        SPI.beginTransaction(SPISettings(IMU_SPI_FREQ, MSBFIRST, SPI_MODE0));
        SPI.transfer(ISM_FIFO_DATA_OUT_TAG | 0x80);
        for (int b = 0; b < 7; b++) word[b] = SPI.transfer(0);
        SPI.endTransaction();
        digitalWrite(IMU_SPI_CS, HIGH);

        uint8_t tag = word[0] >> 3;
        int16_t x = (int16_t)(word[1] | (word[2] << 8));
        int16_t y = (int16_t)(word[3] | (word[4] << 8));
        int16_t z = (int16_t)(word[5] | (word[6] << 8));

        if (tag == 0x01) {  // Gyroscope sample
            latest_gyro.gx = x * GYRO_SENSITIVITY_MDPS_PER_LSB / 1000.0f;
            latest_gyro.gy = y * GYRO_SENSITIVITY_MDPS_PER_LSB / 1000.0f;
            latest_gyro.gz = z * GYRO_SENSITIVITY_MDPS_PER_LSB / 1000.0f;
            contact_oracle_push(latest_gyro.gx, latest_gyro.gy, latest_gyro.gz);
        } else if (tag == 0x02) {  // Accelerometer sample
            latest_accel.ax = x * ACCEL_SENSITIVITY_MG_PER_LSB * 0.00981f;  // mg to m/s²
            latest_accel.ay = y * ACCEL_SENSITIVITY_MG_PER_LSB * 0.00981f;
            latest_accel.az = z * ACCEL_SENSITIVITY_MG_PER_LSB * 0.00981f;
        }
    }
}
```

### 2.5 VL53L5CX I2C Driver

Use the ST VL53L5CX Arduino library (available on GitHub: `stm32duino/VL53L5CX`). Follow the library README for PlatformIO installation.

**tof_driver.h:**
```c
#pragma once
#include <Arduino.h>
#include <Wire.h>

struct ToFFrame {
    uint16_t distances_mm[64];  // 8×8 grid, row-major, row 0 = nearest row in FOV
    uint32_t capture_timestamp_us;
    uint8_t  valid;             // 1 if all center zones have valid status
};

void tof_init();                         // Initializes VL53L5CX in 8×8 at 15Hz
bool tof_check_ready();                  // Polls INT pin, returns true if new frame ready
ToFFrame tof_get_latest();               // Returns last captured frame
```

**tof_driver.cpp key steps:**
```c
#include "VL53L5CX_api.h"

static VL53L5CX_Configuration dev;
static ToFFrame latest_frame;

void tof_init() {
    // Power cycle: pull LPn LOW then HIGH
    pinMode(TOF_LPN, OUTPUT);
    pinMode(TOF_INT, INPUT_PULLUP);
    digitalWrite(TOF_LPN, LOW);
    delay(10);
    digitalWrite(TOF_LPN, HIGH);
    delay(100);  // boot time

    Wire.begin(TOF_SDA, TOF_SCL);   // GPIO21=SDA, GPIO22=SCL
    Wire.setClock(400000);          // 400kHz fast mode

    dev.platform.address = TOF_I2C_ADDR;

    // Load firmware and initialize (takes ~500ms due to firmware upload)
    vl53l5cx_init(&dev);

    // Set 8×8 resolution
    vl53l5cx_set_resolution(&dev, VL53L5CX_RESOLUTION_8X8);

    // Set 15Hz ranging frequency
    vl53l5cx_set_ranging_frequency_hz(&dev, 15);

    // Disable sharpener (slight improvement in accuracy for flat objects)
    vl53l5cx_set_sharpener_percent(&dev, 0);

    vl53l5cx_start_ranging(&dev);
    Serial.println("VL53L5CX initialized: 8x8 @ 15Hz");
}

bool tof_check_ready() {
    // Fast path: check INT pin first (avoids I2C transaction)
    if (digitalRead(TOF_INT) == HIGH) return false;

    uint8_t is_ready = 0;
    vl53l5cx_check_data_ready(&dev, &is_ready);
    if (!is_ready) return false;

    VL53L5CX_ResultsData results;
    vl53l5cx_get_ranging_data(&dev, &results);

    latest_frame.capture_timestamp_us = micros();
    latest_frame.valid = 1;

    for (int i = 0; i < 64; i++) {
        // distance_mm is valid if range_sigma_mm[i] < 35 and target_status[i] == 5
        if (results.target_status[i] == 5 && results.range_sigma_mm[i] < 35) {
            latest_frame.distances_mm[i] = results.distance_mm[i];
        } else {
            latest_frame.distances_mm[i] = 0xFFFF;  // invalid sentinel
            if (i == 27 || i == 28 || i == 35 || i == 36) {
                latest_frame.valid = 0;  // center zone invalid
            }
        }
    }
    return true;
}
```

**Note on zone indexing:** In 8×8 mode, zone index maps as: `idx = row * 8 + col`. Center zones (3,3), (3,4), (4,3), (4,4) correspond to indices 27, 28, 35, 36. These are the grasp depth measurement zones.

### 2.6 Contact Oracle (C++ on ESP32)

This is the real contact oracle — it runs at the full FIFO read rate on the ESP32 (called inside `imu_fifo_read_batch()` on every FIFO word), NOT at 50Hz. The Python class in the main report is only a monitoring visualization on the RPi5 side; the actual fast detection is here.

**contact_oracle.h:**
```c
#pragma once
#include <Arduino.h>

void contact_oracle_init(float threshold_dps);
void contact_oracle_push(float gx, float gy, float gz);  // Called for EVERY gyro sample from FIFO
bool contact_oracle_triggered();                          // Latched true after RMS threshold
bool contact_oracle_event();                              // True for one 50Hz cycle on rising edge
float contact_oracle_rms();                               // Current RMS value
void contact_oracle_reset();                              // Call when skill transitions to REACH
```

**contact_oracle.cpp:**
```c
#include "contact_oracle.h"
#include "config.h"

static float gyro_buffer[CONTACT_WINDOW];  // ring buffer of magnitude samples
static int   buf_head = 0;
static int   buf_count = 0;
static float rms_sum_sq = 0.0f;           // running sum of squares
static float threshold = 0.0f;
static bool  triggered = false;
static bool  event_flag = false;
static float current_rms = 0.0f;

void contact_oracle_init(float threshold_dps) {
    threshold = threshold_dps;
    memset(gyro_buffer, 0, sizeof(gyro_buffer));
}

void contact_oracle_push(float gx, float gy, float gz) {
    if (triggered) return;  // Latch until reset

    float mag = sqrtf(gx*gx + gy*gy + gz*gz);

    // Remove oldest sample from running sum
    if (buf_count == CONTACT_WINDOW) {
        rms_sum_sq -= gyro_buffer[buf_head] * gyro_buffer[buf_head];
    } else {
        buf_count++;
    }

    // Add new sample
    gyro_buffer[buf_head] = mag;
    rms_sum_sq += mag * mag;
    buf_head = (buf_head + 1) % CONTACT_WINDOW;

    if (buf_count == CONTACT_WINDOW) {
        current_rms = sqrtf(rms_sum_sq / CONTACT_WINDOW);
        if (current_rms > threshold) {
            triggered = true;
            event_flag = true;
        }
    }
}

bool contact_oracle_triggered() { return triggered; }
bool contact_oracle_event()     { bool e = event_flag; event_flag = false; return e; }
float contact_oracle_rms()      { return current_rms; }
void contact_oracle_reset()     { triggered = false; event_flag = false; buf_count = 0; buf_head = 0; rms_sum_sq = 0.0f; }
```

### 2.7 Waypoint Interpolation

The RPi5 sends target positions at 8Hz (every 125ms). The ESP32 `control_task` interpolates at 50Hz (every 20ms via `vTaskDelayUntil`), producing 6–7 smooth intermediate positions per RPi5 command.

**waypoint_interp.h:**
```c
#pragma once

// Call this when a new RPi5 command arrives
// start[]: current logical positions [J0, J1, J2, J3 gripper]
// end[]: target logical positions [J0, J1, J2, J3 gripper]
void interp_set_targets(const float* start, const float* end, uint32_t duration_us);

// Call every 20ms. Writes interpolated logical positions into out[4].
// Returns true while interpolation is in progress, false when target reached.
bool interp_get_current(float* out);
```

**waypoint_interp.cpp:**
```c
static float interp_start[4], interp_end[4];
static uint32_t interp_start_us, interp_duration_us;

void interp_set_targets(const float* start, const float* end, uint32_t duration_us) {
    memcpy(interp_start, start, 4 * sizeof(float));
    memcpy(interp_end,   end,   4 * sizeof(float));
    interp_start_us = micros();
    interp_duration_us = duration_us;
}

bool interp_get_current(float* out) {
    uint32_t elapsed = micros() - interp_start_us;
    if (elapsed >= interp_duration_us) {
        memcpy(out, interp_end, 4 * sizeof(float));
        return false;
    }
    float t = (float)elapsed / (float)interp_duration_us;  // 0.0 to 1.0
    for (int i = 0; i < 4; i++) {
        out[i] = interp_start[i] + t * (interp_end[i] - interp_start[i]);
    }
    return true;
}
```

### 2.8 Hardware Safety Layer

**safety_layer.h:**
```c
#pragma once

// Joint limits in degrees (set from calibration)
extern float JOINT_MIN[4];
extern float JOINT_MAX[4];

void safety_init();
// Clamps all 4 logical joint values in-place. Sets clamped_flag if any were modified.
void safety_clamp(float* joints, bool* clamped_flag);
```

**safety_layer.cpp:**
```c
float JOINT_MIN[4] = {-150.0f, -90.0f, -120.0f,   0.0f};
float JOINT_MAX[4] = { 150.0f,  90.0f,   30.0f, 100.0f};
// ^ These are placeholder values. Set real limits during calibration.
// Logical joint order: J0 base, J1 coupled shoulder, J2 elbow/wrist, J3 gripper percentage.

void safety_clamp(float* joints, bool* clamped) {
    *clamped = false;
    for (int i = 0; i < 4; i++) {
        if (joints[i] < JOINT_MIN[i]) { joints[i] = JOINT_MIN[i]; *clamped = true; }
        if (joints[i] > JOINT_MAX[i]) { joints[i] = JOINT_MAX[i]; *clamped = true; }
    }
}
```

### 2.9 USB Serial Communication

**comms.h:**
```c
#pragma once
#include "config.h"

// Packet types (match the structs in the main report exactly)
// IMPORTANT: sizeof(ControllerTelemetry_t) must == 250
// IMPORTANT: sizeof(RPiCommand_t) must == 20

void comms_init();
void comms_send_telemetry(const ControllerTelemetry_t* pkt);
bool comms_receive_command(RPiCommand_t* cmd_out);  // Returns true if valid packet received
```

**comms.cpp:**
```c
#include "comms.h"
#include <Arduino.h>

// Checksum: sum of all bytes except the checksum field itself, mod 65536
static uint16_t compute_checksum(const uint8_t* data, size_t len) {
    uint16_t sum = 0;
    for (size_t i = 0; i < len; i++) sum += data[i];
    return sum;
}

void comms_send_telemetry(const ControllerTelemetry_t* pkt) {
    // Fill checksum field in the packet before sending
    // The checksum covers all bytes except the checksum field itself
    ControllerTelemetry_t local = *pkt;
    local.checksum = compute_checksum((uint8_t*)&local, sizeof(local) - sizeof(local.checksum));
    Serial.write((uint8_t*)&local, sizeof(local));
}

bool comms_receive_command(RPiCommand_t* cmd_out) {
    if (Serial.available() < COMMAND_SIZE) return false;

    RPiCommand_t cmd;
    Serial.readBytes((uint8_t*)&cmd, COMMAND_SIZE);

    // Verify checksum
    uint8_t expected_chk = 0;
    const uint8_t* raw = (const uint8_t*)&cmd;
    for (int i = 0; i < COMMAND_SIZE - 1; i++) expected_chk += raw[i];
    if (cmd.checksum != expected_chk) return false;

    *cmd_out = cmd;
    return true;
}
```

### 2.10 Main Control Loop (main.cpp)

The ESP32 runs FreeRTOS. Instead of a bare-metal `loop()`, all work is split into two tasks pinned to separate cores. `setup()` initialises peripherals, creates both tasks, then deletes itself (the Arduino loop task). `loop()` is left empty and never reached.

**Architecture overview:**

| Core | Task | Priority | Stack |
|---|---|---|---|
| Core 1 (app_cpu) | `control_task` — 50Hz servo loop | 10 | 8192 B |
| Core 0 (pro_cpu) | `comms_task` — USB serial to RPi5 | 5 | 4096 B |

WiFi and BT are disabled in `setup()` before anything else. This removes all RF interrupt sources from Core 0.

**Full main.cpp:**

```c
#include <Arduino.h>
#include <WiFi.h>                      // for WiFi.mode(WIFI_OFF)
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include "config.h"
#include "servo_bus.h"
#include "ism330dhcx_driver.h"
#include "tof_driver.h"
#include "contact_oracle.h"
#include "waypoint_interp.h"
#include "safety_layer.h"
#include "comms.h"

// ── Shared state (protected by FreeRTOS mutexes) ─────────────────────────────
static ControllerTelemetry_t g_telemetry = {};
static RPiCommand_t           g_last_cmd = {};
static SemaphoreHandle_t      g_telemetry_mutex;
static SemaphoreHandle_t      g_command_mutex;

// ── Core 1: 50Hz Control Task ─────────────────────────────────────────────────
void control_task(void* pvParameters) {
    TickType_t lastWakeTime = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(CONTROL_PERIOD_MS);  // 20ms = 50Hz

    ControllerTelemetry_t local_telem  = {};
    float current_joints[4]  = {0, 0, 0, 0};
    float last_cmd_joints[4] = {0, 0, 0, 0};
    bool  have_first_cmd     = false;

    while (true) {
        // === STEP 1: IMU FIFO read (processes all queued samples this cycle) ===
        imu_fifo_read_batch();  // calls contact_oracle_push() for every gyro sample

        // === STEP 2: ToF frame check (interrupt-driven, fires ~15Hz) ===
        if (tof_check_ready()) {
            ToFFrame frame = tof_get_latest();
            memcpy(local_telem.tof_grid, frame.distances_mm, 64 * sizeof(uint16_t));
            local_telem.tof_timestamp_us = frame.capture_timestamp_us;
            local_telem.tof_valid        = frame.valid;
            local_telem.tof_resolution   = 64;
        }
        // If no new frame, tof_grid keeps the previous frame (tof_valid unchanged)

        // === STEP 3: Servo telemetry (all 5 servos, once per 20ms) ===
        {
            static const uint8_t ids[5] = {
                SERVO_ID_J0, SERVO_ID_J1A, SERVO_ID_J1B, SERVO_ID_J2, SERVO_ID_J3
            };
            for (uint8_t i = 0; i < SERVO_COUNT; i++) {
                ServoTelemetry st = servo_read_telemetry(ids[i]);
                local_telem.servo_pos[i]   = st.pos_deg;
                local_telem.servo_load[i]  = st.load_norm;
                local_telem.servo_speed[i] = st.speed_dps;
                local_telem.servo_temp[i]  = st.temp_c;
            }
        }

        // === STEP 4: Read latest command from shared buffer (non-blocking) ===
        RPiCommand_t incoming = {};
        if (xSemaphoreTake(g_command_mutex, 0) == pdTRUE) {
            incoming = g_last_cmd;
            xSemaphoreGive(g_command_mutex);
        }
        if (incoming.emergency_stop) {
            // Immediately cease all servo motion — block this task forever
            // TODO: send torque-disable packet to all servos before halting
            while (true) { vTaskDelay(portMAX_DELAY); }
        }
        if (incoming.execute) {
            last_cmd_joints[0] = incoming.target_arm[0];          // J0
            last_cmd_joints[1] = incoming.target_arm[1];          // J1 (J1a + J1b identical)
            last_cmd_joints[2] = incoming.target_arm[2];          // J2
            last_cmd_joints[3] = incoming.gripper_command * 100.0f;  // J3: 0–100%
            interp_set_targets(current_joints, last_cmd_joints, 125000);  // 125ms window
            have_first_cmd = true;
            if (incoming.skill_state == 0) {  // REACH — reset contact latch
                contact_oracle_reset();
            }
        }

        // === STEP 5: Waypoint interpolation → safety clamp → servo sync write ===
        if (have_first_cmd) {
            float target[4];
            interp_get_current(target);

            bool clamped = false;
            safety_clamp(target, &clamped);
            local_telem.safety_clamped = clamped ? 1 : 0;

            const uint8_t sync_ids[5] = {
                SERVO_ID_J0, SERVO_ID_J1A, SERVO_ID_J1B, SERVO_ID_J2, SERVO_ID_J3
            };
            float sync_pos[5] = { target[0], target[1], target[1], target[2], target[3] };
            servo_sync_write(sync_ids, sync_pos, 5);
            memcpy(current_joints, target, sizeof(current_joints));
        }

        // === STEP 6: Assemble telemetry → push to shared buffer for comms_task ===
        local_telem.timestamp_us = micros();
        ImuData imu = imu_get_latest();
        local_telem.imu_gyro[0]  = imu.gx;
        local_telem.imu_gyro[1]  = imu.gy;
        local_telem.imu_gyro[2]  = imu.gz;
        local_telem.imu_accel[0] = imu.ax;
        local_telem.imu_accel[1] = imu.ay;
        local_telem.imu_accel[2] = imu.az;
        local_telem.contact_flag = contact_oracle_event() ? 1 : 0;
        local_telem.contact_rms  = contact_oracle_rms();

        if (xSemaphoreTake(g_telemetry_mutex, 0) == pdTRUE) {
            g_telemetry = local_telem;
            xSemaphoreGive(g_telemetry_mutex);
        }

        // === STEP 7: Sleep until next 20ms deadline ===
        vTaskDelayUntil(&lastWakeTime, period);
    }
}

// ── Core 0: USB Serial Comms Task ─────────────────────────────────────────────
void comms_task(void* pvParameters) {
    while (true) {
        // Send latest telemetry snapshot to RPi5
        ControllerTelemetry_t snap = {};
        if (xSemaphoreTake(g_telemetry_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
            snap = g_telemetry;
            xSemaphoreGive(g_telemetry_mutex);
        }
        comms_send_telemetry(&snap);

        // Receive RPi5 command (returns false immediately if none available)
        RPiCommand_t cmd = {};
        if (comms_receive_command(&cmd)) {
            if (xSemaphoreTake(g_command_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
                g_last_cmd = cmd;
                xSemaphoreGive(g_command_mutex);
            }
        }

        vTaskDelay(pdMS_TO_TICKS(1));  // yield 1ms — comms is not timing-critical
    }
}

// ── Arduino entry points ───────────────────────────────────────────────────────
void setup() {
    // Step 1: disable WiFi and BT entirely before anything else.
    // This prevents RF interrupt sources from causing jitter on Core 0's scheduler.
    WiFi.mode(WIFI_OFF);
    btStop();

    Serial.begin(USB_BAUD);  // UART0 via CH340/CP2102 bridge → RPi5

    // Step 2: create mutexes before spawning tasks
    g_telemetry_mutex = xSemaphoreCreateMutex();
    g_command_mutex   = xSemaphoreCreateMutex();

    // Step 3: initialise all peripherals sequentially on Core 0
    safety_init();
    servo_bus_init();      // internally: Serial2.begin(SERVO_BAUD, SERIAL_8N1, SERVO_RX_PIN, SERVO_TX_PIN)
    imu_init();            // internally: SPI.begin(); configure ISM330DHCX registers
    tof_init();            // internally: Wire.begin(TOF_SDA, TOF_SCL); upload VL53L5CX firmware (~500ms)
    contact_oracle_init(CONTACT_THRESHOLD);
    comms_init();

    // Step 4: pin 50Hz control loop to Core 1 (app_cpu)
    xTaskCreatePinnedToCore(
        control_task,
        "ControlTask",
        CONTROL_TASK_STACK,
        NULL,
        CONTROL_TASK_PRIO,   // priority 10 — highest
        NULL,
        1                    // Core 1
    );

    // Step 5: pin serial comms to Core 0 (pro_cpu)
    xTaskCreatePinnedToCore(
        comms_task,
        "CommsTask",
        COMMS_TASK_STACK,
        NULL,
        COMMS_TASK_PRIO,     // priority 5 — lower than control
        NULL,
        0                    // Core 0
    );

    Serial.println("ESP32 ready.");

    // Step 6: delete the Arduino loop() task — all work is in the two tasks above
    vTaskDelete(NULL);
}

void loop() {
    // Intentionally empty. This task is deleted in setup() and never executes.
}
```

### 2.11 Expected Outputs at End of Phase 2

- [ ] `sizeof(ControllerTelemetry_t)` prints exactly 250 in firmware (add a `Serial.println(sizeof(ControllerTelemetry_t))` to `setup()` for one-time verification, then remove)
- [ ] `sizeof(RPiCommand_t)` prints exactly 20 in firmware (same approach)
- [ ] USB serial streaming at 50Hz visible in serial monitor — 250 bytes every 20ms; measure using a packet capture script on RPi5
- [ ] Both FreeRTOS tasks confirmed running: add a `Serial.printf("core=%d\n", xPortGetCoreID())` in each task's first cycle to verify Core 0 / Core 1 assignment
- [ ] Servo sync write commands produce smooth motion — no jitter, no twitching; test by commanding a slow 0→45° sweep on J0
- [ ] IMU WHO_AM_I confirmed 0x6B; gyro RMS spikes clearly visible on serial monitor when tapping the gripper
- [ ] ToF distances match a ruler measurement within ±8mm for objects at 50–300mm
- [ ] Contact oracle triggers reliably when gripper closes against a hard block; does not trigger during free motion
- [ ] Safety clamp prevents any joint from exceeding hardcoded limits; verify by sending an out-of-range command from a test script
- [ ] No WDT (watchdog timer) resets: monitor serial output for `E (xxxx) task_wdt` messages; if present, investigate which task is blocking

---

## Phase 3: Calibration (Week 3)

### Goal
Produce the 6 calibration files consumed by your friend's inference pipeline. Every measurement must be made with the final, assembled hardware in its permanent workspace position. If you move the camera post after this, recalibrate.

### 3.1 Servo Zero and Joint Limit Calibration

**Output:** `calibration/servo_zero_offsets.yaml` and `calibration/joint_limits.yaml`

**Procedure:**
1. Command all servos to position 2047 (center of 4096-step range).
2. Physically measure the actual angle of each link from a reference (use a digital angle gauge or a protractor).
3. Record the angular offset from true zero for each servo — this is the zero offset.
4. Move each joint to its physical minimum (joint touches mechanical stop or reaches safe limit), record the servo step count. Convert to degrees. Repeat for maximum.

**joint_limits.yaml format:**
```yaml
# All values in degrees, referenced to servo zero offset corrected positions
J0_base:
  min: -145.0
  max:  145.0
J1_shoulder:
  min:  -85.0
  max:   85.0
J2_elbow_wrist:
  min: -110.0
  max:   25.0
J3_gripper:
  min:   0.0    # fully open
  max:  95.0    # fully closed
```

Update `JOINT_MIN` and `JOINT_MAX` arrays in `safety_layer.cpp` with these values and re-flash firmware.

### 3.2 Camera Intrinsic Calibration

**Output:** `calibration/camera_intrinsics.yaml`

**Procedure:**
1. Print a 9×6 OpenCV checkerboard (internal corners). Each square should be 25mm × 25mm.
2. On the RPi5, run the following script:

```python
# calibration/camera_calibrate.py
import cv2
import numpy as np
import glob
import yaml

CHECKERBOARD = (9, 6)     # internal corners
SQUARE_SIZE_MM = 25.0     # mm

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM / 1000.0  # convert to meters

obj_points = []
img_points = []

# Capture 30+ frames by holding checkerboard in various positions
# Save frames to images/ directory first, then run this on them
for fname in glob.glob('calib_images/*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_refined)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print(f"Reprojection error: {ret:.4f}px  (target: < 0.5px)")

with open('calibration/camera_intrinsics.yaml', 'w') as f:
    yaml.dump({
        'image_width': 640, 'image_height': 480,
        'K': K.tolist(),
        'dist': dist.tolist(),
        'reprojection_error_px': float(ret)
    }, f)
print("Saved: calibration/camera_intrinsics.yaml")
```

**Acceptance criterion:** Reprojection error < 0.5 pixels. If higher, collect more calibration images with better checkerboard coverage of the frame corners.

### 3.3 Overhead Camera Height (Z_table)

**Output:** `calibration/overhead_height.yaml`

**Procedure:**
1. Place a flat, uniformly colored reference board (A4 paper) on the table surface directly below the camera.
2. Using the calibrated intrinsics, undistort a camera frame.
3. Physically measure the height from the camera lens to the table surface with a tape measure.
4. Verify: using the pinhole projection formula, compute the pixel width of a known-size object (e.g., an A4 sheet = 210mm wide). Compare computed pixel width to measured pixel width.

```python
# calibration/overhead_height_calib.py
import yaml
import numpy as np

# Physically measured with tape measure
Z_table_m = 0.502  # example: 50.2cm

# Verification: measure a known object in pixels
# Place a 100mm × 100mm square on the table
K = np.array(yaml.safe_load(open('calibration/camera_intrinsics.yaml'))['K'])
fx = K[0, 0]
known_width_m = 0.100
pixel_width_measured = 142  # measure with image ruler

computed_width_px = known_width_m * fx / Z_table_m
error_mm = abs(computed_width_px - pixel_width_measured) * Z_table_m / fx * 1000
print(f"Computed: {computed_width_px:.1f}px  Measured: {pixel_width_measured}px  Error: {error_mm:.1f}mm")

with open('calibration/overhead_height.yaml', 'w') as f:
    yaml.dump({'Z_table_m': Z_table_m, 'verification_error_mm': float(error_mm)}, f)
```

**Acceptance criterion:** Verification error < 5mm.

### 3.4 Wrist ToF Offset Calibration

**Output:** `calibration/wrist_tof_offset.yaml`

**Procedure:**
1. Command the arm to a known pre-grasp hover position directly above the table (arm extended, gripper pointing straight down).
2. Place the flat calibration board on the table exactly below the wrist.
3. Record the ToF center zone average (average of zones 27, 28, 35, 36) from the ESP32 telemetry stream.
4. Physically measure the distance from the gripper reference point to the table surface. Use the reference point your friend's pose estimator treats as the end-effector target.
5. `wrist_to_sensor_offset_mm = tof_reading_mm - physical_gripper_reference_to_table_mm`

This value is positive when the ToF sensor face is above the gripper reference point along the sensor beam. Your friend's pose estimator subtracts this offset from the ToF reading to estimate gripper-reference distance to the object/table.

```python
# calibration/wrist_tof_calib.py
import serial, struct, yaml, numpy as np, time

ser = serial.Serial('/dev/ttyACM0', 2000000, timeout=1)
readings = []

print("Recording 30 ToF center zone readings. Keep arm still.")
for _ in range(30):
    raw = ser.read(250)
    if len(raw) == 250:
        tof = struct.unpack_from('64H', raw, offset=4 + 5*4*4)  # skip header fields
        centers = [tof[27], tof[28], tof[35], tof[36]]
        valid = [z for z in centers if z < 0xFFFF and z > 10]
        if valid:
            readings.append(np.mean(valid))
    time.sleep(0.1)

tof_mm = float(np.median(readings))
gripper_ref_to_table_physical_mm = float(input("Physically measured gripper-reference-to-table distance (mm): "))
offset_mm = tof_mm - gripper_ref_to_table_physical_mm

print(f"ToF reading: {tof_mm:.1f}mm  Physical: {gripper_ref_to_table_physical_mm:.1f}mm  Offset: {offset_mm:.1f}mm")
with open('calibration/wrist_tof_offset.yaml', 'w') as f:
    yaml.dump({
        'wrist_to_sensor_offset_mm': offset_mm,
        'tof_reading_mm': tof_mm,
        'gripper_ref_to_table_physical_mm': gripper_ref_to_table_physical_mm
    }, f)
```

### 3.5 Camera-to-Base Transform (T_cam_base)

**Output:** `calibration/camera_to_base_transform.yaml`

**Procedure:**
Use 4 known calibration positions — physical points on the table whose position in the robot base frame you know exactly (measure with a ruler) and whose pixel centroid you can see in the overhead camera image.

```python
# calibration/compute_T_cam_base.py
import numpy as np
import cv2
import yaml

# Define 4 calibration points
# world_pts: position in robot base frame [X, Y, Z] in meters
# Z = 0 for all (they are on the table surface)
world_pts = np.array([
    [0.10,  0.10, 0.0],
    [0.10, -0.10, 0.0],
    [0.25,  0.10, 0.0],
    [0.25, -0.10, 0.0],
], dtype=np.float64)  # MEASURE THESE PRECISELY WITH RULER

# pixel_pts: manually click each point in the overhead camera image
# Use a small program to display image and record click coordinates
pixel_pts = np.array([
    [312, 189],
    [328, 291],
    [198, 186],
    [214, 289],
], dtype=np.float64)  # RECORD THESE FROM IMAGE

K_data = yaml.safe_load(open('calibration/camera_intrinsics.yaml'))
K = np.array(K_data['K'])
dist = np.array(K_data['dist'])
Z_table = yaml.safe_load(open('calibration/overhead_height.yaml'))['Z_table_m']

# Unproject pixel points to camera frame (at known Z = Z_table)
cam_pts = np.zeros((4, 3), dtype=np.float64)
for i, (u, v) in enumerate(pixel_pts):
    cam_pts[i, 0] = (u - K[0, 2]) * Z_table / K[0, 0]
    cam_pts[i, 1] = (v - K[1, 2]) * Z_table / K[1, 1]
    cam_pts[i, 2] = Z_table

# Least-squares rigid transform: find T_cam_base such that world = T_cam_base @ cam
# Using cv2.solvePnP approach or Umeyama algorithm
# Simple: compute mean-centered least squares rotation + translation
cam_center = cam_pts.mean(axis=0)
world_center = world_pts.mean(axis=0)
H = (cam_pts - cam_center).T @ (world_pts - world_center)
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = Vt.T @ U.T
t = world_center - R @ cam_center

T = np.eye(4)
T[:3, :3] = R
T[:3,  3] = t

residuals = []
for i in range(4):
    cam_h = np.append(cam_pts[i], 1.0)
    pred = T @ cam_h
    err = np.linalg.norm(pred[:2] - world_pts[i][:2]) * 1000  # mm
    residuals.append(err)
    print(f"Point {i}: predicted={pred[:3]*1000:.1f}mm  actual={world_pts[i]*1000:.1f}mm  err={err:.1f}mm")

print(f"Mean error: {np.mean(residuals):.1f}mm")

with open('calibration/camera_to_base_transform.yaml', 'w') as f:
    yaml.dump({'T_cam_base': T.tolist(), 'mean_error_mm': float(np.mean(residuals))}, f)
```

**Acceptance criterion:** Mean calibration point error < 8mm.

### 3.6 Expected Outputs at End of Phase 3

- [ ] `calibration/camera_intrinsics.yaml` — reprojection error < 0.5px
- [ ] `calibration/overhead_height.yaml` — verification error < 5mm
- [ ] `calibration/wrist_tof_offset.yaml` — offset recorded, depth predictions match physical measurements within ±5mm
- [ ] `calibration/camera_to_base_transform.yaml` — mean error < 8mm across 4 calibration points
- [ ] `calibration/joint_limits.yaml` — limits verified: arm cannot reach any limit without firmware preventing it
- [ ] `calibration/servo_zero_offsets.yaml` — all joints read 0° at true mechanical zero
- [ ] `calibration/arm_config.yaml` — consolidated DH parameters, joint limits, workspace, servo IDs, and contact threshold for your friend's runtime

---

## Phase 4: Teleoperation Interface and Dataset Collection (Week 4)

### Goal
Collect 30 raw synchronized teleoperation demonstrations in the exact dataset format your friend's pipeline expects. Your friend's pipeline generates the skill labels during segmentation.

### 4.1 Gamepad Teleoperation Interface

```python
# teleop/teleop_interface.py
import pygame
import serial, struct, time, numpy as np

class TeleopInterface:
    """
    Maps gamepad axes to joint velocity commands.
    Left stick: J0 (yaw) + J1 (shoulder)
    Right stick: J2 (elbow/wrist)
    Triggers: J3 gripper open/close
    Button A: start recording
    Button B: stop recording and save
    Button X: emergency stop
    """
    JOINT_VEL_LIMIT_DPS = 30.0  # max joint velocity during teleop
    CMD_HZ = 50

    def __init__(self, serial_port='/dev/ttyACM0'):
        pygame.init()
        pygame.joystick.init()
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        self.ser = serial.Serial(serial_port, 2000000, timeout=0.01)
        self.joints = [0.0, 0.0, 0.0, 0.0]  # J0, J1, J2, J3
        self.recording = False
        self.episode_data = []

    def run_loop(self):
        dt = 1.0 / self.CMD_HZ
        while True:
            t0 = time.monotonic()
            pygame.event.pump()

            # Axis mapping (tune dead zones to your specific gamepad)
            ax0 = self.joy.get_axis(0)  # Left stick X → J0
            ax1 = -self.joy.get_axis(1) # Left stick Y → J1
            ax2 = -self.joy.get_axis(4) # Right stick Y → J2
            lt = (self.joy.get_axis(2) + 1) / 2   # Left trigger → open gripper
            rt = (self.joy.get_axis(5) + 1) / 2   # Right trigger → close gripper

            # Dead zone
            dead = 0.08
            ax0 = ax0 if abs(ax0) > dead else 0.0
            ax1 = ax1 if abs(ax1) > dead else 0.0
            ax2 = ax2 if abs(ax2) > dead else 0.0

            # Integrate velocities
            self.joints[0] = np.clip(self.joints[0] + ax0 * self.JOINT_VEL_LIMIT_DPS * dt, -145, 145)
            self.joints[1] = np.clip(self.joints[1] + ax1 * self.JOINT_VEL_LIMIT_DPS * dt, -85, 85)
            self.joints[2] = np.clip(self.joints[2] + ax2 * self.JOINT_VEL_LIMIT_DPS * dt, -110, 25)
            gripper_cmd = rt  # 0.0=open, 1.0=closed

            # Build command packet
            cmd = struct.pack('<3fBBfBB',
                self.joints[0], self.joints[1], self.joints[2],
                0,    # skill_state (REACH by default during teleop)
                1,    # execute=1
                gripper_cmd,
                0,    # no emergency stop
                0     # checksum (compute below)
            )
            chk = sum(cmd[:-1]) & 0xFF
            cmd = cmd[:-1] + bytes([chk])
            self.ser.write(cmd)

            # Read telemetry
            raw = self.ser.read(250)
            if len(raw) == 250 and self.recording:
                self.episode_data.append(raw)

            # Button handling
            if self.joy.get_button(0):  # A = start recording
                if not self.recording:
                    self.recording = True
                    self.episode_data = []
                    print("Recording started.")
            if self.joy.get_button(1):  # B = stop and save
                if self.recording:
                    self.recording = False
                    self.save_episode()

            # Maintain loop rate
            elapsed = time.monotonic() - t0
            time.sleep(max(0, dt - elapsed))
```

### 4.2 Dataset Recorder

```python
# teleop/dataset_recorder.py
import h5py
import numpy as np
import struct
import cv2
import os
import yaml
from datetime import datetime

TELEMETRY_DTYPE = np.dtype([
    ('timestamp_us',    np.uint32),
    ('servo_pos',       np.float32, (5,)),
    ('servo_load',      np.float32, (5,)),
    ('servo_speed',     np.float32, (5,)),
    ('servo_temp',      np.float32, (5,)),
    ('tof_grid',        np.uint16,  (64,)),
    ('tof_timestamp_us',np.uint32),
    ('tof_resolution',  np.uint8),
    ('tof_valid',       np.uint8),
    ('imu_gyro',        np.float32, (3,)),
    ('imu_accel',       np.float32, (3,)),
    ('contact_flag',    np.uint8),
    ('contact_rms',     np.float32),
    ('safety_clamped',  np.uint8),
    ('checksum',        np.uint16),
])  # VERIFY: np.dtype.itemsize == 250

def save_demo(raw_packets, rgb_frames, frame_timestamps_us, instruction, task_type, demo_id,
              arm_config_path='calibration/arm_config.yaml'):
    # Parse packets into structured array
    telemetry_list = []
    for pkt in raw_packets:
        if len(pkt) == 250:
            t = np.frombuffer(pkt, dtype=TELEMETRY_DTYPE)
            telemetry_list.append(t[0])
    tel = np.array(telemetry_list)

    filename = f"demos/demo_{demo_id:03d}_{task_type.replace(' ', '_')}.h5"
    os.makedirs('demos', exist_ok=True)

    with h5py.File(filename, 'w') as f:
        grp = f.create_group('telemetry')
        grp.create_dataset('servo_pos',       data=tel['servo_pos'],        compression='gzip')
        grp.create_dataset('servo_load',      data=tel['servo_load'],       compression='gzip')
        grp.create_dataset('servo_speed',     data=tel['servo_speed'],      compression='gzip')
        grp.create_dataset('servo_temp',      data=tel['servo_temp'],       compression='gzip')
        grp.create_dataset('tof_grid',        data=tel['tof_grid'],         compression='gzip')
        grp.create_dataset('tof_timestamp_us', data=tel['tof_timestamp_us'])
        grp.create_dataset('tof_resolution', data=tel['tof_resolution'])
        grp.create_dataset('tof_valid',       data=tel['tof_valid'])
        grp.create_dataset('imu_gyro',        data=tel['imu_gyro'],         compression='gzip')
        grp.create_dataset('imu_accel',       data=tel['imu_accel'],        compression='gzip')
        grp.create_dataset('contact_flag',    data=tel['contact_flag'])
        grp.create_dataset('contact_rms',     data=tel['contact_rms'])
        grp.create_dataset('safety_clamped',  data=tel['safety_clamped'])
        grp.create_dataset('checksum',        data=tel['checksum'])
        grp.create_dataset('timestamps_us',   data=tel['timestamp_us'])

        vgrp = f.create_group('video')
        rgb_arr = np.array(rgb_frames, dtype=np.uint8)
        vgrp.create_dataset('rgb_frames',       data=rgb_arr,    compression='gzip')
        vgrp.create_dataset('frame_timestamps_us', data=np.array(frame_timestamps_us, dtype=np.uint64))

        meta = f.create_group('metadata')
        meta.attrs['instruction']  = instruction
        meta.attrs['task_type']    = task_type
        meta.attrs['demo_id']      = demo_id
        meta.attrs['n_telemetry']  = len(tel)
        meta.attrs['n_frames']     = len(rgb_frames)
        meta.attrs['telemetry_hz'] = 50.0
        meta.attrs['video_fps']    = 30.0
        meta.attrs['date_collected'] = datetime.now().isoformat()
        meta.attrs['arm_config'] = arm_config_path

    print(f"Saved: {filename}  ({len(tel)} telemetry samples, {len(rgb_frames)} frames)")
    return filename
```

### 4.3 Demo Collection Protocol

Collect in this order to ensure balanced dataset:

| Demos | Task | Instruction Template |
|---|---|---|
| 001–010 | Single-object pick-place (red) | "pick the red block and place it in the tray" |
| 011–020 | Single-object pick-place (blue/yellow) | "pick the [color] block and place it in the tray" |
| 021–025 | Block stacking | "stack the [color1] block on top of the [color2] block" |
| 026–030 | Color sorting (3 objects on table) | "pick the red block" / "pick the blue block" |

Between each demo: randomize object positions within ±3cm of center. Record each demo end-to-end: from arm at rest, through REACH→GRASP→LIFT→PLACE, to arm returned to rest.

### 4.4 Expected Outputs at End of Phase 4

- [ ] 30 HDF5 demo files in `demos/` directory
- [ ] Each file has correct dtype (250-byte packets parse cleanly, assert `np.dtype.itemsize == 250`)
- [ ] Each demo: 350–500 telemetry samples (7–10 seconds at 50Hz)
- [ ] Each demo: 200–300 RGB frames (7–10 seconds at 30fps)
- [ ] Delivery to friend: zip `demos/` and `calibration/` and send

---

## Phase 5: Sensor Fusion Validation (Week 4–5)

### Goal
Prove that the full 3D pose estimation pipeline is accurate before your friend depends on it.

### 5.1 Contact Oracle Ground Truth Test

**Method:** Hold a solid block against the gripper by hand. Confirm that `contact_flag` goes HIGH in the ESP32 telemetry. Confirm it does NOT go HIGH when the arm moves freely with no contact.

**Quantitative test:** Log `contact_rms` values during free motion vs. contact events. Compute mean and max for each. Set `CONTACT_THRESHOLD` between `(free_max + contact_min) / 2`. Update in `config.h` and reflash.

**Acceptance:** Zero false positives during 60 seconds of free arm motion. Detects contact on 10/10 block-touching events.

### 5.2 3D Pose Estimation Accuracy Test

Write a Python test script:

```python
# tests/test_pose_estimation.py
import serial, struct, yaml, numpy as np, cv2, time

# Load calibration
K = np.array(yaml.safe_load(open('calibration/camera_intrinsics.yaml'))['K'])
Z_table = yaml.safe_load(open('calibration/overhead_height.yaml'))['Z_table_m']
T_cam_base = np.array(yaml.safe_load(open('calibration/camera_to_base_transform.yaml'))['T_cam_base'])
wrist_offset = yaml.safe_load(open('calibration/wrist_tof_offset.yaml'))['wrist_to_sensor_offset_mm']

# Place a colored block at a KNOWN position (measured with ruler from base origin)
# e.g., block at X=0.20, Y=0.05 from base center
KNOWN_X, KNOWN_Y = 0.20, 0.05

# Step 1: Run YOLOv8-nano detection (or use a simple color threshold for this test)
# Step 2: Compute X, Y from overhead camera
# Step 3: Move arm to pre-grasp hover above the block
# Step 4: Read ToF center zones from ESP32 telemetry
# Step 5: Compute Z
# Step 6: Compare computed 3D pose to known position

# Run 10 trials at the same location
errors = []
for trial in range(10):
    # ... (get detection centroid, read telemetry, compute pose)
    computed_pos = compute_pick_pose(centroid, K, Z_table, tof_grid, wrist_offset, T_cam_base)
    error_mm = np.linalg.norm(computed_pos[:2] - np.array([KNOWN_X, KNOWN_Y])) * 1000
    errors.append(error_mm)
    print(f"Trial {trial}: computed={computed_pos[:2]*1000:.1f}mm  known=[{KNOWN_X*1000:.0f},{KNOWN_Y*1000:.0f}]mm  error={error_mm:.1f}mm")

print(f"\nMean XY error: {np.mean(errors):.1f}mm  Max: {np.max(errors):.1f}mm")
```

**Acceptance:** Mean XY error < 8mm, max < 15mm across 10 trials.

### 5.3 Expected Outputs at End of Phase 5

- [ ] `CONTACT_THRESHOLD` value finalized and documented in `config.h`
- [ ] Contact detection test results log (0 false positives, 100% detection rate)
- [ ] 3D pose estimation test results (mean error < 8mm)
- [ ] Short written note on any calibration issues found, sent to friend

---

## Phase 6: Kinematic Simulation (Week 5)

### Goal
A Python simulation that lets both you and your friend test the IK solver, visualize arm motion, and test the inference pipeline without the physical arm.

### 6.1 Forward Kinematics Python Implementation

```python
# simulation/arm_kinematics.py
import numpy as np
from scipy.spatial.transform import Rotation

# DH Parameters from the project report (meters):
# Link 1 (base):          d1=0.065, a1=0,     alpha1=pi/2
# Link 2 (shoulder):      d2=0,     a2=0.130,  alpha2=0
# Link 3 (elbow/wrist):   d3=0,     a3=0.190,  alpha3=0

DH = [
    # (theta_offset, d, a, alpha)  — all in radians/meters
    (0.0,  0.065, 0.000, np.pi/2),
    (0.0,  0.000, 0.130, 0.0),
    (0.0,  0.000, 0.190, 0.0),
]

def dh_matrix(theta, d, a, alpha):
    """4x4 DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1],
    ])

def forward_kinematics(q_deg):
    """
    Compute end-effector position from joint angles.
    q_deg: [q0, q1, q2] in degrees (arm joints only, not gripper)
    Returns: 4x4 homogeneous transform of end-effector in base frame
    """
    q = np.deg2rad(q_deg)
    T = np.eye(4)
    for i, (q_offset, d, a, alpha) in enumerate(DH):
        T = T @ dh_matrix(q[i] + q_offset, d, a, alpha)
    return T

def inverse_kinematics(x, y, z):
    """
    Closed-form IK for the 3-DOF planar arm.
    Returns [q0, q1, q2] in degrees, or None if unreachable.
    Target (x, y, z) in meters in base frame.
    """
    # Step 1: Base rotation
    q0 = np.arctan2(y, x)

    # Step 2: Project to plane of arm
    r = np.sqrt(x**2 + y**2)  # horizontal distance from base axis
    z_from_base = z - DH[0][1]  # subtract base height d1

    L1, L2 = DH[1][2], DH[2][2]  # link lengths a2, a3

    # Two-link planar IK
    D = (r**2 + z_from_base**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(D) > 1.0:
        return None  # Unreachable

    q2 = np.arctan2(-np.sqrt(1 - D**2), D)  # elbow-down configuration
    q1 = np.arctan2(z_from_base, r) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))

    return [np.rad2deg(q0), np.rad2deg(q1), np.rad2deg(q2)]

def check_workspace(x, y, z):
    """Returns True if position is within defined workspace limits."""
    WORKSPACE = {'x': (-0.38, 0.38), 'y': (-0.38, 0.38), 'z': (0.02, 0.35)}
    return (WORKSPACE['x'][0] <= x <= WORKSPACE['x'][1] and
            WORKSPACE['y'][0] <= y <= WORKSPACE['y'][1] and
            WORKSPACE['z'][0] <= z <= WORKSPACE['z'][1])
```

### 6.2 3D Visualization

```python
# simulation/visualize_arm.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from arm_kinematics import forward_kinematics, DH

def get_joint_positions(q_deg):
    """Returns list of 3D positions for base, each joint, and end-effector."""
    q = np.deg2rad(q_deg)
    positions = [np.array([0, 0, 0])]
    T = np.eye(4)
    for i, (q_offset, d, a, alpha) in enumerate(DH):
        from arm_kinematics import dh_matrix
        T = T @ dh_matrix(q[i] + q_offset, d, a, alpha)
        positions.append(T[:3, 3].copy())
    return positions

def plot_arm(q_deg, target=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    pts = get_joint_positions(q_deg)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]

    ax.plot(xs, ys, zs, 'b-o', linewidth=3, markersize=8)
    ax.scatter(xs[0], ys[0], zs[0], c='g', s=100, label='Base')
    ax.scatter(xs[-1], ys[-1], zs[-1], c='r', s=100, label='End-effector')

    if target is not None:
        ax.scatter(*target, c='orange', s=200, marker='*', label='Target')

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.4, 0.4); ax.set_zlim(0, 0.5)
    ax.legend()
    plt.tight_layout()
    return ax

if __name__ == '__main__':
    from arm_kinematics import inverse_kinematics
    target = [0.25, 0.10, 0.05]
    q = inverse_kinematics(*target)
    if q:
        print(f"IK solution: {q}")
        T = forward_kinematics(q)
        print(f"FK check: {T[:3,3]}")
        plot_arm(q, target=target)
        plt.show()
    else:
        print("Target unreachable.")
```

### 6.3 Mock ESP32 Serial for Testing Friend's Inference Code

```python
# simulation/mock_esp32.py
"""
Runs a fake ESP32 that streams synthetic ControllerTelemetry_t packets over a virtual serial port.
Allows friend to test their inference pipeline without hardware.
Usage: python mock_esp32.py --port /dev/ttyUSB0
Creates a pty pair; friend connects to /dev/pts/X printed at startup.
"""
import struct, time, numpy as np, os, pty, serial

def make_fake_telemetry(t):
    """Generate synthetic telemetry at time t."""
    # Slowly oscillating joints to simulate arm motion
    pos = [10*np.sin(t), 20*np.sin(0.5*t), -15*np.cos(0.3*t), 0, 0]
    load = [0.1, 0.15, 0.1, 0.05, 0.0]
    tof = np.full(64, 250, dtype=np.uint16)
    tof[27] = tof[28] = tof[35] = tof[36] = 180  # simulate object below

    pkt = struct.pack('<I', int(t * 1e6))               # timestamp_us
    for v in pos:   pkt += struct.pack('<f', v)          # servo_pos[5]
    for v in load:  pkt += struct.pack('<f', v)          # servo_load[5]
    for v in pos:   pkt += struct.pack('<f', abs(v)*0.1) # servo_speed[5]
    for _ in range(5): pkt += struct.pack('<f', 35.0)    # servo_temp[5]
    pkt += tof.tobytes()                                 # tof_grid[64]
    pkt += struct.pack('<IBBffffffBfBH',
        int(t * 1e6), 64, 1,                            # tof_timestamp_us, resolution, valid
        0.01, 0.01, 0.01,                               # imu_gyro[3]
        0.0, 0.0, -9.8,                                 # imu_accel[3]
        0, 0.5, 0, 0                                    # contact_flag, rms, safety, checksum
    )
    # Fix checksum
    chk = sum(pkt) & 0xFFFF
    pkt = pkt[:-2] + struct.pack('<H', chk)
    assert len(pkt) == 250
    return pkt

master, slave = pty.openpty()
print(f"Mock ESP32 on: {os.ttyname(slave)}")
t = 0.0
while True:
    t0 = time.monotonic()
    pkt = make_fake_telemetry(t)
    os.write(master, pkt)
    t += 0.02
    time.sleep(max(0, 0.02 - (time.monotonic() - t0)))
```

### 6.4 Expected Outputs at End of Phase 6

- [ ] `simulation/arm_kinematics.py` — FK/IK verified: FK(IK(target)) matches target within 0.5mm for 20 random targets
- [ ] `simulation/visualize_arm.py` — 3D visualization working, shows correct arm geometry
- [ ] `simulation/mock_esp32.py` — streaming synthetic 250-byte telemetry at 50Hz; friend confirms their parser reads it correctly

---

## Phase 7: Integration Testing (Week 7–8)

### Goal
End-to-end validation with both your firmware and your friend's inference pipeline running simultaneously.

### 7.1 Integration Checklist

Run these tests in order:

**Step 1 — Serial handshake:**
- [ ] Friend's `controller_serial.py` successfully parses your 250-byte telemetry packets (print servo positions to confirm)
- [ ] Your ESP32 successfully receives and executes friend's 20-byte command packets
- [ ] Emergency stop command halts all servos immediately (test this first — safety)

**Step 2 — Latency measurement:**
- [ ] Friend measures their actual inference loop time on RPi5. Must report: min, median, max over 100 steps
- [ ] You measure actual control loop timing on the ESP32. Jitter must be < 2ms (FreeRTOS with WiFi/BT off; measure with `micros()` delta at top of control_task)

**Step 3 — Waypoint tracking:**
- [ ] Friend sends a sequence of commanded positions; you confirm arm tracks them smoothly
- [ ] Send an intentionally bad position (outside joint limits). Confirm the ESP32 clamps it and sets `safety_clamped=1`

**Step 4 — Contact oracle integration:**
- [ ] While arm is executing GRASP, close gripper on block, verify `contact_flag=1` appears in telemetry within 20ms

**Step 5 — Full task trial:**
- [ ] Run 5 supervised trials of Task 1 (pick-place) with a human guiding pass/fail
- [ ] If < 3/5 succeed, identify whether failure is hardware/firmware or AI/inference

### 7.2 Evaluation Framework Support

Write the evaluation logger that both of you will use:

```python
# evaluation/eval_logger.py
import json, time, csv

class EvalLogger:
    def __init__(self, task_name, trial_id):
        self.task = task_name
        self.trial = trial_id
        self.start_time = time.monotonic()
        self.events = []
        self.result = None

    def log_event(self, event_type, data=None):
        self.events.append({
            'time_s': time.monotonic() - self.start_time,
            'type': event_type,
            'data': data or {}
        })

    def record_result(self, success, failure_mode=None):
        self.result = {'success': success, 'failure_mode': failure_mode}

    def save(self, output_dir='evaluation/results'):
        import os; os.makedirs(output_dir, exist_ok=True)
        fname = f"{output_dir}/{self.task}_trial{self.trial:03d}.json"
        with open(fname, 'w') as f:
            json.dump({
                'task': self.task, 'trial': self.trial,
                'result': self.result, 'events': self.events
            }, f, indent=2)

    @staticmethod
    def compute_summary(results_dir='evaluation/results'):
        import glob
        results = [json.load(open(f)) for f in glob.glob(f'{results_dir}/*.json')]
        for task in ['pick_place', 'stacking', 'sorting']:
            task_results = [r for r in results if r['task'] == task]
            if task_results:
                success_rate = sum(r['result']['success'] for r in task_results) / len(task_results)
                print(f"{task}: {success_rate:.1%} ({sum(r['result']['success'] for r in task_results)}/{len(task_results)})")
```

### 7.3 Expected Outputs at End of Phase 7

- [ ] Integration test checklist above: all 5 steps passing
- [ ] Evaluation results CSV with at least 60 Task 1 trials, 30 Task 2, 40 Task 3
- [ ] GitHub repository: all firmware + calibration scripts + teleoperation code + simulation committed

---

## Appendix A: Verify Struct Sizes

Add this to `setup()` in main.cpp before any other initialization:

```c
Serial.begin(USB_BAUD);
delay(1000);
Serial.printf("sizeof(ControllerTelemetry_t) = %d  (expected 250)\n", sizeof(ControllerTelemetry_t));
Serial.printf("sizeof(RPiCommand_t) = %d  (expected 20)\n", sizeof(RPiCommand_t));
if (sizeof(ControllerTelemetry_t) != 250 || sizeof(RPiCommand_t) != 20) {
    Serial.println("STRUCT SIZE MISMATCH — check padding, halt.");
    while(1);
}
```

Do not proceed with any development until both sizes print correctly.

---

## Appendix B: Deliverables Timeline

| Week | Deliverables |
|---|---|
| Week 2 | Assembled hardware; all sensors verified; firmware boots and streams |
| Week 3 | Complete firmware with contact oracle + safety; all 6 calibration files |
| Week 4 | 30 demo HDF5 files delivered to friend; teleop interface working |
| Week 5 | Sensor fusion validation results; kinematic simulation + mock ESP32 |
| Week 7 | Integration testing complete with friend's inference pipeline |
| Week 8 | Evaluation complete; all code on GitHub |

---

## Appendix C: Common Failure Modes and Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| Servo twitches erratically | Half-duplex direction control timing | Add 5µs delay between TX and RX toggle |
| RPi5 reboots under servo load | Power rail not isolated | Check capacitor placement and rail separation |
| IMU WHO_AM_I returns 0xFF | SPI CS not pulled HIGH before init | Check CS polarity and GPIO init order |
| ToF reads 0xFFFF constantly | LPn pin not asserted HIGH | Check TOF_LPN pin and toggle sequence |
| Checksum errors on USB serial | USB buffer overflow | Reduce telemetry rate or add flow control |
| Struct size mismatch | Compiler padding | Verify `__attribute__((packed))` on struct definition |
| Joint drifts past limit | Safety clamp not applied to interpolated waypoints | Apply clamp AFTER interpolation, not BEFORE |
