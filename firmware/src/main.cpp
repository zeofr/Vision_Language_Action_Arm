#include <Arduino.h>
#include <WiFi.h>                      // For WiFi.mode(WIFI_OFF)
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

// ── Shared State (Protected by FreeRTOS Mutexes) ──────────────────────────────
static ControllerTelemetry_t g_telemetry = {};
static RPiCommand_t           g_last_cmd = {};
static SemaphoreHandle_t      g_telemetry_mutex = nullptr;
static SemaphoreHandle_t      g_command_mutex = nullptr;

// ── Core 1: 50Hz Control Loop Task (High Priority) ───────────────────────────
void control_task(void* pvParameters) {
    TickType_t lastWakeTime = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(CONTROL_PERIOD_MS);  // 20ms = 50Hz

    ControllerTelemetry_t local_telem = {};
    float current_joints[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float last_cmd_joints[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool  have_first_cmd     = false;

    // Static assertions (guaranteeing exact specification struct byte sizes)
    static_assert(sizeof(ControllerTelemetry_t) == 254, "ERROR: ControllerTelemetry_t size must be exactly 254 bytes!");
    static_assert(sizeof(RPiCommand_t) == 20, "ERROR: RPiCommand_t size must be exactly 20 bytes!");

    Serial.println("Control Task: Initialized and running on Core 1.");

    while (true) {
        // === STEP 1: IMU FIFO read (processes all queued samples this cycle) ===
        imu_fifo_read_batch();  // Internally calls contact_oracle_push() for every gyro sample

        // === STEP 2: ToF frame check (interrupt-driven pin check) ===
        if (tof_check_ready()) {
            ToFFrame frame = tof_get_latest();
            memcpy(local_telem.tof_grid, frame.distances_mm, 64 * sizeof(uint16_t));
            local_telem.tof_timestamp_us = frame.capture_timestamp_us;
            local_telem.tof_valid        = frame.valid;
            local_telem.tof_resolution   = 64;
        }
        // If no new frame is ready, tof_grid keeps the previous frame, and validity stays unchanged

        // === STEP 3: Servo telemetry (polled sequentially) ===
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
        bool cmd_updated = false;
        if (xSemaphoreTake(g_command_mutex, 0) == pdTRUE) {
            incoming = g_last_cmd;
            cmd_updated = incoming.execute;  // Only register as new if execute flag is raised
            xSemaphoreGive(g_command_mutex);
        }

        if (incoming.emergency_stop) {
            // Immediate ESTOP override: disable servo torque and lock CPU
            Serial.println(">>> FATAL: EMERGENCY STOP RECEIVED. Torque Disabled. Halting control task. <<<");
            
            // Send torque disable command to all Feetech servos (SCServo register 0x28 is Torque Enable)
            static const uint8_t ids[5] = {
                SERVO_ID_J0, SERVO_ID_J1A, SERVO_ID_J1B, SERVO_ID_J2, SERVO_ID_J3
            };
            for (uint8_t i = 0; i < SERVO_COUNT; i++) {
                // To disable torque, write 0 to register 0x28 (Torque Enable)
                uint8_t pkt[8] = { 0xFF, 0xFF, ids[i], 4, 0x03, 0x28, 0, 0 };
                pkt[7] = ~(ids[i] + 4 + 0x03 + 0x28 + 0) & 0xFF;
                
                // Write command directly to UART2
                digitalWrite(SERVO_TX_ENABLE, HIGH);
                Serial2.write(pkt, 8);
                Serial2.flush();
                digitalWrite(SERVO_TX_ENABLE, LOW);
            }
            
            // Loop forever, halting all motor and control processes
            while (true) {
                vTaskDelay(portMAX_DELAY);
            }
        }

        if (cmd_updated) {
            last_cmd_joints[0] = incoming.target_arm[0];          // J0 Base Yaw
            last_cmd_joints[1] = incoming.target_arm[1];          // J1 coupled shoulder (shared position)
            last_cmd_joints[2] = incoming.target_arm[2];          // J2 Elbow/Wrist
            last_cmd_joints[3] = incoming.gripper_command * 100.0f;  // J3 Gripper scale: 0.0-1.0 mapped to 0-100%
            
            // Smoothly linearize the transition from current angles to the 8Hz goal over a 125ms duration
            interp_set_targets(current_joints, last_cmd_joints, 125000);
            have_first_cmd = true;

            // Reset contact oracle when moving into a new REACH sequence
            if (incoming.skill_state == 0) {
                contact_oracle_reset();
            }
        }

        // === STEP 5: Interpolation → Safety Limits Clamping → Bus command write ===
        if (have_first_cmd) {
            float target[4];
            interp_get_current(target);

            // Safety limit clamping
            bool clamped = false;
            safety_clamp(target, &clamped);
            local_telem.safety_clamped = clamped ? 1 : 0;

            // Write targets out to Feetech serial bus
            const uint8_t sync_ids[5] = {
                SERVO_ID_J0, SERVO_ID_J1A, SERVO_ID_J1B, SERVO_ID_J2, SERVO_ID_J3
            };
            // Coupled shoulder J1 receives identical target sweeps to balance mechanical torque
            float sync_pos[5] = { target[0], target[1], target[1], target[2], target[3] };
            
            servo_sync_write(sync_ids, sync_pos, 5);
            memcpy(current_joints, target, sizeof(current_joints));
        }

        // === STEP 6: Assemble telemetry fields → Push to shared buffer ===
        local_telem.timestamp_us = micros();
        ImuData imu = imu_get_latest();
        local_telem.imu_gyro[0]  = imu.gx;
        local_telem.imu_gyro[1]  = imu.gy;
        local_telem.imu_gyro[2]  = imu.gz;
        local_telem.imu_accel[0] = imu.ax;
        local_telem.imu_accel[1] = imu.ay;
        local_telem.imu_accel[2] = imu.az;
        local_telem.contact_flag = contact_oracle_triggered() ? 1 : 0;
        local_telem.contact_rms  = contact_oracle_rms();

        // Push telemetry snapshot to shared global structure for Core 0 to send
        if (xSemaphoreTake(g_telemetry_mutex, 0) == pdTRUE) {
            g_telemetry = local_telem;
            xSemaphoreGive(g_telemetry_mutex);
        }

        // === STEP 7: Delay until the exact 20ms Control Deadline ===
        vTaskDelayUntil(&lastWakeTime, period);
    }
}

// ── Core 0: 50Hz Serial Comms Task (Lower Priority) ───────────────────────────
void comms_task(void* pvParameters) {
    Serial.println("Comms Task: Initialized and running on Core 0.");

    TickType_t lastWakeTime = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(CONTROL_PERIOD_MS);  // 20ms = 50Hz

    while (true) {
        // Step A: Send the latest telemetry snapshot to the Raspberry Pi 5
        ControllerTelemetry_t snapshot = {};
        if (xSemaphoreTake(g_telemetry_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
            snapshot = g_telemetry;
            xSemaphoreGive(g_telemetry_mutex);
        }
        comms_send_telemetry(&snapshot);

        // Step B: Receive RPi5 command frame (non-blocking)
        RPiCommand_t command = {};
        if (comms_receive_command(&command)) {
            if (xSemaphoreTake(g_command_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
                g_last_cmd = command;
                xSemaphoreGive(g_command_mutex);
            }
        }

        vTaskDelayUntil(&lastWakeTime, period);
    }
}

// ── Arduino Boot Entrypoint ──────────────────────────────────────────────────
void setup() {
    // Step 1: Immediately deactivate ESP32 radio interfaces (WiFi + Bluetooth).
    // This turns off radio-frequency interrupt handlers, ensuring Core 0 scheduler latency remains ultra-low.
    WiFi.mode(WIFI_OFF);
    btStop();

    // Start high-speed USB-serial connection (via onboard CP2102/CH340)
    Serial.begin(USB_BAUD);
    delay(500);

    Serial.println("\n--- ROBOTIC CONTROL FIRMWARE INITIALIZATION ---");

    // Step 2: Create binary mutex objects for safe dual-core data passage
    g_telemetry_mutex = xSemaphoreCreateMutex();
    g_command_mutex   = xSemaphoreCreateMutex();

    if (g_telemetry_mutex == nullptr || g_command_mutex == nullptr) {
        Serial.println(">>> FATAL ERROR: Mutex Allocation Failed! Halting setup. <<<");
        while (true) { delay(1000); }
    }

    // Step 3: Initialize all hardware peripherals sequentially on Core 0
    safety_init();
    servo_bus_init();
    
    // SPI IMU configuration
    imu_init();
    
    // I2C Time-of-Flight calibration and firmware upload (takes ~500ms)
    tof_init();
    
    // High-rate contact oracle
    contact_oracle_init(CONTACT_THRESHOLD);
    
    // Comms parameters
    comms_init();

    Serial.println("All peripheral drivers initialized. Spawning FreeRTOS tasks...");

    // Step 4: Spawn the 50Hz main Control Task on Core 1 (APP CPU)
    xTaskCreatePinnedToCore(
        control_task,
        "ControlTask",
        CONTROL_TASK_STACK,
        NULL,
        CONTROL_TASK_PRIO,
        NULL,
        1  // Pinned to Core 1
    );

    // Step 5: Spawn the Serial Communication Task on Core 0 (PRO CPU)
    xTaskCreatePinnedToCore(
        comms_task,
        "CommsTask",
        COMMS_TASK_STACK,
        NULL,
        COMMS_TASK_PRIO,
        NULL,
        0  // Pinned to Core 0
    );

    Serial.println("System Tasks running. Deleting boot manager task.");

    // Step 6: Delete the default Arduino Setup/Loop task. 
    // This fully hands control to our dual-core task loops, preventing setup stack leak.
    vTaskDelete(NULL);
}

void loop() {
    // This block is never executed since setup deletes itself.
}
