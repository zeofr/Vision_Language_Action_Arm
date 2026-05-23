#include "comms.h"
#include "config.h"
#include <Arduino.h>

void comms_init() {
    // USB serial initialization (Serial.begin) is managed in setup() to ensure 
    // early debug feedback, but we define comms_init() here for compliance with the API.
}

static uint16_t compute_telemetry_checksum(const uint8_t* data, size_t len) {
    uint16_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum;
}

void comms_send_telemetry(const ControllerTelemetry_t* pkt) {
    ControllerTelemetry_t local = *pkt;
    local.magic = TELEMETRY_MAGIC;
    local.checksum = compute_telemetry_checksum((const uint8_t*)&local, sizeof(ControllerTelemetry_t) - sizeof(local.checksum));
    Serial.write((const uint8_t*)&local, sizeof(ControllerTelemetry_t));
}

bool comms_receive_command(RPiCommand_t* cmd_out) {
    // If we don't have enough bytes yet, return immediately (non-blocking)
    if (Serial.available() < (int)sizeof(RPiCommand_t)) {
        return false;
    }
    
    // Read the exact command frame size
    RPiCommand_t cmd;
    Serial.readBytes((uint8_t*)&cmd, sizeof(RPiCommand_t));
    
    // Compute the 8-bit mod-256 checksum over the first 19 bytes of the command frame
    uint8_t expected_checksum = 0;
    const uint8_t* raw_bytes = (const uint8_t*)&cmd;
    for (size_t i = 0; i < sizeof(RPiCommand_t) - 1; i++) {
        expected_checksum += raw_bytes[i];
    }
    
    // If the checksum fails, discard the packet to prevent erratic robot motion
    if (cmd.checksum != expected_checksum) {
        return false;
    }
    
    *cmd_out = cmd;
    return true;
}
