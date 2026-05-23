#include "servo_bus.h"
#include "config.h"

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

void servo_write_deg(uint8_t id, float degrees) {
    uint16_t steps = (uint16_t)((degrees / 0.0879f) + 2047.0f);
    steps = constrain(steps, 0, 4095);

    uint8_t pkt[9];
    pkt[0] = 0xFF;
    pkt[1] = 0xFF;
    pkt[2] = id;
    pkt[3] = 5; // length
    pkt[4] = 0x03; // WRITE_DATA
    pkt[5] = 0x2A; // Goal Position L
    pkt[6] = steps & 0xFF;
    pkt[7] = (steps >> 8) & 0xFF;
    pkt[8] = ~(id + 5 + 0x03 + 0x2A + pkt[6] + pkt[7]) & 0xFF;

    begin_tx();
    Serial2.write(pkt, 9);
    Serial2.flush();
    begin_rx();
}

void servo_sync_write(const uint8_t* ids, const float* positions_deg, uint8_t count) {
    // Sync write packet: 0xFF 0xFF 0xFE L 0x83 start_addr data_len [ID data data]... CHKSUM
    // start_addr=0x2A, data_len=2 (2 bytes per servo for 12-bit position)
    uint8_t param_len = count * 3;  // 3 bytes per servo: ID + 2 position bytes
    uint8_t total_len = param_len + 4;  // +4: start_addr, data_len, + 2 overhead
    uint8_t pkt[3 + 4 + 15 + 1];  // max 5 servos -> count * 3 = 15
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
    Serial2.flush();
    begin_rx();
}

static bool send_read(uint8_t id, uint8_t start_addr, uint8_t data_len,
                      uint8_t* response_buf, uint8_t response_len) {
    // Clear incoming RX buffer leftovers
    while (Serial2.available()) {
        Serial2.read();
    }

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
    Serial2.flush();
    begin_rx();

    // Wait for response: 0xFF 0xFF ID LEN ERR DATA... CHECKSUM
    uint32_t deadline = micros() + 5000;
    uint8_t idx = 0;
    while (micros() < deadline && idx < response_len) {
        if (Serial2.available()) {
            response_buf[idx++] = Serial2.read();
        }
    }

    if (idx != response_len) {
        return false;
    }

    // Basic packet structure check
    if (response_buf[0] != 0xFF || response_buf[1] != 0xFF || response_buf[2] != id) {
        return false;
    }

    // Verify response checksum
    uint8_t sum = 0;
    for (int i = 2; i < response_len - 1; i++) {
        sum += response_buf[i];
    }
    uint8_t calc_checksum = ~sum & 0xFF;
    if (response_buf[response_len - 1] != calc_checksum) {
        return false;
    }

    return true;
}

ServoTelemetry servo_read_telemetry(uint8_t id) {
    ServoTelemetry telem = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint8_t response[14]; // 6 bytes overhead + 8 bytes data (from 0x38 to 0x3F)

    if (send_read(id, 0x38, 8, response, 14)) {
        // Data bytes start at index 5 of the response packet:
        // response[0]=0xFF, response[1]=0xFF, response[2]=ID, response[3]=LEN, response[4]=ERR
        uint8_t* data = &response[5];

        // Present Position (0x38, 0x39)
        uint16_t steps = data[0] | (data[1] << 8);
        telem.pos_deg = ((float)steps - 2047.0f) * 0.0879f;

        // Present Speed (0x3A, 0x3B)
        int16_t raw_speed = (int16_t)(data[2] | (data[3] << 8));
        int16_t speed_val = raw_speed & 0x7FFF;
        if (raw_speed & 0x8000) {
            speed_val = -speed_val;
        }
        telem.speed_dps = (float)speed_val * 0.0879f;

        // Present Load (0x3C, 0x3D)
        int16_t raw_load = (int16_t)(data[4] | (data[5] << 8));
        int16_t load_val = raw_load & 0x03FF;
        telem.load_norm = (float)load_val / 1023.0f;
        if (raw_load & 0x0400) {
            telem.load_norm = -telem.load_norm; // keep sign if direction is indicated
        }

        // Present Voltage (0x3E)
        telem.voltage_v = (float)data[6] * 0.1f;

        // Present Temperature (0x3F)
        telem.temp_c = (float)data[7];
    }
    return telem;
}

bool servo_poll_all(ServoTelemetry* telemetry) {
    static const uint8_t ids[5] = {
        SERVO_ID_J0, SERVO_ID_J1A, SERVO_ID_J1B, SERVO_ID_J2, SERVO_ID_J3
    };
    bool ok = true;
    for (uint8_t i = 0; i < SERVO_COUNT; i++) {
        telemetry[i] = servo_read_telemetry(ids[i]);
    }
    return ok;
}
