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
    pkt[3] = 2;
    pkt[4] = 0x01;
    pkt[5] = ~(id + 2 + 0x01) & 0xFF;

    Serial.printf("  [DBG] sending ping to ID 0x%02X: ", id);
    for (int i = 0; i < 6; i++) Serial.printf("%02X ", pkt[i]);
    Serial.println();

    Serial2.write(pkt, 6);
    Serial2.flush();

    // Drain echo (half-duplex bus reflects TX back on RX)
    delayMicroseconds(600);  // doubled margin
    uint8_t echo_count = 0;
    while (Serial2.available()) { Serial2.read(); echo_count++; }
    Serial.printf("  [DBG] drained %d echo bytes\n", echo_count);

    uint8_t resp[16];
    uint8_t idx = 0;
    uint32_t deadline = micros() + 50000;  // 50ms timeout (5x longer)
    while (micros() < deadline && idx < 16) {
        if (Serial2.available()) resp[idx++] = Serial2.read();
    }

    Serial.printf("  [DBG] received %d bytes: ", idx);
    for (int i = 0; i < idx; i++) Serial.printf("%02X ", resp[i]);
    Serial.println();

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
    pinMode(SERVO_RX, INPUT_PULLUP);  // enable pull-up on RX2 for open-drain TX output
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
