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

MAGIC       = 0xA55AA55A
MAGIC_BYTES = MAGIC.to_bytes(4, 'little')

TELEMETRY_DTYPE = np.dtype([
    ('magic',            np.uint32),
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
PACKET_SIZE = 254
assert TELEMETRY_DTYPE.itemsize == PACKET_SIZE, \
    f"dtype size mismatch: {TELEMETRY_DTYPE.itemsize} != {PACKET_SIZE}"


def verify_checksum(raw: bytes) -> bool:
    computed = sum(raw[:-2]) & 0xFFFF
    received = int.from_bytes(raw[-2:], 'little')
    return computed == received


def find_packet(ser) -> bytes | None:
    """Scan the stream for the magic preamble, then read and checksum-verify
    the full packet. Returns the 254-byte packet or None on timeout."""
    tail = bytearray()
    while True:
        b = ser.read(1)
        if not b:
            return None
        tail += b
        if len(tail) < 4:
            continue
        if bytes(tail[-4:]) != MAGIC_BYTES:
            if len(tail) > 4:
                tail = tail[-4:]
            continue
        # Magic found — read the remaining 250 bytes
        rest = ser.read(PACKET_SIZE - 4)
        if len(rest) < PACKET_SIZE - 4:
            return None
        pkt = bytes(tail[-4:]) + rest
        if verify_checksum(pkt):
            return pkt
        # Checksum failed on this magic match — keep scanning from rest
        tail = bytearray(rest[-3:])


def main():
    print(f"Opening {PORT} at {BAUD} baud...")
    ser = serial.Serial(PORT, BAUD, timeout=2.0)
    try:
        time.sleep(0.5)
        ser.reset_input_buffer()

        packets = []

        print("Syncing to packet stream...")
        pkt_bytes = find_packet(ser)
        if pkt_bytes is None:
            print("  Timeout during sync — no valid packet from ESP32")
            return
        print("  Synced.")

        t_start = time.monotonic()
        print("Reading 100 packets...")
        packets.append(np.frombuffer(pkt_bytes, dtype=TELEMETRY_DTYPE)[0])

        while len(packets) < 100:
            pkt_bytes = find_packet(ser)
            if pkt_bytes is None:
                print(f"  Timeout waiting for packet {len(packets) + 1}")
                continue
            packets.append(np.frombuffer(pkt_bytes, dtype=TELEMETRY_DTYPE)[0])

        elapsed = time.monotonic() - t_start
        rate = len(packets) / elapsed

        print("\n=== Telemetry Verification ===")
        print(f"Packets:   {len(packets)}")
        print(f"Elapsed:   {elapsed:.2f}s")
        print(f"Rate:      {rate:.1f} Hz  (target 50 Hz)  "
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

        checks = [
            ("Rate 45–55 Hz",          45 < rate < 55),
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
    finally:
        ser.close()


if __name__ == '__main__':
    main()
