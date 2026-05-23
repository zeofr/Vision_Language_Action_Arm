#!/usr/bin/env python3
"""
Live servo position monitor.
Auto-detects 254-byte (new firmware with magic) or 250-byte (old firmware) packets.
Usage: python3 servo_monitor.py [/dev/ttyUSB1]
"""
import sys, struct, time, serial

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB1'
BAUD = 2_000_000
MAGIC = bytes([0x5a, 0xa5, 0x5a, 0xa5])


def verify_checksum(raw):
    return (sum(raw[:-2]) & 0xFFFF) == int.from_bytes(raw[-2:], 'little')


def find_magic(s, scan_limit=2000):
    """Scan at most scan_limit bytes for the magic preamble. Returns 254-byte packet or None."""
    tail = bytearray()
    for _ in range(scan_limit):
        b = s.read(1)
        if not b:
            return None
        tail += b
        if len(tail) >= 4 and bytes(tail[-4:]) == MAGIC:
            rest = s.read(250)
            if len(rest) == 250:
                pkt = bytes(tail[-4:]) + rest
                if verify_checksum(pkt):
                    return pkt
            tail = tail[-3:]
    return None


def find_checksum(s, size, scan_limit=None):
    """Slide a size-byte window until checksum passes. Returns packet or None."""
    if scan_limit is None:
        scan_limit = size * 4
    window = bytearray(s.read(size))
    if len(window) < size:
        return None
    for _ in range(scan_limit):
        if verify_checksum(bytes(window)):
            return bytes(window)
        b = s.read(1)
        if not b:
            return None
        window = window[1:] + bytearray(b)
    return None


def parse(pkt, fmt):
    if fmt == 254:
        pos  = struct.unpack_from('<5f', pkt,  8)   # magic(4) + ts(4)
        temp = struct.unpack_from('<5f', pkt, 68)
    else:
        pos  = struct.unpack_from('<5f', pkt,  4)   # ts(4)
        temp = struct.unpack_from('<5f', pkt, 64)
    return pos, temp


def main():
    print(f"Opening {PORT} at {BAUD} baud...")
    s = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(0.5)
    s.reset_input_buffer()

    # Check data is flowing
    probe = s.read(16)
    if not probe:
        print("ERROR: no data from ESP32. Check USB and power.")
        s.close()
        return
    print(f"Data flowing. Detecting packet format...")
    s.reset_input_buffer()

    # Try new firmware first (magic bytes)
    pkt = find_magic(s, scan_limit=2000)
    if pkt:
        fmt = 254
        print("Format: NEW firmware — 254-byte packets with magic.")
    else:
        # Fall back to old firmware (checksum sync)
        s.reset_input_buffer()
        print("No magic found. Trying old 250-byte format...")
        pkt = find_checksum(s, 250)
        if pkt:
            fmt = 250
            print("Format: OLD firmware — 250-byte packets.")
        else:
            print("Could not sync. Raw bytes:", probe.hex())
            print("Run:  cd ~/vla_rob/firmware && pio run -t clean && pio run -t upload")
            s.close()
            return

    print("Synced. Move the arm. Ctrl+C to stop.\n")
    print(f"  {'J0-Yaw':>8} {'J1a-Shldr':>10} {'J1b-Shldr':>10} {'J2-Elbow':>10} {'J3-Grip':>9}    "
          f"{'T0':>5} {'T1':>5} {'T2':>5} {'T3':>5} {'T4':>5}")
    print("─" * 90)

    try:
        while True:
            if fmt == 254:
                pkt = find_magic(s, scan_limit=500)
            else:
                raw = s.read(fmt)
                pkt = raw if len(raw) == fmt and verify_checksum(raw) else None
                if pkt is None:
                    s.reset_input_buffer()
                    pkt = find_checksum(s, fmt)

            if not pkt:
                s.reset_input_buffer()
                continue

            pos, temp = parse(pkt, fmt)
            p = '  '.join(f'{v:+8.2f}' for v in pos)
            t = '  '.join(f'{v:5.1f}' for v in temp)
            print(f"  {p}    {t}", end='\r')

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        s.close()


if __name__ == '__main__':
    main()
