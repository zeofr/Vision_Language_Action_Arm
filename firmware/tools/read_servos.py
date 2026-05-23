#!/usr/bin/env python3
"""
Live servo position reader — SmartElex board direct USB.
Usage: python3 read_servos.py [/dev/ttyUSB0]
"""
import sys, time, serial

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
BAUD = 1_000_000

# J0, J1A, J1B, J2, J3
IDS   = [0x01, 0x02, 0x03, 0x04, 0x05]
NAMES = ['J0-Yaw', 'J1a-Shldr', 'J1b-Shldr', 'J2-Elbow', 'J3-Grip']

POS_REG = 0x38  # Present Position, 2 bytes


def read_pkt(sid):
    length = 4
    instr  = 0x02
    data_len = 2
    chk = (~(sid + length + instr + POS_REG + data_len)) & 0xFF
    return bytes([0xFF, 0xFF, sid, length, instr, POS_REG, data_len, chk])


def parse(resp):
    if len(resp) < 7 or resp[0] != 0xFF or resp[1] != 0xFF:
        return None
    if resp[4] != 0:  # error byte
        return None
    return resp[5] | (resp[6] << 8)


def steps_to_deg(s):
    return s * 360.0 / 4096.0


def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.05)
    time.sleep(0.2)
    ser.reset_input_buffer()

    print(f"Connected to {PORT} at {BAUD} baud. Move the arm. Ctrl+C to stop.\n")
    header = '  '.join(f'{n:>10}' for n in NAMES)
    print(f"  {header}")
    print('─' * 65)

    try:
        while True:
            cols = []
            for sid in IDS:
                ser.reset_input_buffer()
                ser.write(read_pkt(sid))
                resp = ser.read(8)
                pos = parse(resp)
                cols.append(f'{steps_to_deg(pos):+10.2f}' if pos is not None else f'{"---":>10}')
            print('  ' + '  '.join(cols), end='\r')
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        ser.close()


if __name__ == '__main__':
    main()
