"""
USB serial link to the Teensy microcontroller.

Wire protocol
─────────────
Direction   Rate    Packet size
Teensy→RPi  50 Hz   250 bytes  (TELEMETRY_DTYPE)
RPi→Teensy   8 Hz    20 bytes  (COMMAND_DTYPE)

Physical layer: /dev/ttyACM0 @ 2 Mbaud, 8N1.

Checksum: XOR of all bytes excluding the final checksum byte itself.
Magic bytes mark the start of each frame type.

Telemetry fields of interest for the inference loop:
  servo_pos_raw   – uint16[5]   raw servo ticks
  imu_gyro_rms    – float32     RMS angular rate, dps (used by contact oracle)
  contact_flag    – uint8       set by Teensy when imu_gyro_rms > threshold
  tof_grid        – uint16[8,8] VL53L5CX distances in mm
  gripper_pos     – float32     jaw gap mm

Joint-cmd encoding in Command packet:
  joint_cmd_deg10 – int16[4]    joint angles × 10 (e.g. 12.3° → 123)
  gripper_cmd     – uint8       0-100 (% open)
  skill_state     – uint8       matches Skill IntEnum
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import serial   # pyserial

# ── magic constants ───────────────────────────────────────────────────────────
TELEM_MAGIC: int = 0xABCD
CMD_MAGIC:   int = 0xAA

# ── packet dtypes ─────────────────────────────────────────────────────────────
TELEMETRY_DTYPE = np.dtype([
    ('magic',         '<u2'),            #   2  B  → TELEM_MAGIC
    ('seq',           '<u4'),            #   4
    ('timestamp_ms',  '<u4'),            #   4
    ('servo_pos_raw', '<u2', (5,)),      #  10
    ('servo_vel',     '<i2', (5,)),      #  10
    ('servo_load',    '<i2', (5,)),      #  10
    ('imu_accel',     '<f4', (3,)),      #  12  m s⁻²
    ('imu_gyro',      '<f4', (3,)),      #  12  rad s⁻¹
    ('imu_gyro_rms',  '<f4'),            #   4  dps
    ('contact_flag',  'u1'),             #   1
    ('gripper_pos',   '<f4'),            #   4  mm
    ('tof_grid',      '<u2', (8, 8)),    # 128  mm
    ('adc_supply_mv', '<u2'),            #   2  mV
    ('temp_c',        '<i2'),            #   2  0.01 °C
    ('state_flags',   'u1'),             #   1
    ('error_flags',   'u1'),             #   1
    ('servo_temp',    'i1', (5,)),       #   5  °C
    ('reserved',      'u1', (37,)),      #  37
    ('checksum',      'u1'),             #   1
])                                       # ═══ 250 bytes total

COMMAND_DTYPE = np.dtype([
    ('magic',             'u1'),         #   1  → CMD_MAGIC
    ('seq',               'u1'),         #   1
    ('cmd_type',          'u1'),         #   1
    ('joint_cmd_deg10',   '<i2', (4,)),  #   8  degrees × 10
    ('gripper_cmd',       'u1'),         #   1  0-100 %
    ('skill_state',       'u1'),         #   1
    ('flags',             'u1'),         #   1
    ('reserved',          'u1', (5,)),   #   5
    ('checksum',          'u1'),         #   1
])                                       # ═══  20 bytes total


def _xor_checksum(data: bytes | bytearray) -> int:
    """XOR of every byte in *data*."""
    result = 0
    for b in data:
        result ^= b
    return result


class TeensySerial:
    """
    Manages the USB serial link to the Teensy.

    port : str   path to serial device (e.g. '/dev/ttyACM0')
           OR any duck-typed serial-like object (for unit tests).

    The background RX thread runs as a daemon and writes validated
    telemetry packets into self._latest under self._lock.
    """

    BAUD         = 2_000_000
    RX_TIMEOUT_S = 0.05     # 50 ms read timeout on real serial port

    def __init__(self, port) -> None:
        if isinstance(port, str):
            self._ser = serial.Serial(
                port,
                baudrate=self.BAUD,
                timeout=self.RX_TIMEOUT_S,
            )
        else:
            self._ser = port   # duck-typed mock

        self._latest:  Optional[np.ndarray] = None
        self._lock     = threading.Lock()
        self._running  = True
        self._cmd_seq  = 0
        self._rx_errors = 0   # bad checksum count (diagnostic)

        self._rx_thread = threading.Thread(
            target=self._rx_loop, name="teensy-rx", daemon=True
        )
        self._rx_thread.start()

    # ── public API ────────────────────────────────────────────────────

    @property
    def latest_telemetry(self) -> Optional[np.ndarray]:
        """
        Returns the most recently validated telemetry record, or None
        if no valid packet has arrived yet.  Thread-safe.
        """
        with self._lock:
            return self._latest

    @property
    def rx_errors(self) -> int:
        return self._rx_errors

    def send_command(
        self,
        joints_deg:  list[float],
        gripper_pct: int,
        skill_state: int,
        cmd_type:    int = 0,
        flags:       int = 0,
    ) -> None:
        """
        Pack and send a 20-byte command frame.

        joints_deg  : [J0, J1, J2, J3] in degrees (4 elements)
        gripper_pct : 0=fully closed … 100=fully open
        skill_state : Skill IntEnum value
        """
        cmd = np.zeros(1, dtype=COMMAND_DTYPE)
        cmd['magic']           = CMD_MAGIC
        cmd['seq']             = self._cmd_seq & 0xFF
        cmd['cmd_type']        = cmd_type & 0xFF
        cmd['joint_cmd_deg10'] = [int(round(j * 10)) for j in joints_deg[:4]]
        cmd['gripper_cmd']     = int(np.clip(gripper_pct, 0, 100))
        cmd['skill_state']     = int(skill_state) & 0xFF
        cmd['flags']           = int(flags) & 0xFF

        raw = bytearray(cmd.tobytes())
        raw[-1] = _xor_checksum(raw[:-1])
        self._ser.write(bytes(raw))
        self._cmd_seq += 1

    def close(self) -> None:
        self._running = False
        self._rx_thread.join(timeout=1.0)
        self._ser.close()

    # ── background RX loop ────────────────────────────────────────────

    def _rx_loop(self) -> None:
        pkt_size = TELEMETRY_DTYPE.itemsize   # 250
        buf      = bytearray()
        magic_lo = TELEM_MAGIC & 0xFF          # 0xCD
        magic_hi = (TELEM_MAGIC >> 8) & 0xFF  # 0xAB

        while self._running:
            # Respect timeout on real serial; avoid busy-spin on mocks.
            waiting = getattr(self._ser, 'in_waiting', 0)
            if waiting == 0:
                time.sleep(0.002)
                continue

            chunk = self._ser.read(min(waiting, pkt_size))
            if not chunk:
                continue
            buf.extend(chunk)

            # Slide through buffer looking for valid magic-aligned packets.
            while len(buf) >= pkt_size:
                # Fast path: already aligned.
                if buf[0] == magic_lo and buf[1] == magic_hi:
                    raw = bytes(buf[:pkt_size])
                    buf = buf[pkt_size:]

                    expected = _xor_checksum(raw[:-1])
                    if raw[-1] != expected:
                        self._rx_errors += 1
                        continue   # drop corrupt packet, keep buffering

                    pkt = np.frombuffer(raw, dtype=TELEMETRY_DTYPE).copy()
                    with self._lock:
                        self._latest = pkt
                else:
                    # Resync: discard one byte and search again.
                    del buf[0]


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    all_pass = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        global all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {label}{suffix}")

    # ── dtype sizes — the most critical invariant ─────────────────────
    print("=== Dtype byte sizes ===")
    check(
        f"TELEMETRY_DTYPE.itemsize == 250",
        TELEMETRY_DTYPE.itemsize == 250,
        f"got {TELEMETRY_DTYPE.itemsize}",
    )
    check(
        f"COMMAND_DTYPE.itemsize   ==  20",
        COMMAND_DTYPE.itemsize == 20,
        f"got {COMMAND_DTYPE.itemsize}",
    )

    # ── field presence sanity ─────────────────────────────────────────
    print("\n=== Telemetry field names ===")
    telem_fields = TELEMETRY_DTYPE.names
    for field in ('magic', 'seq', 'timestamp_ms', 'servo_pos_raw',
                  'imu_gyro_rms', 'contact_flag', 'tof_grid',
                  'gripper_pos', 'checksum'):
        check(f"field '{field}' present", field in telem_fields)

    print("\n=== Command field names ===")
    cmd_fields = COMMAND_DTYPE.names
    for field in ('magic', 'seq', 'cmd_type', 'joint_cmd_deg10',
                  'gripper_cmd', 'skill_state', 'flags', 'checksum'):
        check(f"field '{field}' present", field in cmd_fields)

    # ── checksum function ─────────────────────────────────────────────
    print("\n=== XOR checksum ===")
    check("xor([0x00])         == 0x00", _xor_checksum(b'\x00') == 0x00)
    check("xor([0xFF])         == 0xFF", _xor_checksum(b'\xFF') == 0xFF)
    check("xor([0xAB, 0xCD])   == 0x66", _xor_checksum(b'\xAB\xCD') == 0x66)
    check("xor([0x01,0x02,0x03])== 0x00", _xor_checksum(b'\x01\x02\x03') == 0x00)

    # ── mock serial class for TeensySerial tests ──────────────────────
    class _MockSerial:
        def __init__(self) -> None:
            self._buf:  bytearray = bytearray()
            self._lock: threading.Lock = threading.Lock()
            self.written: list[bytes] = []

        def feed(self, data: bytes) -> None:
            with self._lock:
                self._buf.extend(data)

        @property
        def in_waiting(self) -> int:
            with self._lock:
                return len(self._buf)

        def read(self, n: int) -> bytes:
            with self._lock:
                chunk = bytes(self._buf[:n])
                del self._buf[:n]
                return chunk

        def write(self, data: bytes) -> int:
            self.written.append(bytes(data))
            return len(data)

        def close(self) -> None:
            pass

    # ── instantiation with mock port ──────────────────────────────────
    print("\n=== Instantiation with mock serial ===")
    mock = _MockSerial()
    ts   = TeensySerial(mock)
    check("TeensySerial created without exception", True)
    check("latest_telemetry is None before any packet", ts.latest_telemetry is None)

    # ── send_command packs correct bytes ──────────────────────────────
    print("\n=== send_command ===")
    ts.send_command([10.0, 20.5, -30.0, 45.0], gripper_pct=60, skill_state=1)

    check("write() called once", len(mock.written) == 1)
    raw_cmd = mock.written[0]
    check(f"command is exactly 20 bytes", len(raw_cmd) == 20)
    check("magic byte is CMD_MAGIC (0xAA)", raw_cmd[0] == CMD_MAGIC)

    # Verify checksum
    expected_cs = _xor_checksum(raw_cmd[:-1])
    check("command checksum is correct",
          raw_cmd[-1] == expected_cs,
          f"got {raw_cmd[-1]:#04x}, expected {expected_cs:#04x}")

    # Parse back the command and verify joint encoding
    cmd_pkt = np.frombuffer(raw_cmd, dtype=COMMAND_DTYPE)
    j10 = cmd_pkt['joint_cmd_deg10'][0]
    check("J0=10.0° → joint_cmd_deg10[0]==100", int(j10[0]) == 100)
    check("J1=20.5° → joint_cmd_deg10[1]==205", int(j10[1]) == 205)
    check("J2=-30.0° → joint_cmd_deg10[2]==-300", int(j10[2]) == -300)
    check("gripper_cmd == 60", int(cmd_pkt['gripper_cmd'][0]) == 60)
    check("skill_state == 1", int(cmd_pkt['skill_state'][0]) == 1)

    # seq increments
    ts.send_command([0, 0, 0, 0], 0, 0)
    check("seq increments on second command",
          mock.written[1][1] == 1)

    # ── RX loop: feed a valid telemetry packet ────────────────────────
    print("\n=== RX loop: valid packet ===")
    telem = np.zeros(1, dtype=TELEMETRY_DTYPE)
    telem['magic']        = TELEM_MAGIC
    telem['seq']          = 42
    telem['timestamp_ms'] = 1234
    telem['contact_flag'] = 1
    telem['imu_gyro_rms'] = 5.5
    telem['tof_grid']     = 200   # all cells 200 mm

    raw_telem = bytearray(telem.tobytes())
    raw_telem[-1] = _xor_checksum(raw_telem[:-1])
    mock.feed(bytes(raw_telem))

    # Allow the daemon thread up to 200 ms to process.
    deadline = time.monotonic() + 0.2
    while ts.latest_telemetry is None and time.monotonic() < deadline:
        time.sleep(0.005)

    pkt = ts.latest_telemetry
    check("latest_telemetry populated after feeding valid packet", pkt is not None)
    if pkt is not None:
        check("seq == 42",           int(pkt['seq'][0])          == 42)
        check("contact_flag == 1",   int(pkt['contact_flag'][0]) == 1)
        check("imu_gyro_rms == 5.5", abs(float(pkt['imu_gyro_rms'][0]) - 5.5) < 1e-5)
        check("tof_grid[0,0] == 200",int(pkt['tof_grid'][0, 0, 0]) == 200)

    # ── RX loop: corrupt packet (bad checksum) is dropped ────────────
    print("\n=== RX loop: corrupt packet ===")
    ts2   = TeensySerial(_MockSerial())
    raw_bad = bytearray(raw_telem)
    raw_bad[-1] ^= 0xFF   # flip all bits in checksum
    ts2._ser.feed(bytes(raw_bad))

    time.sleep(0.1)
    check("Corrupt packet: latest_telemetry remains None",
          ts2.latest_telemetry is None)
    check("rx_errors incremented", ts2.rx_errors == 1)

    ts.close()
    ts2.close()

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
