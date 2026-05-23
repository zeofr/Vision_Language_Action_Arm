import serial
import struct
import threading


_GOAL_POS_ADDR = 0x2A
_DEG_TO_STEPS  = 1.0 / 0.0879
_STEPS_CENTER  = 2047


def _deg_to_steps(deg: float) -> int:
    return max(0, min(4095, int(deg * _DEG_TO_STEPS + _STEPS_CENTER)))


class ServoDriver:
    """
    Direct SCS/STS3215 servo bus driver over SmartElex USB.
    Sends SYNC_WRITE position commands; no response expected.
    Thread-safe for single-writer use.
    """

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 1_000_000):
        self._ser = serial.Serial(port, baud, timeout=0.05)
        self._lock = threading.Lock()

    def sync_write(self, ids: list[int], positions_deg: list[float]) -> None:
        """Send goal positions to all servos in one broadcast packet."""
        count = len(ids)
        assert len(positions_deg) == count

        param_len  = count * 3          # 3 bytes per servo: ID + PosL + PosH
        packet_len = param_len + 4      # +4: instruction + start_addr + data_len + checksum

        pkt = bytearray()
        pkt += b'\xFF\xFF\xFE'          # header + broadcast ID
        pkt.append(packet_len)
        pkt.append(0x83)                # SYNC_WRITE instruction
        pkt.append(_GOAL_POS_ADDR)
        pkt.append(0x02)                # 2 bytes per servo

        chk = 0xFE + packet_len + 0x83 + _GOAL_POS_ADDR + 0x02
        for sid, deg in zip(ids, positions_deg):
            steps = _deg_to_steps(deg)
            lo, hi = steps & 0xFF, (steps >> 8) & 0xFF
            pkt += bytes([sid, lo, hi])
            chk += sid + lo + hi

        pkt.append((~chk) & 0xFF)

        with self._lock:
            self._ser.write(pkt)
            self._ser.flush()

    def ping(self, servo_id: int) -> bool:
        """Returns True if servo responds."""
        chk = (~(servo_id + 2 + 1)) & 0xFF
        with self._lock:
            self._ser.write(bytes([0xFF, 0xFF, servo_id, 2, 1, chk]))
            self._ser.flush()
            resp = self._ser.read(6)
        return (len(resp) == 6 and resp[0] == 0xFF
                and resp[1] == 0xFF and resp[2] == servo_id)

    def close(self) -> None:
        self._ser.close()
