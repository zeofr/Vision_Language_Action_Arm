"""
Skill state machine for the VLA robotic arm.

Skill states:
  REACH = 0   – moving end-effector to pick position
  GRASP = 1   – closing gripper around object
  LIFT  = 2   – lifting object off table
  PLACE = 3   – moving to and releasing at target position

Legal manual transitions (advance() / request_next()):
  REACH → GRASP → LIFT → PLACE

IMU contact override:
  While in GRASP, a contact flag (set by the Teensy telemetry parser
  when IMU RMS angular-rate exceeds threshold) forces an immediate
  transition to LIFT, bypassing any dwell timer.

reset() restores the machine to REACH and clears all flags.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional


class Skill(IntEnum):
    REACH = 0
    GRASP = 1
    LIFT  = 2
    PLACE = 3


# Legal forward transitions — no back-stepping, no skipping.
_TRANSITIONS: dict[Skill, Skill] = {
    Skill.REACH: Skill.GRASP,
    Skill.GRASP: Skill.LIFT,
    Skill.LIFT:  Skill.PLACE,
}


class SkillStateMachine:
    """
    Thread-safe-free (single-threaded) skill state machine.

    Typical call pattern per inference tick:
        sm.notify_contact(telemetry.contact_flag)
        action = policy(sm.state, ...)
        if skill_complete:
            sm.advance()
    """

    def __init__(self) -> None:
        self._state: Skill = Skill.REACH
        self._contact_pending: bool = False
        self._transition_log: list[tuple[Skill, Skill, str]] = []

    # ── read-only properties ──────────────────────────────────────────

    @property
    def state(self) -> Skill:
        return self._state

    @property
    def done(self) -> bool:
        """True after PLACE completes (no further transition available)."""
        return self._state not in _TRANSITIONS

    # ── control interface ─────────────────────────────────────────────

    def notify_contact(self, contact: bool) -> bool:
        """
        Called each tick with the IMU contact flag from telemetry.
        If contact is True while in GRASP, immediately transitions to
        LIFT and returns True.  In all other states the flag is ignored
        and False is returned.
        """
        if contact and self._state is Skill.GRASP:
            self._do_transition(Skill.LIFT, reason="imu_contact")
            return True
        return False

    def advance(self) -> Optional[Skill]:
        """
        Request the next legal transition (called by the policy when the
        current skill phase is judged complete).

        Returns the new Skill, or None if already at PLACE (terminal).
        Raises ValueError on any illegal transition attempt.
        """
        if self.done:
            return None
        next_state = _TRANSITIONS[self._state]
        self._do_transition(next_state, reason="advance")
        return self._state

    def reset(self) -> None:
        """Return to REACH and clear all flags.  Log is preserved."""
        self._do_transition(Skill.REACH, reason="reset")
        self._contact_pending = False

    # ── internal ──────────────────────────────────────────────────────

    def _do_transition(self, target: Skill, reason: str) -> None:
        prev = self._state
        self._state = target
        self._transition_log.append((prev, target, reason))

    def __repr__(self) -> str:
        return f"SkillStateMachine(state={self._state.name}, done={self.done})"


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    all_pass = True

    def check(label: str, condition: bool) -> None:
        global all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        print(f"  [{status}] {label}")

    sm = SkillStateMachine()

    # ── full happy-path sequence ──────────────────────────────────────
    print("=== Full REACH → GRASP → LIFT → PLACE sequence ===")

    check("Initial state is REACH",      sm.state is Skill.REACH)
    check("Not done at REACH",           not sm.done)

    sm.advance()
    check("After advance(): state is GRASP", sm.state is Skill.GRASP)

    sm.advance()
    check("After advance(): state is LIFT",  sm.state is Skill.LIFT)

    sm.advance()
    check("After advance(): state is PLACE", sm.state is Skill.PLACE)

    check("done is True at PLACE",       sm.done)

    ret = sm.advance()
    check("advance() at terminal returns None", ret is None)
    check("State stays PLACE after terminal advance()", sm.state is Skill.PLACE)

    # ── reset ─────────────────────────────────────────────────────────
    print("\n=== reset() ===")

    sm.reset()
    check("After reset(): state is REACH", sm.state is Skill.REACH)
    check("Not done after reset()",        not sm.done)

    # ── IMU contact flag: ignored outside GRASP ───────────────────────
    print("\n=== IMU contact flag handling ===")

    check("State is REACH before contact test", sm.state is Skill.REACH)
    fired = sm.notify_contact(True)
    check("contact=True in REACH is ignored (returns False)", not fired)
    check("State still REACH after spurious contact",         sm.state is Skill.REACH)

    sm.advance()   # → GRASP
    check("State is GRASP",                                   sm.state is Skill.GRASP)

    # contact=False should not transition
    fired = sm.notify_contact(False)
    check("contact=False in GRASP is ignored",                not fired)
    check("State still GRASP",                                sm.state is Skill.GRASP)

    # contact=True in GRASP → immediate LIFT
    fired = sm.notify_contact(True)
    check("contact=True in GRASP fires transition (returns True)", fired)
    check("State is now LIFT after IMU contact",              sm.state is Skill.LIFT)

    # contact=True in LIFT should be ignored
    fired = sm.notify_contact(True)
    check("contact=True in LIFT is ignored",                  not fired)
    check("State still LIFT",                                 sm.state is Skill.LIFT)

    sm.advance()   # → PLACE
    check("advance() from LIFT → PLACE",                      sm.state is Skill.PLACE)

    # ── second reset and re-run to verify repeatability ───────────────
    print("\n=== Reset and repeat (repeatability check) ===")

    sm.reset()
    check("reset() → REACH again", sm.state is Skill.REACH)

    for expected in (Skill.GRASP, Skill.LIFT, Skill.PLACE):
        sm.advance()
        check(f"advance() → {expected.name}", sm.state is expected)

    check("done after second full sequence", sm.done)

    # ── transition log contains all recorded hops ─────────────────────
    print("\n=== Transition log sanity ===")
    check("Log is non-empty", len(sm._transition_log) > 0)
    reasons = {r for _, _, r in sm._transition_log}
    check("Log contains 'advance' entries",  "advance"     in reasons)
    check("Log contains 'reset' entries",    "reset"       in reasons)
    check("Log contains 'imu_contact' entry","imu_contact" in reasons)

    print(f"\nTransition log ({len(sm._transition_log)} entries):")
    for prev, nxt, reason in sm._transition_log:
        print(f"  {prev.name:5} → {nxt.name:5}  [{reason}]")

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
