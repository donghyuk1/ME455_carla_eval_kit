# autopilot_fsm.py
# Tiny FSM layer for MyAutopilot:
# - States: Drive / Stop (easy to extend)
# - Handlers consume cargo every tick: dict with keys you compute in MyAutopilot
# - Keeps a minimal stop dwell time to avoid flicker
# - Provides helper 'maybe_control_override' if you want Stop to hard-brake

from __future__ import annotations
import time
from typing import Dict, Any, Optional, Tuple

from team_code_autopilot.utils.fsm import FSM  # adjust import as needed
# -----------------------------------------------------------------------------
# Cargo schema (convention)
# -----------------------------------------------------------------------------
# Pass a dict each tick with any of these keys (extend as you wish):
#   cargo = {
#       "obstacle": bool,      # True if front AABB has LiDAR hits (your existing logic)
#       "red": bool,           # True if traffic light red (if you compute it; else omit/False)
#       "dt": float,           # seconds since last tick (optional but nice to have)
#       "speed": float,        # m/s (optional)
#       "planner_cmd": str,    # e.g., "LANE_FOLLOW", etc. (optional)
#       "timestamp": float,    # sim time or wall time (optional)
#       # ... add anything your predicates might need later
#   }

# -----------------------------------------------------------------------------
# Internal helpers: we keep small per-FSM memory in fsm._data
# -----------------------------------------------------------------------------
def _get_mem(fsm: FSM) -> Dict[str, Any]:
    # Attach a dict the first time we need it
    if not hasattr(fsm, "_data"):
        setattr(fsm, "_data", {})
    return getattr(fsm, "_data")


# -----------------------------------------------------------------------------
# on_enter / on_exit callbacks
# -----------------------------------------------------------------------------
def on_enter_drive(fsm: FSM) -> None:
    mem = _get_mem(fsm)
    mem["last_enter_drive"] = time.monotonic()

def on_enter_stop(fsm: FSM) -> None:
    mem = _get_mem(fsm)
    mem["stop_since"] = time.monotonic()


# -----------------------------------------------------------------------------
# Handlers (consume cargo each tick)
# -----------------------------------------------------------------------------
def drive_handler(cargo: Dict[str, Any], fsm: FSM) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Stay in Drive unless a blocking condition is present.
    Returns (next_state, optional_output)
    """
    obstacle = bool(cargo.get("obstacle", False))
    red = bool(cargo.get("red", False))

    if obstacle or red:
        # You can put a reason in the output for logging
        reason = "obstacle" if obstacle else "red_light"
        return "Stop", {"reason": reason}
    return "Drive", None  # remain driving


def stop_handler(cargo: Dict[str, Any], fsm: FSM) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Remain in Stop until:
      - min dwell time elapsed, AND
      - no obstacle, AND
      - no red light
    """
    mem = _get_mem(fsm)
    obstacle = bool(cargo.get("obstacle", False))
    red = bool(cargo.get("red", False))
    min_stop_s = float(mem.get("min_stop_s", 0.5))  # tweakable dwell

    stop_since = mem.get("stop_since", time.monotonic())
    dwell = time.monotonic() - stop_since

    if (not obstacle) and (not red) and (dwell >= min_stop_s):
        return "Drive", {"reason": "clear_path"}
    return "Stop", None  # keep holding


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def build_vehicle_fsm(*, min_stop_s: float = 0.5, start_state: str = "Drive") -> FSM:
    """
    Create and configure the FSM for the vehicle.
    States: Drive <-> Stop. Add more by .add(...) with new handlers.
    """
    fsm = FSM()
    fsm.add("Drive", drive_handler, on_enter=on_enter_drive)
    fsm.add("Stop",  stop_handler,  on_enter=on_enter_stop)

    # Optional terminal state example (unused by default)
    fsm.add("Error", end=True)

    # Seed memory
    mem = _get_mem(fsm)
    mem["min_stop_s"] = float(min_stop_s)

    fsm.start(start_state)
    return fsm



if __name__ == "__main__":
    # -------------------------------------------------------------
    # Minimal test for Drive/Stop FSM behavior
    # -------------------------------------------------------------
    print("[Test] Building FSM...")
    fsm = build_vehicle_fsm(min_stop_s=0.5, start_state="Drive")

    timeline = [
        {"t": 0.0, "obstacle": False, "red": False},
        {"t": 1.0, "obstacle": True,  "red": False},   # trigger stop
        {"t": 2.0, "obstacle": True,  "red": False},   # still stop
        {"t": 3.0, "obstacle": False, "red": False},   # should drive again after dwell
        {"t": 4.0, "obstacle": False, "red": False},
    ]

    for tick in timeline:
        state, info = fsm.step(tick)
        print(f"[t={tick['t']:.1f}] state={state} info={info}")

    print("[Test completed]")
