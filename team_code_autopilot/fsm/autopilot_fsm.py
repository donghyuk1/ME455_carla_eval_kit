# autopilot_fsm.py — skeleton for behavioral planning
# Depends on: fsm.py  
# States: Drive, Stop, Detour
# Explanation of each section of the code
#   - Section 1: Output of the FSM (PlannerOutput): output of any state is given by the class PlannerOutput and will be fed to control afterwards
#   - Section 2: Input if the FSM (cargo): explains which information the FSM recieves at each step
#   - Section 3: Internal memory of FSM: used for desicions that require knowledge of prior ticks. The memory is not part of the cargo
#   - Section 4: Adjustable parameters that controll how the behavioural planning behaves
#   - Section 5: Includes helper functions for: 
#                   -- determining time 
#                   -- determining time and reason for stop 
#                   -- how long the ergo vehicle has been in the stop state 
#                   -- if the state should switch form drive to stop 
#                   -- if the state should switch form stop to drive
#                   -- if the state should switch from stop to detour
#   - Section 6: Decide which state should be next and what should be output to control
#   - Section 7: Builds a fully wired FSM plus its configuration

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from .fsm import FSM


# Fixed stop-line distances (in meters) used for TL/stop-sign gating.
# These are the *only* place where "stop line position" is encoded.
# You can tune them as needed.
TL_STOP_LINE_DISTANCE_M = 14.0      # must be tuned 
TL_STOP_LINE_DISTANCE_SMALL = 6.0   # must be tuned - distance from traffic light when car crosses stop line
STOP_SIGN_STOP_LINE_DISTANCE_M = 8.0 # must be tuned


# =============================================================================================================================
# Section 1: Output of the behavioural planning - passed to the caller each tick
# =============================================================================================================================
@dataclass  # class for holding data from dataclasses
class PlannerOutput:
    """Compact directive for the control stack."""
    mode: str                       # Symbolic command: "DRIVE", "STOP", "DETOUR"
    reason: Optional[str] = None    # e.g. "obstacle", "red_light", "stop_sign", "clear"
    target_speed: Optional[float] = None  # Target speed for the controller (0 for STOP, x for DRIVE/DETOUR)
    notes: Optional[Dict[str, Any]] = None  # Any further information (e.g. dwell time, obstacle distance, lane ID)
    waypoint: Optional[Tuple[float, float]] = None  # Next waypoint (x, y)

    # Helper constructors
    @staticmethod
    def stop(reason: str, waypoint: Optional[Tuple[float, float]] = None) -> "PlannerOutput":
        return PlannerOutput(mode="STOP", reason=reason, target_speed=0.0, waypoint=waypoint)

    @staticmethod
    def drive(
        target_speed: Optional[float] = None,
        reason: str = "clear",
        waypoint: Optional[Tuple[float, float]] = None
    ) -> "PlannerOutput":
        return PlannerOutput(mode="DRIVE", reason=reason, target_speed=target_speed, waypoint=waypoint)

    # for detour
    @staticmethod
    def detour(
        target_speed: Optional[float] = None,
        reason: str = "detouring",
        waypoint: Optional[Tuple[float, float]] = None
    ) -> "PlannerOutput":
        return PlannerOutput(mode="DETOUR", reason=reason, target_speed=target_speed, waypoint=waypoint)


# =============================================================================================================================
# Section 2: Input of the FSM (cargo)
# =============================================================================================================================
def build_cargo_from_hdmap(
    *,
    hdmap_obj: Any,
    ego_actor: Any,
    dt: float,
    t: Optional[float],
    cruise_target: float,
    obstacle_distance: Optional[float],
    tl_red_from_vision: Optional[bool],
    tl_distance: Optional[float],
    stop_sign_ahead_from_vision: Optional[bool],
    stop_sign_distance: Optional[float],
) -> Dict[str, Any]:
    """
    Build the cargo dictionary from HDMap + ego + high-level TL/stop-sign signals.

    Cargo keys:

        Timing:
          - "dt": float (time step in seconds)
          - "t":  float | None (absolute time, if available)

        Ego:
          - "speed": float (m/s)

        Gating:
          - "obstacle_ahead": bool
          - "tl_red": bool
          - "tl_distance": float | None
          - "tl_near_stopline": bool
          - "stop_sign_ahead": bool
          - "stop_sign_distance": float | None
          - "ss_near_stopline": bool

        Controller hint:
          - "cruise_target": float

        Nominal path:
          - "waypoint": (x, y) or None

        Detour-related:
          - "orig_lane_free": bool
          - "left_adjacent_lane_free": bool
          - "right_adjacent_lane_free": bool
          - "left_lane_waypoint": (x, y) or None
          - "right_lane_waypoint": (x, y) or None
    """
    cargo: Dict[str, Any] = {}

    # --- Timing ---
    cargo["dt"] = float(dt)
    cargo["t"] = float(t) if t is not None else None

    # --- Ego speed ---
    ego_speed = 0.0
    if ego_actor is not None:
        try:
            v = ego_actor.get_velocity()
            ego_speed = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
        except Exception:
            ego_speed = 0.0
    cargo["speed"] = float(ego_speed)

    # --- Obstacle gating (from HDMap, if available) ---
    obstacle_ahead = False

    # Base safety distance (meters); can be overridden by obstacle_distance
    obstacle_check_distance = 30.0
    if obstacle_distance is not None:
        try:
            obstacle_check_distance = float(obstacle_distance)
        except (TypeError, ValueError):
            obstacle_check_distance = 30.0  # fall back to default

    if hdmap_obj is not None and hasattr(hdmap_obj, "is_obstacle_in_front"):
        try:
            obstacle_ahead = bool(
                hdmap_obj.is_obstacle_in_front(
                    distance=float(obstacle_check_distance),
                )
            )
        except Exception:
            obstacle_ahead = False
    cargo["obstacle_ahead"] = obstacle_ahead


    # --- Traffic light gating ---
    if tl_red_from_vision is not None:
        cargo["tl_red"] = bool(tl_red_from_vision)
    else:
        cargo["tl_red"] = False

    # distance to *relevant* traffic light (if known)
    cargo["tl_distance"] = float(tl_distance) if tl_distance is not None else None

    # Determine "near stop-line" for TLs:
    cargo["tl_near_stopline"] = (
        cargo["tl_red"]
        and (tl_distance is not None)
        and (TL_STOP_LINE_DISTANCE_SMALL < tl_distance <= TL_STOP_LINE_DISTANCE_M)
    )

    # --- Stop-sign gating ---
    if stop_sign_ahead_from_vision is not None:
        cargo["stop_sign_ahead"] = bool(stop_sign_ahead_from_vision)
    else:
        cargo["stop_sign_ahead"] = False

    cargo["stop_sign_distance"] = (
        float(stop_sign_distance) if stop_sign_distance is not None else None
    )

    # Stop-sign near-stopline: purely distance-based
    cargo["ss_near_stopline"] = (
        cargo["stop_sign_ahead"]
        and (stop_sign_distance is not None)
        and (stop_sign_distance <= STOP_SIGN_STOP_LINE_DISTANCE_M)
    )

    # --- Controller hint + next waypoint from HDMap ---
    cargo["cruise_target"] = float(cruise_target)

    # Nominal lane waypoint
    if hdmap_obj is not None and hasattr(hdmap_obj, "get_next_waypoint"):
        try:
            cargo["waypoint"] = hdmap_obj.get_next_waypoint()
        except Exception:
            cargo["waypoint"] = None
    else:
        cargo["waypoint"] = None

    # ==============================
    # Detour-related cargo extension
    # ==============================
    # Original lane "free" flag – if we cannot query, be conservative (False).
    try:
        if hdmap_obj is not None and hasattr(hdmap_obj, "is_obstacle_in_front"):
            obstacle_forward = bool(
                hdmap_obj.is_obstacle_in_front(
                    distance=float(obstacle_check_distance),
                )
            )
            orig_lane_free = not obstacle_forward
        else:
            orig_lane_free = not obstacle_ahead
    except Exception:
        orig_lane_free = False
    cargo["orig_lane_free"] = orig_lane_free

    # Adjacent lane free flags
    left_free = False
    right_free = False
    if hdmap_obj is not None:
        try:
            left_free = bool(hdmap_obj.is_left_lane_free(distance=float(obstacle_check_distance)))
        except (AttributeError, Exception):
            left_free = False


        try:
            right_free = bool(hdmap_obj.is_right_lane_free(distance=float(obstacle_check_distance)))
        except (AttributeError, Exception):
            right_free = False

    cargo["left_adjacent_lane_free"] = left_free
    cargo["right_adjacent_lane_free"] = right_free

    # Adjacent lane candidate waypoints
    left_wp = None
    right_wp = None
    if hdmap_obj is not None:
        try:
            left_wp = hdmap_obj.get_left_lane_waypoint()
        except AttributeError:
            pass
        except Exception:
            pass
        try:
            right_wp = hdmap_obj.get_right_lane_waypoint()
        except AttributeError:
            pass
        except Exception:
            pass

    cargo["left_lane_waypoint"] = (
        (float(left_wp[0]), float(left_wp[1])) if left_wp is not None else None
    )
    cargo["right_lane_waypoint"] = (
        (float(right_wp[0]), float(right_wp[1])) if right_wp is not None else None
    )

    return cargo


# =============================================================================================================================
# Section 3: Internal FSM memory keys
# =============================================================================================================================

# clock_s:        internal clock of the ego vehicle (time in seconds)
# entered_stop_s: timestamp at which the ego vehicle stopped, used for dwell time
# stop_reason:    reason for stopping

MEM = {  # dictionary used as it can be expanded easily
    "clock_s": "clock_s",             # monotonically increasing clock if cargo.t not supplied
    "entered_stop_s": "entered_stop_s",
    "stop_reason": "stop_reason",     # "obstacle" | "red_light" | "stop_sign"
    "last_stop_sign_clear_s": "last_stop_sign_clear_s",  # when we last *left* STOP due to a stop sign

    # for detouring
    "stop_stationary_ticks": "stop_stationary_ticks",  # stationary ticks in Stop (with noise in velocity)
    "detour_probe_ticks": "detour_probe_ticks",        # ticks since entering Detour (for periodic probes)
    "orig_lane_waypoint": "orig_lane_waypoint",        # stored original-lane waypoint (x, y)
    "detour_side": "detour_side",                      # "left" or "right"
    "stop_from_detour": "stop_from_detour",             # to remember if we stopped in detour

    # NEW: cache for the last detour-lane waypoint
    "detour_lane_waypoint": "detour_lane_waypoint",
    "entered_detour_s": "entered_detour_s",  # when we entered Detour (for min-detour time)
}



# =============================================================================================================================
# Section 4: Configuration (tunable)
# =============================================================================================================================
@dataclass  # Possibly not needed
class PlannerConfig:
    # Minimum time in seconds the ego vehicle must stay stopped to avoid flicker
    # (used for obstacle and red-light stops)
    min_stop_s: float = 0.5

    # Mandatory legal stop duration at a stop sign
    stop_sign_min_s: float = 2.0

    # After leaving a stop sign, ignore stop-sign gating for this many seconds
    # to avoid re-stopping on the same sign you just cleared.
    stop_sign_rearm_s: float = 3.0

    # for detouring
    stationary_speed_thresh: float = 0.15   # m/s, considered "stationary" in Stop
    detour_after_ticks: int = 20           # ticks stationary in Stop → go Detour (no TL/SS)
    detour_probe_every: int = 10           # check every X ticks in Detour if original lane is free
    detour_target_speed: Optional[float] = None  # optional speed in Detour (None = use cruise_target)

    # NEW: obstacle-distance relaxation when entering Detour
    # factor < 1.0 shrinks the obstacle lookahead distance,
    # duration controls for how many seconds after entering Detour this applies.
    detour_relax_obstacle_factor: float = 0.1
    detour_relax_obstacle_duration_s: float = 1000.0
    pass


# =============================================================================================================================
# Section 5: Utility helpers
# =============================================================================================================================
def _now_s(cargo: Dict[str, Any], data: Dict[str, Any]) -> float:
    # Determines the current time in seconds. To do so, it checks if cargo includes t, and if
    # it does, it will return that value. If it doesn't exist, it estimates the time using dt.

    """Prefer cargo['t']; otherwise integrate dt locally."""
    if "t" in cargo and cargo["t"] is not None:
        return float(cargo["t"])
    data[MEM["clock_s"]] = float(data.get(MEM["clock_s"], 0.0)) + float(cargo.get("dt", 0.0) or 0.0)
    return data[MEM["clock_s"]]


def _begin_stop(data: Dict[str, Any], now_s: float, reason: str) -> None:
    # Records two things when the vehicle enters the stop state:
    #   The current time by entered_stop_s from MEM
    #   The reason for stopping by stop_reason from MEM

    data[MEM["entered_stop_s"]] = now_s
    data[MEM["stop_reason"]] = reason
    data[MEM["stop_stationary_ticks"]] = 0  # reset stationary-tick counter for this stop episode


def _stop_elapsed_s(data: Dict[str, Any], now_s: float) -> float:
    # Calculates how long the ego vehicle has been in the stop state.
    # Looks up time we entered stop (eventually derived from _begin_stop) 
    # and subtracts it from the current time (derived from _now_s)

    t0 = float(data.get(MEM["entered_stop_s"], now_s))
    return max(0.0, now_s - t0)


def _recently_cleared_stop_sign(data: Dict[str, Any], now_s: float, cfg: PlannerConfig) -> bool:
    """
    Returns True if we have left a stop-sign stop less than stop_sign_rearm_s seconds ago.
    Used to avoid immediately re-stopping at the same sign.
    """
    last_clear = data.get(MEM["last_stop_sign_clear_s"], None)
    if last_clear is None:
        return False

    try:
        last_clear = float(last_clear)
    except (TypeError, ValueError):
        return False

    return (now_s - last_clear) < cfg.stop_sign_rearm_s


def _should_stop(
    cargo: Dict[str, Any],
    data: Dict[str, Any],
    cfg: PlannerConfig,
    now_s: float,
) -> Tuple[bool, Optional[str]]:
    """
    Decide if we must *enter* STOP from DRIVE (or remain logically stopped),
    based on current cargo and recent stop-sign history.
    Returns (need_stop, reason) where reason is:
        "obstacle" | "red_light" | "stop_sign" | None
    """

    # 1) Obstacle has highest priority
    if cargo.get("obstacle_ahead", False):
        return True, "obstacle"

    # 2) Red light near its stop line
    if cargo.get("tl_red", False) and cargo.get("tl_near_stopline", False):
        return True, "red_light"

    # 3) Stop sign near its stop line – but only if we did NOT just clear a stop sign
    recently_cleared_ss = _recently_cleared_stop_sign(data, now_s, cfg)
    if (
        not recently_cleared_ss
        and cargo.get("stop_sign_ahead", False)
        and cargo.get("ss_near_stopline", False)
    ):
        return True, "stop_sign"

    # Otherwise, no stop needed.
    return False, None


def _clear_to_go_from_stop(
    cargo: Dict[str, Any],
    data: Dict[str, Any],
    cfg: PlannerConfig,
    now_s: float,
) -> bool:
    """
    Decide if we can leave STOP and go back to motion (Drive or Detour).

    We look at:
      - stored stop reason (from MEM),
      - dwell time in STOP,
      - current cargo conditions.
    """
    reason = data.get(MEM["stop_reason"], None)
    dwell = _stop_elapsed_s(data, now_s)

    # Gate 1: Has the stop condition ended?
    obstacle_clear = not cargo.get("obstacle_ahead", False)
    red_clear = not cargo.get("tl_red", False)
    ss_clear = not (
        cargo.get("stop_sign_ahead", False)
        and cargo.get("ss_near_stopline", False)
    )

    # Per-reason logic
    if reason == "obstacle":
        cond_cleared = obstacle_clear
        dwell_needed = cfg.min_stop_s

    elif reason == "red_light":
        cond_cleared = red_clear
        dwell_needed = cfg.min_stop_s

    elif reason == "stop_sign":
        # For stop signs, require the legal dwell AND an obstacle-free path.
        cond_cleared = obstacle_clear
        dwell_needed = cfg.stop_sign_min_s

    else:
        # Unknown reason – be conservative
        cond_cleared = obstacle_clear and red_clear and ss_clear
        dwell_needed = cfg.min_stop_s

    return (dwell >= dwell_needed) and cond_cleared


# added for detour
def _should_detour_from_stop(
    cargo: Dict[str, Any],
    data: Dict[str, Any],
    cfg: PlannerConfig,
) -> bool:
    """
    Decide whether we should switch from STOP to DETOUR.

    Conditions:
      - No active red-light or stop-sign gating.
      - At least one adjacent lane is free.
      - We have been stationary in STOP for >= detour_after_ticks.
    """
    # Only consider detour if there is NO red-light/stop-sign condition active
    no_tl_or_ss = (
        not cargo.get("tl_red", False)
        and not (
            cargo.get("stop_sign_ahead", False)
            and cargo.get("ss_near_stopline", False)
        )
    )
    if not no_tl_or_ss:
        data[MEM["stop_stationary_ticks"]] = 0
        return False

    # require at least one adjacent lane to be free (left or right)
    left_free = bool(cargo.get("left_adjacent_lane_free", False))
    right_free = bool(cargo.get("right_adjacent_lane_free", False))
    if not (left_free or right_free):
        # If no candidate lane exists, do not accumulate "stuck" ticks toward detour
        data[MEM["stop_stationary_ticks"]] = 0
        return False

    # Increment "stationary ticks" while in Stop (speed below threshold)
    if float(cargo["speed"]) <= float(cfg.stationary_speed_thresh):
        data[MEM["stop_stationary_ticks"]] = int(data.get(MEM["stop_stationary_ticks"], 0)) + 1
    else:
        data[MEM["stop_stationary_ticks"]] = 0

    # Compare ticks against threshold
    return data.get(MEM["stop_stationary_ticks"], 0) >= int(
        getattr(cfg, "detour_after_ticks", 20)
    )


# =============================================================================================================================
# Section 6: State handlers
# =============================================================================================================================
def drive_handler(
    cargo: Dict[str, Any],
    data: Dict[str, Any],
    cfg: PlannerConfig,
) -> Tuple[str, PlannerOutput]:
    """Free-flow behaviour unless a gating condition requires STOP."""
    now = _now_s(cargo, data)
    waypoint = cargo.get("waypoint", None)

    # Check if any gating condition says we must enter STOP
    need_stop, reason = _should_stop(cargo, data, cfg, now)
    if need_stop:
        # Record time and reason for stopping
        _begin_stop(data, now, reason)
        data[MEM["stop_from_detour"]] = False
        return "Stop", PlannerOutput.stop(reason=reason, waypoint=waypoint)

    # Otherwise continue driving (optionally at cruise_target)
    target_v = cargo.get("cruise_target", None)
    return "Drive", PlannerOutput.drive(
        target_speed=target_v,
        reason="free_drive",
        waypoint=waypoint,
    )


def stop_handler(
    cargo: Dict[str, Any],
    data: Dict[str, Any],
    cfg: PlannerConfig,
) -> Tuple[str, PlannerOutput]:
    """Hold STOP until both dwell and clearance conditions are satisfied, or detour is chosen."""
    now = _now_s(cargo, data)

    # If we *still* should be stopping due to current inputs, remain stopped (this also
    # refreshes the stop reason if it changed in place, but we keep the original dwell start).
    still_stop, current_reason = _should_stop(cargo, data, cfg, now)
    stored_reason = data.get(MEM["stop_reason"], current_reason)

    if still_stop and current_reason and (current_reason != stored_reason):
        # Stop cause changed in place (e.g. red -> obstacle): update reason only.
        data[MEM["stop_reason"]] = current_reason
        stored_reason = current_reason

    waypoint = cargo.get("waypoint", None)

    # Check whether it's safe and legal to go
    if _clear_to_go_from_stop(cargo, data, cfg, now):
        # Stop-sign bookkeeping
        if stored_reason == "stop_sign":
            data[MEM["last_stop_sign_clear_s"]] = now

        stopped_from_detour = bool(data.get(MEM["stop_from_detour"], False))

        if stopped_from_detour:
            # We were in Detour before this STOP, so resume Detour behaviour

            detour_side = data.get(MEM["detour_side"], None)
            left_wp = cargo.get("left_lane_waypoint", None)
            right_wp = cargo.get("right_lane_waypoint", None)

            # Default: fall back to current nominal waypoint
            detour_wp = waypoint

            if detour_side == "left" and left_wp is not None:
                detour_wp = left_wp
            elif detour_side == "right" and right_wp is not None:
                detour_wp = right_wp
            else:
                # Fallback: recompute choice if side is unknown
                left_free = bool(cargo.get("left_adjacent_lane_free", False))
                right_free = bool(cargo.get("right_adjacent_lane_free", False))
                if left_free and left_wp is not None:
                    detour_wp = left_wp
                elif right_free and right_wp is not None:
                    detour_wp = right_wp

            # After you decide detour_wp for resuming Detour:
            data[MEM["detour_side"]] = detour_side
            data[MEM["detour_lane_waypoint"]] = detour_wp
            data[MEM["detour_probe_ticks"]] = 0

            return "Detour", PlannerOutput.detour(
                target_speed=getattr(cfg, "detour_target_speed", None),
                reason="resume_detour_after_stop",
                waypoint=detour_wp,
            )


        # Default case: Stop came from Drive → go back to Drive
        return "Drive", PlannerOutput.drive(
            reason="clear_after_stop",
            waypoint=waypoint,
        )

    # --- Detour branch (only if not clear to drive forward) ---
    if _should_detour_from_stop(cargo, data, cfg):
        # reset the periodic probe counter for Detour state
        data[MEM["detour_probe_ticks"]] = 0

        # store original-lane waypoint so we can switch back later
        data[MEM["orig_lane_waypoint"]] = waypoint

        # Decide which adjacent lane to use for the detour.
        left_free = bool(cargo.get("left_adjacent_lane_free", False))
        right_free = bool(cargo.get("right_adjacent_lane_free", False))
        left_wp = cargo.get("left_lane_waypoint", None)
        right_wp = cargo.get("right_lane_waypoint", None)

        detour_side = None
        detour_wp = waypoint  # fallback: original lane

        # Simple priority: prefer left if free, else right.
        if left_free and left_wp is not None:
            detour_side = "left"
            detour_wp = left_wp
        elif right_free and right_wp is not None:
            detour_side = "right"
            detour_wp = right_wp

        if detour_side is not None:
            data[MEM["detour_side"]] = detour_side

        if detour_side is not None:
            data[MEM["detour_side"]] = detour_side
            # Cache the first detour lane waypoint we actually chose
            data[MEM["detour_lane_waypoint"]] = detour_wp

            return "Detour", PlannerOutput.detour(
                target_speed=getattr(cfg, "detour_target_speed", None),
                reason="stalled_no_TL_SS",
                waypoint=detour_wp,
            )


    # Otherwise, remain in STOP
    return "Stop", PlannerOutput.stop(
        reason=stored_reason,
        waypoint=waypoint,
    )


# added for detour
def detour_handler(
    cargo: Dict[str, Any],
    data: Dict[str, Any],
    cfg: PlannerConfig,
) -> Tuple[str, PlannerOutput]:
    """
    Detour state:
      - If conditions require stop: go to Stop.
      - Otherwise, periodically check if the original lane is free; if so, go back to Drive.
      - Else remain in Detour (control uses current/alt waypoints provided via cargo).
    """
    now = _now_s(cargo, data)

    # --- Detour-entry timing for obstacle relaxation ---
    # Initialise detour entry time on the first tick in Detour.
    if MEM["entered_detour_s"] not in data or data[MEM["entered_detour_s"]] is None:
        data[MEM["entered_detour_s"]] = now

    try:
        detour_elapsed = float(now - float(data[MEM["entered_detour_s"]]))
    except (TypeError, ValueError):
        detour_elapsed = 0.0

    relax_window = float(getattr(cfg, "detour_relax_obstacle_duration_s", 0.0))

    # First, check whether a gating condition would normally require STOP.
    need_stop, reason = _should_stop(cargo, data, cfg, now)

    # During the relax window, IGNORE obstacle-based stops,
    # but still obey traffic lights and stop signs.
    if need_stop and reason == "obstacle" and relax_window > 0.0 and detour_elapsed < relax_window:
        need_stop = False
        reason = None

    if need_stop:
        # We really must stop (red light, stop sign, or obstacle after relax window).
        _begin_stop(data, now, reason)
        data[MEM["stop_from_detour"]] = True
        # Clear detour-entry timestamp when leaving Detour
        data.pop(MEM["entered_detour_s"], None)
        return "Stop", PlannerOutput.stop(reason=reason)

    # No gating condition: we are free to continue detouring or go back to Drive.

    # Original-lane waypoint (stored when entering Detour)
    orig_wp = data.get(MEM["orig_lane_waypoint"], cargo.get("waypoint", None))

    # Which side are we currently detouring to?
    detour_side = data.get(MEM["detour_side"], None)

    # Adjacent-lane availability from cargo
    left_free = bool(cargo.get("left_adjacent_lane_free", False))
    right_free = bool(cargo.get("right_adjacent_lane_free", False))

    # Use boolean logic: if we moved left, original lane is now to our right;
    # if we moved right, original lane is now to our left.
    if detour_side == "left":
        orig_lane_free = right_free
    elif detour_side == "right":
        orig_lane_free = left_free
    else:
        # Fallback if detour_side is unknown: use whatever orig_lane_free was computed upstream.
        orig_lane_free = bool(cargo.get("orig_lane_free", False))

    # --- Periodic probe to return to Drive (original waypoints) ------------
    probe_every = int(getattr(cfg, "detour_probe_every", 10))  # ticks
    data[MEM["detour_probe_ticks"]] = int(data.get(MEM["detour_probe_ticks"], 0)) + 1

    if data[MEM["detour_probe_ticks"]] >= probe_every:
        # reset counter and check original lane
        data[MEM["detour_probe_ticks"]] = 0
        if orig_lane_free:
            # Optionally enforce a minimum dwell time in Detour before returning to Drive.
            min_detour_time_s = float(getattr(cfg, "detour_relax_obstacle_duration_s", 0.0) or 0.0)
            entered_detour_s = data.get(MEM["entered_detour_s"], None)
            allow_exit = True
            if min_detour_time_s > 0.0 and entered_detour_s is not None:
                elapsed_detour = float(now) - float(entered_detour_s)
                if elapsed_detour < min_detour_time_s:
                    # Too early to leave Detour: keep detouring even if the original lane is free.
                    allow_exit = False

            if allow_exit:
                # We are done detouring → clear detour bookkeeping
                data.pop(MEM["orig_lane_waypoint"], None)
                data.pop(MEM["detour_side"], None)
                data.pop(MEM["detour_lane_waypoint"], None)  # NEW
                data.pop(MEM["entered_detour_s"], None)

                return "Drive", PlannerOutput.drive(
                    target_speed=cargo.get("cruise_target", None),
                    reason="orig_lane_clear",
                    waypoint=orig_wp,
                )



    # --- Stay in Detour: select detour-lane waypoint -----------------------
    left_wp = cargo.get("left_lane_waypoint", None)
    right_wp = cargo.get("right_lane_waypoint", None)

    # Retrieve cached detour waypoint (if any)
    cached_detour_wp = data.get(MEM["detour_lane_waypoint"], None)

    if detour_side == "left":
        if left_wp is not None:
            detour_wp = left_wp
        elif cached_detour_wp is not None:
            detour_wp = cached_detour_wp
        else:
            # Last resort fallback
            detour_wp = cargo.get("waypoint", None)

    elif detour_side == "right":
        if right_wp is not None:
            detour_wp = right_wp
        elif cached_detour_wp is not None:
            detour_wp = cached_detour_wp
        else:
            detour_wp = cargo.get("waypoint", None)

    else:
        # Side unknown: reuse cached detour wp if sensible, otherwise fall back to any free side
        if cached_detour_wp is not None:
            detour_wp = cached_detour_wp
        elif left_free and left_wp is not None:
            detour_wp = left_wp
        elif right_free and right_wp is not None:
            detour_wp = right_wp
        else:
            detour_wp = cargo.get("waypoint", None)

    # Update cache if we got a fresh side-lane waypoint this tick
    if detour_wp is not None:
        data[MEM["detour_lane_waypoint"]] = detour_wp

    # If state is not switched, it stays in Detour
    return "Detour", PlannerOutput.detour(
        target_speed=getattr(cfg, "detour_target_speed", None),
        reason="detouring",
        waypoint=detour_wp,
    )



# =============================================================================================================================
# Section 7: Wiring / factory
# =============================================================================================================================
def build_vehicle_fsm(
    *,
    start_state: str = "Drive",
    config: Optional[PlannerConfig] = None
) -> Tuple[FSM, PlannerConfig]:
    """
    Returns:
        fsm: the configured FSM instance
        cfg: the PlannerConfig actually used (so the caller can keep a handle)
    """

    # Uses either configuration or the values specified in PlannerConfig
    cfg = config or PlannerConfig()
    fsm = FSM()

    # Register states: tells the FSM that a state exists and which function to call when it’s active
    fsm.add_state("Drive", lambda cargo, data: drive_handler(cargo, data, cfg))
    fsm.add_state("Stop",  lambda cargo, data: stop_handler(cargo, data, cfg))
    # added for detour
    fsm.add_state("Detour", lambda cargo, data: detour_handler(cargo, data, cfg))

    def on_enter_stop(data):
        data.setdefault("enter_counts", {}).setdefault("Stop", 0)
        data["enter_counts"]["Stop"] += 1

    def on_exit_stop(data):
        data.pop(MEM["entered_stop_s"], None)
        data.pop(MEM["stop_reason"], None)
        data.pop(MEM["stop_stationary_ticks"], None)
        data.pop(MEM["stop_from_detour"], None)

    # added for detour
    def on_enter_detour(data):
        data.setdefault("enter_counts", {}).setdefault("Detour", 0)
        data["enter_counts"]["Detour"] += 1
        # Reset probe counter and record the time we entered Detour.
        data[MEM["detour_probe_ticks"]] = 0
        # Use the current clock_s (maintained by _now_s) as our Detour entry timestamp.
        data[MEM["entered_detour_s"]] = float(data.get(MEM["clock_s"], 0.0))

    def on_exit_detour(data):
        # Clear Detour-specific timers and markers.
        data.pop(MEM["detour_probe_ticks"], None)
        data.pop(MEM["entered_detour_s"], None)

    fsm.on_enter("Stop", on_enter_stop)
    fsm.on_exit("Stop", on_exit_stop)
    fsm.on_enter("Detour", on_enter_detour)
    fsm.on_exit("Detour", on_exit_detour)


    # Start
    fsm.start(start_state)
    # Provide a small scratch-pad for our internal timers if needed
    fsm._data.setdefault(MEM["clock_s"], 0.0)
    return fsm, cfg
