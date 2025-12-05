# autopilot_fsm.py — skeleton for behavioral planning
# Depends on: fsm.py  
# States: Drive, Stop, later: detour
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
#   - Section 6: Decide which state should be next and what should be output to control
#   - Section 7: Builds a fully wired FSM plus its configuration

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from .fsm import FSM


# =============================================================================================================================
# Section 1: Output of the behavioural planning - passed to the caller each tick
# =============================================================================================================================
@dataclass  # class for holding data from dataclasses
class PlannerOutput:
    """Compact directive for the control stack."""
    mode: str                       # Symbolic command: "DRIVE", "STOP", later "DETOUR"
    reason: Optional[str] = None    # Symbolic command to see reason: e.g. "obstacle", "red_light", "stop_sign", "clear"
    target_speed: Optional[float] = None  # Target speed for the controller (0 for STOP, x for DRIVE, y for DETOUR)
    notes: Optional[Dict[str, Any]] = None  # Any further information that might be important (e.g. dwell time, obstacle distance, lane ID)
    waypoint: Optional[Tuple[float, float]] = None  # Next waypoint (x, y)

    # =========================
    # 2 Helper constructs to simplify creating a state of the ergo vehicle
    @staticmethod   # to show the reason for stop
    def stop(reason: str, waypoint: Optional[Tuple[float, float]] = None) -> "PlannerOutput":
        return PlannerOutput(mode="STOP", reason=reason, target_speed=0.0, waypoint=waypoint)

    @staticmethod   # to show the reason for drive
    def drive(target_speed: Optional[float] = None, reason: str = "clear", waypoint: Optional[Tuple[float, float]] = None) -> "PlannerOutput":
        return PlannerOutput(mode="DRIVE", reason=reason, target_speed=target_speed, waypoint=waypoint)


# =============================================================================================================================
# Section 2: Cargo contract (what the FSM recieves each step)
# =============================================================================================================================

# Time information:
#   dt: time step of the simulation in seconds since the previous tick - used to keep 
#       track of how long the vehicle has been in a state
#   t:  absolute simulation time - can be used so that dt must not be integrated
# Ego-vehicle state:
#   speed: current speed of the vehicle, used to determine if the ergo vehicle is stopped
# Obstacle gating: if any condition is true, the obstacle must stop
#   obstacle_ahead:    computed from LIDAR and the safety box, true if an obstacle is ahead 
#   obstacle_distance: distance to the closest obstacle ahead, computed from HD map
# Traffic light gating:
#   tl_red:           true if trafic light ahead is red
#   tl_near_stopline: true when the stop line for the trafic light is approached, stop lign 
#                     position recieved from HD map or RGB segmentation data - so that ergo 
#                     vehicle only stops when it is close enough to the stop line
# Stop-sign gating:
#   stop_sign_ahead:  true if stop sign is ahead
#   ss_near_stopline: true when the stop line for the stop sign is approached, stop lign 
#                     position recieved from HD map or RGB segmentation data - so that ergo 
#                     vehicle only stops when it is close enough to the stop line
# 
# OPTIONALS:
# Controller hints:
#   cruise_target: desired speed in state drive

"""
Expected keys in `cargo` (one tick of fused perception + map context):

# Time information
- dt: float                       # [s] sim/control step 
- t: Optional[float]              # [s] absolute sim clock (optional but preferred) 

# Ego-vehicle state
- speed: float                    # [m/s] current ego speed

# Obstacle gating
- obstacle_ahead: bool            # fused occupancy in safety AABB / path
- obstacle_distance: Optional[float]  # [m] nearest obstacle along path (optional)

# Traffic light gating
- tl_red: bool                    # True if the relevant light governing our lane is red
- tl_near_stopline: bool          # True when we are approaching/at the stop line

# Stop-sign gating
- stop_sign_ahead: bool           # True if a stop sign applies to our approach
- ss_near_stopline: bool          # True when at/approaching the stop line for the sign

# Controller hints (optional)
- cruise_target: Optional[float]  # [m/s] desired free-flow speed for DRIVE

# Next waypoint
- waypoint: Tuple[float, float]   # next waypoint

HDMap compatibility notes (with respect to hdmap.HDMap):

- `dt` is NOT produced by HDMap - is a planning/configuration choice, not an HDMap output.
- `t` is NOT produced by HDMap - derived from CARLA simulation clock

- `speed` are NOT produced by HDMap; must be derived from CARLA's world/ego actor (e.g. ego.get_velocity())

- `obstacle_ahead` is going to be determined via hd.is_obstacle_in_front
- `obstacle_distance` is currently NOT provided by hdmap.py 

- `tl_red` and `tl_near_stopline` are NOT provided by hdmap.py

- `stop_sign_ahead` and `ss_near_stopline` are NOT provided by hdmap.py

- `cruise_target` is a planning/configuration choice, not an HDMap output.

- `waypoint`

"""

def build_cargo_from_hdmap(
    *,
    hdmap_obj: Any,
    ego_actor: Any,
    dt: float,
    t: Optional[float] = None,
    cruise_target: Optional[float] = None,
    obstacle_distance: Optional[float] = None,
    obstacle_check_distance: float = 10.0, # adjustable: how far ahead we look, included in hdmap.py too
    obstacle_fov_deg: float = 30.0, # adjustable: field of view for deg, included in hdmap.py too
) -> Dict[str, Any]:
    """
    Helper to construct a `cargo` dict using the HDMap implementation
    and a CARLA ego vehicle actor. It notes which inputs are
    currently missing from hdmap.py

    Parameters
    ----------
    hdmap_obj:
        Instance of hdmap.HDMap 
    ego_actor:
        CARLA ego vehicle actor (to compute speed).
    dt:
        Time step [s] since last control/FSM tick.
    t:
        Absolute simulation time [s] (if available).
    cruise_target:
        Desired drive speed [m/s].
    obstacle_distance:
        Optional distance to nearest obstacle ahead [m]. Not provided by HDMap.
    obstacle_check_distance:
        Look-ahead distance [m] for hdmap_obj.is_obstacle_in_front.
    obstacle_fov_deg:
        Field-of-view angle [deg] for hdmap_obj.is_obstacle_in_front.

    Returns
    -------
    cargo : Dict[str, Any]
        Dictionary in the exact format expected by the FSM.
    """

    # --- Time information ---
    # to get we might have to do:
    # snapshot = world.get_snapshot()
    # t = snapshot.timestamp.elapsed_seconds
    # dt = t - last_t
    # cargo["t"] = t
    # cargo["dt"] = dt
    cargo: Dict[str, Any] = {
        "dt": float(dt),
        "t": float(t) if t is not None else None,
    }

    # --- Ego speed from CARLA ego_actor ---
    # NOTE: HDMap does not directly expose speed. We compute it from the
    #       CARLA velocity vector here so that the FSM receives a scalar [m/s].
    vel = ego_actor.get_velocity()
    speed = (vel.x ** 2 + vel.y ** 2) ** 0.5 # square root, possibly z must be excluded
    # dongghk : changed speed to only calculate in 2D plane (x,y), z is vertical axis
    
    cargo["speed"] = float(speed)

    # --- Obstacle gating from HDMap ---
    obstacle_ahead = bool(
        hdmap_obj.is_obstacle_in_front(
            distance=float(obstacle_check_distance),
            fov_deg=float(obstacle_fov_deg),
        )
    )
    cargo["obstacle_ahead"] = obstacle_ahead

    # --- HDMap currently (NOT yet provided by HDMap) ---
    cargo["obstacle_distance"] = obstacle_distance # TODO: implement

    # --- Traffic light gating (NOT yet provided by HDMap) ---
    # TODO: Implement traffic-light perception (map + actors) and fill these:
    try:
        tl_red = bool(hdmap_obj.is_traffic_light_red())
    except AttributeError:
        # Failsafe: if method missing, behave as if no red light known.
        tl_red = False
    cargo["tl_red"] = tl_red

    cargo["tl_near_stopline"] = False  # TODO: requires stop-line distance check

    # --- Stop-sign gating (NOT yet provided by HDMap) ---
    # TODO: Implement stop-sign perception and fill these:
    cargo["stop_sign_ahead"] = False   # TODO: not available in current HDMap
    cargo["ss_near_stopline"] = False  # TODO: requires stop-line distance check

    # --- Controller hint ---
    cargo["cruise_target"] = cruise_target

    cargo["waypoint"] = hdmap_obj.get_next_waypoint()

    return cargo


# =============================================================================================================================
# Section 3: Internal FSM memory keys
# =============================================================================================================================

# clock_s:        internal clock of the ergo vehicle (time in seconds)
# entered_stop_s: timestamp at which the ergo vehicle stopped, used for dwell time
# stop_reason:    reason for stopping

MEM = { # dictionary used as it can be expanded easily
    "clock_s": "clock_s",             # monotonically increasing clock if cargo.t not supplied
    "entered_stop_s": "entered_stop_s",
    "stop_reason": "stop_reason",     # "obstacle" | "red_light" | "stop_sign"
}


# =============================================================================================================================
# Section 4: Configuration (tunable)
# =============================================================================================================================
@dataclass # Possibly not needed
class PlannerConfig:
    # Minimum time in seconds the ergo vehicle must stay stopped in order to enter the stop state.
    # Used as perception output fluctuates between ticks. To prevent flickering behaviour, this will
    # be used to specify the minimum time the vehicle must remain stopped after the vehicle enters the
    # stop state. This is used for everything except stop signs
    min_stop_s: float = 0.5

    # Mandatory legal stop duration when the ego-vehicle stops at a stop sign. Used exclucively for stop signs
    stop_sign_min_s: float = 2.0

    # While at a red light we must remain stopped until it turns not red AND min_stop_s elapsed.
    # For obstacles we must remain until obstacle clears AND min_stop_s elapsed.
    # For stop signs, we must remain at/near the stop line for at least stop_sign_min_s,
    # and only then we can proceed if the path is clear and we have right-of-way.
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

def _stop_elapsed_s(data: Dict[str, Any], now_s: float) -> float:
    # Calculates how long the ergo vehicle has been in the stop state.
    # Looks up time we entered stop (eventually derived from _begin_stop) 
    # and subtracts it from the current time (derived from _now_s)

    t0 = float(data.get(MEM["entered_stop_s"], now_s))
    return max(0.0, now_s - t0)

def _should_stop(cargo: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    # Decides if the FSM should switch from Drive to Stop state, based on the cargo input.
    # Checks gating conditions from cargo contract (Obstacle gating, Traffic light gating,
    # Stop-sign gating) and outputs when the ergo vehicle must stop.

    """Predicate deciding if we must *enter* STOP from DRIVE this tick."""
    # If there’s an obstacle in front it must stop
    if cargo.get("obstacle_ahead", False):
        return True, "obstacle"
    # If the traffic light is red and we’re near its stop line it must stop.
    if cargo.get("tl_red", False) and cargo.get("tl_near_stopline", False):
        return True, "red_light"
    # If there’s a stop sign ahead and we’re near its stop line it must stop.
    if cargo.get("stop_sign_ahead", False) and cargo.get("ss_near_stopline", False):
        return True, "stop_sign"
    # Otherwise, no stop needed.
    return False, None

def _clear_to_go_from_stop(cargo: Dict[str, Any], data: Dict[str, Any], cfg: PlannerConfig, now_s: float) -> bool:
    # Decides if ergo vehicle can can drive again. Examines the reason it stopped from MEM, how
    # long it has been stopped from _stop_elapsed_s, the current cargo (e.g. traffic light state), 
    # and the rules in the configuration section (section 4). Returns true if all conditions for 
    # leaving the stop are satisfied.
    
    """Remain in STOP until both (a) gating condition cleared and (b) dwell satisfied."""
    # reason the ergo vehicle stopped
    reason = data.get(MEM["stop_reason"], None)
    # time since the ergo vehicle stopped
    dwell = _stop_elapsed_s(data, now_s)

    # Gate 1: Checks if the stop condition has ended
    obstacle_clear = not cargo.get("obstacle_ahead", False)
    red_clear      = not cargo.get("tl_red", False)
    ss_clear       = not (cargo.get("stop_sign_ahead", False) and cargo.get("ss_near_stopline", False))

    # If obstacle: Wait until obstacle is gone and dwell >= min_stop_s
    if reason == "obstacle":
        cond_cleared = obstacle_clear
        dwell_needed = cfg.min_stop_s
    # If red light: Wait until light is green and dwell >= min_stop_s
    elif reason == "red_light":
        cond_cleared = red_clear
        dwell_needed = cfg.min_stop_s
    # If stop sign: Wait until dwell >= stop_sign_min_s and no obstacle ahead
    elif reason == "stop_sign":
        # For stop signs, we require the legal dwell at/near stop line
        # AND no other blocking condition (obstacle on path).
        cond_cleared = ss_clear and obstacle_clear
        dwell_needed = cfg.stop_sign_min_s
        # TODO : dongghk : this will never be cleared if stopped at the stop sign (ss_clear is false)
        # should change the logic  

    # If reason is unknown, stay stopped
    else:
        # Fallback: be conservative
        cond_cleared = obstacle_clear and red_clear and ss_clear
        dwell_needed = cfg.min_stop_s

    # Return True only if both the condition for either reason and the dwell time are satisfied
    return (dwell >= dwell_needed) and cond_cleared


# =============================================================================================================================
# Section 6: State handlers
# =============================================================================================================================
def drive_handler(cargo: Dict[str, Any], data: Dict[str, Any], cfg: PlannerConfig) -> Tuple[str, PlannerOutput]:
    # Follow way points unless state switch is required

    """Free flow unless a gating condition requires STOP."""
    now = _now_s(cargo, data)
    waypoint = cargo.get("waypoint", None)
    need_stop, reason = _should_stop(cargo) # checks if any gating contidion is active
    if need_stop: # if condition is active, marks the time and reason
        _begin_stop(data, now, reason)  # time when stop is entered
        # returns state name, PlannerOutput, so reason
        return "Stop", PlannerOutput.stop(reason=reason, waypoint=waypoint) # outputs stop, more outputs can be added

    # Otherwise continue to DRIVE; ideally with specified speed (cruise_target)
    target_v = cargo.get("cruise_target", None)

    # returns state name, PlannerOutput , so target speed, and reason for now
    return "Drive", PlannerOutput.drive(target_speed=target_v, reason="clear", waypoint=waypoint) # Can add more outputs, such as notes


def stop_handler(cargo: Dict[str, Any], data: Dict[str, Any], cfg: PlannerConfig) -> Tuple[str, PlannerOutput]:
    # Stay stop until state switch is required

    """Hold STOP until both dwell and clearance conditions are satisfied."""
    now = _now_s(cargo, data) # loads current time

    # If we *still* should be stopping due to current inputs, remain stopped (this also
    # refreshes the stop reason if it changed in place, but we keep the original dwell start).
    still_stop, current_reason = _should_stop(cargo) # checks if we should still be stopped
    stored_reason = data.get(MEM["stop_reason"], current_reason or "unknown") # loads the current reason for the stop in the next tick

    if still_stop and current_reason and (current_reason != stored_reason): 
        # If the stop cause changed (e.g., rolled from red -> obstacle), update the recorded reason without touching the stop time
        data[MEM["stop_reason"]] = current_reason
        stored_reason = current_reason

    waypoint = cargo.get("waypoint", None)
    
    # Check whether it's safe and legal to go
    if _clear_to_go_from_stop(cargo, data, cfg, now): # if it is legal to drive, switch to drive
        return "Drive", PlannerOutput.drive(reason="clear_after_stop", waypoint=waypoint)

    # Otherwise, remain in STOP
    # returns state name, PlannerOutput, so reason for now (speed == 0 is specified in the class definition already)
    return "Stop", PlannerOutput.stop(reason=stored_reason, waypoint=waypoint)


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
    fsm.add_state("Drive", lambda cargo, data: drive_handler(cargo, data, cfg)) # function must be added
    fsm.add_state("Stop",  lambda cargo, data: stop_handler(cargo, data, cfg)) # function must be added

    # Optional: enter/exit hooks (useful for logging/telemetry/markers)
    def on_enter_stop(data):
        data.setdefault("enter_counts", {}).setdefault("Stop", 0)
        data["enter_counts"]["Stop"] += 1

    def on_exit_stop(data):
        data.pop(MEM["entered_stop_s"], None)
        data.pop(MEM["stop_reason"], None)

    fsm.on_enter("Stop", on_enter_stop)
    fsm.on_exit("Stop", on_exit_stop)

    # Start
    fsm.start(start_state)
    # Provide a small scratch-pad for our internal timers if needed
    fsm._data.setdefault(MEM["clock_s"], 0.0)
    return fsm, cfg


# =========================
# Minimal usage example (for quick local test / can be deleted)
# =========================
# =========================
# Extended usage example: test all stop reasons
# =========================
def run_scenario(name: str, timeline):
    """
    Helper to run one scenario:
    - Builds a fresh FSM and config
    - Steps through all cargo inputs in 'timeline'
    - Prints the state and PlannerOutput at each tick
    """
    from pprint import pprint

    print("\n" + "=" * 80)
    print(f"SCENARIO: {name}")
    print("=" * 80)

    # Build a fresh FSM for this scenario
    fsm, cfg = build_vehicle_fsm()

    # Print a header so the output is easy to read
    print("tick | state | mode  | reason         | target_speed | waypoint")
    print("-----+-------+-------+---------------+--------------+---------")

    for i, cargo in enumerate(timeline):
        state, out = fsm.step(cargo)

        print(
            f"{i:03d}  | "
            f"{state:5s} | "
            f"{out.mode:5s} | "
            f"{(out.reason or '-'):13s} | "
            f"{(out.target_speed if out.target_speed is not None else -1):12.2f} | "
            f"{out.waypoint}"
        )


def build_obstacle_scenario():
    """
    Scenario 1: Obstacle appears and then clears.

    Expect:
    - Start in DRIVE
    - When obstacle_ahead=True -> switch to STOP (reason='obstacle')
    - After obstacle disappears and we waited long enough -> back to DRIVE
    """
    timeline = []

    # Phase A: free driving, no obstacle
    for _ in range(5):
        timeline.append(dict(
            dt=0.1,
            speed=5.0,
            obstacle_ahead=False,
            tl_red=False,
            tl_near_stopline=False,
            stop_sign_ahead=False,
            ss_near_stopline=False,
            cruise_target=6.0,
        ))

    # Phase B: obstacle appears -> should trigger STOP
    timeline.append(dict(
        dt=0.1,
        speed=5.0,
        obstacle_ahead=True,   # <-- obstacle in front now
        tl_red=False,
        tl_near_stopline=False,
        stop_sign_ahead=False,
        ss_near_stopline=False,
        cruise_target=6.0,
    ))

    # Phase C: obstacle is gone, but vehicle still stopped for some time
    # cfg.min_stop_s = 0.5 by default, with dt=0.1 -> need at least 5 ticks
    for _ in range(5):
        timeline.append(dict(
            dt=0.1,
            speed=0.0,
            obstacle_ahead=False,  # obstacle cleared
            tl_red=False,
            tl_near_stopline=False,
            stop_sign_ahead=False,
            ss_near_stopline=False,
            cruise_target=6.0,
        ))

    # Phase D: continue driving again
    for _ in range(5):
        timeline.append(dict(
            dt=0.1,
            speed=3.0,
            obstacle_ahead=False,
            tl_red=False,
            tl_near_stopline=False,
            stop_sign_ahead=False,
            ss_near_stopline=False,
            cruise_target=6.0,
        ))

    return timeline


def build_red_light_scenario():
    """
    Scenario 2: Red traffic light ahead at the stop line.

    Expect:
    - Start in DRIVE (green light)
    - Turn light red near stop line -> STOP (reason='red_light')
    - Remain stopped while red
    - When light turns green and we've waited long enough -> DRIVE again
    """
    timeline = []

    # Phase A: approaching intersection, green light
    for _ in range(5):
        timeline.append(dict(
            dt=0.1,
            speed=6.0,
            obstacle_ahead=False,
            tl_red=False,          # green light
            tl_near_stopline=False,
            stop_sign_ahead=False,
            ss_near_stopline=False,
            cruise_target=7.0,
        ))

    # Phase B: close to stop line; light turns red -> should trigger STOP
    timeline.append(dict(
        dt=0.1,
        speed=5.0,
        obstacle_ahead=False,
        tl_red=True,             # <-- red light
        tl_near_stopline=True,   # <-- near its stop line
        stop_sign_ahead=False,
        ss_near_stopline=False,
        cruise_target=7.0,
    ))

    # Phase C: we are stopped at the red light for a while
    for _ in range(5):
        timeline.append(dict(
            dt=0.1,
            speed=0.0,
            obstacle_ahead=False,
            tl_red=True,             # still red
            tl_near_stopline=True,
            stop_sign_ahead=False,
            ss_near_stopline=False,
            cruise_target=7.0,
        ))

    # Phase D: light turns green, we stay near stop line but can now go after dwell
    for _ in range(5):
        timeline.append(dict(
            dt=0.1,
            speed=0.0,               # still basically stopped
            obstacle_ahead=False,
            tl_red=False,            # <-- turns green
            tl_near_stopline=True,   # still near stop line
            stop_sign_ahead=False,
            ss_near_stopline=False,
            cruise_target=7.0,
        ))

    # Phase E: we drive away
    for _ in range(5):
        timeline.append(dict(
            dt=0.1,
            speed=4.0,
            obstacle_ahead=False,
            tl_red=False,
            tl_near_stopline=False,
            stop_sign_ahead=False,
            ss_near_stopline=False,
            cruise_target=7.0,
        ))

    return timeline


def build_stop_sign_scenario():
    """
    Scenario 3: Stop sign ahead at stop line.

    Expect:
    - Start in DRIVE, no sign
    - stop_sign_ahead=True & ss_near_stopline=True -> STOP (reason='stop_sign')
    - Must wait at least stop_sign_min_s (default 2.0 s) AND have no obstacle ahead
    - Then we can DRIVE again
    """
    timeline = []

    # Phase A: driving normally, no stop sign
    for _ in range(5):
        timeline.append(dict(
            dt=0.5,                 # larger dt here so we hit 2.0s quickly
            speed=6.0,
            obstacle_ahead=False,
            tl_red=False,
            tl_near_stopline=False,
            stop_sign_ahead=False,
            ss_near_stopline=False,
            cruise_target=6.0,
        ))

    # Phase B: stop sign appears and we're at its stop line -> STOP
    timeline.append(dict(
        dt=0.5,
        speed=4.0,
        obstacle_ahead=False,
        tl_red=False,
        tl_near_stopline=False,
        stop_sign_ahead=True,     # <-- sign ahead
        ss_near_stopline=True,    # <-- at/near stop line
        cruise_target=6.0,
    ))

    # Phase C: we remain at the stop sign for a while; must wait stop_sign_min_s
    # Default stop_sign_min_s = 2.0 s, and dt=0.5 -> 4 ticks => 2.0 seconds.
    for _ in range(4):
        timeline.append(dict(
            dt=0.5,
            speed=0.0,
            obstacle_ahead=False,   # path clear
            tl_red=False,
            tl_near_stopline=False,
            stop_sign_ahead=True,
            ss_near_stopline=True,  # still at stop line
            cruise_target=6.0,
        ))

    # Phase D: we start rolling again; sign is effectively passed (no longer ahead)
    for _ in range(5):
        timeline.append(dict(
            dt=0.5,
            speed=3.0,
            obstacle_ahead=False,
            tl_red=False,
            tl_near_stopline=False,
            stop_sign_ahead=False,  # sign no longer governs us
            ss_near_stopline=False,
            cruise_target=6.0,
        ))

    return timeline


if __name__ == "__main__":
    # Build and run all three test scenarios
    obstacle_timeline = build_obstacle_scenario()
    red_light_timeline = build_red_light_scenario()
    stop_sign_timeline = build_stop_sign_scenario()

    run_scenario("Obstacle ahead", obstacle_timeline)
    run_scenario("Red traffic light", red_light_timeline)
    run_scenario("Stop sign", stop_sign_timeline)


