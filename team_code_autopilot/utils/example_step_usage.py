from fsm import FSM

def on_enter_drive(fsm): print("[enter] Drive")
def on_exit_drive(fsm):  print("[exit ] Drive")
def on_enter_stop(fsm):  print("[enter] Stop")

def drive_handler(cargo, fsm):
    # cargo is provided every tick, e.g. {"red": bool, "near_obstacle": bool}
    if cargo.get("red") or cargo.get("near_obstacle"):
        return "Stop", {"reason": "safety"}
    return "Drive", None  # stay

def stop_handler(cargo, fsm):
    if not cargo.get("red") and not cargo.get("near_obstacle"):
        return "Drive", {"reason": "clear"}
    return "Stop", None  # stay

fsm = FSM()
fsm.add("Drive", drive_handler, on_enter=on_enter_drive, on_exit=on_exit_drive)
fsm.add("Stop",  stop_handler,  on_enter=on_enter_stop)
fsm.add("Error", end=True)  # example end state

fsm.start("Drive")

# Simulated ticks
timeline = [
    {"red": False, "near_obstacle": False},  # Drive
    {"red": True},                            # -> Stop
    {"red": True},                            # Stop (stay)
    {"red": False},                           # -> Drive
]

for i, cargo in enumerate(timeline):
    state, out = fsm.step(cargo)
    print(f"t={i:02d} state={state} output={out}")