# fsm.py  – small step-based finite-state machine
#
# Old script is adjusted to be compatible with autopilot_fsm
#
# Changed due to missmatches with the autopilot_fsm
# fsm has:
#   States are registered with add(name, handler, on_enter, on_exit, end=False)
#   Handlers have the signature handler(cargo, fsm)
#   on_enter / on_exit callbacks are called as cb(fsm)
#   There is no _data scratchpad
# autopilot wants:
#   fsm.add_state(name, handler)
#   fsm.on_enter(name, cb) and fsm.on_exit(name, cb) where cb(data)
#   scratch dictionary fsm._data that the planner uses to store things like clock_s, entered_stop_s, stop_reason
# 
# Essentially,
#   - Added self._data to store timers, memory, etc.
#   - Deleated add function to have seperate functions for add, end, and enter
#   - Adjusted step to accomodate for the changes
#   - cleaned up layout

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple

Handler = Callable[[Dict[str, Any], Dict[str, Any]], Any]
EnterCb = Callable[[Dict[str, Any]], None]
ExitCb = Callable[[Dict[str, Any]], None]


class FSM:
    def __init__(self) -> None:
        # state_name -> handler(cargo, data)
        self._handlers: Dict[str, Handler] = {}

        # state_name -> enter/exit callbacks(data)
        self._on_enter: Dict[str, EnterCb] = {}
        self._on_exit: Dict[str, ExitCb] = {}

        # end states (no transitions)
        self._end_states = set()

        # current state
        self._current: Optional[str] = None
        self._started: bool = False

        # shared scratchpad for timers, memory, etc.
        self._data: Dict[str, Any] = {}

        # record visited states
        self.history = []

    # ---------------------------------------------------------------
    # State registration API
    # ---------------------------------------------------------------
    
    # CHANGED API
    def add_state(self, name: str, handler: Handler, *, end: bool = False) -> None:
        """Register a state and its handler(cargo, data)."""
        self._handlers[name] = handler
        if end:
            self._end_states.add(name)

    def on_enter(self, name: str, cb: EnterCb) -> None:
        """Register a callback invoked after entering `name`."""
        self._on_enter[name] = cb

    def on_exit(self, name: str, cb: ExitCb) -> None:
        """Register a callback invoked before leaving `name`."""
        self._on_exit[name] = cb

    # ---------------------------------------------------------------
    # Start and state inspection
    # ---------------------------------------------------------------

    def start(self, name: str) -> None:
        """Begin execution in the given state."""
        if name not in self._handlers:
            raise KeyError(f"FSM.start: unknown state {name!r}")

        self._current = name
        self._started = True
        self.history.append(name)

        cb = self._on_enter.get(name)
        if cb:
            cb(self._data)

    @property
    def state(self) -> Optional[str]:
        return self._current


    # ---------------------------------------------------------------
    # Step function (Execution)
    # ---------------------------------------------------------------

    def step(self, cargo: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Run one FSM tick.

        Handler return convention:
            None                  → stay, output=None
            next_state (str)      → transition, output=None
            (next_state, output)  → transition with output

        Returns:
            (current_state_name, output)
        """
        if not self._started or self._current is None:
            raise RuntimeError("FSM.step called before FSM.start()")

        state = self._current

        # Terminal state: no transitions allowed
        if state in self._end_states:
            return state, None

        handler = self._handlers.get(state)
        if handler is None:
            raise KeyError(f"FSM.step: no handler for state {state!r}")

        # Run handler
        result = handler(cargo, self._data)

        # Changed
        # Decode handler output
        next_state = state
        output = None

        # Normalize return
        if isinstance(result, tuple) and len(result) == 2:
            next_state, output = result
        else:
            next_state, output = result, None

        # If handler didn’t propose a transition, stick to current
        if next_state is None:
            return state, output

        # If staying in same state, no enter/exit
        if next_state == state:
            return state, output

        # Validate next state
        if next_state not in self._handlers:
            raise KeyError(f"FSM.step: handler returned unknown state '{next_state}'")


        # Exit callback
        cb = self._on_exit.get(state)
        if cb:
            cb(self._data)

        # Switch state
        self._current = next_state
        self.history.append(next_state)

        # Enter callback
        cb = self._on_enter.get(next_state)
        if cb:
            cb(self._data)

        return self._current, output
