# step_fsm.py


class FSM:
    def __init__(self):
        # state_name -> handler(cargo, fsm) -> next or (next, output)
        self.handlers = {}
        # state_name -> callbacks
        self._on_enter = {}
        self._on_exit = {}
        self.end_states = set()
        self._current = None
        self._started = False
        self.history = []

    # --- Registration ---------------------------------------------------
    def add(self, name, handler=None, *, end=False, on_enter=None, on_exit=None):
        """
        Register a state.
        - handler: callable(cargo, fsm) -> next_state | (next_state, output)
          (handler can be None for end states)
        - end=True: marks terminal states (no handler required)
        - on_enter/on_exit: optional callbacks (fsm) -> None
        """
        self.handlers[name] = handler
        if end:
            self.end_states.add(name)
        if on_enter:
            self._on_enter[name] = on_enter
        if on_exit:
            self._on_exit[name] = on_exit

    def start(self, name):
        if name not in self.handlers:
            raise NotImplementedError(f"Unknown start state '{name}'")
        self._current = name
        self._started = True
        self.history.append(name)
        cb = self._on_enter.get(name)
        if cb:
            cb(self)

    # --- Execution ------------------------------------------------------
    @property
    def state(self):
        return self._current

    def step(self, cargo):
        """
        Consume one tick of input (cargo), optionally transition, and return (state, output).
        - If in an end state: stays there and returns (state, None).
        - If handler returns None or same state: we remain in that state.
        - If handler returns (next, output): we transition and return output.
        """
        if not self._started:
            raise NotImplementedError("Call .start(name) before .step(cargo)")

        state = self._current

        # Terminal: no handler needed; remain here
        if state in self.end_states:
            return state, None

        handler = self.handlers.get(state)
        if handler is None:
            raise NotImplementedError(f"State '{state}' has no handler")

        result = handler(cargo, self)

        # Normalize return
        if isinstance(result, tuple):
            next_state, output = result
        else:
            next_state, output = result, None

        # If handler didn’t propose a transition, stick to current
        if next_state is None:
            return state, output

        # If staying in same state, no enter/exit
        if next_state == state:
            return state, output

        # Transition
        if next_state not in self.handlers:
            raise NotImplementedError(f"Handler returned unknown state '{next_state}'")



        # exit current
        cb = self._on_exit.get(self.state)
        if cb:
            cb(self)

        self._current = next_state
        self.history.append(next_state)

        # enter next
        cb = self._on_enter.get(self.state)
        if cb:
            cb(self)

        return self._current, output

