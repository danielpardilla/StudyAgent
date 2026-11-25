# agent.py
import time
import csv
import os
from enum import Enum

# Define the states of our 4-state Finite State Machine (FSM)
class FocusState(Enum):
    IDLE = 1       # No one is at the desk
    STUDYING = 2   # Person is at the desk WITH study materials
    ON_BREAK = 3   # Person is at the desk WITHOUT study materials
    DISTRACTED = 4 # Person is studying AND phone is visible

# Canonical numeric IDs for plotting (0..3 on the Y-axis)
STATE_ID = {
    FocusState.IDLE: 0,
    FocusState.ON_BREAK: 1,
    FocusState.STUDYING: 2,
    FocusState.DISTRACTED: 3
}

class FocusAgent:
    """World Model with 'patience' to prevent flicker and with logging for evaluation."""
    
    def __init__(self, distraction_threshold_seconds=10):
        # --- Internal State ---
        self.current_state = FocusState.IDLE
        self.distraction_timer_start = None
        self.alert_triggered = False
        
        # --- State Stability (Patience) Timestamps ---
        self.last_seen_person = 0.0
        self.last_seen_study_material = 0.0
        self.last_seen_distraction = 0.0
        
        # --- Study Timer Variables ---
        self.total_study_time = 0.0
        self.last_frame_time = time.time()  # delta between frames
        
        # --- Tuned Thresholds for Patience ---
        self.PERSISTENCE_THRESHOLD = 10.0        # person/book absent must persist >=10s
        self.DISTRACTION_RESET_THRESHOLD = 2.0    # phone must be gone >=2s to clear
        
        # --- Config ---
        self.DISTRACTION_THRESHOLD = distraction_threshold_seconds  # seconds to trigger alert
        
        # --- Logging (dense per second + on-transition) ---
        self.start_time = time.time()
        self._last_dense_log_time = self.start_time - 1.0  # forces a row at ~t=0
        self.log_dense = []         # [{time_s:int, state_id:int, state:str}, ...]
        self.log_transitions = []   # same schema but only when state changes
        # seed initial transition row
        self.log_transitions.append({"time_s": 0, "state_id": STATE_ID[self.current_state],
                                     "state": self.current_state.name})

    def _log_transition(self, now):
        t_rel = round(now - self.start_time, 2)
        self.log_transitions.append({
            "time_s": t_rel,
            "state_id": STATE_ID[self.current_state],
            "state": self.current_state.name
        })

    def _log_dense_if_needed(self, now):
        if now - self._last_dense_log_time >= 1.0:
            t_rel_int = int(now - self.start_time)
            self.log_dense.append({
                "time_s": t_rel_int,
                "state_id": STATE_ID[self.current_state],
                "state": self.current_state.name
            })
            self._last_dense_log_time = now

    def export_logs(self, out_basename="logs/run"):
        os.makedirs(os.path.dirname(out_basename), exist_ok=True)
        dense_path = f"{out_basename}_dense.csv"
        tran_path  = f"{out_basename}_transitions.csv"
        with open(dense_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["time_s", "state_id", "state"])
            w.writeheader()
            w.writerows(self.log_dense)
        with open(tran_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["time_s", "state_id", "state"])
            w.writeheader()
            w.writerows(self.log_transitions)
        return dense_path, tran_path

    def update_state(self, detections):
        """Core agent update: ingest percepts and update FSM with patience + logging."""
        # --- time bookkeeping ---
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # --- raw percepts ---
        has_person_raw = 'person' in detections
        has_distraction_raw = 'cell phone' in detections
        # study-material 
        has_study_material_raw = 'book' in detections
        
        if has_person_raw:
            self.last_seen_person = current_time
        if has_study_material_raw:
            self.last_seen_study_material = current_time
        if has_distraction_raw:
            self.last_seen_distraction = current_time

        # --- persistent beliefs (flicker-proof for absence) ---
        person_is_present = (current_time - self.last_seen_person) <= self.PERSISTENCE_THRESHOLD
        study_material_is_present = (current_time - self.last_seen_study_material) <= self.PERSISTENCE_THRESHOLD
        distraction_is_gone = (current_time - self.last_seen_distraction) >= self.DISTRACTION_RESET_THRESHOLD
        
        # --- FSM logic ---
        prev_state = self.current_state  # track for transition logging
        
        if self.current_state == FocusState.IDLE:
            if person_is_present and study_material_is_present:
                self.current_state = FocusState.STUDYING
                print("[Agent Log] State changed to: STUDYING")
            elif person_is_present and not study_material_is_present:
                self.current_state = FocusState.ON_BREAK
                print("[Agent Log] State changed to: ON_BREAK")

        elif self.current_state == FocusState.STUDYING:
            if not person_is_present:
                self.current_state = FocusState.IDLE
                print("[Agent Log] State changed to: IDLE")
            elif has_distraction_raw:  # instant on phone presence
                self.current_state = FocusState.DISTRACTED
                if self.distraction_timer_start is None:
                    self.distraction_timer_start = time.time()
                    self.alert_triggered = False
                    print("[Agent Log] State changed to: DISTRACTED (Timer Started)")
            elif not study_material_is_present:
                self.current_state = FocusState.ON_BREAK
                print("[Agent Log] State changed to: ON_BREAK")

        elif self.current_state == FocusState.ON_BREAK:
            if not person_is_present:
                self.current_state = FocusState.IDLE
                print("[Agent Log] State changed to: IDLE")
            elif study_material_is_present:
                self.current_state = FocusState.STUDYING
                print("[Agent Log] State changed to: STUDYING")

        elif self.current_state == FocusState.DISTRACTED:
            if not person_is_present:
                self.current_state = FocusState.IDLE
                self.distraction_timer_start = None
                print("[Agent Log] State changed to: IDLE")
            elif distraction_is_gone:
                self.current_state = FocusState.STUDYING
                self.distraction_timer_start = None
                print("[Agent Log] State changed to: STUDYING (Distraction cleared)")
            elif not study_material_is_present:
                self.current_state = FocusState.ON_BREAK
                self.distraction_timer_start = None
                print("[Agent Log] State changed to: ON_BREAK")
            else:
                if not self.alert_triggered and self.distraction_timer_start is not None:
                    elapsed_time = time.time() - self.distraction_timer_start
                    if elapsed_time > self.DISTRACTION_THRESHOLD:
                        print(f"[Agent Log] ACTION: Triggering Alert! (Distracted for {elapsed_time:.0f}s)")
                        self.alert_triggered = True
                        # Note: return action *after* logging below
                        action = "TRIGGER_ALERT"
                        # logging + timers before return
                        if self.current_state == FocusState.STUDYING:
                            self.total_study_time += delta_time
                        # transition logging if needed
                        if self.current_state != prev_state:
                            self._log_transition(current_time)
                        self._log_dense_if_needed(current_time)
                        return action

        # --- study time accumulation ---
        if self.current_state == FocusState.STUDYING:
            self.total_study_time += delta_time

        # --- logging points ---
        if self.current_state != prev_state:
            self._log_transition(current_time)
        self._log_dense_if_needed(current_time)

        return "NONE"
