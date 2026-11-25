# main.py
import argparse
import os
import time
import cv2
from ultralytics import YOLO
from plyer import notification
from playsound import playsound
from agent import FocusAgent, FocusState  # Import our agent's brain

# --- Helper function to format time ---
def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02}:{secs:02}"

# --- Actuators ---
def trigger_alert_notification():
    try:
        notification.notify(
            title='Focus Alert!',
            message='Distraction detected. Please focus on your studies.',
            app_name='Smart Study Zone',
            timeout=10
        )
    except Exception as e:
        print(f"Error sending notification: {e}")

def trigger_alert_sound():
    try:
        playsound('alert.mp3', block=False)
    except Exception as e:
        print(f"Error playing sound: {e} (Make sure 'alert.mp3' is in the folder)")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["final", "baseline"], default="final",
                  help="final: heuristics + conf=0.6; baseline: no persistence, no reset, low conf")
    p.add_argument("--conf", type=float, default=None,
                  help="Override YOLO confidence threshold (default depends on mode)")
    ts = time.strftime("logs/run_%Y%m%d_%H%M%S")
    p.add_argument("--log-basename", default=ts, help="Where to save CSV logs (without extension)")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.log_basename), exist_ok=True)

    # --- 1. Initialize Sensor (YOLO) and Agent ---
    model = YOLO('yolov8m.pt')

    # COCO classes: person=0, cell phone=67, book=73.
    CLASSES_WE_CARE_ABOUT = [0, 67, 73]

    agent = FocusAgent(distraction_threshold_seconds=5)

    # Mode-dependent thresholds
    if args.mode == "baseline":
        # Disable persistence and reset; use low confidence
        agent.PERSISTENCE_THRESHOLD = 0.0
        agent.DISTRACTION_RESET_THRESHOLD = 0.0
        conf = 0.2 if args.conf is None else args.conf
        print("[Mode] Baseline reflex agent (no persistence), YOLO conf =", conf)
    else:
        # Heuristic-augmented agent (your default)
        conf = 0.6 if args.conf is None else args.conf
        print("[Mode] Final agent (persistence & reset), YOLO conf =", conf)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Agent is running... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- PERCEPT PROCESSOR: Run YOLO detection on the frame ---
        results = model(frame, classes=CLASSES_WE_CARE_ABOUT, conf=conf, verbose=False)

        detections = []
        annotated_frame = frame.copy()

        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            detections.append(class_name)
            
            # Draw box ONLY for 'cell phone'
            if class_name == 'cell phone':
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "DISTRACTION", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        action_to_take = agent.update_state(detections)

        if action_to_take == "TRIGGER_ALERT":
            trigger_alert_notification()
            trigger_alert_sound()

        # --- Display / Visualization ---
        state_text = f"STATE: {agent.current_state.name}"
        study_time_str = format_time(agent.total_study_time)
        timer_text = f"Study Time: {study_time_str}"
        
        color = (0, 255, 0)  # Green for STUDYING
        if agent.current_state == FocusState.DISTRACTED:
            color = (0, 0, 255)      # Red for DISTRACTED
        elif agent.current_state == FocusState.IDLE:
            color = (128, 128, 128)  # Grey for IDLE
        elif agent.current_state == FocusState.ON_BREAK:
            color = (255, 150, 0)    # Orange for ON_BREAK

        # UI panel (top-left)
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (5, 5), (350, 90), (0, 0, 0), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0)
        cv2.putText(annotated_frame, state_text, (15, 40),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2)
        cv2.putText(annotated_frame, timer_text, (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Flashing alert bar (top)
        if agent.current_state == FocusState.DISTRACTED:
            pulse = (time.time() % 1) > 0.5
            bar_color = (0, 0, 255) if pulse else (0, 0, 150)
            cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], 45), bar_color, -1)
            alert_text = "== DISTRACTION DETECTED =="
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(annotated_frame, alert_text, (text_x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Smart Study Zone Agent", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

    # Export logs 
    dense_path, tran_path = agent.export_logs(args.log_basename)
    final_time_str = format_time(agent.total_study_time)
    print(f"Agent shutting down. Total Study Time: {final_time_str}")
    print(f"Saved logs:\n  Dense: {dense_path}\n  Transitions: {tran_path}")

if __name__ == "__main__":
    main()
