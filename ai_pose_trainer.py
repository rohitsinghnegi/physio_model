
import cv2
import json
from datetime import datetime
from pathlib import Path
from pose_utils import PoseDetector
from exercise_logic import ExerciseLogic
from feedback_engine import FeedbackEngine

class AIPoseTrainer:
    def __init__(self):
        """Initialize the AI Pose Trainer"""
        self.pose_detector = PoseDetector()
        self.exercise_logic = ExerciseLogic()
        self.feedback_engine = FeedbackEngine()
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.current_exercise = "tree_pose"  # Set tree pose as default
        self.available_exercises = ["tree_pose", "squat", "pushup", "jumping_jack"]
        self.exercise_index = 0
        self.log_file = Path('exercise_log.json')
        self._init_log_file()

    def _init_log_file(self):
        """Initialize the log file if it doesn't exist or is corrupted"""
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            raise ValueError("Invalid log file format")
                    except (json.JSONDecodeError, ValueError):
                        # File is corrupted, recreate it
                        with open(self.log_file, 'w') as f:
                            json.dump([], f)
            else:
                with open(self.log_file, 'w') as f:
                    json.dump([], f)
        except Exception as e:
            print(f"Error initializing log file: {e}")  # For debugging

    def run(self):
        """Main loop for the pose trainer"""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                keypoints = self.pose_detector.detect_pose(frame)

                if keypoints:
                    # For simplicity, we'll use the first detected person
                    angles = self.pose_detector.calculate_all_angles(keypoints[0])
                    
                    # Process the current exercise
                    form_ok = self.exercise_logic.check_form(self.current_exercise, angles)
                    if form_ok:
                        rep_completed = self.exercise_logic.count_reps(self.current_exercise, angles)
                        if rep_completed:
                            print(f"Rep completed! Count: {self.exercise_logic.rep_count}")  # Debug print
                            self.feedback_engine.queue_feedback(f"Rep {self.exercise_logic.rep_count} completed!")
                            self.log_exercise_data(angles, "rep_completed")

                    # Display information
                    self._display_info(frame, angles)

                # Add instruction text
                cv2.putText(frame, "Press 'n' to switch exercise", (10, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('AI Pose Trainer', frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # q or ESC
                    break
                elif key == ord('n'):  # Switch exercise
                    self.exercise_index = (self.exercise_index + 1) % len(self.available_exercises)
                    self.current_exercise = self.available_exercises[self.exercise_index]
                    self.exercise_logic.reset()  # Reset rep count and state
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.feedback_engine.stop()

    def _display_info(self, frame, angles):
        """Display exercise info on the frame"""
        # Exercise name and rep count
        cv2.putText(frame, f"Exercise: {self.current_exercise.title().replace('_', ' ')}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Reps: {self.exercise_logic.rep_count}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Get and display confidence score and feedback
        confidence, feedback = self.exercise_logic.get_form_feedback(self.current_exercise, angles)
        
        # Display confidence score with color based on value
        color = (0, 255, 0) if confidence >= 80 else (0, 165, 255) if confidence >= 50 else (0, 0, 255)
        cv2.putText(frame, f"Form Score: {confidence}%", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display feedback messages
        y_pos = 200
        if isinstance(feedback, list):
            for msg in feedback:
                cv2.putText(frame, msg, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                y_pos += 40
        else:
            cv2.putText(frame, feedback, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
        # Instructions
        cv2.putText(frame, "Press 'n' to switch exercise", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def log_exercise_data(self, angles, status):
        """Log exercise data to a JSON file"""
        try:
            # Convert numpy float32 to regular float for JSON serialization
            serializable_angles = {}
            for k, v in angles.items():
                try:
                    if v is not None:
                        serializable_angles[k] = float(v)
                    else:
                        serializable_angles[k] = None
                except:
                    serializable_angles[k] = None
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'exercise': self.current_exercise,
                'rep_count': self.exercise_logic.rep_count,
                'status': status,
                'angles': serializable_angles
            }
            
            # Read existing data
            try:
                if self.log_file.exists():
                    with open(self.log_file, 'r') as f:
                        try:
                            data = json.load(f)
                            if not isinstance(data, list):
                                data = []
                        except json.JSONDecodeError:
                            data = []
                else:
                    data = []
            except:
                data = []
            
            # Append new entry
            data.append(log_entry)
            
            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=4)
                
        except Exception as e:
            print(f"Error logging exercise data: {e}")  # For debugging

if __name__ == "__main__":
    trainer = AIPoseTrainer()
    trainer.run()
