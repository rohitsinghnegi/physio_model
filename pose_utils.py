
import cv2
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.pt'):
        """Initialize the pose detector with a YOLO model"""
        self.model = YOLO(model_path)

    def detect_pose(self, frame):
        """Detect poses in a given frame and return keypoints"""
        results = self.model.predict(source=frame, show=False, stream=True, verbose=False)
        all_keypoints = []

        for result in results:
            if len(result.keypoints.data) > 0:
                # First person detected
                kpts = result.keypoints.data[0]
                if len(kpts) >= 17:
                    keypoints = kpts.cpu().numpy()
                    all_keypoints.append(keypoints)
                    
                    # Draw skeleton on frame
                    self._draw_skeleton(frame, keypoints)
                    
                    # Draw confidence values
                    for i, kp in enumerate(keypoints):
                        x, y, conf = kp
                        if conf > 0.5:  # Only show high confidence points
                            cv2.putText(frame, f"{conf:.2f}", (int(x), int(y)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
        return all_keypoints

    def _draw_skeleton(self, frame, keypoints):
        """Draw the skeleton on the frame"""
        # Define connections between keypoints
        skeleton = [
            (5, 7), (7, 9),     # Left arm
            (6, 8), (8, 10),    # Right arm
            (5, 6),             # Shoulders
            (5, 11), (6, 12),   # Spine
            (11, 12),           # Hips
            (11, 13), (13, 15), # Left leg
            (12, 14), (14, 16)  # Right leg
        ]

        # Draw connecting lines
        for pair in skeleton:
            if keypoints[pair[0]][2] > 0.5 and keypoints[pair[1]][2] > 0.5:
                pt1 = (int(keypoints[pair[0]][0]), int(keypoints[pair[0]][1]))
                pt2 = (int(keypoints[pair[1]][0]), int(keypoints[pair[1]][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for kp in keypoints:
            x, y, conf = kp
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    @staticmethod
    def calculate_angle(p1, p2, p3):
        """Calculate the angle between three points, handling potential visibility issues"""
        # Check if keypoints are visible (confidence > 0.5)
        if p1[2] < 0.5 or p2[2] < 0.5 or p3[2] < 0.5:
            return None
            
        a = np.array(p1[:2])
        b = np.array(p2[:2])
        c = np.array(p3[:2])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        return angle

    def calculate_all_angles(self, keypoints):
        """Calculate all relevant exercise angles from keypoints"""
        if len(keypoints) < 17:
            return {}

        angles = {
            # Shoulders (angle between upper arm and torso)
            'left_shoulder': self.calculate_angle(keypoints[7], keypoints[5], keypoints[11]),  # left elbow, shoulder, hip
            'right_shoulder': self.calculate_angle(keypoints[8], keypoints[6], keypoints[12]), # right elbow, shoulder, hip
            
            # Elbows
            'left_elbow': self.calculate_angle(keypoints[5], keypoints[7], keypoints[9]),    # left shoulder, elbow, wrist
            'right_elbow': self.calculate_angle(keypoints[6], keypoints[8], keypoints[10]),  # right shoulder, elbow, wrist
            
            # Hips
            'left_hip': self.calculate_angle(keypoints[5], keypoints[11], keypoints[13]),    # left shoulder, hip, knee
            'right_hip': self.calculate_angle(keypoints[6], keypoints[12], keypoints[14]),   # right shoulder, hip, knee
            
            # Knees
            'left_knee': self.calculate_angle(keypoints[11], keypoints[13], keypoints[15]),  # left hip, knee, ankle
            'right_knee': self.calculate_angle(keypoints[12], keypoints[14], keypoints[16]), # right hip, knee, ankle
        }
        
        return angles
