import cv2
import numpy as np
import tensorflow as tf

class PoseTFLiteDetector:
    def __init__(self, model_path='model.tflite'):
        """Initialize the TF Lite pose detector"""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        img = cv2.resize(image, (self.input_shape[1], self.input_shape[2]))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
        
    def detect_pose(self, frame):
        """Detect poses in the frame"""
        # Preprocess the image
        input_data = self.preprocess_image(frame)
        
        # Set the input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get outputs
        keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return self._process_keypoints(keypoints[0])
    
    def _process_keypoints(self, keypoints):
        """Process keypoints from model output"""
        processed_keypoints = []
        if len(keypoints) > 0:
            # Convert keypoints to the format expected by the application
            keypoints_array = np.array([[k[0], k[1], k[2]] for k in keypoints])
            processed_keypoints.append(keypoints_array)
        return processed_keypoints
        
    def draw_skeleton(self, frame, keypoints):
        """Draw the skeleton on the frame"""
        if not keypoints:
            return
            
        # Define the pairs of keypoints that form the skeleton
        skeleton = [
            (5, 7), (7, 9),     # Left arm
            (6, 8), (8, 10),    # Right arm
            (5, 6),             # Shoulders
            (5, 11), (6, 12),   # Spine
            (11, 12),           # Hips
            (11, 13), (13, 15), # Left leg
            (12, 14), (14, 16)  # Right leg
        ]
        
        keypoints = keypoints[0]  # Get first person's keypoints
        
        # Draw the skeleton
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

def test_tflite_model():
    """Test the TF Lite model with webcam"""
    detector = PoseTFLiteDetector()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect poses
        keypoints = detector.detect_pose(frame)
        
        # Draw skeleton if poses detected
        if keypoints:
            detector.draw_skeleton(frame, keypoints)
        
        # Display the frame
        cv2.imshow('TF Lite Pose Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_tflite_model()