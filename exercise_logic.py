from pose_utils import PoseDetector

class ExerciseLogic:
    def __init__(self):
        """Initialize exercise logic with thresholds and state"""
        self.thresholds = {
            'tree_pose': {
                'knee_bend': 25,  # Even more forgiving knee bend angle
                'balance_duration': 1.5,  # Reduced hold time
                'hip_variance': 20,  # More forgiving hip alignment
                'confidence_threshold': 0.6  # Minimum confidence for keypoints
            },
            'squat': {
                'down': 100,  # More forgiving squat depth
                'up': 130,    # Even more forgiving top position
                'knee_alignment': 30,  # Maximum knee spread
                'confidence_threshold': 0.5
            },
            'pushup': {
                'down': 100,  # More forgiving bottom position
                'up': 150,    # More forgiving top position
                'hip_variance': 20,  # Allowed hip sag
                'confidence_threshold': 0.5
            },
            'jumping_jack': {
                'down': 30,   # More forgiving bottom position
                'up': 90,     # More forgiving top position
                'confidence_threshold': 0.5
            }
        }
        self.exercise_state = "start"  # Initial state
        self.rep_count = 0
        self.pose_start_time = None
        self.last_pose_end_time = None
        self.form_status = {"confidence": 0, "feedback": []}

    def detect_exercise(self, angles):
        """Returns True if the current form is correct for the given exercise"""
        right_knee = angles.get('right_knee')
        left_knee = angles.get('left_knee')
        right_hip = angles.get('right_hip')
        left_hip = angles.get('left_hip')
        
        if all(x is not None for x in [right_knee, left_knee, right_hip, left_hip]):
            knee_diff = abs(right_knee - left_knee)
            hip_diff = abs(right_hip - left_hip)
            
            # Check for tree pose form
            if knee_diff > self.thresholds['tree_pose']['knee_bend'] and hip_diff < self.thresholds['tree_pose']['hip_variance']:
                return True

        return False  # Form is not correct

    def check_form(self, exercise, angles):
        """Check form for the given exercise"""
        if exercise == 'squat':
            right_knee = angles.get('right_knee')
            left_knee = angles.get('left_knee')
            if right_knee is not None and left_knee is not None:
                knee_angle = (right_knee + left_knee) / 2
                if self.exercise_state == 'up':
                    # Check if we're in a squat position
                    return knee_angle <= self.thresholds['squat']['down']
                else:
                    # Check if we're back to standing
                    return knee_angle >= self.thresholds['squat']['up']
        
        elif exercise == 'pushup':
            right_elbow = angles.get('right_elbow')
            left_elbow = angles.get('left_elbow')
            if right_elbow is not None and left_elbow is not None:
                elbow_angle = (right_elbow + left_elbow) / 2
                if self.exercise_state == 'up':
                    return elbow_angle <= self.thresholds['pushup']['down']
                else:
                    return elbow_angle >= self.thresholds['pushup']['up']

        elif exercise == 'jumping_jack':
            right_shoulder = angles.get('right_shoulder')
            left_shoulder = angles.get('left_shoulder')
            if right_shoulder is not None and left_shoulder is not None:
                shoulder_angle = (right_shoulder + left_shoulder) / 2
                if self.exercise_state == 'up':
                    return shoulder_angle >= self.thresholds['jumping_jack']['up']
                else:
                    return shoulder_angle <= self.thresholds['jumping_jack']['down']

        return False

    def count_reps(self, exercise, angles):
        """Count reps based on state transitions with improved reliability"""
        from time import time
        current_time = time()
        
        if exercise == 'tree_pose':
            right_knee = angles.get('right_knee')
            left_knee = angles.get('left_knee')
            right_hip = angles.get('right_hip')
            left_hip = angles.get('left_hip')
            
            if all(x is not None for x in [right_knee, left_knee, right_hip, left_hip]):
                knee_diff = abs(right_knee - left_knee)
                hip_diff = abs(right_hip - left_hip)
                
                if (knee_diff > self.thresholds['tree_pose']['knee_bend'] and 
                    hip_diff < self.thresholds['tree_pose']['hip_variance']):
                    
                    if self.exercise_state == 'start':
                        self.pose_start_time = current_time
                        self.exercise_state = 'holding'
                        print("Started holding tree pose")  # Debug print
                    elif self.exercise_state == 'holding':
                        if current_time - self.pose_start_time >= self.thresholds['tree_pose']['balance_duration']:
                            if (self.last_pose_end_time is None or 
                                current_time - self.last_pose_end_time > 1.0):
                                self.rep_count += 1
                                self.last_pose_end_time = current_time
                                print(f"Tree pose held! Count: {self.rep_count}")  # Debug print
                                return True
                else:
                    if self.exercise_state == 'holding':
                        print("Lost tree pose position")  # Debug print
                    self.exercise_state = 'start'
                    self.pose_start_time = None
                    
        elif exercise == 'squat':
            right_knee = angles.get('right_knee')
            left_knee = angles.get('left_knee')
            if right_knee is not None and left_knee is not None:
                knee_angle = (right_knee + left_knee) / 2
                print(f"Current knee angle: {knee_angle:.1f}, State: {self.exercise_state}")  # Debug print

                # Going down
                if self.exercise_state == 'up' and knee_angle < self.thresholds['squat']['down']:
                    if self.pose_start_time is None:
                        self.pose_start_time = current_time
                        print("Starting squat down")  # Debug print
                    elif current_time - self.pose_start_time >= 0.2:
                        self.exercise_state = 'down'
                        self.pose_start_time = None
                        print("Reached squat bottom")  # Debug print

                # Coming up
                elif self.exercise_state == 'down' and knee_angle > self.thresholds['squat']['up']:
                    if self.pose_start_time is None:
                        self.pose_start_time = current_time
                        print("Starting squat up")  # Debug print
                    elif current_time - self.pose_start_time >= 0.2:
                        if self.last_pose_end_time is None or current_time - self.last_pose_end_time > 0.8:
                            self.exercise_state = 'up'
                            self.rep_count += 1
                            self.last_pose_end_time = current_time
                            self.pose_start_time = None
                            print(f"Squat complete! Count: {self.rep_count}")  # Debug print
                            return True

                # Reset if stuck in a state too long
                if self.pose_start_time and current_time - self.pose_start_time > 2.0:
                    print("Reset due to timeout")  # Debug print
                    self.pose_start_time = None
                    self.exercise_state = 'up'
        
        elif exercise == 'pushup':
            right_elbow = angles.get('right_elbow')
            left_elbow = angles.get('left_elbow')
            if right_elbow is not None and left_elbow is not None:
                elbow_angle = (right_elbow + left_elbow) / 2
                if self.exercise_state == 'up' and elbow_angle < self.thresholds['pushup']['down']:
                    self.exercise_state = 'down'
                elif self.exercise_state == 'down' and elbow_angle > self.thresholds['pushup']['up']:
                    self.exercise_state = 'up'
                    self.rep_count += 1
                    return True # Rep completed

        elif exercise == 'jumping_jack':
            right_shoulder = angles.get('right_shoulder')
            left_shoulder = angles.get('left_shoulder')
            if right_shoulder is not None and left_shoulder is not None:
                shoulder_angle = (right_shoulder + left_shoulder) / 2
                if self.exercise_state == 'up' and shoulder_angle > self.thresholds['jumping_jack']['up']:
                    self.exercise_state = 'down' # Arms are up
                elif self.exercise_state == 'down' and shoulder_angle < self.thresholds['jumping_jack']['down']:
                    self.exercise_state = 'up' # Arms are down
                    self.rep_count += 1
                    return True # Rep completed

        return False

    def get_form_feedback(self, exercise, angles):
        """Get confidence score and feedback for current form"""
        confidence = 0
        feedback = []
        
        if exercise == 'tree_pose':
            right_knee = angles.get('right_knee')
            left_knee = angles.get('left_knee')
            right_hip = angles.get('right_hip')
            left_hip = angles.get('left_hip')
            
            if all(x is not None for x in [right_knee, left_knee, right_hip, left_hip]):
                knee_diff = abs(right_knee - left_knee)
                hip_diff = abs(right_hip - left_hip)
                
                # Calculate confidence score
                knee_score = min(100, (knee_diff / self.thresholds['tree_pose']['knee_bend']) * 100)
                hip_score = max(0, 100 - (hip_diff / self.thresholds['tree_pose']['hip_variance']) * 100)
                confidence = int((knee_score + hip_score) / 2)
                
                # Generate feedback
                if knee_diff < self.thresholds['tree_pose']['knee_bend']:
                    feedback.append("Raise your knee higher")
                if hip_diff > self.thresholds['tree_pose']['hip_variance']:
                    feedback.append("Keep your hips level")
                
        elif exercise == 'squat':
            right_knee = angles.get('right_knee')
            left_knee = angles.get('left_knee')
            
            if right_knee is not None and left_knee is not None:
                knee_angle = (right_knee + left_knee) / 2
                knee_diff = abs(right_knee - left_knee)
                
                # Calculate confidence score
                depth_score = min(100, (knee_angle / self.thresholds['squat']['down']) * 100)
                alignment_score = max(0, 100 - (knee_diff / self.thresholds['squat']['knee_alignment']) * 100)
                confidence = int((depth_score + alignment_score) / 2)
                
                # Generate feedback
                if knee_angle > self.thresholds['squat']['down']:
                    feedback.append("Squat deeper")
                if knee_diff > self.thresholds['squat']['knee_alignment']:
                    feedback.append("Keep knees aligned")
                
        elif exercise == 'pushup':
            right_elbow = angles.get('right_elbow')
            left_elbow = angles.get('left_elbow')
            right_hip = angles.get('right_hip')
            left_hip = angles.get('left_hip')
            
            if all(x is not None for x in [right_elbow, left_elbow, right_hip, left_hip]):
                elbow_angle = (right_elbow + left_elbow) / 2
                hip_diff = abs(right_hip - left_hip)
                
                # Calculate confidence score
                depth_score = min(100, (elbow_angle / self.thresholds['pushup']['down']) * 100)
                hip_score = max(0, 100 - (hip_diff / self.thresholds['pushup']['hip_variance']) * 100)
                confidence = int((depth_score + hip_score) / 2)
                
                # Generate feedback
                if elbow_angle > self.thresholds['pushup']['down']:
                    feedback.append("Lower your chest")
                if hip_diff > self.thresholds['pushup']['hip_variance']:
                    feedback.append("Keep your body straight")
                
        elif exercise == 'jumping_jack':
            right_shoulder = angles.get('right_shoulder')
            left_shoulder = angles.get('left_shoulder')
            
            if right_shoulder is not None and left_shoulder is not None:
                shoulder_angle = (right_shoulder + left_shoulder) / 2
                
                # Calculate confidence score
                confidence = int(min(100, (shoulder_angle / self.thresholds['jumping_jack']['up']) * 100))
                
                # Generate feedback
                if shoulder_angle < self.thresholds['jumping_jack']['up']:
                    feedback.append("Raise arms higher")
        
        # Default feedback if none generated
        if not feedback:
            feedback = ["Form looks good!"]
            
        return confidence, feedback

    def reset(self):
        """Reset rep count and state"""
        self.rep_count = 0
        self.exercise_state = "start"
        self.pose_start_time = None
        self.last_pose_end_time = None
        self.form_status = {"confidence": 0, "feedback": []}
