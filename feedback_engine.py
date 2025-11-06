
import pyttsx3
import time
from threading import Thread, Lock
from queue import Queue
import random

class FeedbackEngine:
    def __init__(self, cooldown=2):
        """Initialize the feedback engine with text-to-speech and a processing queue"""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.feedback_queue = Queue()
        self.lock = Lock()
        self.running = True
        self.last_feedback_time = 0
        self.cooldown = cooldown
        
        self.corrections = {
            'squat': ["Go lower!", "Keep your chest up!", "Knees over toes!"],
            'pushup': ["Lower your chest!", "Keep your back straight!", "Go all the way down!"],
            'plank': ["Keep your back flat!", "Don't drop your hips!", "Engage your core!"],
            'jumping_jack': ["Jump wider!", "Raise arms higher!", "More energy!"],
            'lunge': ["Lower your back knee!", "Keep your front knee over your ankle!", "Step further!"]
        }
        
        self.praises = {
            'squat': ["Perfect squat!", "Great depth!", "Excellent form!"],
            'pushup': ["Strong pushup!", "Great form!", "Keep it up!"],
            'plank': ["Solid plank!", "Rock solid!", "Amazing hold!"],
            'jumping_jack': ["Great rhythm!", "Full energy!", "Looking good!"],
            'lunge': ["Perfect lunge!", "Great balance!", "Strong stance!"]
        }

        self.feedback_thread = Thread(target=self._process_queue)
        self.feedback_thread.daemon = True
        self.feedback_thread.start()

    def _process_queue(self):
        """Process the feedback queue in a separate thread"""
        while self.running:
            if not self.feedback_queue.empty():
                with self.lock:
                    current_time = time.time()
                    if current_time - self.last_feedback_time >= self.cooldown:
                        feedback = self.feedback_queue.get()
                        self.engine.say(feedback)
                        self.engine.runAndWait()
                        self.last_feedback_time = current_time
            time.sleep(0.1)

    def queue_feedback(self, feedback):
        """Add feedback to the queue"""
        self.feedback_queue.put(feedback)

    def get_correction(self, exercise):
        """Get a random correction for a given exercise"""
        if exercise in self.corrections:
            return random.choice(self.corrections[exercise])
        return "Check your form."

    def get_praise(self, exercise):
        """Get a random praise for a given exercise"""
        if exercise in self.praises:
            return random.choice(self.praises[exercise])
        return "Good job!"

    def stop(self):
        """Stop the feedback engine thread"""
        self.running = False
        self.feedback_thread.join()
