import cv2
import mediapipe as mp
import numpy as np

class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def count_fingers(self, hand_landmarks):
        """Count the number of extended fingers"""
        # Tip ids for thumb, index, middle, ring, pinky
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        
        # Get landmark points
        landmarks = hand_landmarks.landmark
        
        # Thumb (different logic - check if tip is to the right/left of IP joint)
        if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers (check if tip is above PIP joint)
        for id in range(1, 5):
            if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
    def recognize_gesture(self, hand_landmarks):
        """Recognize specific hand gestures"""
        fingers = self.count_fingers(hand_landmarks)
        total_fingers = sum(fingers)
        
        # Gesture recognition logic
        if total_fingers == 0:
            return "Fist ✊"
        elif total_fingers == 5:
            return "Open Palm ✋"
        elif fingers == [1, 0, 0, 0, 0]:
            return "Thumbs Up 👍"
        elif fingers == [0, 1, 1, 0, 0]:
            return "Peace ✌️"
        elif fingers == [0, 1, 0, 0, 0]:
            return "Pointing ☝️"
        elif fingers == [0, 1, 1, 1, 0]:
            return "Three Fingers"
        elif fingers == [0, 1, 1, 1, 1]:
            return "Four Fingers"
        elif fingers == [1, 1, 0, 0, 1]:
            return "Rock 🤘"
        else:
            return f"Fingers: {total_fingers}"
    
    def run(self, camera_source=0):
        """Main function to run gesture recognition
        
        Args:
            camera_source: 0 for laptop webcam, or IP address for phone camera
                          Example: "http://192.168.1.100:8080/video"
        """
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            print("For phone camera, make sure:")
            print("1. Phone and computer are on same WiFi")
            print("2. IP Webcam app is running on phone")
            print("3. URL is correct (e.g., http://192.168.1.100:8080/video)")
            return
        
        print("Hand Gesture Recognition Started!")
        print(f"Camera source: {camera_source}")
        print("Press 'q' to quit")
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks and recognize gesture
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Recognize gesture
                    gesture = self.recognize_gesture(hand_landmarks)
                    
                    # Display gesture on screen
                    cv2.putText(
                        frame, 
                        f"Gesture: {gesture}", 
                        (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
            
            # Display instructions
            cv2.putText(
                frame, 
                "Press 'q' to quit", 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
            # Show the frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    recognizer = HandGestureRecognizer()
    
    # Check if IP address provided as command line argument
    if len(sys.argv) > 1:
        camera_source = sys.argv[1]
        print(f"Using camera source: {camera_source}")
    else:
        camera_source = 0
        print("Using default webcam. To use phone camera, run:")
        print("python hand_gesture_recognition.py http://YOUR_PHONE_IP:8080/video")
    
    recognizer.run(camera_source)
