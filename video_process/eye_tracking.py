import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# Configuration parameters
MAX_HEAD_ANGLE = 20  # Increased from 15 for more flexibility
MIN_EYE_OPENNESS = 0.25  # Adjusted eye openness threshold
GAZE_SMOOTHING_WINDOW = 5  # Number of frames for smoothing gaze direction
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames for stable eye state
GAZE_CONSEC_FRAMES = 5  # Number of consecutive frames for stable gaze direction

# Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks (using more points for better accuracy)
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left Mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
])

class EyeTracker:
    def __init__(self):
        self.size = (640, 480)
        self.focal_length = self.size[1]
        self.center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype="double")
        self.dist_coeffs = np.zeros((4, 1))
        
        # State tracking
        self.eye_state_history = deque(maxlen=EYE_AR_CONSEC_FRAMES)
        self.gaze_history = deque(maxlen=GAZE_CONSEC_FRAMES)
        self.last_eye_contact_time = time.time()
        
    def get_head_pose(self, landmarks):
        try:
            image_points = np.array([
                landmarks[1],     # Nose tip
                landmarks[152],   # Chin
                landmarks[263],   # Left eye left corner
                landmarks[33],    # Right eye right corner
                landmarks[287],   # Left Mouth corner
                landmarks[57]     # Right mouth corner
            ], dtype="double")

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, self.camera_matrix, 
                self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                return angles  # pitch, yaw, roll
        except:
            pass
        return [0, 0, 0]  # Return neutral angles if detection fails
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        # Horizontal eye landmarks
        p1 = eye_landmarks[0]
        p2 = eye_landmarks[8]
        
        # Vertical eye landmarks (top and bottom)
        p3 = eye_landmarks[12]
        p4 = eye_landmarks[4]
        p5 = eye_landmarks[11]
        p6 = eye_landmarks[5]
        
        # Calculate distances
        horiz_dist = np.linalg.norm(p2 - p1)
        vert_dist1 = np.linalg.norm(p4 - p3)
        vert_dist2 = np.linalg.norm(p6 - p5)
        
        # Use average of vertical distances
        avg_vert_dist = (vert_dist1 + vert_dist2) / 2
        
        # Avoid division by zero
        if horiz_dist == 0:
            return 0.0
            
        ear = avg_vert_dist / horiz_dist
        return ear
    
    def get_eye_landmarks(self, landmarks, indices):
        return np.array([landmarks[i] for i in indices], dtype="double")
    
    def is_looking_at_camera(self, pitch, yaw, roll):
        # More flexible thresholds for natural head movements
        return (abs(yaw) < MAX_HEAD_ANGLE and 
                abs(pitch) < MAX_HEAD_ANGLE * 1.5 and 
                abs(roll) < MAX_HEAD_ANGLE * 2)
    
    def analyze_frame(self, frame):
        frame = cv2.resize(frame, self.size)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [(int(lm.x * self.size[0]), int(lm.y * self.size[1])) 
                            for lm in face_landmarks.landmark]
                
                # Get head pose
                pitch, yaw, roll = self.get_head_pose(landmarks)
                
                # Get eye landmarks and calculate EAR
                left_eye = self.get_eye_landmarks(landmarks, LEFT_EYE_INDICES)
                right_eye = self.get_eye_landmarks(landmarks, RIGHT_EYE_INDICES)
                left_ear = self.calculate_eye_aspect_ratio(left_eye)
                right_ear = self.calculate_eye_aspect_ratio(right_eye)
                
                # Determine eye state
                eyes_open = (left_ear > MIN_EYE_OPENNESS and 
                           right_ear > MIN_EYE_OPENNESS)
                self.eye_state_history.append(eyes_open)
                
                # Stable eye state (open/closed) over several frames
                stable_eyes_open = (sum(self.eye_state_history) == 
                                  len(self.eye_state_history))
                
                # Gaze direction
                looking_at_camera = self.is_looking_at_camera(pitch, yaw, roll)
                self.gaze_history.append(looking_at_camera)
                
                # Stable gaze direction over several frames
                stable_gaze = (sum(self.gaze_history) == 
                             len(self.gaze_history))
                
                # Eye contact requires stable eyes open and stable gaze
                eye_contact = stable_eyes_open and stable_gaze
                
                # Update last eye contact time
                if eye_contact:
                    self.last_eye_contact_time = time.time()
                
                return eye_contact, pitch, yaw, roll, left_ear, right_ear
        
        return False, 0, 0, 0, 0, 0

def simulate_eye_tracking_score(video_path):
    print(f"Analyzing eye contact for video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    tracker = EyeTracker()
    
    frame_count = 0
    eye_contact_frames = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        eye_contact, _, _, _, _, _ = tracker.analyze_frame(frame)
        frame_count += 1
        
        if eye_contact:
            eye_contact_frames += 1
    
    cap.release()
    
    if frame_count == 0:
        return 0.0
    
    score = (eye_contact_frames / frame_count) * 100
    print(f"Eye contact score: {score:.2f}% for video {video_path}")
    return round(score, 2)