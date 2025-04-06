# Import all required libraries
import cv2
import sounddevice as sd
import numpy as np
import queue
import time
import mediapipe as mp
import whisper
import torch
import warnings
from scipy.io.wavfile import write
from datetime import datetime
from typing import List, Tuple, Dict
import csv
import subprocess,os

# Global Audio Recorder Class
def extract_audio(video_file, output_audio="output.wav"):
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file '{video_file}' not found.")

    command = [
        "ffmpeg", "-i", video_file, "-q:a", "0", "-map", "a", output_audio, "-y"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if not os.path.exists(output_audio):  # Check if output.wav exists
        raise RuntimeError(f"FFmpeg failed to extract audio. Error: {result.stderr.decode()}")

    print("Audio extracted successfully.")
    return output_audio

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result['text']


class AssessmentMonitor:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize detectors
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        # Drawing specifications
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=(128, 0, 128),
            thickness=2,
            circle_radius=1
        )
        
        # Eye and Iris tracking indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Suspicious movement counters
        self.suspicious_head_movements = 0
        self.suspicious_eye_movements = 0
        
        # Structured activity log
        self.activity_log = []
        
        # Previous states
        self.prev_head_direction = "Forward"
        self.prev_gaze_direction = "Looking Forward"
        
        # Movement thresholds
        self.head_direction_change_timeout = 1.5   # seconds
        self.eye_direction_change_timeout = 0.4    # seconds
        self.last_head_change_time = time.time()
        self.last_eye_change_time = time.time()
        
        # Timestamp reference
        self.start_time = time.time()
        
        # Frame counter for logging
        self.frame_count = 0
        self.log_interval = 15  # Log every 15 frames

    # During processing of each frame
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate eye aspect ratio for blinking/gaze detection"""
        points = []
        for idx in eye_indices:
            point = landmarks.landmark[idx]
            points.append([point.x, point.y])
        points = np.array(points)
        horizontal = np.linalg.norm(points[0] - points[8])
        vertical1 = np.linalg.norm(points[2] - points[10])
        vertical2 = np.linalg.norm(points[3] - points[9])
        return (vertical1 + vertical2) / (2.0 * horizontal)
    
    def detect_gaze_direction(self, ear_left, ear_right):
        """Determine gaze direction based on eye aspect ratios"""
        if ear_left < 0.2 and ear_right < 0.2:
            return "Eyes Closed"
        elif abs(ear_left - ear_right) > 0.1:
            return "Looking Sideways"
        else:
            return "Looking Forward"
    
    def detect_eye_direction(self, pupil_x, pupil_y, eye_points):
        """Determine eye gaze direction based on pupil position and eye landmarks"""
        x, y, w, h = cv2.boundingRect(eye_points)
        eye_center_x = x + w // 2
        eye_center_y = y + h // 2
        displacement_x = pupil_x - eye_center_x
        displacement_y = pupil_y - eye_center_y
        
        normalized_x = displacement_x / (w / 2)
        normalized_y = displacement_y / (h / 2)
        
        h_threshold = 0.35
        v_threshold = 0.4
        
        h_dir = "Right" if normalized_x > h_threshold else "Left" if normalized_x < -h_threshold else "Center"
        v_dir = "Down" if normalized_y > v_threshold else "Up" if normalized_y < -v_threshold else "Center"
        
        directions = []
        if h_dir != "Center":
            directions.append(h_dir)
        if v_dir != "Center":
            directions.append(v_dir)
        return " ".join(directions) if directions else "Center"
    
    def detect_head_pose(self, face_landmarks, image_shape):
        """Calculate head pose angles and direction"""
        img_h, img_w = image_shape[:2]
        face_2d = []
        face_3d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in [33, 263, 1, 61, 291, 199]:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h/2],
            [0, focal_length, img_w/2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        
        if y < -5:   direction = "Looking Left"
        elif y > 5:  direction = "Looking Right"
        elif x < -4: direction = "Looking Down"
        elif x > 6:  direction = "Looking Up"
        else:         direction = "Forward"
        
        return x, y, z, direction
    
    def is_suspicious_head_movement(self, current_direction):
        """Detect rapid head movements indicating suspicious behavior"""
        current_time = time.time()
        
        if (current_direction != self.prev_head_direction and
            current_time - self.last_head_change_time < self.head_direction_change_timeout):
            self.last_head_change_time = current_time
            self.prev_head_direction = current_direction
            return True
        
        if current_direction != self.prev_head_direction:
            self.last_head_change_time = current_time
            self.prev_head_direction = current_direction
            return True
        
        return False
    
    def is_suspicious_eye_movement(self, current_gaze, left_eye_dir, right_eye_dir):
        """Detect suspicious eye movements (rapid shifts, mismatched directions)"""
        current_time = time.time()
        
        if (current_gaze != self.prev_gaze_direction and
            current_time - self.last_eye_change_time < self.eye_direction_change_timeout):
            self.last_eye_change_time = current_time
            self.prev_gaze_direction = current_gaze
            return True
        
        if (left_eye_dir != right_eye_dir and
            "Center" not in left_eye_dir and
            "Center" not in right_eye_dir and
             (("Left" in left_eye_dir and "Right" in right_eye_dir) or
             ("Right" in left_eye_dir and "Left" in right_eye_dir) or
             ("Up" in left_eye_dir and "Down" in right_eye_dir) or
             ("Down" in left_eye_dir and "Up" in right_eye_dir))):
            return True
        
        if current_gaze != self.prev_gaze_direction:
            self.last_eye_change_time = current_time
            self.prev_gaze_direction = current_gaze
        
        return False
    
    def log_movement_status(self, suspicious_head, suspicious_eye, additional_info=""):
        """Log suspicious movements in the activity log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.activity_log.append({
            "timestamp": timestamp,
            "head_movement": "True" if suspicious_head else "False",
            "eye_movement": "True" if suspicious_eye else "False",
            "additional_info": additional_info
        })

    def process_frame(self, frame):
        """Process a single frame and return analysis results"""
        if frame is None:
            return None, None
        
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Detect faces
        face_detection_results = self.face_detection.process(image)
        face_mesh_results = self.face_mesh.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        self.frame_count += 1
        should_log = (self.frame_count % self.log_interval == 0)
        
        suspicious_head = False
        suspicious_eye = False
        additional_info = ""
        
        results = {
            'num_faces': 0,
            'face_visible': False,
            'head_pose': None,
            'gaze_direction': None,
            'left_eye_direction': None,
            'right_eye_direction': None,
            'iris': None,
            'warnings': [],
            'suspicious_head_count': self.suspicious_head_movements,
            'suspicious_eye_count': self.suspicious_eye_movements
        }
        
        # Check for faces
        if face_detection_results.detections:
            results['num_faces'] = len(face_detection_results.detections)
            results['face_visible'] = True
            if results['num_faces'] > 1:
                results['warnings'].append("Multiple faces detected")
                additional_info += f"Multiple faces detected ({results['num_faces']}); "
        
        if face_mesh_results.multi_face_landmarks:
            landmarks = face_mesh_results.multi_face_landmarks[0]
            
            # Calculate head pose
            img_h, img_w, _ = image.shape
            x, y, z, head_dir = self.detect_head_pose(landmarks, image.shape)
            results['head_pose'] = {
                'x': np.round(x, 2),
                'y': np.round(y, 2),
                'z': np.round(z, 2),
                'direction': head_dir
            }
            
            # Check for suspicious head movement
            if self.is_suspicious_head_movement(head_dir):
                self.suspicious_head_movements += 1
                results['suspicious_head_count'] = self.suspicious_head_movements
                suspicious_head = True
                additional_info += f"Head: {head_dir}, Angles: x={x:.1f}, y={y:.1f}, z={z:.1f}; "
            
            # Calculate eye aspect ratios
            ear_left = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE)
            ear_right = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE)
            results['gaze_direction'] = self.detect_gaze_direction(ear_left, ear_right)
            
            # Calculate iris positions
            mesh_points = np.array([(int(p.x * img_w), int(p.y * img_h)) for p in landmarks.landmark])
            left_iris = mesh_points[self.LEFT_IRIS]
            right_iris = mesh_points[self.RIGHT_IRIS]
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(left_iris)
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(right_iris)
            results['iris'] = {
                'left_center': (int(l_cx), int(l_cy)),
                'left_radius': int(l_radius),
                'right_center': (int(r_cx), int(r_cy)),
                'right_radius': int(r_radius)
            }
            
            # Calculate eye directions
            left_eye_points = mesh_points[self.LEFT_EYE]
            right_eye_points = mesh_points[self.RIGHT_EYE]
            left_eye_dir = self.detect_eye_direction(
                results['iris']['left_center'][0],
                results['iris']['left_center'][1],
                left_eye_points
            )
            right_eye_dir = self.detect_eye_direction(
                results['iris']['right_center'][0],
                results['iris']['right_center'][1],
                right_eye_points
            )
            results['left_eye_direction'] = left_eye_dir
            results['right_eye_direction'] = right_eye_dir
            
            # Check for suspicious eye movements
            if self.is_suspicious_eye_movement(results['gaze_direction'], left_eye_dir, right_eye_dir):
                self.suspicious_eye_movements += 1
                results['suspicious_eye_count'] = self.suspicious_eye_movements
                suspicious_eye = True
                additional_info += f"Gaze: {results['gaze_direction']}, Left: {left_eye_dir}, Right: {right_eye_dir}"
            
            # Log activity
            if should_log or suspicious_head or suspicious_eye:
                self.log_movement_status(suspicious_head, suspicious_eye, additional_info.strip("; "))
            
            # Draw landmarks and annotations
            self.draw_annotations(image, landmarks, results)
        
        return image, results

    def draw_annotations(self, image, landmarks, results):
        """Draw visual annotations on the image"""
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.drawing_spec
        )
        
        if results['iris']:
            cv2.circle(image, results['iris']['left_center'], results['iris']['left_radius'], (255, 0, 0), 2)
            cv2.circle(image, results['iris']['right_center'], results['iris']['right_radius'], (255, 0, 0), 2)
        
        # Display metrics
        cv2.putText(image, f"Faces: {results['num_faces']}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        if results['head_pose']:
            cv2.putText(image, f"Head: {results['head_pose']['direction']}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(image, f"Overall Gaze: {results['gaze_direction']}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(image, f"Left Eye: {results.get('left_eye_direction', 'N/A')}", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(image, f"Right Eye: {results.get('right_eye_direction', 'N/A')}", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Display counters
        cv2.putText(image, f"Suspicious Head Movements: {results['suspicious_head_count']}", 
                    (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(image, f"Suspicious Eye Movements: {results['suspicious_eye_count']}", 
                    (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Display warnings
        y_pos = 330
        for warning in results['warnings']:
            cv2.putText(image, f"Warning: {warning}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            y_pos += 40

    def save_activity_log_to_csv(self, filename="movement_log.csv"):
        """Save activity log to a CSV file"""
        if not self.activity_log:
            print("No activities logged.")
            return
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ["timestamp", "head_movement", "eye_movement", "additional_info"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.activity_log)
        
        print(f"\nActivity log saved to {filename}")
        print("Log Format Sample:")
        print("-" * 50)
        for entry in self.activity_log[:5]:
            print(f"{entry['timestamp']},{entry['head_movement']},{entry['eye_movement']}")
        if len(self.activity_log) > 5:
            print("...")
        
        total_entries = len(self.activity_log)
        suspicious_head_entries = sum(1 for entry in self.activity_log if entry["head_movement"] == "True")
        suspicious_eye_entries = sum(1 for entry in self.activity_log if entry["eye_movement"] == "True")
        print(f"\nTotal log entries: {total_entries}")
        print(f"Entries with suspicious head movement: {suspicious_head_entries}")
        print(f"Entries with suspicious eye movement: {suspicious_eye_entries}")

    def calculate_final_score(self, base_score=100):
        head_movement_deduction = (self.suspicious_head_movements // 1) * 0.1
        eye_movement_deduction = (self.suspicious_eye_movements // 2) * 0.1
        
        # Calculate final score
        final_score = base_score - head_movement_deduction - eye_movement_deduction
        
        # Ensure score doesn't go below 0
        final_score = max(0, final_score)
        
        # Create deduction details
        deduction_details = {
            "head_movements_detected": self.suspicious_head_movements,
            "head_movement_deduction": head_movement_deduction,
            "eye_movements_detected": self.suspicious_eye_movements,
            "eye_movement_deduction": eye_movement_deduction,
            "total_deduction": head_movement_deduction + eye_movement_deduction,
            "original_score": base_score,
            "final_score": final_score
        }
        
        return final_score, deduction_details
    
def main():
    # Initialize components
    video_file = "input.mp4"
    cap = cv2.VideoCapture(video_file)  # Use default camera (0)
    monitor = AssessmentMonitor()
    
    # Configure video display
    cv2.namedWindow("Assessment Monitor", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Assessment Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Process the frame
            start_time = time.time()
            processed_frame, results = monitor.process_frame(frame)
            
            # Calculate FPS
            fps = 1 / (time.time() - start_time)
            cv2.putText(processed_frame, f'FPS: {int(fps)}', (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            cv2.imshow('Assessment Monitor', processed_frame)

            # Display current score
            final_score, _ = monitor.calculate_final_score()
            cv2.putText(processed_frame, f'Current Score: {final_score}', (20, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            cv2.imshow('Assessment Monitor', processed_frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit loop on 'q' press
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        final_score, details = monitor.calculate_final_score()

        print("\n" + "="*50)
        print("ASSESSMENT SCORE REPORT")
        print("="*50)
        print(f"Final Score: {details['final_score']}/100")
        print("-"*50)
        print("Deduction Details:")
        print(f"Total Head Movements Detected: {details['head_movements_detected']}")
        print(f"Head Movement Deduction: -{details['head_movement_deduction']} points")
        print(f"Total Eye Movements Detected: {details['eye_movements_detected']}")
        print(f"Eye Movement Deduction: -{details['eye_movement_deduction']} points")
        print(f"Total Deduction: -{details['total_deduction']} points")
        print("="*50)


        audio_file = extract_audio(video_file)
        transcription = transcribe_audio(audio_file)
        print("\nTranscribed Text:\n", transcription)

# Entry point
if __name__ == "__main__":
    main()