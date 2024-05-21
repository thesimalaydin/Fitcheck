import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Açı hesaplama fonksiyonu
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = cv2.VideoCapture(video_source)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.current_arm = 'LEFT'  # Default arm selection
        self.counter = 0
        self.stage = None
        
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        # ComboBox for selecting the arm
        self.arm_selection = ttk.Combobox(window, values=['LEFT', 'RIGHT'], state="readonly")
        self.arm_selection.pack()
        self.arm_selection.bind('<<ComboboxSelected>>', self.select_arm)
        self.arm_selection.current(0)  # default to left arm
        
        self.update()
        
    def select_arm(self, event):
        self.current_arm = self.arm_selection.get()
        self.counter = 0  # reset counter when arm selection changes
        self.stage = None  # reset stage
    
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            self.draw_landmarks(image, results)
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(image))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.window.after(10, self.update)

    def draw_landmarks(self, image, results):
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = landmarks[getattr(mp_pose.PoseLandmark, f'{self.current_arm}_SHOULDER').value]
            elbow = landmarks[getattr(mp_pose.PoseLandmark, f'{self.current_arm}_ELBOW').value]
            wrist = landmarks[getattr(mp_pose.PoseLandmark, f'{self.current_arm}_WRIST').value]
            
            # Calculate angle
            angle = calculate_angle(
                [shoulder.x, shoulder.y], 
                [elbow.x, elbow.y], 
                [wrist.x, wrist.y])
            
            # Curl counter logic
            if angle > 160:
                self.stage = "down"
            if angle < 30 and self.stage == 'down':
                self.stage = "up"
                self.counter += 1
            
            # Display counter
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, self.stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

if __name__ == '__main__':
    root = tk.Tk()
    App(root, "Pose Estimation Exercise Tracker")
    root.mainloop()
