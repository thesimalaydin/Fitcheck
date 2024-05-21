import sys
import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer, QTime, Qt
from PyQt5.QtGui import QImage, QPixmap

from utils import score_table
from types_of_exercise import TypeOfExercise

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class ExerciseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FitCheck")
        self.setGeometry(100, 100, 1280, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #87CEFA;  /* Light blue background */
            }
            QLabel {
                border: 2px solid black;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #1E90FF;  /* Dodger blue button */
                color: white;
                font-size: 18px;
                padding: 10px;
                border-radius: 5px;
                border: 2px solid #1E90FF;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #1C86EE;
            }
        """)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.video_layout = QHBoxLayout()
        self.layout.addLayout(self.video_layout)
        
        self.image_label = QLabel(self)
        self.video_layout.addWidget(self.image_label)
        
        self.video_label = QLabel(self)
        self.video_layout.addWidget(self.video_label)
        
        self.start_button = QPushButton("Egzersize Başla", self)
        self.layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_exercise)
        
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        self.video_cap = None
        self.counter = 0
        self.status = True
        self.exercise_type = "squat"  # Change to desired exercise type
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.countdown_time = QTime(0, 0, 5)  # 5 seconds countdown
        self.is_counting_down = False
        
        # Hide the webcam feed initially
        self.image_label.hide()
        
    def start_exercise(self):
        self.video_cap = cv2.VideoCapture('SquatAvatar.mp4')
        
        self.counter = 0
        self.status = True
        self.countdown_time = QTime(0, 0, 5)
        self.is_counting_down = True
        self.countdown_timer.start(1000)
        self.video_timer.start(30)
        
        # Show the webcam feed
        self.image_label.show()
        self.frame_timer.start(30)
        
    def update_countdown(self):
        self.countdown_time = self.countdown_time.addSecs(-1)
        if self.countdown_time == QTime(0, 0, 0):
            self.countdown_timer.stop()
            self.is_counting_down = False
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame)
        
        if results.pose_landmarks and not self.is_counting_down:
            landmarks = results.pose_landmarks.landmark
            self.counter, self.status = TypeOfExercise(landmarks).calculate_exercise(
                self.exercise_type, self.counter, self.status
            )
        
            frame = score_table(self.exercise_type, frame, self.counter, self.status)
        
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2),
            )
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if self.is_counting_down:
            frame = self.overlay_countdown(frame)
        
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(image))
        
    def update_video_frame(self):
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # Resize the video frame to match the webcam frame size
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(image))
        
    def overlay_countdown(self, frame):
        countdown_text = self.countdown_time.toString("ss")
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(countdown_text, font, 2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        
        # Draw a semi-transparent rectangle as the background
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x - 20, text_y - text_size[1] - 20), (text_x + text_size[0] + 20, text_y + 20), (0, 0, 0), -1)
        alpha = 0.6  # Transparency factor.
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw the countdown text
        cv2.putText(frame, countdown_text, (text_x, text_y), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
        return frame
        
    def closeEvent(self, event):
        self.frame_timer.stop()
        self.video_timer.stop()
        self.countdown_timer.stop()
        if self.cap is not None:
            self.cap.release()
        if self.video_cap is not None:
            self.video_cap.release()
        self.pose.close()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExerciseApp()
    window.show()
    sys.exit(app.exec_())
