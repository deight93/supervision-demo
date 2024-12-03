import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import supervision as sv
from ultralytics import YOLO

# GUI 클래스
class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 기본 설정
        self.setWindowTitle("Supervision Demo with Roboflow")
        self.setGeometry(100, 100, 800, 600)

        # UI 요소
        self.label = QLabel("Output", self)
        self.label.setScaledContents(True)

        self.model_select = QComboBox(self)
        self.model_select.addItems(["YOLOv8n (Nano)", "YOLOv8s (Small)"])

        self.load_video_button = QPushButton("Load Video", self)
        self.start_stream_button = QPushButton("Start Stream", self)
        self.start_button = QPushButton("Start Detection", self)
        self.stop_button = QPushButton("Stop Detection", self)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.model_select)
        self.layout.addWidget(self.load_video_button)
        self.layout.addWidget(self.start_stream_button)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        # 동작 연결
        self.load_video_button.clicked.connect(self.load_video)
        self.start_stream_button.clicked.connect(self.start_stream)
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)

        # 변수 초기화
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.model = None
        self.frame_source = None

    def load_video(self):
        # 동영상 파일 선택
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if filename:
            self.frame_source = filename

    def start_stream(self):
        # 핸드폰 스트림 URL 설정
        self.frame_source = "http://<휴대폰_IP>:4747/video"

    def start_detection(self):
        # 선택된 모델 로드
        selected_model = self.model_select.currentText()
        model_map = {"YOLOv8n (Nano)": "yolov8n.pt", "YOLOv8s (Small)": "yolov8s.pt"}
        self.model = YOLO(model_map[selected_model])

        # 입력 소스 설정
        if self.frame_source:
            self.cap = cv2.VideoCapture(self.frame_source)
            self.timer.start(30)  # 30ms마다 업데이트

    def stop_detection(self):
        # 탐지 중지
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.label.clear()

    def update_frame(self):
        # 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        # YOLO 탐지 실행
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        # Annotator를 이용한 시각화
        # box_annotator = sv.BoxAnnotator()  # BoxAnnotator 초기화
        triangle_annotator = sv.TriangleAnnotator()
        frame = triangle_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)

        # 화면 업데이트
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        qimg = QImage(frame_rgb.data, width, height, width * channel, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

# 실행 코드
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())