import sys
import cv2
import face_recognition
import pickle
import os
import uuid
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLineEdit,
    QMessageBox,
    QFileDialog,
    QListWidget,
    QComboBox,
)
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt6.QtMultimedia import QMediaDevices

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание лиц")
        self.setGeometry(100, 100, 1000, 600)

        # данные для базы (энкодинги, UUID, имена в формате uuid: имя)
        self.known_encodings = []
        self.known_ids = []
        self.known_names = {} 
        self.db_file = "faces.pkl"

        # погрешность для распознавания лица
        self.tolerance = 0.6

        # уникальные распознавания
        self.session_unique_ids = []

        # стейт
        self.is_camera_running = False
        self.is_registering = False
        self.process_this_frame = True
        self.face_data = []
        self.current_name = ""

        self.initUI()
        self.load_database()
        self.update_camera_list()

    # инициализация интерфейса
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()

        self.video_label = QLabel("Распознавание по камере выключено. Нажмите \"Включить камеру\"")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: black; color: white; font-size: 20px;"
        )
        self.video_label.setMinimumSize(640, 480)

        reg_layout = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Введите имя")

        self.btn_register_cam = QPushButton("Добавить с камеры")
        self.btn_register_cam.clicked.connect(self.start_registration_cam)

        self.btn_register_photo = QPushButton("Добавить по фото")
        self.btn_register_photo.clicked.connect(self.register_from_photo)

        reg_layout.addWidget(self.name_input)
        reg_layout.addWidget(self.btn_register_cam)
        reg_layout.addWidget(self.btn_register_photo)

        control_layout = QHBoxLayout()

        self.camera_selector = QComboBox()
        self.camera_selector.setMinimumWidth(200)

        self.btn_refresh_cameras = QPushButton("Обновить")
        self.btn_refresh_cameras.setFixedWidth(80)
        self.btn_refresh_cameras.clicked.connect(self.update_camera_list)

        self.btn_start = QPushButton("Включить камеру")
        self.btn_start.clicked.connect(self.toggle_camera)

        self.btn_recognize_photo = QPushButton("Распознать на фото")
        self.btn_recognize_photo.clicked.connect(self.recognize_on_photo)

        control_layout.addWidget(QLabel("Устройство:"))
        control_layout.addWidget(self.camera_selector)
        control_layout.addWidget(self.btn_refresh_cameras)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_recognize_photo)

        left_panel.addWidget(self.video_label)
        left_panel.addLayout(reg_layout)
        left_panel.addLayout(control_layout)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Уникальные лица:"))

        self.session_list_widget = QListWidget()
        self.session_list_widget.setMinimumWidth(250)
        right_panel.addWidget(self.session_list_widget)

        session_buttons_layout = QHBoxLayout()
        self.btn_clear_session = QPushButton("Очистить")
        self.btn_clear_session.clicked.connect(self.clear_session)

        self.btn_save_session = QPushButton("Сохранить в файл")
        self.btn_save_session.clicked.connect(self.save_session_to_file)

        session_buttons_layout.addWidget(self.btn_clear_session)
        session_buttons_layout.addWidget(self.btn_save_session)
        right_panel.addLayout(session_buttons_layout)

        outer_layout.addLayout(left_panel, 3)
        outer_layout.addLayout(right_panel, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    # загружает базу лиц
    def load_database(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "rb") as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get("encodings", [])
                    self.known_ids = data.get("ids", [])
                    self.known_names = data.get("names", {})

                print(f"Загружено {len(self.known_encodings)} лиц из базы")
            except Exception as e:
                print(f"Ошибка при загрузке базы: {e}")

    # сохраняет базу лиц
    def save_database(self):
        data = {
            "encodings": self.known_encodings,
            "ids": self.known_ids,
            "names": self.known_names,
        }
        with open(self.db_file, "wb") as f:
            pickle.dump(data, f)

    # сохраняет новое лицо в базу
    def save_new_face(self, face_encoding, name):
        uid = str(uuid.uuid4())
        self.known_encodings.append(face_encoding)
        self.known_ids.append(uid)
        self.known_names[uid] = name

        self.save_database()
        QMessageBox.information(
            self,
            "Успех",
            f"Пользователь '{name}' (ID: {uid.split('-')[0]}) успешно добавлен",
        )

    # сравнивает энкодинг лица с сохранёнными, если не найдено, то имя будет "Неизвестный"
    def match_face(self, face_encoding):
        if not self.known_encodings:
            return "Неизвестный", None

        face_distances = face_recognition.face_distance(
            self.known_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] <= self.tolerance:
            matched_id = self.known_ids[best_match_index]
            return self.known_names.get(matched_id, "Неизвестный"), matched_id

        return "Неизвестный", None

    # находит лица на изображении
    def identify_faces(self, frame, scale=1.0):
        if scale != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = frame

        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detections = []
        for face_encoding, (top, right, bottom, left) in zip(
            face_encodings, face_locations
        ):
            name, matched_id = self.match_face(face_encoding)

            # Если лицо распознано (есть UUID), добавляем в лог сессии
            if matched_id:
                self.add_to_session_list(matched_id, name)

            if scale != 1.0:
                inv_scale = 1.0 / scale
                top, right, bottom, left = (
                    int(top * inv_scale),
                    int(right * inv_scale),
                    int(bottom * inv_scale),
                    int(left * inv_scale),
                )

            detections.append((top, right, bottom, left, name))

        return detections

    # получает энкодинг одного лица на кадре (для сохранения)
    def get_single_face_encoding(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            return None, "Лицо не найдено, попробуйте снова"
        if len(face_locations) > 1:
            return None, "В кадре несколько лиц, а должно быть одно"

        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        return face_encoding, None

    # добавляет в список уникальных обнаружений
    def add_to_session_list(self, uid, name):
        if uid not in self.session_unique_ids:
            self.session_unique_ids.append(uid)
            short_uid = uid.split("-")[0]
            self.session_list_widget.addItem(f"{name} (ID: {short_uid})")

    # очищает список уникальных обнаружений
    def clear_session(self):
        self.session_unique_ids.clear()
        self.session_list_widget.clear()
        QMessageBox.information(self, "Сессия", "Список обнаруженных за сессию очищен")

    # сохраняет уникальные обнаружения в файл
    def save_session_to_file(self):
        if not self.session_unique_ids:
            QMessageBox.warning(
                self, "Внимание", "Список сессии пуст"
            )
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"session_{timestamp}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Журнал распознавания лиц за сессию: {timestamp}\n")
                f.write("=" * 40 + "\n")
                for uid in self.session_unique_ids:
                    name = self.known_names.get(uid, "Неизвестный")
                    f.write(f"UUID: {uid} | Имя: {name}\n")

            QMessageBox.information(
                self, "Успех", f"Сессия успешно сохранена в файл:\n{filename}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{e}")

    def update_camera_list(self):
        """Обновляет выпадающий список доступных камер"""
        if self.is_camera_running:
            QMessageBox.warning(
                self, "Внимание", "Выключите камеру перед обновлением списка."
            )
            return

        self.camera_selector.clear()
        cameras = QMediaDevices.videoInputs()

        if not cameras:
            self.camera_selector.addItem("Камеры не найдены", -1)
            return

        for i, cam in enumerate(cameras):
            description = cam.description()
            self.camera_selector.addItem(f"{i}: {description}", i)

    # включение и выключение чтения
    def toggle_camera(self):
        if not self.is_camera_running:
            cam_index = self.camera_selector.currentData()
            if cam_index is None or cam_index == -1:
                QMessageBox.critical(self, "Ошибка", "Камера не выбрана или не найдена.")
                return
            
            self.cap = cv2.VideoCapture(cam_index)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Ошибка", f"Не удалось получить доступ к камере (Индекс {cam_index}).")
                return
            
            self.is_camera_running = True
            self.btn_start.setText("Выключить камеру")
            self.camera_selector.setEnabled(False)
            self.btn_refresh_cameras.setEnabled(False)
            self.timer.start(20)
        else:
            self.timer.stop()
            self.cap.release()
            self.is_camera_running = False
            self.video_label.clear()
            self.video_label.setText("Камера выключена")
            self.video_label.setStyleSheet("background-color: black; color: white; font-size: 20px;")
            self.btn_start.setText("Включить камеру")
            self.camera_selector.setEnabled(True)
            self.btn_refresh_cameras.setEnabled(True)


    # отрисовка кадров с камеры
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        if self.is_registering:
            self.register_face_cam(frame)
            self.is_registering = False
            return

        if self.process_this_frame:
            self.face_data = self.identify_faces(frame, scale=0.25)

        self.process_this_frame = not self.process_this_frame

        self.display_image(frame, self.face_data)

    # регистрация в базу с камеры
    def start_registration_cam(self):
        if not self.is_camera_running:
            QMessageBox.warning(self, "Внимание", "Сначала включите камеру")
            return

        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Внимание", "Введите имя человека")
            return

        self.current_name = name
        self.is_registering = True
        self.name_input.clear()
        self.video_label.setText("Идет регистрация")
        self.video_label.setStyleSheet(
            "background-color: black; color: yellow; font-size: 20px;"
        )

    # финализация регистрации в базу с камеры
    def register_face_cam(self, frame):
        face_encoding, error = self.get_single_face_encoding(frame)
        if error:
            QMessageBox.warning(self, "Ошибка", error)
            return

        self.save_new_face(face_encoding, self.current_name)
        self.current_name = ""

    # регистрация в базу с фото
    def register_from_photo(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(
                self, "Внимание", "Введите имя человека перед выбором фото"
            )
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите фото", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_name:
            return

        frame = cv2.imread(file_name)
        if frame is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
            return

        face_encoding, error = self.get_single_face_encoding(frame)
        if error:
            QMessageBox.warning(self, "Ошибка", error)
            return

        self.save_new_face(face_encoding, name)
        self.name_input.clear()

    # распознание на фото
    def recognize_on_photo(self):
        if self.is_camera_running:
            self.toggle_camera()

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите фото для распознавания",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_name:
            return

        frame = cv2.imread(file_name)
        if frame is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
            return

        detections = self.identify_faces(frame, scale=1.0)
        self.display_image(frame, detections)

    # рисует изображение с камеры/фото в окно, а также рисует рамку вокруг распознанных лиц
    def display_image(self, frame, detections=[]):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_img)

        painter = QPainter(pixmap)
        painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))

        for top, right, bottom, left, name in detections:
            color = QColor(0, 255, 0) if name != "Неизвестный" else QColor(255, 0, 0)

            painter.setPen(color)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(left, top, right - left, bottom - top)

            text_rect = QRect(left, bottom, right - left, 30)
            painter.fillRect(text_rect, color)

            painter.setPen(QColor(255, 255, 255))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, name)

        painter.end()

        self.video_label.setStyleSheet("")
        self.video_label.setText("")
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec())
