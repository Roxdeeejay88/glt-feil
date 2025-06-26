import sys
import cv2
import numpy as np
import json
import socket
import threading
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout,
                             QTabWidget, QPushButton, QLabel, QLineEdit, QComboBox,
                             QSpinBox, QDoubleSpinBox, QTextEdit, QGraphicsView, QGraphicsScene, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRectF, Qt as QtEnum
import uuid
import os
from datetime import datetime


class Signals(QObject):
    update_gui = pyqtSignal(list, str)  # Für erkannte Bauteile und Protokollmeldungen
    update_counts = pyqtSignal(int, int)  # Für Zähler (aktuelle/gesamte Bauteile)
    update_training_contour = pyqtSignal(list)  # Für visuelle Rückmeldung der Trainingskontur


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bauteilerkennung und Laserkommunikation")
        self.setGeometry(100, 100, 1400, 800)  # Erhöhte Breite für dreispaltiges Layout

        # Signale
        self.signals = Signals()
        self.signals.update_gui.connect(self.update_gui)
        self.signals.update_counts.connect(self.update_counts)
        self.signals.update_training_contour.connect(self.update_training_contour)

        # Konfigurationsdatei
        self.config_file = "settings.json"
        self.templates = {}  # Sicherstellen, dass templates immer initialisiert ist
        self.load_settings()

        # Kamera
        self.cap = None
        self.camera_thread = None
        self.camera_running = False
        self.recognition_enabled = False  # Neue Variable für manuelles Starten
        self.frame = None

        # Erkennung
        self.contours = []
        self.recognized_parts = []
        self.total_parts = 0
        self.current_id = 1
        self.training_mode = False
        self.recognition_paused = False
        self.manual_points = []
        self.training_contour = None
        self.rectangle_points = []  # Für Rechteck-Zeichnen

        # Kalibrierung
        self.mm_per_pixel = self.settings.get("mm_per_pixel", 0.1)  # Standardwert, kann angepasst werden

        # TCP/IP
        self.tcp_thread = None
        self.tcp_running = False
        self.tcp_socket = None

        # GUI
        self.init_ui()

        # Timer für Live-Vorschau
        self.timer_interval = self.settings.get("timer_interval", 30)  # Standard 30 ms (ca. 33 fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.timer_interval)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        grid_layout = QGridLayout(main_widget)

        # Spalte 1: Sidebar (links)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        self.coord_display = QTextEdit()
        self.coord_display.setReadOnly(True)
        sidebar_layout.addWidget(QLabel("Erkannte Bauteile:"))
        sidebar_layout.addWidget(self.coord_display)

        self.part_count_label = QLabel("Aktuell erkannt: 0 | Gesamt: 0")
        sidebar_layout.addWidget(self.part_count_label)

        self.start_stop_button = QPushButton("Erkennung starten")
        self.start_stop_button.clicked.connect(self.toggle_recognition)
        sidebar_layout.addWidget(self.start_stop_button)

        reset_button = QPushButton("Zurücksetzen")
        reset_button.clicked.connect(self.reset_counts)
        sidebar_layout.addWidget(reset_button)

        sidebar.setLayout(sidebar_layout)
        grid_layout.addWidget(sidebar, 0, 0, 2, 1)  # Spalte 0, über beide Reihen

        # Spalte 2: Kamerabild (mitte)
        self.view = QGraphicsView()
        self.view.setMinimumSize(800, 600)  # Minimale Größe für besseres Kamerabild
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.view.mousePressEvent = self.mouse_press_event
        self.view.mouseMoveEvent = self.mouse_move_event
        self.view.mouseReleaseEvent = self.mouse_release_event
        grid_layout.addWidget(self.view, 0, 1, 1, 1)  # Spalte 1, obere Reihe

        # Spalte 3: Tabs und Steuerung (rechts)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        tabs = QTabWidget()
        content_layout.addWidget(tabs)

        # Tab 1: Anlernen
        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab)
        self.train_button = QPushButton("Neues Bauteil anlernen")
        self.train_button.clicked.connect(self.start_training)
        train_layout.addWidget(self.train_button)

        self.manual_train_button = QPushButton("Manuell nachzeichnen")
        self.manual_train_button.clicked.connect(self.start_manual_training)
        train_layout.addWidget(self.manual_train_button)

        self.rectangle_train_button = QPushButton("Rechteck ziehen")
        self.rectangle_train_button.clicked.connect(self.start_rectangle_training)
        train_layout.addWidget(self.rectangle_train_button)

        self.template_name = QLineEdit("Bauteil_1")
        train_layout.addWidget(QLabel("Bauteilname:"))
        train_layout.addWidget(self.template_name)

        confirm_button = QPushButton("Kontur bestätigen")
        confirm_button.clicked.connect(self.confirm_training)
        confirm_button.setEnabled(False)
        self.confirm_button = confirm_button
        train_layout.addWidget(confirm_button)

        instructions_label = QLabel(
            "Automatisches Anlernen: Halte ein Bauteil vor die Kamera, Kontur wird rot markiert. Bestätige sie.\nManuelles Anlernen: Setze Punkte entlang der Kontur, Rechtsklick zum Abschließen.\nRechteck: Ziehe ein Rechteck um das Bauteil, Links-Release zum Speichern.")
        instructions_label.setWordWrap(True)
        train_layout.addWidget(instructions_label)

        train_tab.setLayout(train_layout)
        tabs.addTab(train_tab, "Anlernen")

        # Tab 2: Einstellungen
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        self.field_x = QSpinBox()
        self.field_x.setRange(10, 1000)
        self.field_x.setValue(self.settings.get("field_x", 50))  # Anpassung auf 50 mm
        settings_layout.addWidget(QLabel("Feldgröße X (mm):"))
        settings_layout.addWidget(self.field_x)

        self.field_y = QSpinBox()
        self.field_y.setRange(10, 1000)
        self.field_y.setValue(self.settings.get("field_y", 50))  # Anpassung auf 50 mm
        settings_layout.addWidget(QLabel("Feldgröße Y (mm):"))
        settings_layout.addWidget(self.field_y)

        self.start_pos = QComboBox()
        self.start_pos.addItems(["Mitte", "Oben-Links", "Oben-Rechts", "Unten-Links", "Unten-Rechts"])
        self.start_pos.setCurrentText(self.settings.get("start_pos", "Mitte"))
        settings_layout.addWidget(QLabel("Startkoordinate:"))
        settings_layout.addWidget(self.start_pos)

        self.axis = QComboBox()
        self.axis.addItems(["X-Achse", "Y-Achse"])
        self.axis.setCurrentText(self.settings.get("axis", "X-Achse"))
        settings_layout.addWidget(QLabel("Winkel-Achse:"))
        settings_layout.addWidget(self.axis)

        self.sensitivity = QSpinBox()
        self.sensitivity.setRange(0, 100)
        self.sensitivity.setValue(self.settings.get("sensitivity", 50))
        settings_layout.addWidget(QLabel("Empfindlichkeit (%):"))
        settings_layout.addWidget(self.sensitivity)

        self.mm_per_pixel_spin = QDoubleSpinBox()
        self.mm_per_pixel_spin.setRange(0.01, 1.0)
        self.mm_per_pixel_spin.setSingleStep(0.01)
        self.mm_per_pixel_spin.setValue(self.settings.get("mm_per_pixel", 0.1))
        self.mm_per_pixel_spin.valueChanged.connect(self.update_mm_per_pixel)
        settings_layout.addWidget(QLabel("mm pro Pixel:"))
        settings_layout.addWidget(self.mm_per_pixel_spin)

        self.timer_interval_spin = QSpinBox()
        self.timer_interval_spin.setRange(10, 1000)  # 10 ms bis 1000 ms
        self.timer_interval_spin.setValue(self.settings.get("timer_interval", 30))
        self.timer_interval_spin.valueChanged.connect(self.update_timer_interval)
        settings_layout.addWidget(QLabel("Refresh-Rate (ms):"))
        settings_layout.addWidget(self.timer_interval_spin)

        self.ip_address = QLineEdit(self.settings.get("ip_address", "192.168.1.100"))
        settings_layout.addWidget(QLabel("SPS IP-Adresse:"))
        settings_layout.addWidget(self.ip_address)

        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(self.settings.get("port", 12345))
        settings_layout.addWidget(QLabel("Port:"))
        settings_layout.addWidget(self.port)

        save_settings = QPushButton("Einstellungen speichern")
        save_settings.clicked.connect(self.save_settings)
        settings_layout.addWidget(save_settings)

        settings_tab.setLayout(settings_layout)
        tabs.addTab(settings_tab, "Einstellungen")

        # Kalibrierungsbutton
        calib_button = QPushButton("Kamera kalibrieren")
        calib_button.clicked.connect(self.calibrate_camera)
        content_layout.addWidget(calib_button)

        content.setLayout(content_layout)
        grid_layout.addWidget(content, 0, 2, 2, 1)  # Spalte 2, über beide Reihen

        main_widget.setLayout(grid_layout)

    def load_settings(self):
        self.settings = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.settings = json.load(f)
            except json.JSONDecodeError as e:
                self.signals.update_gui.emit([],
                                             f"Fehler beim Laden der Einstellungen: {str(e)}. Standardwerte verwendet.")

        # Lade Vorlagen
        templates_dir = self.settings.get("templates_dir", "templates")
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)
        for f in os.listdir(templates_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(templates_dir, f), 'r') as file:
                        name = f.replace(".json", "")
                        self.templates[name] = json.load(file)
                except json.JSONDecodeError as e:
                    self.signals.update_gui.emit([], f"Fehler beim Laden der Vorlage {f}: {str(e)}. Vorlage ignoriert.")
                except Exception as e:
                    self.signals.update_gui.emit([],
                                                 f"Unbekannter Fehler beim Laden von {f}: {str(e)}. Vorlage ignoriert.")

    def save_settings(self):
        self.settings.update({
            "field_x": self.field_x.value(),
            "field_y": self.field_y.value(),
            "start_pos": self.start_pos.currentText(),
            "axis": self.axis.currentText(),
            "sensitivity": self.sensitivity.value(),
            "mm_per_pixel": self.mm_per_pixel_spin.value(),
            "timer_interval": self.timer_interval_spin.value(),
            "ip_address": self.ip_address.text(),
            "port": self.port.value(),
            "templates_dir": "templates"
        })
        with open(self.config_file, 'w') as f:
            json.dump(self.settings, f, indent=4)
        self.signals.update_gui.emit([], "Einstellungen gespeichert.")

    def update_mm_per_pixel(self, value):
        self.mm_per_pixel = value
        self.save_settings()

    def update_timer_interval(self, value):
        self.timer_interval = value
        self.timer.start(self.timer_interval)
        self.save_settings()

    def start_camera(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.signals.update_gui.emit([], "Fehler: Kamera konnte nicht geöffnet werden.")
                return
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            self.start_tcp()

    def camera_loop(self):
        while self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                if self.recognition_enabled:
                    self.process_frame()
            time.sleep(0.01)

    def process_frame(self):
        if self.frame is None or self.recognition_paused:
            return
        frame = self.frame.copy()

        # Bildvorverarbeitung
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                       2)  # Anpassung für bessere Konturen
        sensitivity = self.sensitivity.value() / 100.0
        thresh = cv2.convertScaleAbs(thresh, alpha=sensitivity, beta=0)

        # Debugging-Ausgabe
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.signals.update_gui.emit([],
                                     f"Processing frame, recognition_enabled: {self.recognition_enabled}, contours found: {len(contours)}")

        # Definiere das Erkennungsfeld
        h, w = self.frame.shape[:2]  # Verwende die tatsächliche Frame-Größe
        self.signals.update_gui.emit([], f"Frame size: {w}x{h}")
        field_w = int(self.field_x.value() / self.mm_per_pixel)
        field_h = int(self.field_y.value() / self.mm_per_pixel)
        # Begrenze das Feld auf die Bildgröße, damit es korrekt zentriert werden kann
        field_w = min(field_w, w)
        field_h = min(field_h, h)
        if field_w <= 0 or field_h <= 0:
            self.signals.update_gui.emit([], f"Ungültige Feldgröße: field_w={field_w}, field_h={field_h}")
            return
        start_pos = self.start_pos.currentText()
        if start_pos == "Mitte":
            x0 = (w - field_w) // 2  # Zentrierung basierend auf tatsächlicher Frame-Größe
            y0 = (h - field_h) // 2
        elif start_pos == "Oben-Links":
            x0, y0 = 0, 0
        elif start_pos == "Oben-Rechts":
            x0, y0 = w - field_w, 0
        elif start_pos == "Unten-Links":
            x0, y0 = 0, h - field_h
        elif start_pos == "Unten-Rechts":
            x0, y0 = w - field_w, h - field_h

        # Schneide das Bild auf das Erkennungsfeld
        x0 = max(0, min(x0, w - field_w))
        y0 = max(0, min(y0, h - field_h))
        field = thresh[y0:y0 + field_h, x0:x0 + field_w] if field_w > 0 and field_h > 0 else thresh
        self.signals.update_gui.emit([], f"Field position: x0={x0}, y0={y0}, width={field_w}, height={field_h}")

        # Konturenerkennung nur im Feld
        contours, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = []
        self.recognized_parts = []

        for cnt in contours:
            # Anpassen der Konturkoordinaten zum Originalbild
            cnt[:, :, 0] += x0
            cnt[:, :, 1] += y0
            area = cv2.contourArea(cnt)
            x, y, w_cont, h_cont = cv2.boundingRect(cnt)  # Koordinaten und Größe der Kontur
            self.signals.update_gui.emit([],
                                         f"Contour at ({x}, {y}), size ({w_cont}, {h_cont}), area: {area}, template count: {len(self.templates)}")
            if area > 100:  # Mindestfläche
                if self.training_mode and not self.manual_points and not self.rectangle_points:  # Automatisches Anlernen
                    self.training_contour = cnt
                    self.signals.update_gui.emit([], f"Training contour set at ({x}, {y}), size ({w_cont}, {h_cont})")
                    self.signals.update_training_contour.emit([cnt])
                    break
                for name, template in self.templates.items():
                    match = self.match_template(cnt, template)
                    self.signals.update_gui.emit([], f"Match with {name}: {match}")
                    if match > 0.8:  # Schwellenwert
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            angle = self.calculate_angle(cnt)
                            part = {
                                "id": self.current_id,
                                "x": (cx - x0) * self.mm_per_pixel,  # Relative Koordinaten im Feld
                                "y": (cy - y0) * self.mm_per_pixel,
                                "angle": angle
                            }
                            self.recognized_parts.append(part)
                            self.signals.update_gui.emit([],
                                                         f"Part detected: ID {part['id']}, X {part['x']:.2f}, Y {part['y']:.2f}")
                            self.current_id += 1
                            self.total_parts += 1
                            self.contours.append(cnt)
                            self.send_tcp_data(part)

        # Zeichne Konturen und IDs auf das Originalbild
        for i, cnt in enumerate(self.contours):
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)  # Grüne Konturen für erkannte Bauteile
            part = self.recognized_parts[i]
            cv2.putText(frame, f"ID: {part['id']}",
                        (int((part['x'] / self.mm_per_pixel) + x0), int((part['y'] / self.mm_per_pixel) + y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if self.training_contour is not None:
            cv2.drawContours(frame, [self.training_contour], -1, (255, 0, 0), 2)  # Rote Kontur im Trainingsmodus

        self.frame = frame  # Aktualisiere das Frame mit den gezeichneten Konturen
        # Sende Daten an Hauptthread
        self.signals.update_gui.emit(self.recognized_parts, "")
        self.signals.update_counts.emit(len(self.recognized_parts), self.total_parts)

    def match_template(self, contour, template):
        # Einfaches Feature-Matching
        return 1.0 if cv2.matchShapes(contour, np.array(template["contour"]), 1, 0.0) < 0.1 else 0.0

    def calculate_angle(self, contour):
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        if self.axis.currentText() == "Y-Achse":
            angle = angle + 90
        return angle % 360

    def update_frame(self):
        if self.frame is None:
            self.start_camera()
            return
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w = self.frame.shape[:2]  # Verwende die tatsächliche Frame-Größe
        image = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
        self.scene.clear()
        self.scene.addPixmap(pixmap)

        # Zeichne das Feld als Overlay
        field_w = int(self.field_x.value() / self.mm_per_pixel)
        field_h = int(self.field_y.value() / self.mm_per_pixel)
        # Begrenze das Feld auf die Bildgröße, damit es korrekt zentriert werden kann
        field_w = min(field_w, w)
        field_h = min(field_h, h)
        start_pos = self.start_pos.currentText()
        if start_pos == "Mitte":
            x0 = (w - field_w) // 2  # Zentrierung basierend auf tatsächlicher Frame-Größe
            y0 = (h - field_h) // 2
        elif start_pos == "Oben-Links":
            x0, y0 = 0, 0
        elif start_pos == "Oben-Rechts":
            x0, y0 = w - field_w, 0
        elif start_pos == "Unten-Links":
            x0, y0 = 0, h - field_h
        elif start_pos == "Unten-Rechts":
            x0, y0 = w - field_w, h - field_h
        x0 = max(0, min(x0, w - field_w))
        y0 = max(0, min(y0, h - field_h))
        self.scene.addRect(QRectF(x0, y0, field_w, field_h), QPen(QColor(0, 0, 255), 2))  # Blau mit QColor

        if self.training_contour:
            self.draw_training_contour()
        if self.manual_points:
            self.draw_manual_points()
        if self.rectangle_points:
            self.draw_rectangle()

    def update_gui(self, parts, message):
        self.recognized_parts = parts
        # Kein clear(), füge einfach hinzu
        if parts or message:
            for part in parts:
                self.coord_display.append(
                    f"ID: {part['id']}, X: {part['x']:.2f} mm, Y: {part['y']:.2f} mm, Winkel: {part['angle']:.2f}°")
            if message:
                self.coord_display.append(message)

    def update_counts(self, current, total):
        self.part_count_label.setText(f"Aktuell erkannt: {current} | Gesamt: {total}")

    def update_training_contour(self, contours):
        self.training_contour = contours[0] if contours else None
        self.confirm_button.setEnabled(bool(self.training_contour))
        self.update_frame()

    def draw_training_contour(self):
        if self.training_contour:
            contour = self.training_contour
            points = [(p[0][0], p[0][1]) for p in contour]
            poly = self.scene.addPolygon([[x, y] for x, y in points], QPen(QColor(255, 0, 0), 2))  # Rot mit QColor
            self.scene.update()

    def draw_manual_points(self):
        for x, y in self.manual_points:
            self.scene.addEllipse(x - 2, y - 2, 4, 4, QPen(QColor(0, 255, 0), 1),
                                  QPen(QColor(0, 255, 0)).brush())  # Grün mit QColor
        self.scene.update()

    def draw_rectangle(self):
        if len(self.rectangle_points) == 2:
            x0, y0 = self.rectangle_points[0]
            x1, y1 = self.rectangle_points[1]
            rect = self.scene.addRect(QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)),
                                      QPen(QColor(255, 255, 0), 2))  # Gelb mit QColor
            self.scene.update()
        elif len(self.rectangle_points) == 1:
            # Zeichne temporäres Rechteck während des Ziehens
            x0, y0 = self.rectangle_points[0]
            pos = self.view.mapFromGlobal(self.cursor().pos())
            scene_pos = self.view.mapToScene(pos)
            x1, y1 = int(scene_pos.x()), int(scene_pos.y())
            rect = self.scene.addRect(QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)),
                                      QPen(QColor(255, 255, 0), 2, QtEnum.PenStyle.DashLine))  # Gestricheltes Rechteck
            self.scene.update()

    def start_training(self):
        self.training_mode = True
        # Recognition should remain active so contours can be detected
        self.recognition_paused = False
        self.training_contour = None
        self.rectangle_points = []
        self.signals.update_gui.emit([],
                                     "Bitte Bauteil in die Kamera halten. Kontur wird automatisch im definierten Feld erkannt. Drücke 'Anlernen abschließen' oder 'Kontur bestätigen'.")
        self.train_button.setText("Anlernen abschließen")
        self.train_button.clicked.disconnect()
        self.train_button.clicked.connect(self.finish_training)
        self.confirm_button.setEnabled(False)

    def finish_training(self):
        if not self.training_contour:
            self.signals.update_gui.emit([],
                                         "Keine Kontur erkannt. Versuchen Sie es erneut oder wählen Sie manuell/Rechteck.")
            return
        name = self.template_name.text()
        template = {
            "contour": self.training_contour.tolist(),
            "features": {}  # Platzhalter für Feature-Vektoren
        }
        self.templates[name] = template
        with open(os.path.join("templates", f"{name}.json"), 'w') as f:
            json.dump(template, f)
        self.signals.update_gui.emit([], f"Bauteil '{name}' gespeichert.")
        self.training_mode = False
        self.recognition_paused = False
        self.training_contour = None
        self.rectangle_points = []
        self.train_button.setText("Neues Bauteil anlernen")
        self.train_button.clicked.disconnect()
        self.train_button.clicked.connect(self.start_training)
        self.confirm_button.setEnabled(False)

    def confirm_training(self):
        if self.training_contour:
            reply = QMessageBox.question(self, "Kontur bestätigen",
                                         "Möchten Sie die erkannte Kontur speichern?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.finish_training()
            else:
                self.signals.update_gui.emit([], "Kontur wurde nicht gespeichert. Versuchen Sie es erneut.")

    def start_manual_training(self):
        self.training_mode = True
        self.recognition_paused = True
        self.manual_points = []
        self.rectangle_points = []
        self.signals.update_gui.emit([],
                                     "Setze Punkte entlang der Kontur des Bauteils. Rechtsklick zum Abschließen und Speichern.")

    def start_rectangle_training(self):
        self.training_mode = True
        self.recognition_paused = True
        self.manual_points = []
        self.rectangle_points = []
        self.signals.update_gui.emit([], "Ziehe ein Rechteck um das Bauteil. Links-Release zum Speichern.")

    def mouse_press_event(self, event):
        if self.training_mode:
            pos = self.view.mapToScene(event.pos())
            if event.button() == Qt.MouseButton.LeftButton:
                if self.rectangle_points is not None and not self.rectangle_points:
                    self.rectangle_points.append((int(pos.x()), int(pos.y())))
                else:
                    self.manual_points.append([int(pos.x()), int(pos.y())])
            elif event.button() == Qt.MouseButton.RightButton and self.manual_points:
                if len(self.manual_points) > 2:
                    contour = np.array(self.manual_points, dtype=np.int32).reshape((-1, 1, 2))
                    name = self.template_name.text()
                    template = {
                        "contour": contour.tolist(),
                        "features": {}
                    }
                    self.templates[name] = template
                    with open(os.path.join("templates", f"{name}.json"), 'w') as f:
                        json.dump(template, f)
                    self.signals.update_gui.emit([], f"Manuelles Bauteil '{name}' gespeichert.")
                self.training_mode = False
                self.recognition_paused = False
                self.manual_points = []
                self.rectangle_points = []
            self.update_frame()

    def mouse_move_event(self, event):
        if self.training_mode and len(self.rectangle_points) == 1:
            pos = self.view.mapToScene(event.pos())
            self.rectangle_points = [self.rectangle_points[0], (int(pos.x()), int(pos.y()))]
            self.update_frame()

    def mouse_release_event(self, event):
        if self.training_mode and event.button() == Qt.MouseButton.LeftButton and len(self.rectangle_points) == 2:
            self.finish_rectangle_training()

    def finish_rectangle_training(self):
        if len(self.rectangle_points) == 2:
            x0, y0 = self.rectangle_points[0]
            x1, y1 = self.rectangle_points[1]
            contour = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32).reshape((-1, 1, 2))
            name = self.template_name.text()
            template = {
                "contour": contour.tolist(),
                "features": {}
            }
            self.templates[name] = template
            with open(os.path.join("templates", f"{name}.json"), 'w') as f:
                json.dump(template, f)
            self.signals.update_gui.emit([], f"Rechteck-Bauteil '{name}' gespeichert.")
            self.training_mode = False
            self.recognition_paused = False
            self.rectangle_points = []

    def calibrate_camera(self):
        self.signals.update_gui.emit([],
                                     "Kalibrierung: Bitte Schachbrettmuster (z. B. 8x6) vor die Kamera halten oder Referenzobjekt mit bekannter Größe.")
        if self.frame is None:
            self.signals.update_gui.emit([], "Fehler: Kein Kamerabild verfügbar.")
            return

        # Rudimentäre Kalibrierung (Platzhalter)
        ref_mm = 50.0  # Beispiel: Referenzobjekt ist 50 mm breit
        ref_pixels = 500  # Beispiel: Gemessene Pixel
        self.mm_per_pixel = ref_mm / ref_pixels
        self.settings["mm_per_pixel"] = self.mm_per_pixel
        self.save_settings()
        self.signals.update_gui.emit([], f"Kalibrierung abgeschlossen. mm/Pixel: {self.mm_per_pixel:.4f}")

    def start_tcp(self):
        if not self.tcp_running:
            self.tcp_running = True
            self.tcp_thread = threading.Thread(target=self.tcp_loop)
            self.tcp_thread.daemon = True
            self.tcp_thread.start()

    def tcp_loop(self):
        while self.tcp_running:
            try:
                self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.ip_address.text(), self.port.value()))
                self.signals.update_gui.emit([], "TCP-Verbindung hergestellt.")
                while self.tcp_running:
                    time.sleep(1)  # Heartbeat
            except Exception as e:
                self.signals.update_gui.emit([], f"TCP-Fehler: {str(e)}. Wiederverbinden...")
                time.sleep(5)
            finally:
                if self.tcp_socket:
                    self.tcp_socket.close()

    def send_tcp_data(self, part):
        if self.tcp_socket:
            try:
                msg = f"{part['id']},{part['x']:.2f},{part['y']:.2f},{part['angle']:.2f}\n"
                self.tcp_socket.send(msg.encode())
                self.signals.update_gui.emit([], f"Gesendet: {msg.strip()}")
            except Exception as e:
                self.signals.update_gui.emit([], f"Sendefehler: {str(e)}")

    def toggle_recognition(self):
        self.recognition_enabled = not self.recognition_enabled
        self.start_stop_button.setText("Erkennung stoppen" if self.recognition_enabled else "Erkennung starten")
        status = "gestartet" if self.recognition_enabled else "gestoppt"
        self.signals.update_gui.emit([], f"Erkennung {status}.")

    def reset_counts(self):
        self.total_parts = 0
        self.current_id = 1
        self.recognized_parts = []
        self.signals.update_gui.emit([], "Zähler zurückgesetzt.")
        self.signals.update_counts.emit(0, 0)

    def closeEvent(self, event):
        self.camera_running = False
        self.tcp_running = False
        if self.cap:
            self.cap.release()
        if self.tcp_socket:
            self.tcp_socket.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
