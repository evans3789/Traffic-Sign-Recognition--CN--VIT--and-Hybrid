# traffic_sign_ui.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Quiet TF INFO logs

import sys
import time
import cv2
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import Qt, QTimer, QDateTime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QTextEdit, QSlider, QSpinBox, QFormLayout,
    QCheckBox, QMessageBox, QFrame, QTableWidget, QTableWidgetItem
)

# -------------------- Globals / Model --------------------
thd = 0.45  # EXACTLY 45% confidence threshold, 
FONT = cv2.FONT_HERSHEY_SIMPLEX

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'model.h5')
try:
    mdl = tf.keras.models.load_model(model_path, compile=False)  # inference-only
except Exception as e:
    mdl = None
    print(f"[ERROR] Failed to load model from {model_path}: {e}")

# -------------------- Detection helpers --------------------
def _make_mask(imgBGR, erode_dilate=True, boost=False):
    """Simple red/blue mask; if boost=True, relax S/V and add a white interior mask."""
    hsv = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    s_floor, v_floor = (43, 46)
    if boost:
        s_floor, v_floor = (25, 30)

    blue = cv2.inRange(hsv, (100, s_floor, v_floor), (124, 255, 255))
    red = cv2.inRange(hsv, (0, s_floor, v_floor), (10, 255, 255)) | \
          cv2.inRange(hsv, (156, s_floor, v_floor), (180, 255, 255))
    mask = blue | red

    if boost:
        white = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
        mask = mask | white

    if erode_dilate:
        k = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, k, iterations=1 if not boost else 0)
        mask = cv2.dilate(mask, k, iterations=1)

    return mask

def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    """Return rectangles of roughly square-ish blobs; minimal gating for good recall."""
    rects = []
    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return rects
    max_area = img_bin.shape[0] * img_bin.shape[1] if max_area < 0 else max_area
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        if (w / h) < wh_ratio and (h / w) < wh_ratio:
            rects.append((x, y, w, h))
    return rects

def preprocessing(img):
    """Stable 32x32 grayscale with light denoise + CLAHE (good in rain too)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return (gray / 255.0).astype('float32')

def predict_image(img):
    if mdl is None:
        return -1, 0.0
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img).reshape(1, 32, 32, 1)
    preds = mdl.predict(img, verbose=0)
    return int(np.argmax(preds)), float(np.max(preds))

def getClassName(i):
    names = [
        'Speed Limit 20 km/h','Speed Limit 30 km/h','Speed Limit 50 km/h',
        'Speed Limit 60 km/h','Speed Limit 70 km/h','Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h','Speed Limit 100 km/h','Speed Limit 120 km/h',
        'No passing','No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection','Priority road','Yield','Stop',
        'No vehicles','Vehicles over 3.5 metric tons prohibited','No entry',
        'General caution','Dangerous curve to the left','Dangerous curve to the right',
        'Double curve','Bumpy road','Slippery road','Road narrows on the right',
        'Road work','Traffic signals','Pedestrians','Children crossing',
        'Bicycles crossing','Beware of ice/snow','Wild animals crossing',
        'End of all speed and passing limits','Turn right ahead','Turn left ahead',
        'Ahead only','Go straight or right','Go straight or left','Keep right',
        'Keep left','Roundabout mandatory','End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return names[i] if 0 <= i < len(names) else 'Unknown'

# -------------------- Label drawing (constant 12px) --------------------
def _compute_font_scale_for_12px():
    target_px, test_scale = 12, 0.5
    (_, h), _ = cv2.getTextSize("Ag", FONT, test_scale, 1)
    return max(0.4, float(target_px) / max(1, h))

def _draw_label_below_const(img, x, y, w, h, text, font_scale, thickness=1):
    if not text:
        return  # never draw an empty band
    (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, thickness)
    pad = 4
    bx1, by1 = int(x), int(y + h + pad)
    bx2, by2 = int(x + max(w, tw) + pad * 2), int(y + h + pad + th + bl + pad)
    if by2 > img.shape[0]:
        by2 = int(y - pad)
        by1 = int(y - (pad + th + bl + pad))
    bx1 = max(0, bx1); by1 = max(0, by1)
    bx2 = min(img.shape[1] - 1, bx2); by2 = min(img.shape[0] - 1, by2)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
    tx, ty = bx1 + pad, by2 - pad - bl
    cv2.putText(img, text, (int(tx), int(ty)),      FONT, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (int(tx), int(ty)),      FONT, font_scale, (255, 255, 255), thickness,     cv2.LINE_AA)

# -------------------- Main UI --------------------
class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Sign Detection UI")
        self.resize(1200, 780)

        # State
        self.cap = None
        self.is_image = False
        self.cols = None
        self.rows = None
        self.prev_time = time.time()
        self.frame_count = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.morph_enabled = True
        self.rain_boost_on = False           # manual only
        self.show_debug_probs = False        # optional p=… overlay
        self.current_source_label = "None"
        self._placeholder_pix = None
        self.font_scale_12 = _compute_font_scale_for_12px()  # constant for all labels

        # Left controls
        left_panel = self._build_left_controls()

        # Center column: viewer + detections table
        center_col = QWidget(); center_v = QVBoxLayout(center_col)
        center_v.setContentsMargins(0, 0, 0, 0); center_v.setSpacing(6)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color:#202225; color:#bbb;")
        center_v.addWidget(self.video_label, 1)
        self._load_placeholder()

        # Detections header + table
        row = QHBoxLayout()
        lbl = QLabel("Detections"); lbl.setStyleSheet("font-weight:bold;")
        row.addWidget(lbl); row.addStretch(1)
        btn_clear = QPushButton("Clear Detections"); btn_clear.clicked.connect(self._clear_detections)
        row.addWidget(btn_clear)
        center_v.addLayout(row)

        self.det_table = QTableWidget(0, 4)
        self.det_table.setHorizontalHeaderLabels(["Time", "Label", "Confidence", "Box (x,y,w,h)"])
        self.det_table.verticalHeader().setVisible(False)
        self.det_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.det_table.setSelectionBehavior(self.det_table.SelectRows)
        self.det_table.setAlternatingRowColors(True)
        self.det_table.setMinimumHeight(170)
        self.det_table.setStyleSheet("""
            QTableWidget {
                background-color: #0f1115;
                color: #e9edf1;
                gridline-color: #2b3440;
                alternate-background-color: #151a22;
            }
            QHeaderView::section {
                background-color: #1b2230;
                color: #ffffff;
                font-weight: 600;
                padding: 6px;
                border: 0px;
                border-bottom: 1px solid #2b3440;
            }
            QTableWidget::item:selected {
                background-color: #2b3440;
                color: #ffffff;
            }
        """)
        center_v.addWidget(self.det_table, 0)

        # Right logs
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet("background-color:#111; color:#ddd;")
        self.log.setMinimumWidth(300)

        # Root layout
        root = QWidget(); root_h = QHBoxLayout(root)
        root_h.addLayout(left_panel, 0)
        root_h.addWidget(self._vline())
        root_h.addWidget(center_col, 1)
        root_h.addWidget(self._vline())
        root_h.addWidget(self.log, 0)
        self.setCentralWidget(root)

        # Timer loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)

        # Logs
        self.append_log("App started.")
        self.append_log(f"Model loaded from: {model_path}" if mdl else "⚠️ Model failed to load.")

    # ---------- UI builders ----------
    def _vline(self):
        l = QFrame(); l.setFrameShape(QFrame.VLine); l.setFrameShadow(QFrame.Sunken)
        return l
    def _hline(self):
        l = QFrame(); l.setFrameShape(QFrame.HLine); l.setFrameShadow(QFrame.Sunken)
        return l

    def _build_left_controls(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8); layout.setSpacing(10)

        btn_camera = QPushButton("Use Camera"); btn_camera.clicked.connect(self.use_camera)
        cam_form = QFormLayout()
        self.cam_index = QSpinBox(); self.cam_index.setRange(0, 10); self.cam_index.setValue(0)
        cam_form.addRow("Camera Index:", self.cam_index)

        btn_open = QPushButton("Open Image/Video"); btn_open.clicked.connect(self.open_file)

        self.th_slider = QSlider(Qt.Horizontal); self.th_slider.setRange(0, 100)
        self.th_slider.setValue(int(thd * 100)); self.th_slider.valueChanged.connect(self._on_threshold_changed)
        self.th_spin = QSpinBox(); self.th_spin.setRange(0, 100)
        self.th_spin.setValue(int(thd * 100)); self.th_spin.valueChanged.connect(self._on_threshold_changed)
        self.th_slider.valueChanged.connect(self.th_spin.setValue)
        self.th_spin.valueChanged.connect(self.th_slider.setValue)
        th_form = QFormLayout(); th_form.addRow("Confidence %:", self.th_spin)

        self.chk_morph = QCheckBox("Erode/Dilate (Noise cleanup)")
        self.chk_morph.setChecked(True); self.chk_morph.stateChanged.connect(self._on_morph_toggle)

        self.chk_rain = QCheckBox("Rain / Low-light boost (manual)")
        self.chk_rain.setChecked(False); self.chk_rain.stateChanged.connect(self._on_rain_toggle)

        self.chk_debug = QCheckBox("Show debug probs (p=...)")
        self.chk_debug.setChecked(False); self.chk_debug.stateChanged.connect(self._on_debug_toggle)

        btn_stop = QPushButton("Stop / Release"); btn_stop.clicked.connect(self.stop_stream)
        btn_quit = QPushButton("Quit")
        btn_quit.setStyleSheet("background-color:#b00; color:white; font-weight:bold;")
        btn_quit.clicked.connect(self.close)

        layout.addWidget(btn_camera)
        layout.addLayout(cam_form)
        layout.addWidget(btn_open)
        layout.addWidget(self._hline())
        layout.addWidget(QLabel("Detection Threshold"))
        layout.addWidget(self.th_slider)
        layout.addLayout(th_form)
        layout.addWidget(self.chk_morph)
        layout.addWidget(self.chk_rain)
        layout.addWidget(self.chk_debug)
        layout.addWidget(self._hline())
        layout.addWidget(btn_stop)
        layout.addWidget(btn_quit)
        layout.addStretch(1)
        return layout

    # ---------- Placeholder ----------
    def _load_placeholder(self):
        p = os.path.join(base_path, "./placeholder.jpg")
        if os.path.exists(p):
            self._placeholder_pix = QPixmap(p)
            self._apply_placeholder_scale()
        else:
            self.video_label.setText("Video / Image will appear here")

    def _apply_placeholder_scale(self):
        if self._placeholder_pix:
            self.video_label.setPixmap(
                self._placeholder_pix.scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        pix = self.video_label.pixmap()
        if pix:
            self.video_label.setPixmap(
                pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        elif self._placeholder_pix:
            self._apply_placeholder_scale()

    # ---------- Source management ----------
    def use_camera(self):
        idx = self.cam_index.value()
        self.start_capture(idx)
        self.current_source_label = f"Camera[{idx}]"
        self.append_log(f"Using camera {idx}")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image or video", "",
            "Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not path: return
        ext = os.path.splitext(path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.stop_stream()
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Error", "Failed to load image.")
                return
            self.cap = img; self.is_image = True
            self.rows, self.cols = img.shape[:2]
            self.current_source_label = os.path.basename(path)
            self.append_log(f"Opened image: {path}")
            processed, cand_count, det_count, det_list = self.process_frame(img)  # GUI thread
            self.show_frame(processed)
            self._append_detections(det_list)
            self.append_log(f"Source: {self.current_source_label} | candidates: {cand_count} | accepted: {det_count}")
        else:
            self.start_capture(path)
            self.current_source_label = os.path.basename(path)
            self.append_log(f"Opened video: {path}")

    def start_capture(self, source):
        self.stop_stream()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap = None
            QMessageBox.warning(self, "Error", "Failed to open camera/video.")
            self.append_log("Failed to open source.", warn=True)
            return
        self.is_image = False
        self.cols = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.rows = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.prev_time = time.time(); self.frame_count = 0
        self.timer.start(0)
        self.append_log(f"Capture started ({self.cols}x{self.rows}).")

    def stop_stream(self):
        if self.timer.isActive(): self.timer.stop()
        if isinstance(self.cap, cv2.VideoCapture) and self.cap.isOpened(): self.cap.release()
        self.cap = None; self.is_image = False
        self.append_log("Capture stopped and resources released.")

    # ---------- Timer loop ----------
    def _on_timer(self):
        if self.cap is None or self.is_image: return
        ok, frame = self.cap.read()
        if not ok:
            self.append_log("End of stream or read failed. Stopping.", warn=True)
            self.stop_stream()
            return
        processed, cand_count, det_count, det_list = self.executor.submit(self.process_frame, frame).result()
        self.show_frame(processed)
        self._append_detections(det_list)
        self.append_log(f"Source: {self.current_source_label} | candidates: {cand_count} | accepted: {det_count}")

        self.frame_count += 1
        now = time.time()
        if (now - self.prev_time) >= 1.0:
            fps = self.frame_count / (now - self.prev_time)
            self.append_log(f"FPS: {fps:.2f}")
            self.prev_time = now; self.frame_count = 0

    # ---------- Core processing (worker thread safe) ----------
    def process_frame(self, frame):
        global thd
        rows, cols = frame.shape[:2]

        # Manual boost only (predictable behavior)
        mask = _make_mask(frame, erode_dilate=self.morph_enabled, boost=self.rain_boost_on)

        # Resolution-aware gating
        min_area = rows * cols * 0.0003            # ~0.03% of image area
        min_side_px = int(0.02 * min(rows, cols))  # >=2% of short side

        rects = contour_detect(mask, min_area=min_area)

        img_bbx = frame.copy()
        det_count = 0
        det_list = []
        # Keep candidate-wise scores for fallback
        all_candidates = []  # (prob, label, (x,y,w,h))

        for (x, y, w, h) in rects:
            if max(w, h) < min_side_px:
                continue

            # draw proposal
            cv2.rectangle(img_bbx, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # square crop
            xc, yc, size = int(x + w / 2), int(y + h / 2), max(w, h)
            x1 = max(0, int(xc - size / 2)); y1 = max(0, int(yc - size / 2))
            x2 = min(cols, int(xc + size / 2)); y2 = min(rows, int(yc + size / 2))
            crop = frame[y1:y2, x1:x2]

            classIndex, prob = predict_image(crop)
            label = getClassName(classIndex)
            all_candidates.append((prob, label, (x, y, w, h)))

            if self.show_debug_probs:
                cv2.putText(img_bbx, f"p={prob:.2f}", (x, max(0, y - 6)),
                            FONT, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

            # Accept if >= 45% confidence
            if prob >= thd:
                # Count "Priority road" as accepted but do not draw or record it
                if label == "Priority road":
                    det_count += 1
                    continue
                det_count += 1
                _draw_label_below_const(img_bbx, x, y, w, h, label, self.font_scale_12)
                det_list.append((label, prob, (x, y, w, h)))

        # ---- Guaranteed-accept fallback: ensure accepted >= 1 when we have candidates ----
        if det_count == 0 and len(all_candidates) > 0:
            # Choose best non-Priority if possible; else best overall
            best_non_priority = max(
                (c for c in all_candidates if c[1] != "Priority road"),
                default=None,
                key=lambda t: t[0]
            )
            best_overall = max(all_candidates, key=lambda t: t[0])

            chosen = best_non_priority if best_non_priority is not None else best_overall
            prob, label, (x, y, w, h) = chosen

            # Count it as accepted
            det_count = 1

            # Draw label only if not Priority road; otherwise just keep the red box already drawn
            if label != "Priority road":
                _draw_label_below_const(img_bbx, x, y, w, h, label, self.font_scale_12)
                det_list.append((label, prob, (x, y, w, h)))
            # If Priority road, we intentionally do not draw/add label per your rule

        # Subtle indicator when manual boost is enabled
        if self.rain_boost_on:
            cv2.putText(img_bbx, "BOOST", (10, 30), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return img_bbx, len(rects), det_count, det_list

    # ---------- Display / Logging / Detections ----------
    def show_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def append_log(self, text, warn=False):
        ts = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        prefix = "⚠️ " if warn else ""
        self.log.append(f"[{ts}] {prefix}{text}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _append_detections(self, det_list):
        if not det_list: return
        for label, prob, (x, y, w, h) in det_list:
            row = self.det_table.rowCount()
            self.det_table.insertRow(row)
            ts = QDateTime.currentDateTime().toString("hh:mm:ss.zzz")
            self.det_table.setItem(row, 0, QTableWidgetItem(ts))
            self.det_table.setItem(row, 1, QTableWidgetItem(label))
            self.det_table.setItem(row, 2, QTableWidgetItem(f"{prob*100:.2f}%"))
            self.det_table.setItem(row, 3, QTableWidgetItem(f"({x},{y},{w},{h})"))
        self.det_table.scrollToBottom()

    # ---------- Controls ----------
    def _clear_detections(self):
        self.det_table.setRowCount(0)

    def _on_threshold_changed(self, val):
        global thd
        thd = val / 100.0
        self.append_log(f"Threshold set to {thd:.2f}")

    def _on_morph_toggle(self, state):
        self.morph_enabled = (state == Qt.Checked)
        self.append_log(f"Erode/Dilate {'enabled' if self.morph_enabled else 'disabled'}.")

    def _on_rain_toggle(self, state):
        self.rain_boost_on = (state == Qt.Checked)
        self.append_log(f"Rain/Low-light boost {'ON' if self.rain_boost_on else 'OFF'}.")

    def _on_debug_toggle(self, state):
        self.show_debug_probs = (state == Qt.Checked)
        self.append_log(f"Debug probs {'ON' if self.show_debug_probs else 'OFF'}.")

    def closeEvent(self, event):
        try:
            self.stop_stream()
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        event.accept()

# -------------------- Entry --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoApp()
    win.show()
    sys.exit(app.exec_())
