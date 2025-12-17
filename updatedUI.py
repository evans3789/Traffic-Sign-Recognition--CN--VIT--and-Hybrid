import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
    QCheckBox, QMessageBox, QFrame, QTableWidget, QTableWidgetItem, QComboBox
)

# ------------------- GLOBALS -------------------
thd = 0.45
FONT = cv2.FONT_HERSHEY_SIMPLEX
base_path = os.path.dirname(os.path.abspath(__file__))

# ------------------- CLASS NAMES -------------------
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

# ------------------- LOAD MODELS -------------------
models = {}
for name, fname in [("CNN", "cnn_model.h5"), ("ViT", "vit_model.h5"), ("Hybrid", "hybrid_model.h5")]:
    path = os.path.join(base_path, fname)
    try:
        models[name] = tf.keras.models.load_model(path, compile=False)
        print(f"[INFO] Loaded {name} model from {path}")
    except Exception as e:
        models[name] = None
        print(f"[WARN] Failed to load {name} model: {e}")

current_model_name = "Automatic"

# ------------------- IMAGE PREPROCESSING -------------------
def preprocessing(img):
    """Stable 32x32 grayscale with light denoise + CLAHE (good in rain too)."""
    # Convert to grayscale only if image has 3 channels
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # already grayscale

    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return (gray / 255.0).astype('float32')


def predict_image(img, model_name):
    # ------- AUTOMATIC MODE -------
    if model_name == "Automatic":
        best_prob = 0
        best_idx = -1
        best_model_name = None

        for name, mdl in models.items():
            if mdl is None:
                continue

            input_channels = mdl.input_shape[-1]

            # Preprocess dynamically
            if input_channels == 3:
                rgb = cv2.resize(img, (32, 32))
                x = (rgb / 255.0).astype("float32").reshape(1, 32, 32, 3)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = preprocessing(gray)
                gray = cv2.resize(gray, (32, 32))
                x = gray.reshape(1, 32, 32, 1)

            p = mdl.predict(x, verbose=0)
            idx = int(np.argmax(p))
            prob = float(np.max(p))

            if prob > best_prob:
                best_prob = prob
                best_idx = idx
                best_model_name = name

        return best_idx, best_prob, best_model_name

    # ------- MANUAL MODEL MODE -------
    mdl = models.get(model_name)
    if mdl is None:
        return -1, 0.0, model_name

    input_channels = mdl.input_shape[-1]

    if input_channels == 3:
        rgb = cv2.resize(img, (32, 32))
        x = (rgb / 255.0).astype("float32").reshape(1, 32, 32, 3)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = preprocessing(gray)
        gray = cv2.resize(gray, (32, 32))
        x = gray.reshape(1, 32, 32, 1)

    p = mdl.predict(x, verbose=0)
    return int(np.argmax(p)), float(np.max(p)), model_name



# ------------------- MASK / CONTOUR -------------------
def _make_mask(imgBGR, erode_dilate=True, boost=False):
    hsv = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    s_floor, v_floor = (43, 46)
    if boost: s_floor, v_floor = (25,30)
    blue = cv2.inRange(hsv, (100,s_floor,v_floor), (124,255,255))
    red = cv2.inRange(hsv,(0,s_floor,v_floor),(10,255,255)) | cv2.inRange(hsv,(156,s_floor,v_floor),(180,255,255))
    mask = blue | red
    if boost:
        white = cv2.inRange(hsv,(0,0,180),(180,60,255))
        mask |= white
    if erode_dilate:
        k = np.ones((3,3),np.uint8)
        mask = cv2.erode(mask,k,iterations=1 if not boost else 0)
        mask = cv2.dilate(mask,k,iterations=1)
    return mask

def contour_detect(img_bin, min_area, max_area=-1, wh_ratio=2.0):
    rects=[]
    contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return rects
    max_area = img_bin.shape[0]*img_bin.shape[1] if max_area<0 else max_area
    for c in contours:
        area = cv2.contourArea(c)
        if not(min_area<=area<=max_area): continue
        x,y,w,h = cv2.boundingRect(c)
        if w==0 or h==0: continue
        if (w/h)<wh_ratio and (h/w)<wh_ratio: rects.append((x,y,w,h))
    return rects

# ------------------- DRAW LABEL -------------------
def _compute_font_scale_for_12px():
    target_px,test_scale=12,0.5
    (_,h),_=cv2.getTextSize("Ag", FONT, test_scale,1)
    return max(0.4,float(target_px)/max(1,h))

def _draw_label_below_const(img, x, y, w, h, text, font_scale, thickness=1):
    if not text:
        return
    # Compute text size
    (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, thickness)
    pad = 4
    # Draw label above the box if near bottom
    if y + h + th + 10 > img.shape[0]:
        y_label = max(0, y - 10)
    else:
        y_label = y + h + th + 5

    # Background rectangle
    cv2.rectangle(
        img,
        (x, y_label - th - pad),
        (x + tw + pad * 2, y_label + pad),
        (0, 0, 255),
        -1,
    )
    # Text
    cv2.putText(
        img,
        text,
        (x + pad, y_label),
        FONT,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ------------------- MAIN UI -------------------
class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Sign Detection UI")
        self.resize(1200,780)
        self.cap = None
        self.is_image = False
        self.prev_time = time.time()
        self.frame_count = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.morph_enabled = True
        self.rain_boost_on = False
        self.show_debug_probs = False
        self.current_source_label = "None"
        self._placeholder_pix = None
        self.font_scale_12 = _compute_font_scale_for_12px()
        self.current_model_name = current_model_name
        left_panel = self._build_left_controls()
        center_col = QWidget(); center_v = QVBoxLayout(center_col)
        # --- CREATE THREE MODEL TABLES ---
        self.cnn_table = self._make_det_table("cnn")
        self.vit_table = self._make_det_table("vit")
        self.hybrid_table = self._make_det_table("hybrid")

        center_v.setContentsMargins(0,0,0,0); center_v.setSpacing(6)
        self.video_label = QLabel(); self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color:#202225;color:#bbb;")
        center_v.addWidget(self.video_label,1)
        self._load_placeholder()
        row = QHBoxLayout()
        lbl = QLabel("Detections"); lbl.setStyleSheet("font-weight:bold;")
        row.addWidget(lbl); row.addStretch(1)
        btn_clear = QPushButton("Clear Detections"); btn_clear.clicked.connect(self._clear_detections)
        row.addWidget(btn_clear)
        center_v.addLayout(row)
        # --- THREE MODEL TABLES ---
        # ---- THREE MODEL TABLES WITH LABELS ----
        cnn_box = QVBoxLayout()
        cnn_label = QLabel("CNN")
        cnn_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        cnn_label.setAlignment(Qt.AlignCenter)
        cnn_box.addWidget(cnn_label)
        cnn_box.addWidget(self.cnn_table)

        vit_box = QVBoxLayout()
        vit_label = QLabel("ViT")
        vit_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        vit_label.setAlignment(Qt.AlignCenter)
        vit_box.addWidget(vit_label)
        vit_box.addWidget(self.vit_table)

        hybrid_box = QVBoxLayout()
        hybrid_label = QLabel("Hybrid")
        hybrid_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        hybrid_label.setAlignment(Qt.AlignCenter)
        hybrid_box.addWidget(hybrid_label)
        hybrid_box.addWidget(self.hybrid_table)

        table_row = QHBoxLayout()
        table_row.addLayout(cnn_box)
        table_row.addLayout(vit_box)
        table_row.addLayout(hybrid_box)

        center_v.addLayout(table_row)


        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setStyleSheet("background-color:#111;color:#ddd;")
        self.log.setMinimumWidth(300)
        root = QWidget(); root_h = QHBoxLayout(root)
        root_h.addLayout(left_panel,0)
        root_h.addWidget(self._vline())
        root_h.addWidget(center_col,1)
        root_h.addWidget(self._vline())
        root_h.addWidget(self.log,0)
        self.setCentralWidget(root)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.append_log("App started.")
        self.append_log("Models loaded: " + ", ".join([k for k,v in models.items() if v is not None]))
    
    def _make_det_table(self, title):
        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["Time", "Label", "Confidence", "Box (x,y,w,h)"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(180)
        table.setStyleSheet("font-size: 11px;")
        table.setObjectName(f"{title}_table")

        return table

    # UI helpers
    def _vline(self): l=QFrame(); l.setFrameShape(QFrame.VLine); l.setFrameShadow(QFrame.Sunken); return l
    def _hline(self): l=QFrame(); l.setFrameShape(QFrame.HLine); l.setFrameShadow(QFrame.Sunken); return l
    def _build_left_controls(self):
        layout=QVBoxLayout(); layout.setContentsMargins(8,8,8,8); layout.setSpacing(10)
        btn_camera=QPushButton("Use Camera"); btn_camera.clicked.connect(self.use_camera)
        cam_form=QFormLayout(); self.cam_index=QSpinBox(); self.cam_index.setRange(0,10); self.cam_index.setValue(0); cam_form.addRow("Camera Index:",self.cam_index)
        btn_open=QPushButton("Open Image/Video"); btn_open.clicked.connect(self.open_file)
        self.th_slider=QSlider(Qt.Horizontal); self.th_slider.setRange(0,100); self.th_slider.setValue(int(thd*100))
        self.th_slider.valueChanged.connect(self._on_threshold_changed)
        self.th_spin=QSpinBox(); self.th_spin.setRange(0,100); self.th_spin.setValue(int(thd*100))
        self.th_spin.valueChanged.connect(self._on_threshold_changed)
        self.th_slider.valueChanged.connect(self.th_spin.setValue); self.th_spin.valueChanged.connect(self.th_slider.setValue)
        th_form=QFormLayout(); th_form.addRow("Confidence %:",self.th_spin)
        self.chk_morph=QCheckBox("Erode/Dilate (Noise cleanup)"); self.chk_morph.setChecked(True); self.chk_morph.stateChanged.connect(self._on_morph_toggle)
        self.chk_rain=QCheckBox("Rain / Low-light boost (manual)"); self.chk_rain.setChecked(False); self.chk_rain.stateChanged.connect(self._on_rain_toggle)
        self.chk_debug=QCheckBox("Show debug probs (p=...)"); self.chk_debug.setChecked(False); self.chk_debug.stateChanged.connect(self._on_debug_toggle)
        self.model_combo=QComboBox(); self.model_combo.addItems(["Automatic"]+list(models.keys())); self.model_combo.currentTextChanged.connect(self._on_model_change)
        btn_stop=QPushButton("Stop / Release"); btn_stop.clicked.connect(self.stop_stream)
        btn_quit=QPushButton("Quit"); btn_quit.setStyleSheet("background-color:#b00;color:white;font-weight:bold;"); btn_quit.clicked.connect(self.close)
        layout.addWidget(btn_camera)
        layout.addLayout(cam_form)
        layout.addWidget(btn_open)
        layout.addWidget(self.model_combo)
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

    def _load_placeholder(self):
        p=os.path.join(base_path,"placeholder.jpg")
        if os.path.exists(p):
            self._placeholder_pix=QPixmap(p)
            self._apply_placeholder_scale()
        else:
            self.video_label.setText("Video / Image will appear here")
    def _apply_placeholder_scale(self):
        if self._placeholder_pix:
            self.video_label.setPixmap(self._placeholder_pix.scaled(self.video_label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def resizeEvent(self,event):
        super().resizeEvent(event)
        pix=self.video_label.pixmap()
        if pix:
            self.video_label.setPixmap(pix.scaled(self.video_label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        elif self._placeholder_pix:
            self._apply_placeholder_scale()

    def use_camera(self):
        idx=self.cam_index.value(); self.start_capture(idx); self.current_source_label=f"Camera[{idx}]"; self.append_log(f"Using camera {idx}")
    
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image or video", "",
            "Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()

        # ---------- Image ----------
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.stop_stream()
            img = cv2.imread(path)
            if img is None:
                QMessageBox.warning(self, "Error", "Failed to load image.")
                return
            self.is_image = True
            self.current_source_label = os.path.basename(path)
            processed, cand_count, det_count, det_list = self.process_frame(img)
            self.show_frame(processed)
            self._append_detections(det_list)
            self.append_log(
                f"Source: {self.current_source_label} | candidates: {cand_count} | accepted: {det_count}"
            )

    # ---------- Video ----------
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            self.start_capture(path)
            self.is_image = False
            self.current_source_label = os.path.basename(path)
            self.append_log(f"Opened video file: {path}")

        else:
            QMessageBox.warning(self, "Unsupported", f"Unsupported file type: {ext}")


    def start_capture(self, source):
        self.stop_stream()
        self.is_image = False  # Ensure not in image mode
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            self.cap = None
            QMessageBox.warning(self, "Error", "Failed to open camera/video.")
            self.append_log("Failed to open source.", warn=True)
            return

        self.cols = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.rows = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.prev_time = time.time()
        self.frame_count = 0

        # Start the timer
        self.timer.start(30)  # 30ms ~ 33fps
        self.append_log(f"Capture started ({self.cols}x{self.rows}).")

    def stop_stream(self):
        if self.timer.isActive():
            self.timer.stop()
        if isinstance(self.cap, cv2.VideoCapture) and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.is_image = False
        self.append_log("Capture stopped and resources released.")


    def _on_timer(self):
        if self.cap is None or self.is_image:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.append_log("End of video or camera feed.")
            self.stop_stream()
            return

        processed, cand_count, det_count, det_list = self.process_frame(frame)
        self.show_frame(processed)

        if det_list:
            self._append_detections(det_list)

        self.frame_count += 1

        self.frame_count+=1

    def process_frame(self,frame):
        img_disp=frame.copy()
        mask=_make_mask(frame,self.morph_enabled,self.rain_boost_on)
        rects=contour_detect(mask,50)
        detections=[]
        for x, y, w, h in rects:
            roi = frame[y:y+h, x:x+w]
            # idx, prob = predict_image(roi, self.current_model_name)
            idx, prob, used_model = predict_image(roi, self.current_model_name)

            if prob < thd:
                continue

            label = getClassName(idx)
            cv2.rectangle(img_disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            _draw_label_below_const(img_disp, x, y, w, h, f"{label} ({prob:.2f})", self.font_scale_12)
            # detections.append((time.strftime("%H:%M:%S"), label, prob, (x, y, w, h)))
            detections.append((time.strftime("%H:%M:%S"), label, prob, (x, y, w, h), used_model))

        
        
        return img_disp,len(rects),len(detections),detections

    def show_frame(self, frame):
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _append_detections(self, det_list):
        for t, l, p, b, model_used in det_list:
            # Pick table based on model
            if model_used == "CNN":
                table = self.cnn_table
            elif model_used == "ViT":
                table = self.vit_table
            elif model_used == "Hybrid":
                table = self.hybrid_table
            else:
                continue

            row = table.rowCount()
            table.insertRow(row)

            table.setItem(row, 0, QTableWidgetItem(t))
            table.setItem(row, 1, QTableWidgetItem(l))
            table.setItem(row, 2, QTableWidgetItem(f"{p:.2f}"))
            table.setItem(row, 3, QTableWidgetItem(str(b)))

            table.scrollToBottom()

            # Log
            self.append_log(f"[{model_used}] Detected: {l} ({p:.2f}) at {b}")

        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    # Keep UI responsive during updates
    QApplication.processEvents()


    def _clear_detections(self):
        self.cnn_table.setRowCount(0)
        self.vit_table.setRowCount(0)
        self.hybrid_table.setRowCount(0)
        self.append_log("All detection tables cleared.")

    def _on_threshold_changed(self,val):
        global thd; thd=self.th_slider.value()/100.0
    def _on_morph_toggle(self,state): self.morph_enabled=(state>0)
    def _on_rain_toggle(self,state): self.rain_boost_on=(state>0)
    def _on_debug_toggle(self,state): self.show_debug_probs=(state>0)
    def _on_model_change(self,text): self.current_model_name=text; self.append_log(f"Model selected: {text}")

    def append_log(self,msg,warn=False): self.log.append(f"[{QDateTime.currentDateTime().toString('HH:mm:ss')}] {'WARN: ' if warn else ''}{msg}")

# ------------------- MAIN -------------------
if __name__=="__main__":
    app=QApplication(sys.argv)
    win=VideoApp(); win.show()
    sys.exit(app.exec_())
