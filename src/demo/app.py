"""app.py — Main Window"""
import os
import json
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog,
    QProgressBar, QSizePolicy, QFrame, QLineEdit,
)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

from player import VideoPlayer
from timeline import TimelinePanel, ACTION_NAMES, ACTION_COLORS
from inference import InferenceThread

_FONT = QFont("Microsoft YaHei", 10)
_FONT_SMALL = QFont("Microsoft YaHei", 9)
_FONT_MONO = QFont("Consolas", 10)

_PREFS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_session.json")


def _load_prefs():
    try:
        with open(_PREFS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_prefs(data: dict):
    try:
        with open(_PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _btn(text, min_w=80):
    b = QPushButton(text)
    b.setFont(_FONT)
    b.setMinimumWidth(min_w)
    b.setFixedHeight(32)
    return b


def _label(text, font=None, color="#CCCCCC"):
    l = QLabel(text)
    l.setFont(font or _FONT)
    l.setStyleSheet(f"color: {color};")
    return l


STYLESHEET = """
QMainWindow, QWidget {
    background-color: #12121C;
    color: #CCCCCC;
}
QPushButton {
    background-color: #2A2A3E;
    color: #CCCCCC;
    border: 1px solid #3A3A5A;
    border-radius: 4px;
    padding: 4px 12px;
}
QPushButton:hover { background-color: #3A3A5A; }
QPushButton:pressed { background-color: #4A4A6A; }
QPushButton:disabled { color: #555; border-color: #2A2A3E; }
QSlider::groove:horizontal {
    height: 4px;
    background: #2A2A3E;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 12px; height: 12px;
    margin: -4px 0;
    background: #7C7CFF;
    border-radius: 6px;
}
QSlider::sub-page:horizontal { background: #5555CC; border-radius: 2px; }
QProgressBar {
    background: #2A2A3E;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: #888;
}
QProgressBar::chunk { background: #5555CC; border-radius: 3px; }
QLineEdit {
    background: #1E1E2E;
    border: 1px solid #3A3A5A;
    border-radius: 4px;
    color: #CCCCCC;
    padding: 2px 6px;
}
QScrollBar:horizontal {
    height: 8px;
    background: #1E1E2E;
}
QScrollBar::handle:horizontal {
    background: #3A3A5A;
    border-radius: 4px;
    min-width: 20px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QToolTip {
    background-color: #2A2A3E;
    color: #CCCCCC;
    border: 1px solid #5555CC;
    font-family: "Microsoft YaHei";
    font-size: 9pt;
}
"""


class MainWindow(QMainWindow):
    def __init__(self, rally_dir=None, config_path=None, weights_path=None,
                 person_model=None, pose_model=None):
        super().__init__()
        self.setWindowTitle("Tennis Action Recognition Demo")
        self.setMinimumSize(1100, 760)
        self.setStyleSheet(STYLESHEET)

        prefs = _load_prefs()
        self._rally_dir    = rally_dir    or prefs.get("rally_dir",    "")
        self._config_path  = config_path  or prefs.get("config_path",  "")
        self._weights_path = weights_path or prefs.get("weights_path", "")
        self._person_model = person_model or prefs.get("person_model", "")
        self._pose_model   = pose_model   or prefs.get("pose_model",   "")
        self._inference_thread = None
        self._result = None

        self._player = VideoPlayer(self)
        self._player.frame_changed.connect(self._on_frame)
        self._player.finished.connect(self._on_video_end)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Video Frame Viewport
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet("background: #000; border-radius: 6px;")
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setMinimumHeight(360)
        root.addWidget(self._video_label, stretch=1)

        # Loading Spinner Overlay Cover Layer
        self._loading_label = QLabel(self._video_label)
        self._loading_label.setAlignment(Qt.AlignCenter)
        self._loading_label.setStyleSheet(
            "background: #000; color: #888; font-size: 18px; border-radius: 6px;"
        )
        self._loading_label.setFont(QFont("Microsoft YaHei", 14))
        self._loading_label.hide()
        self._loading_dots = 0
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(400)
        self._loading_timer.timeout.connect(self._tick_loading)

        # Timeline Component Panel
        self._timeline = TimelinePanel()
        root.addWidget(self._timeline)

        # Progress Slider Bar
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.sliderPressed.connect(self._player.pause)
        self._slider.sliderMoved.connect(self._player.seek)
        root.addWidget(self._slider)

        # Playback Control Interface Bar
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        self._btn_prev = _btn("◀ Prev Frame", 110)
        self._btn_play = _btn("▶ Play", 90)
        self._btn_next = _btn("Next Frame ▶", 110)
        self._lbl_time = _label("00:00.000 / 00:00.000", _FONT_MONO, "#888")
        self._lbl_action = _label("—", _FONT, "#CCCCCC")
        self._lbl_action.setMinimumWidth(120)

        self._btn_prev.clicked.connect(lambda: self._player.step(-1))
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_next.clicked.connect(lambda: self._player.step(1))

        ctrl.addWidget(self._btn_prev)
        ctrl.addWidget(self._btn_play)
        ctrl.addWidget(self._btn_next)
        ctrl.addSpacing(16)
        ctrl.addWidget(self._lbl_time)
        ctrl.addSpacing(16)
        ctrl.addWidget(_label("Current Prediction: ", _FONT_SMALL, "#888"))
        ctrl.addWidget(self._lbl_action)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # Horizontal Divider Line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #2A2A3E;")
        root.addWidget(line)

        # Directory File Picker Bar
        file_row = QHBoxLayout()
        file_row.setSpacing(6)

        self._edit_rally = QLineEdit(self._rally_dir)
        self._edit_rally.setPlaceholderText("Rally Directory Path")
        self._edit_rally.setFont(_FONT_SMALL)
        btn_rally = _btn("Select Rally", 100)
        btn_rally.clicked.connect(self._pick_rally)

        self._edit_cfg = QLineEdit(self._config_path)
        self._edit_cfg.setPlaceholderText("Config YAML Path")
        self._edit_cfg.setFont(_FONT_SMALL)
        btn_cfg = _btn("Select Config", 110)
        btn_cfg.clicked.connect(self._pick_config)

        self._edit_weights = QLineEdit(self._weights_path)
        self._edit_weights.setPlaceholderText("Weights .pth Path")
        self._edit_weights.setFont(_FONT_SMALL)
        btn_w = _btn("Select Weights", 120)
        btn_w.clicked.connect(self._pick_weights)

        self._btn_infer = _btn("▶ Run Inference", 120)
        self._btn_infer.setStyleSheet(
            "QPushButton { background: #3A3A8A; border-color: #5555CC; }"
            "QPushButton:hover { background: #4A4AAA; }"
        )
        self._btn_infer.clicked.connect(self._start_inference)

        file_row.addWidget(self._edit_rally, 3)
        file_row.addWidget(btn_rally)
        file_row.addWidget(self._edit_cfg, 2)
        file_row.addWidget(btn_cfg)
        file_row.addWidget(self._edit_weights, 2)
        file_row.addWidget(btn_w)
        file_row.addWidget(self._btn_infer)
        root.addLayout(file_row)

        # Optional YOLO Weights Selection Row
        yolo_row = QHBoxLayout()
        yolo_row.setSpacing(6)

        self._edit_person = QLineEdit(self._person_model)
        self._edit_person.setPlaceholderText("person detection model .pt (Optional, falls back to pose_data.json if empty)")
        self._edit_person.setFont(_FONT_SMALL)
        btn_person = _btn("Select Person", 110)
        btn_person.clicked.connect(self._pick_person)

        self._edit_pose = QLineEdit(self._pose_model)
        self._edit_pose.setPlaceholderText("pose estimation model .pt (Optional)")
        self._edit_pose.setFont(_FONT_SMALL)
        btn_pose = _btn("Select Pose", 100)
        btn_pose.clicked.connect(self._pick_pose)

        yolo_row.addWidget(_label("YOLO: ", _FONT_SMALL, "#888"))
        yolo_row.addWidget(self._edit_person, 3)
        yolo_row.addWidget(btn_person)
        yolo_row.addWidget(self._edit_pose, 3)
        yolo_row.addWidget(btn_pose)
        yolo_row.addStretch()
        root.addLayout(yolo_row)

        # Thread Performance Status and Progress Bars
        status_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._lbl_status = _label("Ready", _FONT_SMALL, "#666")
        status_row.addWidget(self._progress, 1)
        status_row.addSpacing(8)
        status_row.addWidget(self._lbl_status)
        root.addLayout(status_row)

        # Color Legends
        legend = QHBoxLayout()
        legend.addWidget(_label("Legend: ", _FONT_SMALL, "#666"))
        for i, name in enumerate(ACTION_NAMES):
            c = ACTION_COLORS[i].name()
            dot = QLabel("■")
            dot.setStyleSheet(f"color: {c}; font-size: 14px;")
            legend.addWidget(dot)
            legend.addWidget(_label(name, _FONT_SMALL, "#AAA"))
            legend.addSpacing(8)
        legend.addStretch()
        root.addLayout(legend)

        # Automatically spin up clip file loaders if arguments were parsed
        if self._rally_dir:
            self._load_video()

    # ── File Dialog Selectors ─────────────────────────────────

    def _pick_rally(self):
        d = QFileDialog.getExistingDirectory(self, "Select Rally Directory", self._rally_dir or ".")
        if d:
            self._rally_dir = d
            self._edit_rally.setText(d)
            self._load_video()

    def _pick_config(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Configuration Profile", ".", "YAML (*.yaml *.yml)")
        if f:
            self._config_path = f
            self._edit_cfg.setText(f)

    def _pick_weights(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Weight Matrix File", ".", "PyTorch (*.pth *.pt)")
        if f:
            self._weights_path = f
            self._edit_weights.setText(f)

    def _pick_person(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Person Detection Layer", ".", "YOLO (*.pt)")
        if f:
            self._person_model = f
            self._edit_person.setText(f)

    def _pick_pose(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Pose Estimation Layer", ".", "YOLO (*.pt)")
        if f:
            self._pose_model = f
            self._edit_pose.setText(f)

    # ── Video Loading Handlers ────────────────────────────────

    def _load_video(self):
        video_path = os.path.join(self._rally_dir, "raw_clip.mp4")
        if not os.path.exists(video_path):
            self._lbl_status.setText("Missing raw_clip.mp4 targets")
            return
        ok = self._player.load(video_path)
        if not ok:
            self._lbl_status.setText("Failed to safely stream target clip matrix")
            return
        total = self._player.total_frames
        self._slider.setRange(0, max(total - 1, 0))
        self._lbl_status.setText(f"Loaded clip: {total} frames @ {self._player.fps:.1f} fps")
        self._player.seek(0)

        # Ingest Ground Truth structural tags
        anno_path = os.path.join(self._rally_dir, "annotations.json")
        if os.path.exists(anno_path):
            with open(anno_path, "r", encoding="utf-8") as f:
                anno = json.load(f)
            self._timeline.load_gt(total, self._player.fps, anno)

    # ── Playback State Logic ──────────────────────────────────

    def _toggle_play(self):
        self._player.toggle()
        self._btn_play.setText("⏸ Pause" if self._player._playing else "▶ Play")

    def _on_frame(self, frame_idx, rgb):
        # Override baseline matrix loops with tracked arrays upon inference success
        if (self._result and "annotated_rgb" in self._result
                and frame_idx < len(self._result["annotated_rgb"])):
            rgb = self._result["annotated_rgb"][frame_idx]

        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        label_size = self._video_label.size()
        self._video_label.setPixmap(
            pix.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        # Synchronize scrub bars and time stamps
        self._slider.blockSignals(True)
        self._slider.setValue(frame_idx)
        self._slider.blockSignals(False)

        fps = self._player.fps
        total = self._player.total_frames
        cur_s = frame_idx / fps
        tot_s = total / fps
        self._lbl_time.setText(
            f"{int(cur_s//60):02d}:{cur_s%60:06.3f} / {int(tot_s//60):02d}:{tot_s%60:06.3f}"
        )

        # Shift timeline pointers
        self._timeline.set_cursor(frame_idx)

        # Direct current classification parameters into layout fields
        if self._result and frame_idx < len(self._result["per_frame_preds"]):
            pid = self._result["per_frame_preds"][frame_idx]
            name = ACTION_NAMES[pid] if 0 <= pid < len(ACTION_NAMES) else "Unknown"
            color = ACTION_COLORS[pid].name() if 0 <= pid < len(ACTION_COLORS) else "#888"
            self._lbl_action.setText(name)
            self._lbl_action.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _on_video_end(self):
        self._btn_play.setText("▶ Play")

    # ── Inference Execution ───────────────────────────────────

    def _start_inference(self):
        rally = self._edit_rally.text().strip()
        cfg = self._edit_cfg.text().strip()
        weights = self._edit_weights.text().strip()

        if not rally or not os.path.isdir(rally):
            self._lbl_status.setText("Please select a valid Rally path directory link")
            return
        if not cfg or not os.path.exists(cfg):
            self._lbl_status.setText("Please point to valid YAML parameters profiles")
            return
        if not weights or not os.path.exists(weights):
            self._lbl_status.setText("Please point to valid target checkpoint parameters .pth")
            return

        self._btn_infer.setEnabled(False)
        self._btn_play.setEnabled(False)
        self._progress.setValue(0)

        # Drop viewports into blacked overlays and wipe historical data bars
        self._player.pause()
        self._btn_play.setText("▶ Play")
        self._show_loading()
        self._timeline.reset_predictions()

        person = self._edit_person.text().strip()
        pose   = self._edit_pose.text().strip()

        if person and pose:
            self._lbl_status.setText("Inference executing (YOLO live detection streams)...")
        else:
            self._lbl_status.setText("Inference executing (Parsing cached pose_data.json layers)...")

        self._inference_thread = InferenceThread(
            rally, cfg, weights,
            person_model_path=person or None,
            pose_model_path=pose or None,
            parent=self,
        )
        self._inference_thread.progress.connect(self._on_infer_progress)
        self._inference_thread.result.connect(self._on_infer_result)
        self._inference_thread.error.connect(self._on_infer_error)
        self._inference_thread.start()

    def _on_infer_progress(self, pct):
        self._progress.setValue(pct)
        self._lbl_status.setText(f"Processing matrices... {pct}%")

    def _on_infer_result(self, result):
        self._result = result
        self._hide_loading()
        self._btn_infer.setEnabled(True)
        self._btn_play.setEnabled(True)
        self._progress.setValue(100)
        acc = result["accuracy"] * 100
        self._lbl_status.setText(f"Inference verified successfully — Model Accuracy Score: {acc:.1f}%")

        total = result["total_frames"]
        fps = result["fps"]
        self._timeline.load_predictions(total, fps, result["per_frame_preds"])

    def _on_infer_error(self, msg):
        self._hide_loading()
        self._btn_infer.setEnabled(True)
        self._btn_play.setEnabled(True)
        self._lbl_status.setText("Inference framework broken down. Inspect stdout terminal outputs.")
        print("[INFERENCE ERROR]\n", msg)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._loading_label.setGeometry(self._video_label.geometry())

    def _tick_loading(self):
        dots = "●" * (self._loading_dots % 4 + 1) + "○" * (3 - self._loading_dots % 4)
        self._loading_label.setText(f"Loading matrices  {dots}")
        self._loading_dots += 1

    def _show_loading(self):
        self._loading_label.setGeometry(self._video_label.geometry())
        self._loading_label.raise_()
        self._loading_label.show()
        self._loading_dots = 0
        self._loading_timer.start()

    def _hide_loading(self):
        self._loading_timer.stop()
        self._loading_label.hide()

    def closeEvent(self, event):
        _save_prefs({
            "rally_dir":    self._edit_rally.text().strip(),
            "config_path":  self._edit_cfg.text().strip(),
            "weights_path": self._edit_weights.text().strip(),
            "person_model": self._edit_person.text().strip(),
            "pose_model":   self._edit_pose.text().strip(),
        })
        self._player.release()
        if self._inference_thread and self._inference_thread.isRunning():
            self._inference_thread.terminate()
        super().closeEvent(event)