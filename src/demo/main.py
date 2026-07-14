"""main.py — Demo Entry Point"""
import sys
import os
import argparse

# torch must be imported before PyQt5, otherwise CUDA DLL loading sequence conflicts will occur on Windows
import torch  # noqa: F401

# Ensure src/demo is included in the system paths
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

# Fix issue where PyQt5 cannot resolve platform plugins on Windows environments
import PyQt5
_qt_plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
if os.path.isdir(_qt_plugins):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_plugins

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from app import MainWindow


def main():
    parser = argparse.ArgumentParser(description="Tennis Action Recognition Demo")
    parser.add_argument("--rally",   default="", help="Path directory to Rally target files")
    parser.add_argument("--config",  default="", help="Path to config YAML configurations")
    parser.add_argument("--weights", default="", help="Path to model checkpoint weights .pth")
    parser.add_argument("--person",  default="", help="Path to person detection YOLO model .pt")
    parser.add_argument("--pose",    default="", help="Path to pose estimation YOLO model .pt")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))

    win = MainWindow(
        rally_dir=args.rally   or None,
        config_path=args.config or None,
        weights_path=args.weights or None,
        person_model=args.person or None,
        pose_model=args.pose   or None,
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()