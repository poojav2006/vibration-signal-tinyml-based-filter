# dashboard_ui.py
# Layout and styling for the Intelligent Vibration Monitor dashboard.
# Built with PyQt6 with 30-mark academic innovation panels including Accuracy Before/After.

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QSplitter
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from plots import WaveformPlotWidget, FFTPlotWidget

DARK_STYLESHEET = """
QMainWindow {
    background-color: #0d1117;
}
QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}
QGroupBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: bold;
    color: #58a6ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding-left: 8px;
    padding-right: 8px;
}
QLabel {
    color: #c9d1d9;
}
QPushButton {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 5px;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
}
QPushButton:pressed {
    background-color: #1f6feb;
    color: #ffffff;
}
QPushButton#btn_start {
    background-color: #238636;
    color: #ffffff;
    border: 1px solid #2ea043;
}
QPushButton#btn_start:hover {
    background-color: #2ea043;
}
QPushButton#btn_stop {
    background-color: #da3633;
    color: #ffffff;
    border: 1px solid #f85149;
}
QPushButton#btn_stop:hover {
    background-color: #f85149;
}
QPushButton#btn_eval_demo {
    background-color: #8957e5;
    color: #ffffff;
    border: 1px solid #a371f7;
    font-weight: bold;
}
QPushButton#btn_eval_demo:hover {
    background-color: #a371f7;
}
QComboBox {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 8px;
    color: #c9d1d9;
}
QSlider::groove:horizontal {
    border: 1px solid #30363d;
    height: 6px;
    background: #21262d;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #1f6feb;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #58a6ff;
    border: 1px solid #58a6ff;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
QTableWidget {
    background-color: #0d1117;
    border: 1px solid #30363d;
    gridline-color: #21262d;
    color: #c9d1d9;
}
QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    padding: 4px;
    border: 1px solid #30363d;
    font-weight: bold;
}
"""


class DashboardUI:
    def setup_ui(self, main_window: QMainWindow):
        main_window.setWindowTitle("INTELLIGENT REAL-TIME VIBRATION MONITOR - TinyML Adaptive Filtering")
        main_window.resize(1420, 940)
        main_window.setStyleSheet(DARK_STYLESHEET)
        
        central_widget = QWidget(main_window)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # 1. Top Status Header
        top_header = self._create_top_header()
        main_layout.addWidget(top_header)
        
        # 2. Main Middle Area (Splitter)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        
        self.waveform_widget = WaveformPlotWidget(max_samples=256)
        self.fft_widget = FFTPlotWidget()
        left_layout.addWidget(self.waveform_widget, stretch=3)
        left_layout.addWidget(self.fft_widget, stretch=2)
        
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        
        intel_group = self._create_intelligence_panel()
        right_layout.addWidget(intel_group)
        
        innovations_group = self._create_innovations_panel()
        right_layout.addWidget(innovations_group)
        
        timeline_group = self._create_timeline_panel()
        right_layout.addWidget(timeline_group)
        
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([880, 540])
        main_layout.addWidget(splitter, stretch=1)
        
        control_bar = self._create_control_panel()
        main_layout.addWidget(control_bar)
        
        main_window.setCentralWidget(central_widget)

    def _create_top_header(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 4px;")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 4, 10, 4)
        
        title = QLabel("INTELLIGENT REAL-TIME VIBRATION MONITOR")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #58a6ff;")
        layout.addWidget(title)
        layout.addStretch(1)
        
        self.lbl_badge_system = QLabel("SYSTEM: RUNNING")
        self.lbl_badge_system.setStyleSheet("background-color: #238636; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;")
        layout.addWidget(self.lbl_badge_system)
        
        self.lbl_badge_mode = QLabel("MODE: SOFTWARE TEST")
        self.lbl_badge_mode.setStyleSheet("background-color: #1f6feb; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;")
        layout.addWidget(self.lbl_badge_mode)
        
        self.lbl_badge_fs = QLabel("SAMPLING: 100 Hz")
        self.lbl_badge_fs.setStyleSheet("background-color: #30363d; color: #c9d1d9; padding: 3px 8px; border-radius: 4px; font-weight: bold;")
        layout.addWidget(self.lbl_badge_fs)
        
        self.lbl_badge_ml = QLabel("ML: ACTIVE")
        self.lbl_badge_ml.setStyleSheet("background-color: #8957e5; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;")
        layout.addWidget(self.lbl_badge_ml)
        
        self.lbl_badge_flt = QLabel("FILTER: ACTIVE")
        self.lbl_badge_flt.setStyleSheet("background-color: #00ff88; color: black; padding: 3px 8px; border-radius: 4px; font-weight: bold;")
        layout.addWidget(self.lbl_badge_flt)
        
        return frame

    def _create_intelligence_panel(self) -> QGroupBox:
        group = QGroupBox("TinyML Disturbance Intelligence & Adaptive Filter Selection")
        layout = QGridLayout(group)
        layout.setSpacing(5)
        
        layout.addWidget(QLabel("Detected Disturbance:"), 0, 0)
        self.lbl_detected_class = QLabel("Class 1: 8-12 Hz Tonal")
        self.lbl_detected_class.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.lbl_detected_class.setStyleSheet("color: #00f0ff;")
        layout.addWidget(self.lbl_detected_class, 0, 1)
        
        layout.addWidget(QLabel("Dominant Peak:"), 1, 0)
        self.lbl_dominant_freq = QLabel("10.0 Hz")
        self.lbl_dominant_freq.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_dominant_freq.setStyleSheet("color: #ff007f;")
        layout.addWidget(self.lbl_dominant_freq, 1, 1)
        
        layout.addWidget(QLabel("Active Causal Filter:"), 2, 0)
        self.lbl_active_filter = QLabel("Causal 2nd-Order Notch")
        self.lbl_active_filter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_active_filter.setStyleSheet("color: #00ff88;")
        layout.addWidget(self.lbl_active_filter, 2, 1)
        
        layout.addWidget(QLabel("Filter Parameter:"), 3, 0)
        self.lbl_filter_param = QLabel("fc = 10.0 Hz (Q = 8.0)")
        self.lbl_filter_param.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.lbl_filter_param, 3, 1)
        
        layout.addWidget(QLabel("Edge Model Rules:"), 4, 0)
        self.lbl_model_info = QLabel("Shallow Decision Tree (0 bytes malloc, 100% C++ Parity)")
        self.lbl_model_info.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.lbl_model_info, 4, 1)
        
        return group

    def _create_innovations_panel(self) -> QGroupBox:
        group = QGroupBox("Signal Accuracy (Before vs After) & ISO 10816 Machinery Health")
        layout = QGridLayout(group)
        layout.setSpacing(6)
        
        # Accuracy Before vs After Column
        layout.addWidget(QLabel("<b>Waveform Accuracy:</b>"), 0, 0, 1, 2)
        
        layout.addWidget(QLabel("Accuracy BEFORE Filtering:"), 1, 0)
        self.lbl_acc_before = QLabel("73.86%")
        self.lbl_acc_before.setStyleSheet("color: #e6edf3; font-weight: bold;")
        layout.addWidget(self.lbl_acc_before, 1, 1)
        
        layout.addWidget(QLabel("Accuracy AFTER Filtering:"), 2, 0)
        self.lbl_acc_after = QLabel("85.61%")
        self.lbl_acc_after.setStyleSheet("color: #00ff88; font-weight: bold;")
        layout.addWidget(self.lbl_acc_after, 2, 1)
        
        layout.addWidget(QLabel("Net Accuracy Gain:"), 3, 0)
        self.lbl_acc_gain = QLabel("+11.75%")
        self.lbl_acc_gain.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.lbl_acc_gain.setStyleSheet("color: #00ff88;")
        layout.addWidget(self.lbl_acc_gain, 3, 1)
        
        # A/B & ISO 10816 Column
        layout.addWidget(QLabel("<b>A/B Benchmark & ISO Health:</b>"), 0, 2, 1, 2)
        
        layout.addWidget(QLabel("TinyML Adaptive SNR Gain:"), 1, 2)
        self.lbl_adaptive_snr_gain = QLabel("+0.00 dB")
        self.lbl_adaptive_snr_gain.setStyleSheet("color: #00ff88; font-weight: bold;")
        layout.addWidget(self.lbl_adaptive_snr_gain, 1, 3)
        
        layout.addWidget(QLabel("Static Fixed Filter Gain:"), 2, 2)
        self.lbl_static_snr_gain = QLabel("+0.00 dB")
        self.lbl_static_snr_gain.setStyleSheet("color: #ff8c00; font-weight: bold;")
        layout.addWidget(self.lbl_static_snr_gain, 2, 3)
        
        layout.addWidget(QLabel("ISO 10816 Machine State:"), 3, 2)
        self.lbl_iso_filtered = QLabel("Zone A (Normal / Safe)")
        self.lbl_iso_filtered.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_iso_filtered.setStyleSheet("color: #00ff88;")
        layout.addWidget(self.lbl_iso_filtered, 3, 3)
        
        return group

    def _create_timeline_panel(self) -> QGroupBox:
        group = QGroupBox("Adaptive Decisions & Filtered Signal Features")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 10, 6, 6)
        
        self.table_timeline = QTableWidget(0, 5)
        self.table_timeline.setHorizontalHeaderLabels(["Time", "Noise Type", "Selected Filter", "SNR Gain", "Accuracy Gain"])
        self.table_timeline.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_timeline.setAlternatingRowColors(True)
        self.table_timeline.verticalHeader().setVisible(False)
        self.table_timeline.setFixedHeight(120)
        layout.addWidget(self.table_timeline)
        
        features_frame = QFrame()
        features_frame.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 4px;")
        feat_layout = QGridLayout(features_frame)
        feat_layout.setContentsMargins(8, 4, 8, 4)
        feat_layout.setSpacing(6)
        
        title_lbl = QLabel("<b>FILTERED SIGNAL FEATURES (Live 128-Sample Window):</b>")
        title_lbl.setStyleSheet("color: #58a6ff; font-weight: bold;")
        feat_layout.addWidget(title_lbl, 0, 0, 1, 3)
        
        self.lbl_filt_mean = QLabel("Mean: <b>--</b>")
        self.lbl_filt_mean.setStyleSheet("color: #c9d1d9;")
        feat_layout.addWidget(self.lbl_filt_mean, 1, 0)
        
        self.lbl_filt_std = QLabel("Std Dev: <b>--</b>")
        self.lbl_filt_std.setStyleSheet("color: #c9d1d9;")
        feat_layout.addWidget(self.lbl_filt_std, 1, 1)
        
        self.lbl_filt_rms = QLabel("RMS: <b>--</b>")
        self.lbl_filt_rms.setStyleSheet("color: #00ff88; font-weight: bold;")
        feat_layout.addWidget(self.lbl_filt_rms, 1, 2)
        
        self.lbl_filt_var = QLabel("Variance: <b>--</b>")
        self.lbl_filt_var.setStyleSheet("color: #c9d1d9;")
        feat_layout.addWidget(self.lbl_filt_var, 2, 0)
        
        self.lbl_filt_dom_freq = QLabel("Dom Freq: <b>--</b>")
        self.lbl_filt_dom_freq.setStyleSheet("color: #d2a8ff; font-weight: bold;")
        feat_layout.addWidget(self.lbl_filt_dom_freq, 2, 1)
        
        self.lbl_filt_status = QLabel("Status: <b>Filtered</b>")
        self.lbl_filt_status.setStyleSheet("color: #00ff88; font-weight: bold;")
        feat_layout.addWidget(self.lbl_filt_status, 2, 2)
        
        layout.addWidget(features_frame)
        return group

    def _create_control_panel(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 6px;")
        layout = QHBoxLayout(frame)
        layout.setSpacing(8)
        
        self.btn_start = QPushButton("START")
        self.btn_start.setObjectName("btn_start")
        self.btn_pause = QPushButton("PAUSE")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_reset = QPushButton("RESET")
        
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_reset)
        
        self.btn_eval_demo = QPushButton("RUN EVALUATOR DEMO")
        self.btn_eval_demo.setObjectName("btn_eval_demo")
        layout.addWidget(self.btn_eval_demo)
        
        self.lbl_demo_status = QLabel("")
        self.lbl_demo_status.setStyleSheet("color: #a371f7; font-weight: bold;")
        layout.addWidget(self.lbl_demo_status)
        
        layout.addWidget(QLabel("Disturbance:"))
        self.combo_noise = QComboBox()
        self.combo_noise.addItems([
            "Class 1 (8-12 Hz Tonal)",
            "Class 2 (18-22 Hz Tonal)",
            "Class 3 (Impulse Noise)",
            "Class 4 (Low-Freq Drift)",
            "Random Mode"
        ])
        layout.addWidget(self.combo_noise)
        
        self.chk_ab_compare = QCheckBox("Show Static A/B Comparison")
        self.chk_ab_compare.setStyleSheet("color: #ff8c00; font-weight: bold;")
        layout.addWidget(self.chk_ab_compare)
        
        self.chk_filter_enable = QCheckBox("Filter Enabled")
        self.chk_filter_enable.setChecked(True)
        layout.addWidget(self.chk_filter_enable)
        
        self.btn_export = QPushButton("Export Log CSV")
        layout.addWidget(self.btn_export)
        
        return frame