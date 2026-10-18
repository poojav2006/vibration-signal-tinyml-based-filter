"""
plots.py
========
PyQtGraph plotting widgets with innovative DSP overlays:
- WaveformPlotWidget: Live scrolling comparison of Noisy, TinyML Filtered, Reference, and Static Baseline.
- FFTPlotWidget: Real-time FFT magnitude spectrum with dynamic |H(f)| Bode transfer function overlay.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt


BG_COLOR = "#12151c"
TEXT_COLOR = "#d1d5db"
CYAN = "#00f0ff"
GREEN = "#00ff88"
ORANGE = "#ff8c00"
GRAY = "#6b7280"
MAGENTA = "#ff007f"
GOLD = "#ffd700"


class WaveformPlotWidget(QWidget):
    """
    Scrolling multi-channel waveform plot supporting A/B static vs adaptive comparison.
    """
    def __init__(self, parent=None, max_samples=256):
        super().__init__(parent)
        self.max_samples = max_samples
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(BG_COLOR)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Samples (Last 2.5s)', color=TEXT_COLOR)
        self.plot_widget.setLabel('left', 'Acceleration (g)', color=TEXT_COLOR)
        self.plot_widget.addLegend(offset=(10, 10))
        
        # Curves
        self.curve_noisy = self.plot_widget.plot(
            pen=pg.mkPen(color=CYAN, width=1.5), 
            name="Noisy Vibration"
        )
        self.curve_filtered = self.plot_widget.plot(
            pen=pg.mkPen(color=GREEN, width=2.2), 
            name="TinyML Adaptive Filter"
        )
        self.curve_static = self.plot_widget.plot(
            pen=pg.mkPen(color=ORANGE, width=1.4, style=Qt.PenStyle.DotLine), 
            name="Static Fixed Filter (A/B Test)"
        )
        self.curve_clean = self.plot_widget.plot(
            pen=pg.mkPen(color=GRAY, width=1.2, style=Qt.PenStyle.DashLine), 
            name="Reference (Clean)"
        )
        
        # Initially hide static curve until enabled
        self.curve_static.setVisible(False)
        layout.addWidget(self.plot_widget)

    def set_ab_mode_visible(self, visible: bool):
        self.curve_static.setVisible(visible)

    def update_data(self, clean: np.ndarray, noisy: np.ndarray, filtered: np.ndarray, static_filt: np.ndarray = None):
        x = np.arange(len(noisy))
        self.curve_clean.setData(x, clean)
        self.curve_noisy.setData(x, noisy)
        self.curve_filtered.setData(x, filtered)
        if static_filt is not None and self.curve_static.isVisible():
            self.curve_static.setData(x, static_filt)


class FFTPlotWidget(QWidget):
    """
    Real-time FFT magnitude spectrum plot with dynamic |H(f)| Bode transfer function overlay.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(BG_COLOR)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Frequency (Hz)', color=TEXT_COLOR)
        self.plot_widget.setLabel('left', 'FFT Magnitude (g)', color=TEXT_COLOR)
        self.plot_widget.setXRange(0, 50, padding=0.02)
        self.plot_widget.addLegend(offset=(10, 10))
        
        # Highlight disturbance regions
        self.band_c1 = pg.LinearRegionItem(values=[8.0, 12.0], orientation='vertical', brush=pg.mkBrush(0, 240, 255, 25), movable=False)
        self.band_c2 = pg.LinearRegionItem(values=[18.0, 22.0], orientation='vertical', brush=pg.mkBrush(255, 140, 0, 25), movable=False)
        self.plot_widget.addItem(self.band_c1)
        self.plot_widget.addItem(self.band_c2)
        
        # Spectrum curve
        self.curve_spectrum = self.plot_widget.plot(
            pen=pg.mkPen(color=CYAN, width=1.8),
            fillLevel=0.0,
            fillBrush=pg.mkBrush(0, 240, 255, 35),
            name="Vibration FFT"
        )
        
        # Real-time Bode Transfer Function |H(f)| curve
        self.curve_bode = self.plot_widget.plot(
            pen=pg.mkPen(color=GOLD, width=2.0, style=Qt.PenStyle.DashLine),
            name="Active Filter |H(f)| Bode"
        )
        
        # Dominant peak scatter marker
        self.peak_scatter = pg.ScatterPlotItem(
            size=10, 
            pen=pg.mkPen(color="#ffffff", width=1.5), 
            brush=pg.mkBrush(MAGENTA),
            symbol='o'
        )
        self.plot_widget.addItem(self.peak_scatter)
        
        self.peak_label = pg.TextItem(text="", color=MAGENTA, anchor=(0.5, 1.5))
        self.plot_widget.addItem(self.peak_label)
        
        layout.addWidget(self.plot_widget)

    def update_spectrum(
        self, 
        freqs: np.ndarray, 
        magnitudes: np.ndarray, 
        dom_freq: float,
        bode_freqs: np.ndarray = None,
        bode_mag: np.ndarray = None
    ):
        self.curve_spectrum.setData(freqs, magnitudes)
        
        # Peak indicator
        max_mag = float(np.max(magnitudes)) if len(magnitudes) > 0 else 1.0
        if len(freqs) > 1 and dom_freq > 0:
            idx = int(np.argmin(np.abs(freqs - dom_freq)))
            peak_mag = float(magnitudes[idx])
            self.peak_scatter.setData(pos=[(dom_freq, peak_mag)])
            self.peak_label.setText(f"Peak: {dom_freq:.1f} Hz")
            self.peak_label.setPos(dom_freq, peak_mag)
        else:
            self.peak_scatter.clear()
            self.peak_label.setText("")
            
        # Scale and plot analytical Bode transfer response
        if bode_freqs is not None and bode_mag is not None:
            scaled_bode = bode_mag * max(max_mag * 1.15, 0.1)
            self.curve_bode.setData(bode_freqs, scaled_bode)