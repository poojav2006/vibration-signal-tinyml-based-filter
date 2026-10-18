"""
simulation.py
=============
Real-time streaming simulation engine for adaptive vibration filtering.

Features:
- Filtered Signal Feature Extraction (Mean, Std, RMS, Variance, Dominant Frequency).
- Accuracy Before vs After Filtering: Computes real-time waveform reconstruction accuracy.
- A/B Comparative Benchmark: Concurrently evaluates Traditional Static Filter vs TinyML Adaptive Filter.
- ISO 10816-1 Industrial Vibration Severity Grading (Zone A Normal to Zone D Critical Danger).
- 1-Click Evaluator Auto-Pilot Demonstration state machine.
- Microsecond timing profiling.
"""

import os
import time
import joblib
from typing import Dict, Any, Optional, Callable, List
import numpy as np

from data_loader import load_and_prepare_vibration
from signal_generator import StreamingNoiseSource, CLASS_NAMES, CLASS_LABELS
from feature_extractor import extract_features_from_window, extract_fft_spectrum, FEATURE_NAMES
from adaptive_filter import (
    CausalImpulseFilter,
    CausalBiquadFilter,
    design_causal_notch,
    design_causal_highpass,
    get_filter_frequency_response,
    calculate_mse,
    calculate_snr,
    calculate_signal_accuracy
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def compute_iso_10816_zone(rms_val: float) -> Dict[str, str]:
    if rms_val < 1.05:
        return {
            'zone': 'Zone A/B (Normal / Good)',
            'color': '#00ff88',
            'status': 'SAFE',
            'description': 'Permissible for long-term continuous operation'
        }
    elif rms_val <= 1.25:
        return {
            'zone': 'Zone C (Warning / Alert)',
            'color': '#d29922',
            'status': 'WARNING',
            'description': 'Elevated vibration; remedial maintenance advised'
        }
    else:
        return {
            'zone': 'Zone D (Critical / Danger)',
            'color': '#f85149',
            'status': 'DANGER',
            'description': 'Severe vibration; risk of mechanical damage'
        }


class VibrationSimulationEngine:
    def __init__(
        self,
        fs: float = 100.0,
        window_size: int = 128,
        hop_size: int = 64,
        wconfid: int = 1,
        pctid: int = 20,
        csv_path: Optional[str] = None
    ):
        self.fs = fs
        self.window_size = window_size
        self.hop_size = hop_size
        
        _, self.reference_signal, self.meta = load_and_prepare_vibration(
            csv_path=csv_path,
            wconfid=wconfid,
            pctid=pctid
        )
        self.total_reference_samples = len(self.reference_signal)
        self.sample_pointer = 0
        
        model_path = os.path.join(MODELS_DIR, 'decision_tree.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Run train_model.py first.")
        self.model = joblib.load(model_path)
        
        self.noise_source = StreamingNoiseSource(fs=fs, seed=42)
        
        self.clean_history = np.zeros(window_size, dtype=np.float64)
        self.noisy_history = np.zeros(window_size, dtype=np.float64)
        self.filtered_history = np.zeros(window_size, dtype=np.float64)
        self.static_filtered_history = np.zeros(window_size, dtype=np.float64)
        
        self.impulse_filter = CausalImpulseFilter(window_len=5, threshold_factor=2.2)
        b_hp, a_hp = design_causal_highpass(fc=3.5, fs=fs, order=2)
        self.highpass_filter = CausalBiquadFilter(b_hp, a_hp)
        self.notch_filter = None
        self.active_notch_freq = 10.0
        self._update_notch_filter(10.0)
        
        b_st, a_st = design_causal_notch(f_notch=10.0, fs=fs, Q=8.0)
        self.static_notch = CausalBiquadFilter(b_st, a_st)
        self.static_notch.reset(init_val=1.0)
        
        self.current_predicted_class = 1
        self.current_dominant_freq = 10.0
        self.current_features = np.zeros(5, dtype=np.float32)
        self.filtered_features = np.zeros(5, dtype=np.float32)
        self.samples_since_last_inference = 0
        self.total_samples_processed = 0
        
        self.class_counters = {1: 0, 2: 0, 3: 0, 4: 0}
        self.timeline_events: List[Dict[str, Any]] = []
        
        self.last_feature_time_us = 0.0
        self.last_inference_time_us = 0.0
        self.last_filter_time_us = 0.0
        
        self.latest_metrics = {
            'mse_before': 0.0,
            'mse_after': 0.0,
            'snr_before': 0.0,
            'snr_after': 0.0,
            'snr_gain': 0.0,
            'accuracy_before': 75.0,
            'accuracy_after': 88.0,
            'accuracy_gain': 13.0,
            'filtered_mean': 0.9956,
            'filtered_std': 0.0260,
            'filtered_rms': 0.9959,
            'filtered_var': 0.00067,
            'filtered_dom_freq': 3.125,
            'static_snr_after': 0.0,
            'static_snr_gain': 0.0,
            'rms_noisy': 0.0,
            'rms_filtered': 0.0,
            'peak': 0.0,
            'iso_zone_noisy': compute_iso_10816_zone(1.0),
            'iso_zone_filtered': compute_iso_10816_zone(1.0)
        }
        
        self.demo_active = False
        self.demo_step = 0
        self.demo_timer_samples = 0
        self.demo_scorecard: Optional[Dict[str, Any]] = None

    def _update_notch_filter(self, center_freq: float):
        self.active_notch_freq = float(np.clip(center_freq, 5.0, 45.0))
        b, a = design_causal_notch(f_notch=self.active_notch_freq, fs=self.fs, Q=8.0)
        self.notch_filter = CausalBiquadFilter(b, a)
        self.notch_filter.reset(init_val=1.0)

    def set_noise_mode(self, noise_class: int, amplitude: Optional[float] = None, freq: Optional[float] = None):
        self.noise_source.set_class(noise_class, amplitude=amplitude, freq=freq)

    def start_evaluator_demo(self):
        self.demo_active = True
        self.demo_step = 1
        self.demo_timer_samples = 0
        self.set_noise_mode(1, amplitude=0.35, freq=10.0)

    def step_single_sample(self) -> Dict[str, Any]:
        if self.demo_active:
            self.demo_timer_samples += 1
            if self.demo_timer_samples == 800:
                self.demo_step = 2
                self.set_noise_mode(3, amplitude=1.8)
            elif self.demo_timer_samples == 1600:
                self.demo_step = 3
                self.set_noise_mode(4, amplitude=0.6, freq=1.5)
            elif self.demo_timer_samples == 2400:
                self.demo_step = 4
                self.set_noise_mode(2, amplitude=0.35, freq=20.0)
            elif self.demo_timer_samples >= 3200:
                self.demo_active = False
                self.demo_scorecard = {
                    'total_detections': sum(self.class_counters.values()),
                    'class_breakdown': dict(self.class_counters),
                    'final_accuracy_before': self.latest_metrics['accuracy_before'],
                    'final_accuracy_after': self.latest_metrics['accuracy_after'],
                    'final_accuracy_gain': self.latest_metrics['accuracy_gain'],
                    'final_snr_gain': self.latest_metrics['snr_gain'],
                    'static_snr_gain': self.latest_metrics['static_snr_gain'],
                    'adaptive_advantage_db': self.latest_metrics['snr_gain'] - self.latest_metrics['static_snr_gain'],
                    'iso_status': self.latest_metrics['iso_zone_filtered']['status']
                }

        clean_sample = float(self.reference_signal[self.sample_pointer])
        self.sample_pointer = (self.sample_pointer + 1) % self.total_reference_samples
        
        noise_sample = self.noise_source.next_sample()
        noisy_sample = clean_sample + noise_sample
        
        t0_filt = time.perf_counter_ns()
        if self.current_predicted_class in (1, 2):
            filtered_sample = self.notch_filter.filter_sample(noisy_sample)
        elif self.current_predicted_class == 3:
            filtered_sample = self.impulse_filter.filter_sample(noisy_sample)
        elif self.current_predicted_class == 4:
            baseline = float(np.mean(self.noisy_history[-32:])) if self.total_samples_processed > 32 else 1.0
            hp_out = self.highpass_filter.filter_sample(noisy_sample - baseline)
            filtered_sample = hp_out + baseline
        else:
            filtered_sample = noisy_sample
        t1_filt = time.perf_counter_ns()
        self.last_filter_time_us = (t1_filt - t0_filt) / 1000.0
        
        static_filtered_sample = self.static_notch.filter_sample(noisy_sample)
        
        self.clean_history = np.roll(self.clean_history, -1)
        self.clean_history[-1] = clean_sample
        
        self.noisy_history = np.roll(self.noisy_history, -1)
        self.noisy_history[-1] = noisy_sample
        
        self.filtered_history = np.roll(self.filtered_history, -1)
        self.filtered_history[-1] = filtered_sample
        
        self.static_filtered_history = np.roll(self.static_filtered_history, -1)
        self.static_filtered_history[-1] = static_filtered_sample
        
        self.samples_since_last_inference += 1
        self.total_samples_processed += 1
        
        inference_triggered = False
        
        if self.samples_since_last_inference >= self.hop_size:
            self.samples_since_last_inference = 0
            inference_triggered = True
            
            t0_feat = time.perf_counter_ns()
            self.current_features = extract_features_from_window(self.noisy_history, fs=self.fs)
            t1_feat = time.perf_counter_ns()
            self.last_feature_time_us = (t1_feat - t0_feat) / 1000.0
            
            # Extract 5 features from the filtered signal window
            self.filtered_features = extract_features_from_window(self.filtered_history, fs=self.fs)
            
            t0_infer = time.perf_counter_ns()
            pred_class = int(self.model.predict(self.current_features.reshape(1, -1))[0])
            t1_infer = time.perf_counter_ns()
            self.last_inference_time_us = (t1_infer - t0_infer) / 1000.0
            
            self.current_predicted_class = pred_class
            self.class_counters[pred_class] += 1
            self.current_dominant_freq = float(self.current_features[4])
            
            if pred_class in (1, 2):
                self._update_notch_filter(self.current_dominant_freq)
            elif pred_class == 3:
                self.impulse_filter.reset(init_val=filtered_sample)
            elif pred_class == 4:
                self.highpass_filter.reset(init_val=0.0)
                
            mse_bef = calculate_mse(self.clean_history, self.noisy_history)
            mse_aft = calculate_mse(self.clean_history, self.filtered_history)
            snr_bef = calculate_snr(self.clean_history, self.noisy_history - self.clean_history)
            snr_aft = calculate_snr(self.clean_history, self.filtered_history - self.clean_history)
            
            acc_bef = calculate_signal_accuracy(self.clean_history, self.noisy_history)
            acc_aft = calculate_signal_accuracy(self.clean_history, self.filtered_history)
            acc_gain = acc_aft - acc_bef
            
            static_snr_aft = calculate_snr(self.clean_history, self.static_filtered_history - self.clean_history)
            
            rms_noisy = float(np.sqrt(np.mean(self.noisy_history ** 2)))
            rms_filtered = float(np.sqrt(np.mean(self.filtered_history ** 2)))
            
            self.latest_metrics = {
                'mse_before': mse_bef,
                'mse_after': mse_aft,
                'snr_before': snr_bef,
                'snr_after': snr_aft,
                'snr_gain': snr_aft - snr_bef,
                'accuracy_before': acc_bef,
                'accuracy_after': acc_aft,
                'accuracy_gain': acc_gain,
                'filtered_mean': float(self.filtered_features[0]),
                'filtered_std': float(self.filtered_features[1]),
                'filtered_rms': float(self.filtered_features[2]),
                'filtered_var': float(self.filtered_features[3]),
                'filtered_dom_freq': float(self.filtered_features[4]),
                'static_snr_after': static_snr_aft,
                'static_snr_gain': static_snr_aft - snr_bef,
                'rms_noisy': rms_noisy,
                'rms_filtered': rms_filtered,
                'peak': float(np.max(np.abs(self.filtered_history))),
                'iso_zone_noisy': compute_iso_10816_zone(rms_noisy),
                'iso_zone_filtered': compute_iso_10816_zone(rms_filtered)
            }
            
            filter_desc = "Notch" if pred_class in (1, 2) else ("Adaptive Median" if pred_class == 3 else "High-Pass")
            
            self.timeline_events.append({
                'time_s': round(self.total_samples_processed / self.fs, 2),
                'actual_class': self.noise_source.current_class,
                'predicted_class': pred_class,
                'class_name': CLASS_NAMES[pred_class],
                'filter_name': filter_desc,
                'snr_gain_db': round(self.latest_metrics['snr_gain'], 2),
                'accuracy_gain_pct': round(acc_gain, 2),
                'static_gain_db': round(self.latest_metrics['static_snr_gain'], 2)
            })
            if len(self.timeline_events) > 50:
                self.timeline_events.pop(0)
                
        return {
            'clean': clean_sample,
            'noisy': noisy_sample,
            'filtered': filtered_sample,
            'static_filtered': static_filtered_sample,
            'inference_triggered': inference_triggered,
            'predicted_class': self.current_predicted_class,
            'dominant_freq': self.current_dominant_freq
        }