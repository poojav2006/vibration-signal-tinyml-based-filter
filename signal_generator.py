"""
signal_generator.py
===================
Parametric randomized noise generator for vibration disturbance simulation.

Four distinct disturbance classes:
- Class 1: Narrowband / tonal interference (8.0 - 12.0 Hz)
- Class 2: High-frequency tonal interference (18.0 - 22.0 Hz)
- Class 3: Impulse noise (1 - 10 impulses, randomized width, polarity, amplitude)
- Class 4: Low-frequency drift / wander (0.5 - 3.0 Hz)

Supports both window-based generation (N=128) and continuous streaming generation (maintaining phase).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


CLASS_LABELS = {
    1: "Class 1: 8-12 Hz Tonal",
    2: "Class 2: 18-22 Hz Tonal",
    3: "Class 3: Impulse Noise",
    4: "Class 4: Low-Freq Drift"
}

CLASS_NAMES = {
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4"
}


def generate_class1_noise(
    n_samples: int = 128,
    fs: float = 100.0,
    rng: Optional[np.random.Generator] = None,
    f_center: Optional[float] = None,
    amplitude: Optional[float] = None,
    phase: Optional[float] = None,
    t_start: float = 0.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Class 1: 8-12 Hz Narrowband / Tonal Interference.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    freq = f_center if f_center is not None else rng.uniform(8.0, 12.0)
    amp = amplitude if amplitude is not None else rng.uniform(0.12, 0.45)
    phi = phase if phase is not None else rng.uniform(0.0, 2.0 * np.pi)
    
    t = t_start + np.arange(n_samples) / fs
    noise = amp * np.sin(2.0 * np.pi * freq * t + phi)
    
    info = {
        'class_id': 1,
        'class_name': CLASS_NAMES[1],
        'frequency': float(freq),
        'amplitude': float(amp),
        'phase': float(phi),
        't_end': float(t[-1] + 1.0 / fs)
    }
    return noise, info


def generate_class2_noise(
    n_samples: int = 128,
    fs: float = 100.0,
    rng: Optional[np.random.Generator] = None,
    f_center: Optional[float] = None,
    amplitude: Optional[float] = None,
    phase: Optional[float] = None,
    t_start: float = 0.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Class 2: 18-22 Hz High-Frequency Tonal Interference.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    freq = f_center if f_center is not None else rng.uniform(18.0, 22.0)
    amp = amplitude if amplitude is not None else rng.uniform(0.12, 0.45)
    phi = phase if phase is not None else rng.uniform(0.0, 2.0 * np.pi)
    
    t = t_start + np.arange(n_samples) / fs
    noise = amp * np.sin(2.0 * np.pi * freq * t + phi)
    
    info = {
        'class_id': 2,
        'class_name': CLASS_NAMES[2],
        'frequency': float(freq),
        'amplitude': float(amp),
        'phase': float(phi),
        't_end': float(t[-1] + 1.0 / fs)
    }
    return noise, info


def generate_class3_noise(
    n_samples: int = 128,
    fs: float = 100.0,
    rng: Optional[np.random.Generator] = None,
    num_impulses: Optional[int] = None,
    amplitude: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Class 3: Impulse Noise.
    Random impulses with randomized position, amplitude, polarity, and width.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    k_impulses = num_impulses if num_impulses is not None else rng.integers(1, 11)
    base_amp = amplitude if amplitude is not None else rng.uniform(0.5, 2.2)
    
    noise = np.zeros(n_samples, dtype=np.float64)
    positions = []
    
    # Avoid overlapping placement if possible
    chosen_pos = rng.choice(n_samples, size=min(k_impulses, n_samples), replace=False)
    
    for pos in chosen_pos:
        polarity = rng.choice([-1.0, 1.0])
        imp_amp = base_amp * rng.uniform(0.7, 1.3) * polarity
        width = rng.choice([1, 2, 3])  # impulse duration 1 to 3 samples
        for w in range(width):
            idx = pos + w
            if idx < n_samples:
                # Decaying or triangular envelope for multi-sample impulse
                weight = 1.0 if w == 0 else (0.5 if w == 1 else 0.25)
                noise[idx] += imp_amp * weight
        positions.append(int(pos))
        
    info = {
        'class_id': 3,
        'class_name': CLASS_NAMES[3],
        'num_impulses': int(k_impulses),
        'amplitude': float(base_amp),
        'positions': positions
    }
    return noise, info


def generate_class4_noise(
    n_samples: int = 128,
    fs: float = 100.0,
    rng: Optional[np.random.Generator] = None,
    f_center: Optional[float] = None,
    amplitude: Optional[float] = None,
    phase: Optional[float] = None,
    t_start: float = 0.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Class 4: Low-Frequency Drift (0.5 - 3.0 Hz).
    """
    if rng is None:
        rng = np.random.default_rng()
        
    freq = f_center if f_center is not None else rng.uniform(0.5, 3.0)
    amp = amplitude if amplitude is not None else rng.uniform(0.25, 0.90)
    phi = phase if phase is not None else rng.uniform(0.0, 2.0 * np.pi)
    
    t = t_start + np.arange(n_samples) / fs
    noise = amp * np.sin(2.0 * np.pi * freq * t + phi)
    
    info = {
        'class_id': 4,
        'class_name': CLASS_NAMES[4],
        'frequency': float(freq),
        'amplitude': float(amp),
        'phase': float(phi),
        't_end': float(t[-1] + 1.0 / fs)
    }
    return noise, info


def generate_noise(
    noise_class: int,
    n_samples: int = 128,
    fs: float = 100.0,
    rng: Optional[np.random.Generator] = None,
    amplitude: Optional[float] = None,
    frequency: Optional[float] = None,
    t_start: float = 0.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate noise for any given class (1, 2, 3, or 4).
    """
    if noise_class == 1:
        return generate_class1_noise(n_samples, fs, rng, f_center=frequency, amplitude=amplitude, t_start=t_start)
    elif noise_class == 2:
        return generate_class2_noise(n_samples, fs, rng, f_center=frequency, amplitude=amplitude, t_start=t_start)
    elif noise_class == 3:
        return generate_class3_noise(n_samples, fs, rng, amplitude=amplitude)
    elif noise_class == 4:
        return generate_class4_noise(n_samples, fs, rng, f_center=frequency, amplitude=amplitude, t_start=t_start)
    else:
        raise ValueError(f"Invalid noise class: {noise_class}. Must be 1, 2, 3, or 4.")


def inject_noise(
    clean_signal: np.ndarray,
    noise_class: int,
    fs: float = 100.0,
    rng: Optional[np.random.Generator] = None,
    amplitude: Optional[float] = None,
    frequency: Optional[float] = None,
    t_start: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Add specified noise to a clean signal segment.
    
    Returns:
        (noisy_signal, noise, info)
    """
    noise, info = generate_noise(
        noise_class=noise_class,
        n_samples=len(clean_signal),
        fs=fs,
        rng=rng,
        amplitude=amplitude,
        frequency=frequency,
        t_start=t_start
    )
    noisy_signal = clean_signal + noise
    return noisy_signal, noise, info


class StreamingNoiseSource:
    """
    Stateful noise generator for real-time continuous sample-by-sample streaming.
    Preserves continuous phase across sample updates.
    """
    def __init__(self, fs: float = 100.0, seed: Optional[int] = 42):
        self.fs = fs
        self.rng = np.random.default_rng(seed)
        self.current_class = 1
        self.phase = 0.0
        self.freq = 10.0
        self.amplitude = 0.25
        self.sample_idx = 0
        self.set_class(1)

    def set_class(self, noise_class: int, amplitude: Optional[float] = None, freq: Optional[float] = None):
        self.current_class = noise_class
        if noise_class == 1:
            self.freq = freq if freq is not None else float(self.rng.uniform(8.0, 12.0))
            self.amplitude = amplitude if amplitude is not None else float(self.rng.uniform(0.15, 0.40))
        elif noise_class == 2:
            self.freq = freq if freq is not None else float(self.rng.uniform(18.0, 22.0))
            self.amplitude = amplitude if amplitude is not None else float(self.rng.uniform(0.15, 0.40))
        elif noise_class == 3:
            self.amplitude = amplitude if amplitude is not None else float(self.rng.uniform(0.8, 2.0))
        elif noise_class == 4:
            self.freq = freq if freq is not None else float(self.rng.uniform(0.5, 2.8))
            self.amplitude = amplitude if amplitude is not None else float(self.rng.uniform(0.3, 0.8))

    def randomize_parameters(self):
        """Randomize parameters within current class or pick random class."""
        self.set_class(self.current_class)

    def next_sample(self) -> float:
        """Produce the next noise sample causally."""
        t = self.sample_idx / self.fs
        self.sample_idx += 1
        
        if self.current_class in (1, 2, 4):
            val = self.amplitude * np.sin(2.0 * np.pi * self.freq * t + self.phase)
            return float(val)
        else: # Class 3 Impulse
            # Poisson/Bernoulli arrival probability ~ 3% per sample (average 3-4 impulses per 128 samples)
            if self.rng.random() < 0.035:
                polarity = self.rng.choice([-1.0, 1.0])
                val = self.amplitude * float(polarity) * float(self.rng.uniform(0.8, 1.4))
                return float(val)
            return 0.0


if __name__ == '__main__':
    print("--- Testing Signal Generator ---")
    for c in [1, 2, 3, 4]:
        noise, info = generate_noise(c, n_samples=128)
        print(f"Generated {info['class_name']}: shape={noise.shape}, min={noise.min():.3f}, max={noise.max():.3f}, std={noise.std():.3f}")
