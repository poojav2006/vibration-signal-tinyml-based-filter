"""
multi_model_trainer.py
======================
Train and benchmark 6 diverse Machine Learning classifiers for vibration disturbance classification:
1. Decision Tree (CART) - Ultralight TinyML Baseline
2. Random Forest (Ensemble Bagging) - Variance Reduction
3. Gradient Boosting (GBDT) - Sequential Boosting
4. Support Vector Machine (SVM - RBF) - Optimal Margin Boundary
5. Multilayer Perceptron (MLP / Tiny Neural Net) - Deep Learning Baseline
6. K-Nearest Neighbors (KNN) - Instance-Based Non-Parametric Metric

Evaluates on identical independent test set (zero data leakage) and computes:
- Accuracy (Train, Val, Test)
- Macro & Weighted F1-Scores
- Microsecond Inference Latency
- Serialized Model Footprint (KB)
- Embedded TinyML Suitability Rating
- Confusion Matrix per model
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_raw_dataset, get_available_segments, extract_segment_x, resample_50_to_100, get_default_data_path
from signal_generator import generate_noise, CLASS_NAMES, CLASS_LABELS
from feature_extractor import extract_features_from_window, slice_into_windows, FEATURE_NAMES
from train_model import build_labeled_dataset

MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


MODEL_SPECS = {
    'decision_tree': {
        'name': 'Decision Tree (CART)',
        'category': 'Tree-based / Symbolic',
        'pipeline': DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, random_state=42, criterion='gini'),
        'tinyml_score': 98,
        'embedded_status': 'Excellent (Native C++ Header, 0 Bytes Malloc)',
        'hardware_target': 'ESP32 / ARM Cortex-M0+/M4 (Direct Form)',
        'description': 'Direct if-else decision rules with minimal ROM/RAM footprint and deterministic execution time.'
    },
    'random_forest': {
        'name': 'Random Forest',
        'category': 'Ensemble (Bagging)',
        'pipeline': RandomForestClassifier(n_estimators=50, max_depth=7, min_samples_leaf=3, random_state=42, n_jobs=-1),
        'tinyml_score': 74,
        'embedded_status': 'Moderate (Requires ~120 KB Flash for 50 trees)',
        'hardware_target': 'ESP32-S3 / ARM Cortex-M7 / Raspberry Pi',
        'description': 'Ensemble of 50 decision trees voting in parallel. Reduces variance and boosts generalization.'
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting (GBDT)',
        'category': 'Ensemble (Boosting)',
        'pipeline': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42),
        'tinyml_score': 78,
        'embedded_status': 'Good (Sequential shallow trees, ~45 KB Flash)',
        'hardware_target': 'ESP32 / Edge Microcontrollers',
        'description': 'Sequential tree boosting correcting residual errors iteratively. High tabular accuracy.'
    },
    'svm': {
        'name': 'Support Vector Machine (SVM - RBF)',
        'category': 'Kernel Machine',
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42))
        ]),
        'tinyml_score': 62,
        'embedded_status': 'Moderate (Support vectors require SRAM storage)',
        'hardware_target': 'ARM Cortex-M4F / Edge Gateway',
        'description': 'Max-margin hyperplane in infinite-dimensional RBF space. Highly robust decision boundaries.'
    },
    'mlp_neural_net': {
        'name': 'Tiny Neural Network (MLP)',
        'category': 'Deep Learning / Connectionist',
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True))
        ]),
        'tinyml_score': 92,
        'embedded_status': 'Excellent (Easily mapped to CMSIS-NN / TFLite Micro)',
        'hardware_target': 'ESP32 / STM32 (CMSIS-NN int8 quantized)',
        'description': 'Feedforward neural network: 5 inputs -> 16 Dense (ReLU) -> 8 Dense (ReLU) -> 4 Softmax.'
    },
    'knn': {
        'name': 'K-Nearest Neighbors (KNN)',
        'category': 'Instance-Based / Non-Parametric',
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
        ]),
        'tinyml_score': 35,
        'embedded_status': 'Poor for Microcontrollers (Stores entire training set in RAM)',
        'hardware_target': 'Gateway / Server / PC Only',
        'description': 'Lazy learning calculating Euclidean distances to all training exemplars. Memory intensive.'
    }
}


def train_and_benchmark_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics_dir = os.path.join(RESULTS_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    print('Loading raw dataset from UCI Accelerometer CSV...')
    df = load_raw_dataset()
    segments = get_available_segments(df)
    
    rng_partition = np.random.default_rng(2026)
    perm = rng_partition.permutation(len(segments))
    
    n_total = len(segments)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    
    train_segs = [segments[i] for i in perm[:n_train]]
    val_segs = [segments[i] for i in perm[n_train:n_train + n_val]]
    test_segs = [segments[i] for i in perm[n_train + n_val:]]
    
    print(f'Segments -> Train: {len(train_segs)}, Val: {len(val_segs)}, Test: {len(test_segs)}')
    
    print('Generating Training windows...')
    X_train, y_train, _, _, _ = build_labeled_dataset(df, train_segs, windows_per_segment=15, rng_seed=101)
    
    print('Generating Validation windows...')
    X_val, y_val, _, _, _ = build_labeled_dataset(df, val_segs, windows_per_segment=10, rng_seed=202)
    
    print('Generating Independent Test windows...')
    X_test, y_test, clean_test, noisy_test, meta_test = build_labeled_dataset(df, test_segs, windows_per_segment=10, rng_seed=303)
    
    print(f'Window Counts -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
    
    benchmark_results = {}
    
    print('\n========================================================================')
    print('TRAINING & BENCHMARKING 6 MACHINE LEARNING CLASSIFIERS')
    print('========================================================================\n')
    
    for key, spec in MODEL_SPECS.items():
        print(f'--> Training: {spec["name"]} ({spec["category"]}) ...')
        model = spec['pipeline']
        
        # Fit
        t0_train = time.perf_counter()
        model.fit(X_train, y_train)
        t1_train = time.perf_counter()
        train_duration_s = t1_train - t0_train
        
        # Predict on Test
        test_preds = model.predict(X_test)
        val_preds = model.predict(X_val)
        train_preds = model.predict(X_train)
        
        test_acc = accuracy_score(y_test, test_preds)
        val_acc = accuracy_score(y_val, val_preds)
        train_acc = accuracy_score(y_train, train_preds)
        macro_f1 = f1_score(y_test, test_preds, average='macro')
        weighted_f1 = f1_score(y_test, test_preds, average='weighted')
        
        # Latency Benchmark: 500 single-window predictions
        latencies_us = []
        for i in range(min(500, len(X_test))):
            sample = X_test[i:i+1]
            t0_inf = time.perf_counter_ns()
            _ = model.predict(sample)
            t1_inf = time.perf_counter_ns()
            latencies_us.append((t1_inf - t0_inf) / 1000.0)
        mean_latency_us = float(np.median(latencies_us))
        
        # Save model
        model_filename = f'{key}.joblib'
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_path)
        file_size_kb = os.path.getsize(model_path) / 1024.0
        
        cm = confusion_matrix(y_test, test_preds, labels=[1, 2, 3, 4]).tolist()
        report = classification_report(
            y_test, test_preds,
            labels=[1, 2, 3, 4],
            target_names=['Class 1 (8-12Hz)', 'Class 2 (18-22Hz)', 'Class 3 (Impulse)', 'Class 4 (Drift)'],
            output_dict=True
        )
        
        benchmark_results[key] = {
            'key': key,
            'name': spec['name'],
            'category': spec['category'],
            'train_acc': round(float(train_acc * 100.0), 2),
            'val_acc': round(float(val_acc * 100.0), 2),
            'test_acc': round(float(test_acc * 100.0), 2),
            'macro_f1': round(float(macro_f1 * 100.0), 2),
            'weighted_f1': round(float(weighted_f1 * 100.0), 2),
            'latency_us': round(mean_latency_us, 1),
            'file_size_kb': round(file_size_kb, 1),
            'tinyml_score': spec['tinyml_score'],
            'embedded_status': spec['embedded_status'],
            'hardware_target': spec['hardware_target'],
            'description': spec['description'],
            'confusion_matrix': cm,
            'report': report,
            'model_file': model_filename
        }
        
        print(f'    Test Accuracy: {test_acc * 100.0:.2f}% | Macro F1: {macro_f1 * 100.0:.2f}% | Latency: {mean_latency_us:.1f} us | Size: {file_size_kb:.1f} KB')
        
    # Save benchmark JSON
    benchmark_path = os.path.join(metrics_dir, 'multi_model_benchmark.json')
    with open(benchmark_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f'\nSaved comprehensive benchmark report to: {benchmark_path}')
    
    # Also save test dataset package if not present
    test_pkg_path = os.path.join(MODELS_DIR, 'test_dataset.joblib')
    joblib.dump({
        'X_test': X_test,
        'y_test': y_test,
        'clean_test': clean_test,
        'noisy_test': noisy_test,
        'meta_test': meta_test
    }, test_pkg_path)
    
    print('All 6 models successfully trained, evaluated, and saved!')
    return benchmark_results

if __name__ == '__main__':
    train_and_benchmark_all()
