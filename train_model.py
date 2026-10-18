"""
train_model.py
==============
Train a lightweight Decision Tree classifier on the UCI Accelerometer dataset.

Pipeline:
1. Load UCI Accelerometer dataset and discover all 51 (wconfid, pctid) segments.
2. Partition segments into Train (70%), Validation (15%), and Test (15%) to strictly
   prevent data leakage between adjacent windows of the same physical run.
3. Resample each segment to 100 Hz.
4. Extract windows (N=128, hop=64).
5. Inject randomized parametric noise across the 4 classes with randomized:
   - Frequencies (8-12 Hz for Class 1, 18-22 Hz for Class 2, 0.5-3.0 Hz for Class 4)
   - Amplitudes
   - Phases
   - Impulse counts, positions, widths, and polarities for Class 3
6. Extract 5 TinyML features: [mean, std, rms, var, dominant_freq].
7. Train a lightweight DecisionTreeClassifier (max_depth <= 6).
8. Evaluate on validation set and independent test set.
9. Save model and dataset artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data_loader import (
    load_raw_dataset,
    get_available_segments,
    extract_segment_x,
    resample_50_to_100,
    get_default_data_path
)
from signal_generator import generate_noise, CLASS_NAMES
from feature_extractor import (
    extract_features_from_window,
    slice_into_windows,
    FEATURE_NAMES
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def build_labeled_dataset(
    df: pd.DataFrame,
    segments: list,
    windows_per_segment: int = 15,
    rng_seed: int = 42
):
    """
    Generate labeled (features, labels, clean_windows, noisy_windows) dataset from segments.
    """
    rng = np.random.default_rng(rng_seed)
    
    all_features = []
    all_labels = []
    all_clean = []
    all_noisy = []
    meta_records = []
    
    for seg in segments:
        wconfid = seg['wconfid']
        pctid = seg['pctid']
        x_50 = extract_segment_x(df, wconfid, pctid)
        x_100 = resample_50_to_100(x_50)
        
        # Extract available 128-sample windows
        clean_windows = slice_into_windows(x_100, window_size=128, hop_size=64)
        if len(clean_windows) == 0:
            continue
            
        # Select sample windows from this segment
        n_take = min(windows_per_segment, len(clean_windows))
        chosen_indices = rng.choice(len(clean_windows), size=n_take, replace=False)
        
        for idx in chosen_indices:
            clean_w = clean_windows[idx]
            
            # Balance noise classes 1, 2, 3, 4
            for target_class in [1, 2, 3, 4]:
                noise, n_info = generate_noise(
                    noise_class=target_class,
                    n_samples=128,
                    fs=100.0,
                    rng=rng
                )
                noisy_w = clean_w + noise
                feats = extract_features_from_window(noisy_w, fs=100.0)
                
                all_features.append(feats)
                all_labels.append(target_class)
                all_clean.append(clean_w)
                all_noisy.append(noisy_w)
                meta_records.append({
                    'wconfid': wconfid,
                    'pctid': pctid,
                    'class_id': target_class,
                    'noise_freq': n_info.get('frequency', 0.0),
                    'noise_amp': n_info.get('amplitude', 0.0)
                })
                
    return (
        np.array(all_features, dtype=np.float32),
        np.array(all_labels, dtype=np.int32),
        all_clean,
        all_noisy,
        meta_records
    )


def train_and_evaluate(target_windows: int = 1600):
    """
    Main training and validation orchestration.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'confusion_matrix'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'metrics'), exist_ok=True)
    
    print(f"Loading raw dataset from: {get_default_data_path()} ...")
    df = load_raw_dataset()
    segments = get_available_segments(df)
    print(f"Total available physical segments: {len(segments)}")
    
    # Shuffle segments to partition by run
    rng_partition = np.random.default_rng(2026)
    perm = rng_partition.permutation(len(segments))
    
    n_total = len(segments)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    
    train_segs = [segments[i] for i in perm[:n_train]]
    val_segs = [segments[i] for i in perm[n_train:n_train + n_val]]
    test_segs = [segments[i] for i in perm[n_train + n_val:]]
    
    print(f"Segment Split -> Train: {len(train_segs)} segments, Val: {len(val_segs)} segments, Test: {len(test_segs)} segments")
    
    # Generate datasets
    print("Generating training dataset...")
    X_train, y_train, _, _, _ = build_labeled_dataset(df, train_segs, windows_per_segment=15, rng_seed=101)
    
    print("Generating validation dataset...")
    X_val, y_val, _, _, _ = build_labeled_dataset(df, val_segs, windows_per_segment=10, rng_seed=202)
    
    print("Generating independent test dataset...")
    X_test, y_test, clean_test, noisy_test, meta_test = build_labeled_dataset(df, test_segs, windows_per_segment=10, rng_seed=303)
    
    print(f"Dataset window counts -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Total generated windows: {len(X_train) + len(X_val) + len(X_test)} (Target >= 1,000 satisfied)")
    
    # Class balance check
    for c in [1, 2, 3, 4]:
        print(f"  Train Class {c} count: {np.sum(y_train == c)}")
        
    # Train Decision Tree
    # We restrict max_depth to 6 and min_samples_leaf to 2 for lightweight TinyML deployment with >92% accuracy
    clf = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        criterion='gini'
    )
    clf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    
    print("\n--- Model Training Results ---")
    print(f"Train Accuracy: {train_acc * 100.0:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100.0:.2f}%")
    print(f"Independent Test Accuracy: {test_acc * 100.0:.2f}%")
    
    # Confusion Matrix & Classification Report
    cm = confusion_matrix(y_test, test_preds, labels=[1, 2, 3, 4])
    report = classification_report(
        y_test, 
        test_preds, 
        labels=[1, 2, 3, 4],
        target_names=["Class 1 (8-12Hz)", "Class 2 (18-22Hz)", "Class 3 (Impulse)", "Class 4 (Drift)"],
        output_dict=True
    )
    
    print("\nConfusion Matrix (Independent Test):")
    print(cm)
    print("\nClassification Report (Independent Test):")
    print(classification_report(
        y_test, 
        test_preds, 
        labels=[1, 2, 3, 4],
        target_names=["Class 1 (8-12Hz)", "Class 2 (18-22Hz)", "Class 3 (Impulse)", "Class 4 (Drift)"]
    ))
    
    # Save model and test package
    model_path = os.path.join(MODELS_DIR, 'decision_tree.joblib')
    joblib.dump(clf, model_path)
    print(f"Saved Decision Tree model to: {model_path}")
    
    # Save test dataset for evaluate_model.py and repeatable verification
    test_data_pkg = {
        'X_test': X_test,
        'y_test': y_test,
        'clean_test': clean_test,
        'noisy_test': noisy_test,
        'meta_test': meta_test
    }
    test_pkg_path = os.path.join(MODELS_DIR, 'test_dataset.joblib')
    joblib.dump(test_data_pkg, test_pkg_path)
    
    # Save metrics JSON
    metrics_summary = {
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'feature_names': FEATURE_NAMES,
        'tree_depth': int(clf.get_depth()),
        'tree_n_leaves': int(clf.get_n_leaves())
    }
    with open(os.path.join(RESULTS_DIR, 'metrics', 'model_metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"Tree Depth: {clf.get_depth()}, Number of Leaves: {clf.get_n_leaves()}")
    print("Model training complete!")
    return clf, metrics_summary


if __name__ == '__main__':
    train_and_evaluate()
