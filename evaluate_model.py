"""
evaluate_model.py
=================
Comprehensive independent test evaluation of the TinyML Decision Tree model
and the adaptive digital filtering pipeline.

Evaluates:
1. Model Classification Performance:
   - Independent Test Accuracy
   - Per-Class Precision, Recall, F1-Score
   - Confusion Matrix (printed and saved as image/text)
2. Adaptive Filter Performance across All Classes:
   - Signal Reconstruction Accuracy (%) Before vs After
   - Mean Squared Error (MSE) before & after
   - Signal-to-Noise Ratio (SNR in dB) before & after
   - Net SNR Improvement (dB)
3. Downstream Machine State Classification Accuracy (ISO 10816 Health Condition):
   - Machine State Accuracy Before Filtering vs After Filtering
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_extractor import FEATURE_NAMES
from adaptive_filter import AdaptiveFilterEngine, evaluate_filter_metrics, calculate_signal_accuracy
from simulation import compute_iso_10816_zone
from signal_generator import CLASS_NAMES, CLASS_LABELS


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_comprehensive_evaluation():
    print("=" * 75)
    print("      TINYML MODEL & ADAPTIVE FILTER COMPREHENSIVE EVALUATION")
    print("=" * 75)
    
    model_path = os.path.join(MODELS_DIR, 'decision_tree.joblib')
    test_pkg_path = os.path.join(MODELS_DIR, 'test_dataset.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(test_pkg_path):
        raise FileNotFoundError("Model or test dataset not found. Run train_model.py first.")
        
    clf = joblib.load(model_path)
    pkg = joblib.load(test_pkg_path)
    
    X_test = pkg['X_test']
    y_test = pkg['y_test']
    clean_test = pkg['clean_test']
    noisy_test = pkg['noisy_test']
    
    n_samples = len(y_test)
    print(f"Loaded {n_samples} independent test windows from unseen physical runs.")
    
    # 1. Classification Predictions
    y_pred = clf.predict(X_test)
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, supp = precision_recall_fscore_support(y_test, y_pred, labels=[1, 2, 3, 4])
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
    
    print(f"\nTinyML Disturbance Classification Accuracy: {acc * 100.0:.2f}%\n")
    print(f"{'Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 68)
    for c in [1, 2, 3, 4]:
        idx = c - 1
        print(f"{CLASS_LABELS[c]:<22} | {prec[idx]:<10.4f} | {rec[idx]:<10.4f} | {f1[idx]:<10.4f} | {supp[idx]:<8}")
        
    # Save confusion matrix figure
    cm_dir = os.path.join(RESULTS_DIR, 'confusion_matrix')
    os.makedirs(cm_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(4),
        yticks=np.arange(4),
        xticklabels=[CLASS_NAMES[c] for c in [1, 2, 3, 4]],
        yticklabels=[CLASS_NAMES[c] for c in [1, 2, 3, 4]],
        title=f"Disturbance Confusion Matrix (Accuracy: {acc*100:.1f}%)",
        ylabel="True Class",
        xlabel="Predicted Class"
    )
    for i in range(4):
        for j in range(4):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    plt.tight_layout()
    cm_img_path = os.path.join(cm_dir, 'confusion_matrix.png')
    plt.savefig(cm_img_path, dpi=150)
    plt.close()
    
    # 2. Filter Performance & Accuracy Evaluation
    print("\n" + "=" * 75)
    print("      ACCURACY BEFORE FILTERING vs AFTER FILTERING")
    print("=" * 75)
    
    filter_engine = AdaptiveFilterEngine(fs=100.0)
    
    class_stats = {
        c: {
            'mse_bef': [], 'mse_aft': [], 
            'snr_bef': [], 'snr_aft': [], 'snr_gain': [],
            'acc_bef': [], 'acc_aft': [], 'acc_gain': [],
            'iso_correct_bef': 0, 'iso_correct_aft': 0, 'total': 0
        } for c in [1, 2, 3, 4]
    }
    
    total_iso_bef_correct = 0
    total_iso_aft_correct = 0
    
    for i in range(n_samples):
        true_c = int(y_test[i])
        pred_c = int(y_pred[i])
        clean_w = clean_test[i]
        noisy_w = noisy_test[i]
        dom_f = float(X_test[i, 4])
        
        filtered_w, _, _ = filter_engine.filter_window(
            noisy_w, 
            predicted_class=pred_c,
            dominant_freq=dom_f
        )
        
        m = evaluate_filter_metrics(clean_w, noisy_w, filtered_w)
        
        # Machine Condition Classification Accuracy (ISO 10816 Zone A/B Normal = True)
        true_rms = np.sqrt(np.mean(clean_w ** 2))
        noisy_rms = np.sqrt(np.mean(noisy_w ** 2))
        filt_rms = np.sqrt(np.mean(filtered_w ** 2))
        
        # Ground truth state is Safe / Normal (Zone A/B)
        true_is_safe = (true_rms < 1.05)
        noisy_is_safe = (noisy_rms < 1.05)
        filt_is_safe = (filt_rms < 1.05)
        
        is_bef_correct = (noisy_is_safe == true_is_safe)
        is_aft_correct = (filt_is_safe == true_is_safe)
        
        cs = class_stats[true_c]
        cs['mse_bef'].append(m['mse_before'])
        cs['mse_aft'].append(m['mse_after'])
        cs['snr_bef'].append(m['snr_before'])
        cs['snr_aft'].append(m['snr_after'])
        cs['snr_gain'].append(m['snr_gain_db'])
        cs['acc_bef'].append(m['accuracy_before_pct'])
        cs['acc_aft'].append(m['accuracy_after_pct'])
        cs['acc_gain'].append(m['accuracy_gain_pct'])
        cs['total'] += 1
        if is_bef_correct: cs['iso_correct_bef'] += 1
        if is_aft_correct: cs['iso_correct_aft'] += 1
        
        if is_bef_correct: total_iso_bef_correct += 1
        if is_aft_correct: total_iso_aft_correct += 1
        
    print(f"\n[A] SIGNAL WAVEFORM RECONSTRUCTION ACCURACY:")
    print(f"{'Noise Class':<22} | {'Accuracy BEFORE':<16} | {'Accuracy AFTER':<16} | {'Accuracy GAIN':<14} | {'SNR Gain (dB)':<12}")
    print("-" * 88)
    
    table_rows = []
    overall_acc_bef = []
    overall_acc_aft = []
    
    for c in [1, 2, 3, 4]:
        st = class_stats[c]
        mean_acc_b = np.mean(st['acc_bef'])
        mean_acc_a = np.mean(st['acc_aft'])
        mean_acc_g = np.mean(st['acc_gain'])
        mean_snr_g = np.mean(st['snr_gain'])
        
        overall_acc_bef.extend(st['acc_bef'])
        overall_acc_aft.extend(st['acc_aft'])
        
        print(f"{CLASS_LABELS[c]:<22} | {mean_acc_b:>14.2f}% | {mean_acc_a:>14.2f}% | {mean_acc_g:>+12.2f}% | {mean_snr_g:>+10.2f} dB")
        table_rows.append({
            'class_id': c,
            'class_label': CLASS_LABELS[c],
            'accuracy_before_pct': float(mean_acc_b),
            'accuracy_after_pct': float(mean_acc_a),
            'accuracy_gain_pct': float(mean_acc_g),
            'snr_gain_db': float(mean_snr_g),
            'mse_before': float(np.mean(st['mse_bef'])),
            'mse_after': float(np.mean(st['mse_aft']))
        })
        
    avg_acc_bef = np.mean(overall_acc_bef)
    avg_acc_aft = np.mean(overall_acc_aft)
    avg_acc_gain = avg_acc_aft - avg_acc_bef
    
    print("-" * 88)
    print(f"{'OVERALL AVERAGE':<22} | {avg_acc_bef:>14.2f}% | {avg_acc_aft:>14.2f}% | {avg_acc_gain:>+12.2f}% |")
    
    print(f"\n[B] DOWNSTREAM MACHINE CONDITION (ISO 10816) CLASSIFICATION ACCURACY:")
    iso_acc_bef = (total_iso_bef_correct / n_samples) * 100.0
    iso_acc_aft = (total_iso_aft_correct / n_samples) * 100.0
    iso_acc_gain = iso_acc_aft - iso_acc_bef
    print(f"  Machine Health Accuracy BEFORE Filtering (Corrupted by Noise): {iso_acc_bef:.2f}%")
    print(f"  Machine Health Accuracy AFTER Filtering (Disturbance Cleansed): {iso_acc_aft:.2f}%")
    print(f"  Net Machine Health Diagnosis Accuracy Gain:                    {iso_acc_gain:+.2f}%\n")
    
    eval_summary = {
        'disturbance_classification_accuracy': float(acc),
        'signal_reconstruction_accuracy': {
            'overall_before_pct': float(avg_acc_bef),
            'overall_after_pct': float(avg_acc_aft),
            'overall_gain_pct': float(avg_acc_gain)
        },
        'machine_condition_iso_accuracy': {
            'before_filtering_pct': float(iso_acc_bef),
            'after_filtering_pct': float(iso_acc_aft),
            'gain_pct': float(iso_acc_gain)
        },
        'per_class_filtering': table_rows
    }
    
    summary_path = os.path.join(RESULTS_DIR, 'metrics', 'filter_evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Saved comprehensive evaluation summary to: {summary_path}")
    print("=" * 75)
    return eval_summary


if __name__ == '__main__':
    run_comprehensive_evaluation()