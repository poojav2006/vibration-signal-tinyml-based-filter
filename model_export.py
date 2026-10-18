"""
model_export.py
===============
Exports the trained scikit-learn Decision Tree classifier into a static, zero-allocation
C/C++ header file (`models/tinyml_model.h`) for TinyML execution on embedded devices (ESP32).

Features:
- Pure nested if-else structure matching the exact learned thresholds from the tree
- Zero dynamic memory allocation (no malloc/heap)
- Header guard and clean embedded interface:
      int predict_noise_class(const float features[5]);
- Automated numerical parity verification against Python scikit-learn predictions
"""

import os
import joblib
import numpy as np
from sklearn.tree import _tree

from feature_extractor import FEATURE_NAMES


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def generate_c_tree_code(clf) -> str:
    """
    Traverse scikit-learn Decision Tree and output pure C/C++ code.
    """
    tree = clf.tree_
    feature_names = FEATURE_NAMES
    classes = [int(c) for c in clf.classes_]

    lines = []
    lines.append("/*")
    lines.append(" * ===========================================================================")
    lines.append(" * TINYML DECISION TREE NOISE CLASSIFIER FOR EMBEDDED SYSTEMS (ESP32)")
    lines.append(" * Automatically exported from scikit-learn DecisionTreeClassifier")
    lines.append(" * Features:")
    lines.append(" *   features[0] -> Mean")
    lines.append(" *   features[1] -> Standard Deviation")
    lines.append(" *   features[2] -> RMS")
    lines.append(" *   features[3] -> Kurtosis")
    lines.append(" *   features[4] -> Dominant Frequency (Hz)")
    lines.append(" * Outputs:")
    lines.append(" *   1 -> Class 1 (8-12 Hz Tonal Interference)")
    lines.append(" *   2 -> Class 2 (18-22 Hz Tonal Interference)")
    lines.append(" *   3 -> Class 3 (Impulse Noise)")
    lines.append(" *   4 -> Class 4 (Low-Frequency Drift)")
    lines.append(" * ===========================================================================")
    lines.append(" */")
    lines.append("")
    lines.append("#ifndef TINYML_MODEL_H")
    lines.append("#define TINYML_MODEL_H")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("extern \"C\" {")
    lines.append("#endif")
    lines.append("")
    lines.append("// Feature Indices")
    lines.append("#define FEATURE_MEAN          0")
    lines.append("#define FEATURE_STD           1")
    lines.append("#define FEATURE_RMS           2")
    lines.append("#define FEATURE_KURTOSIS      3")
    lines.append("#define FEATURE_DOMINANT_FREQ 4")
    lines.append("")
    lines.append("/**")
    lines.append(" * Predict noise class from 5-element feature vector.")
    lines.append(" * Memory: 0 bytes dynamic allocation.")
    lines.append(" * Return: 1, 2, 3, or 4.")
    lines.append(" */")
    lines.append("static inline int predict_noise_class(const float features[5]) {")

    def recurse(node: int, depth: int):
        indent = "    " * depth
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            f_idx = tree.feature[node]
            f_name = feature_names[f_idx]
            threshold = float(tree.threshold[node])
            
            lines.append(f"{indent}if (features[{f_idx}] <= {threshold:.6f}f) {{ // {f_name}")
            recurse(tree.children_left[node], depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(tree.children_right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:
            # Leaf node: find class with maximum value count
            values = tree.value[node][0]
            best_cls_idx = int(np.argmax(values))
            best_class = classes[best_cls_idx]
            lines.append(f"{indent}return {best_class}; // counts: {values.astype(int).tolist()}")

    recurse(0, 1)
    lines.append("}")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")
    lines.append("#endif // TINYML_MODEL_H")
    return "\n".join(lines)


def export_and_verify():
    """Export the model and verify numerical parity against python predictions."""
    model_path = os.path.join(MODELS_DIR, 'decision_tree.joblib')
    test_pkg_path = os.path.join(MODELS_DIR, 'test_dataset.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
        
    clf = joblib.load(model_path)
    c_code = generate_c_tree_code(clf)
    
    out_h_path = os.path.join(MODELS_DIR, 'tinyml_model.h')
    with open(out_h_path, 'w', encoding='utf-8') as f:
        f.write(c_code)
    print(f"Exported C++ model to: {out_h_path}")
    
    # Parity verification: simulate C tree evaluation in Python
    if os.path.exists(test_pkg_path):
        pkg = joblib.load(test_pkg_path)
        X_test = pkg['X_test']
        py_preds = clf.predict(X_test)
        
        # Test C++ rule parity by traversing tree structure directly with float32 casting
        tree = clf.tree_
        classes = [int(c) for c in clf.classes_]
        
        c_sim_preds = []
        for row in X_test:
            # Emulate single-precision float C evaluation
            f_row = [np.float32(v) for v in row]
            node = 0
            while tree.feature[node] != _tree.TREE_UNDEFINED:
                f_idx = tree.feature[node]
                th = np.float32(tree.threshold[node])
                if f_row[f_idx] <= th:
                    node = tree.children_left[node]
                else:
                    node = tree.children_right[node]
            best_cls = classes[int(np.argmax(tree.value[node][0]))]
            c_sim_preds.append(best_cls)
            
        c_sim_preds = np.array(c_sim_preds)
        parity = np.mean(py_preds == c_sim_preds) * 100.0
        print(f"C++ / Python Rule Parity on {len(X_test)} test vectors: {parity:.2f}% identical")
        assert parity == 100.0, f"Discrepancy detected in exported rules: {parity}%"
        print("100% Numerical Parity Confirmed!")


if __name__ == '__main__':
    export_and_verify()
