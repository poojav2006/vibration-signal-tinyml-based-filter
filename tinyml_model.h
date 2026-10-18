/*
 * ===========================================================================
 * TINYML DECISION TREE NOISE CLASSIFIER FOR EMBEDDED SYSTEMS (ESP32)
 * Automatically exported from scikit-learn DecisionTreeClassifier
 * Features:
 *   features[0] -> Mean
 *   features[1] -> Standard Deviation
 *   features[2] -> RMS
 *   features[3] -> Kurtosis
 *   features[4] -> Dominant Frequency (Hz)
 * Outputs:
 *   1 -> Class 1 (8-12 Hz Tonal Interference)
 *   2 -> Class 2 (18-22 Hz Tonal Interference)
 *   3 -> Class 3 (Impulse Noise)
 *   4 -> Class 4 (Low-Frequency Drift)
 * ===========================================================================
 */

#ifndef TINYML_MODEL_H
#define TINYML_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

// Feature Indices
#define FEATURE_MEAN          0
#define FEATURE_STD           1
#define FEATURE_RMS           2
#define FEATURE_KURTOSIS      3
#define FEATURE_DOMINANT_FREQ 4

/**
 * Predict noise class from 5-element feature vector.
 * Memory: 0 bytes dynamic allocation.
 * Return: 1, 2, 3, or 4.
 */
static inline int predict_noise_class(const float features[5]) {
    if (features[3] <= 6.803136f) { // kurtosis
        if (features[4] <= 3.515625f) { // dominant_freq
            if (features[2] <= 1.739380f) { // rms
                if (features[3] <= 3.597743f) { // kurtosis
                    if (features[1] <= 0.921918f) { // std
                        return 4; // counts: [0, 0, 0, 1]
                    } else {
                        if (features[2] <= 1.356491f) { // rms
                            return 1; // counts: [0, 0, 0, 0]
                        } else {
                            return 4; // counts: [0, 0, 0, 1]
                        }
                    }
                } else {
                    if (features[0] <= 0.970398f) { // mean
                        return 3; // counts: [0, 0, 0, 0]
                    } else {
                        if (features[1] <= 0.852370f) { // std
                            return 1; // counts: [0, 0, 0, 0]
                        } else {
                            return 4; // counts: [0, 0, 0, 1]
                        }
                    }
                }
            } else {
                if (features[4] <= 1.953125f) { // dominant_freq
                    if (features[3] <= 5.433203f) { // kurtosis
                        if (features[1] <= 1.403084f) { // std
                            return 1; // counts: [0, 0, 0, 0]
                        } else {
                            return 4; // counts: [0, 0, 0, 0]
                        }
                    } else {
                        return 4; // counts: [0, 0, 0, 1]
                    }
                } else {
                    if (features[3] <= 4.810264f) { // kurtosis
                        if (features[4] <= 2.734375f) { // dominant_freq
                            return 4; // counts: [0, 0, 0, 1]
                        } else {
                            return 4; // counts: [0, 0, 0, 0]
                        }
                    } else {
                        return 1; // counts: [0, 0, 0, 0]
                    }
                }
            }
        } else {
            if (features[4] <= 16.796875f) { // dominant_freq
                if (features[1] <= 0.519371f) { // std
                    if (features[3] <= 3.974189f) { // kurtosis
                        if (features[4] <= 12.500000f) { // dominant_freq
                            return 1; // counts: [0, 0, 0, 0]
                        } else {
                            return 2; // counts: [0, 0, 0, 0]
                        }
                    } else {
                        if (features[1] <= 0.319658f) { // std
                            return 3; // counts: [0, 0, 1, 0]
                        } else {
                            return 1; // counts: [0, 0, 0, 0]
                        }
                    }
                } else {
                    if (features[4] <= 8.203125f) { // dominant_freq
                        if (features[3] <= 6.194423f) { // kurtosis
                            return 3; // counts: [0, 0, 0, 0]
                        } else {
                            return 3; // counts: [0, 0, 0, 0]
                        }
                    } else {
                        if (features[4] <= 8.984375f) { // dominant_freq
                            return 1; // counts: [1, 0, 0, 0]
                        } else {
                            return 1; // counts: [0, 0, 0, 0]
                        }
                    }
                }
            } else {
                if (features[3] <= 2.904059f) { // kurtosis
                    if (features[2] <= 1.159698f) { // rms
                        if (features[2] <= 0.986899f) { // rms
                            return 2; // counts: [0, 0, 0, 0]
                        } else {
                            return 2; // counts: [0, 1, 0, 0]
                        }
                    } else {
                        if (features[2] <= 1.231184f) { // rms
                            return 1; // counts: [0, 0, 0, 0]
                        } else {
                            return 2; // counts: [0, 0, 0, 0]
                        }
                    }
                } else {
                    if (features[1] <= 0.368644f) { // std
                        if (features[3] <= 3.840251f) { // kurtosis
                            return 2; // counts: [0, 0, 0, 0]
                        } else {
                            return 3; // counts: [0, 0, 1, 0]
                        }
                    } else {
                        if (features[3] <= 3.046679f) { // kurtosis
                            return 1; // counts: [0, 0, 0, 0]
                        } else {
                            return 2; // counts: [0, 0, 0, 0]
                        }
                    }
                }
            }
        }
    } else {
        if (features[1] <= 0.826211f) { // std
            return 3; // counts: [0, 0, 1, 0]
        } else {
            if (features[4] <= 5.078125f) { // dominant_freq
                return 4; // counts: [0, 0, 0, 1]
            } else {
                if (features[0] <= 0.970230f) { // mean
                    return 3; // counts: [0, 0, 1, 0]
                } else {
                    if (features[3] <= 7.193036f) { // kurtosis
                        return 1; // counts: [1, 0, 0, 0]
                    } else {
                        if (features[0] <= 1.098462f) { // mean
                            return 2; // counts: [0, 0, 0, 0]
                        } else {
                            return 3; // counts: [0, 0, 0, 0]
                        }
                    }
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif // TINYML_MODEL_H