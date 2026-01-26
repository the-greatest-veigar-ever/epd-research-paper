# Research Report: XGBoost Malicious/Benign Watcher

This report documents the performance results of the **XGBoost Malicious/Benign Classifier** implemented for the **Watcher layer**. The system uses behavioral network metadata to detect threats with high sensitivity and low latency.

## 1. Methodology
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Dataset**: CSE-CIC-IDS2018 (Heavily pre-processed)
- **Features**: 21 behavioral indicators including TCP flags, Inter-Arrival Times (IAT), Destination Ports, and Flow Rates.
- **Optimization**: Hyperparameter tuning via `RandomizedSearchCV` and Overfitting protection via `Early Stopping`.
- **Class Imbalance**: Handled via `scale_pos_weight` (3.02) to prioritize Recall.

## 2. Experimental Results

### Performance Metrics
The model was evaluated on a test set of 20,000 samples.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | 0.9675 | High overall reliability |
| **Recall (Malicious)** | **0.9187** | Successfully detects >91% of attacks |
| **Precision (Malicious)** | 0.9452 | Low false positive rate |
| **F1-Score** | 0.9318 | Balanced performance |

### Classification Detailed Report
```text
              precision    recall  f1-score   support

      Benign       0.97      0.98      0.98     15028
   Malicious       0.95      0.92      0.93      4972

    accuracy                           0.97     20000
```

## 3. Operational Efficiency
The Watcher is designed for real-time operation in high-traffic environments.

- **Batch Inference Speed**: 0.00139 ms per log entry.
- **Single-log Live Latency**: **~4.10 ms** (including data mapping and context extraction).
- **Deployment Ready**: Optimized for low CPU/Memory footprint via `tree_method="hist"`.

## 4. Live Verification Output
Verified against external attack metadata (HOIC DDoS scenario):

```text
Scanning enriched test logs...
[watcher-test] Scanning 2 log entries with XGBoost...
[watcher-test][ML] Inference Time: 4.1068 ms
[watcher-test] !!! THREAT DETECTED (Risk: 0.98): Verified_HOIC_Attack !!!
Alerts generated: 1
Alert: DDOS attack-HOIC with Risk: 0.9814
```

---
*Report Generated: 2026-01-25*
