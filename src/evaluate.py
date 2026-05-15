import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, precision_recall_curve)

def best_threshold(y_true, probs):
    p, r, t = precision_recall_curve(y_true, probs)
    f1 = 2*p*r / (p+r+1e-8)
    return float(t[np.argmax(f1[:-1])]) if len(t) > 0 else 0.5

def evaluate(y_true, probs, thr, setting, model_name):
    pred = (probs > thr).astype(int)
    return {
        "Setting":   setting,
        "Model":     model_name,
        "Threshold": round(thr, 4),
        "F1":        round(f1_score(y_true, pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, pred, zero_division=0), 4),
        "AUC":       round(roc_auc_score(y_true, probs), 4),
    }
