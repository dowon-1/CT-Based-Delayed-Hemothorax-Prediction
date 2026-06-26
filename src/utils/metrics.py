import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


def binary_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf.ravel()

    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-9)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = (sensitivity + specificity) / 2

    if len(np.unique(y_true)) < 2:
        auc = np.nan
        auprc = np.nan
    else:
        auc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

    return {
        "threshold": threshold,
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "auprc": float(auprc) if np.isfinite(auprc) else np.nan,
        "confusion_matrix": conf.tolist(),
    }
