import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

@torch.no_grad()
def classification_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    out = {}
    out["acc"] = accuracy_score(y_true, y_pred)
    out["f1"] = f1_score(y_true, y_pred, average="binary")
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["auc"] = float("nan")
    out["cm"] = confusion_matrix(y_true, y_pred).tolist()
    return out
