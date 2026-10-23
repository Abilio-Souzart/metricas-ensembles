"""Métricas de classificação reunidas em uma tabela."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def metricas(y, prob, corte: float = 0.5) -> dict:
    pred = (np.asarray(prob) >= corte).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "acurácia": accuracy_score(y, pred),
        "precisão": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "especificidade": tn / (tn + fp),
        "F1": f1_score(y, pred, zero_division=0),
        "AUC": roc_auc_score(y, prob),
        "VP": tp, "VN": tn, "FP": fp, "FN": fn,
    }
