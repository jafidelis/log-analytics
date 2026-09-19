import numpy as np


def sweep_best_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> dict:
    """Varredura exata: melhor F1 entre todos os cortes possíveis."""
    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    s_sorted = scores[order]
    total_anomalies = int(y_true.sum())

    best = None
    tp = fp = 0
    index = 0
    n = len(s_sorted)

    while index < n:
        threshold = s_sorted[index]
        while index < n and s_sorted[index] == threshold:
            if y_sorted[index]:
                tp += 1
            else:
                fp += 1
            index += 1

        precision = tp / (tp + fp)
        recall = tp / total_anomalies
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if best is None or f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": total_anomalies - tp,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

    return best