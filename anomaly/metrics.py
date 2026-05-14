from typing import Dict, Iterable, Optional

import numpy as np


def compute_image_metrics(labels: Iterable[int], scores: Iterable[float], threshold: Optional[float] = None) -> Dict[str, float]:
    labels = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(scores), dtype=np.float64)
    valid = labels >= 0
    labels = labels[valid]
    scores = scores[valid]
    metrics: Dict[str, float] = {}

    if labels.size == 0:
        return {"num_labeled": 0}
    metrics["num_labeled"] = int(labels.size)
    metrics["num_normal"] = int((labels == 0).sum())
    metrics["num_defect"] = int((labels == 1).sum())

    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
        if len(np.unique(labels)) == 2:
            metrics["auroc"] = float(roc_auc_score(labels, scores))
            metrics["auprc"] = float(average_precision_score(labels, scores))
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            best_idx = int(np.nanargmax(f1))
            metrics["best_f1"] = float(f1[best_idx])
            if best_idx < thresholds.size:
                metrics["best_f1_threshold"] = float(thresholds[best_idx])
        else:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
    except Exception:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    if threshold is not None:
        pred = (scores >= float(threshold)).astype(np.int64)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        metrics.update({"tp": tp, "fp": fp, "tn": tn, "fn": fn})
        metrics["precision_at_threshold"] = tp / max(tp + fp, 1)
        metrics["recall_at_threshold"] = tp / max(tp + fn, 1)
        metrics["f1_at_threshold"] = (2 * tp) / max(2 * tp + fp + fn, 1)
        metrics["false_positive_rate"] = fp / max(fp + tn, 1)
        metrics["false_negative_rate"] = fn / max(fn + tp, 1)
    return metrics
