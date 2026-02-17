# metrics.py
import numpy as np

def compute_iou_metrics(eval_pred):
    """
    eval_pred = (logits, labels)
    logits: numpy array (batch, num_classes, h, w)
    labels: numpy array (batch, h, w)
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    ious = []
    for class_id in range(2):  # для двух классов
        intersection = np.logical_and(preds == class_id, labels == class_id).sum()
        union = np.logical_or(preds == class_id, labels == class_id).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)

    mean_iou = np.nanmean(ious)
    return {"mean_iou": mean_iou}