import numpy as np
import torch
import torch.nn.functional as F


def compute_iou_metrics(eval_pred) -> dict:
    """
    eval_pred = (logits, labels)
    logits: numpy array (batch, num_classes, h, w)
    labels: numpy array (batch, H, W)
    """
    logits, labels = eval_pred

    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels).long()

    target_h, target_w = labels.shape[1:]
    logits_resized = F.interpolate(
        logits,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    preds = logits_resized.argmax(dim=1).numpy()
    labels_np = labels.numpy()

    ious = []
    for class_id in range(2):
        intersection = np.logical_and(preds == class_id, labels_np == class_id).sum()
        union = np.logical_or(preds == class_id, labels_np == class_id).sum()
        ious.append(intersection / union if union > 0 else float("nan"))

    return {"mean_iou": float(np.nanmean(ious))}
