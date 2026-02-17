# metrics.py
import numpy as np
import torch
import torch.nn.functional as F


def compute_iou_metrics(eval_pred):
    """
    eval_pred = (logits, labels)
    logits: numpy array (batch, num_classes, h, w)
    labels: numpy array (batch, H, W)
    """
    logits, labels = eval_pred

    # Конвертируем в тензоры Torch (интерполяция удобнее в torch)
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels).long()

    # Целевой размер из меток
    target_h, target_w = labels.shape[1:]

    # Интерполяция логитов до размера меток (билинейная)
    logits_resized = F.interpolate(
        logits,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )

    # Получаем предсказания (индексы класса)
    preds = logits_resized.argmax(dim=1).numpy()
    labels_np = labels.numpy()

    ious = []
    for class_id in range(2):  # 2 класса: фон и повреждение
        intersection = np.logical_and(preds == class_id, labels_np == class_id).sum()
        union = np.logical_or(preds == class_id, labels_np == class_id).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)

    mean_iou = np.nanmean(ious)
    return {"mean_iou": mean_iou}