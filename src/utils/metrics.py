import torch
import numpy as np

#metric = evaluate.load("mean_iou") - нужно куда-то добавить

def compute_metrics(pred, metric, num_labels, ignore_index):
    logits, labels = pred
    logits_tensor = torch.from_numpy(logits)

    logits_tensor = torch.nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    pred_labels = logits_tensor.detach().cpu().numpy()

    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=num_labels,
        ignore_index=ignore_index,
        reduce_labels=False,
    )

    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()

    return metrics