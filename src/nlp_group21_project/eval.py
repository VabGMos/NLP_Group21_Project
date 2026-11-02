import torch
from dataset import LABEL_NAMES
from utils import to_device


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=3):
    model.eval()
    total_loss = torch.zeros((), device=device)
    n = 0
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch)
        labels = batch["labels"]
        total_loss += criterion(logits, labels).detach() * labels.size(0)
        n += labels.size(0)
        idx = labels * num_classes + logits.argmax(-1)
        conf += torch.bincount(idx, minlength=num_classes**2).view(
            num_classes, num_classes
        )

    total_loss = total_loss.item()
    conf = conf.cpu().float()
    tp = conf.diag()
    precision = tp / conf.sum(0).clamp(min=1)
    recall = tp / conf.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        "loss": total_loss / n,
        "accuracy": (tp.sum() / conf.sum()).item(),
        "macro_f1": f1.mean().item(),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": precision[i].item(),
                "recall": recall[i].item(),
                "f1": f1[i].item(),
            }
            for i in range(num_classes)
        },
        "confusion_matrix": conf.long().tolist(),
    }
