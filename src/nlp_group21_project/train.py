import torch
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np
from typing import Sequence
from utils import Config, to_device, set_seed
from dataset import SNLIDataset, make_dataloader
from tokenizer import BaseTokenizer, SentenceTransformerTokenizer
from eval import evaluate
from models import MLPNLI


MODEL_REGISTRY = {"mlp": MLPNLI}


def build_model(name: str, tokenizer: BaseTokenizer, **kwargs) -> nn.Module:
    return MODEL_REGISTRY[name](tokenizer.vocab_size, tokenizer.pad_token_id, **kwargs)


def train_one_epoch(model, loader, optimizer, criterion, device,
                    scheduler=None, max_grad_norm=1.0, log_every=200):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for step, batch in enumerate(loader, 1):
        batch = to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["labels"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()

        bs = batch["labels"].size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(-1) == batch["labels"]).sum().item()
        n += bs
        if step % log_every == 0:
            print(
                f"  step {step}/{len(loader)}  loss {total_loss/n:.4f}  acc {correct/n:.4f}")
    return {"loss": total_loss / n, "accuracy": correct / n}


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------
def group_kfold_indices(groups: Sequence[str], k: int, seed: int = 42):
    """Yield (train_idx, val_idx) per fold; all items sharing a group land in the same fold."""
    rng = np.random.RandomState(seed)
    unique = sorted(set(groups))
    rng.shuffle(unique)
    fold_of = {g: i % k for i, g in enumerate(unique)}
    fold_ids = np.array([fold_of[g] for g in groups])
    for f in range(k):
        yield np.where(fold_ids != f)[0], np.where(fold_ids == f)[0]


def cross_validate(dataset: SNLIDataset, tokenizer, model_fn, optimizer_fn, criterion,
                   device, k=5, epochs=3, batch_size=64, seed=42, scheduler_fn=None):
    """
    model_fn:     () -> nn.Module                       (fresh model per fold)
    optimizer_fn: (params) -> torch.optim.Optimizer     (fresh optimizer per fold)
    scheduler_fn: (optimizer, steps_per_epoch*epochs) -> scheduler | None  (optional)
    """
    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(
            group_kfold_indices(dataset.premises, k, seed), start=1):
        print(
            f"\n=== Fold {fold}/{k}  (train {len(tr_idx)}, val {len(va_idx)}) ===")
        set_seed(seed + fold)

        train_loader = make_dataloader(
            Subset(dataset, tr_idx), tokenizer, batch_size, shuffle=True)
        val_loader = make_dataloader(
            Subset(dataset, va_idx), tokenizer, batch_size)

        model = model_fn().to(device)
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer, len(
            train_loader) * epochs) if scheduler_fn else None

        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                 scheduler, log_every=10**9)
            va = evaluate(model, val_loader, criterion, device)
            print(f"  epoch {epoch}: train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
                  f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} F1 {va['macro_f1']:.4f}")

        # metrics of the final epoch, no checkpoint picking on the val fold
        fold_results.append(va)

    summary = {}
    for metric in ["loss", "accuracy", "macro_f1"]:
        vals = np.array([r[metric] for r in fold_results])
        summary[metric] = {"mean": vals.mean(), "std": vals.std(ddof=1) if k > 1 else 0.0,
                           "folds": vals.tolist()}

    print(f"\n=== {k}-fold CV summary ===")
    for m, s in summary.items():
        print(f"  {m:<9} {s['mean']:.4f} ± {s['std']:.4f}")
    return summary, fold_results


def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs,
        scheduler=None, ckpt_path="best_model.pt"):
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        tr = train_one_epoch(model, train_loader, optimizer,
                             criterion, device, scheduler)
        va = evaluate(model, val_loader, criterion, device)
        print(f"  train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} macro-F1 {va['macro_f1']:.4f}")
        if va["accuracy"] > best_acc:
            best_acc = va["accuracy"]
            torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


def train(cfg: Config = Config()):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SNLIDataset("train", cfg.max_train_samples)
    val_ds, test_ds = SNLIDataset("validation"), SNLIDataset("test")
    tokenizer = SentenceTransformerTokenizer(
        cfg.tokenizer_name, cfg.max_length)
    criterion = nn.CrossEntropyLoss()

    def model_fn(): return build_model(cfg.model_name, tokenizer)

    def optimizer_fn(params): return torch.optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- optional k-fold cross-validation on the training split ---
    if cfg.cv_folds > 1:
        cross_validate(train_ds, tokenizer, model_fn, optimizer_fn, criterion, device,
                       k=cfg.cv_folds, epochs=cfg.epochs, batch_size=cfg.batch_size, seed=cfg.seed)

    # --- final model: train on full train split, select on validation, report on test ---
    train_loader = make_dataloader(
        train_ds, tokenizer, cfg.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, tokenizer, cfg.batch_size)
    test_loader = make_dataloader(test_ds, tokenizer, cfg.batch_size)

    model = model_fn().to(device)
    optimizer = optimizer_fn(model.parameters())
    model = fit(model, train_loader, val_loader,
                optimizer, criterion, device, cfg.epochs)

    test = evaluate(model, test_loader, criterion, device)
    print(
        f"\nTest: acc {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}")


if __name__ == "__main__":

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str,
                           default="mlp", help="key in MODEL_REGISTRY")
    argparser.add_argument("--tokenizer_name", type=str,
                           default="sentence-transformers/nli-mpnet-base-v2")
    argparser.add_argument("--max_length", type=int, default=128)
    argparser.add_argument("--batch_size", type=int, default=512)
    argparser.add_argument("--epochs", type=int, default=10)
    argparser.add_argument("--lr", type=float, default=3e-4)
    argparser.add_argument("--weight_decay", type=float, default=1e-5)
    argparser.add_argument("--max_train_samples", type=int, default=50000,
                           help="set e.g. 50_000 for quick runs")
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--cv_folds", type=int, default=1,
                           help="set >1 for k-fold cross-validation on the training split")
    args = argparser.parse_args()
    arg_dict = vars(args)

    cfg = Config(**arg_dict)

    print(f"Training with config: {cfg}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    train(cfg)
