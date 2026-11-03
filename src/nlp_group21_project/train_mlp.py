import torch
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np
from typing import Sequence
from attr import asdict
from pathlib import Path

import time
import tqdm
import wandb
from utils import Config, to_device, set_seed
from dataset import SNLIDataset, cache_tokenized_dataset, make_dataloader
from tokenizer import BaseTokenizer, SentenceTransformerTokenizer
from eval import evaluate
from models import MLPNLI, MLPSimple


MODEL_REGISTRY = {"mlp": MLPNLI, "mlp_simple": MLPSimple}


def _wandb_metrics(prefix, metrics):
    logged = {
        f"{prefix}/{name}": metrics[name]
        for name in ("loss", "accuracy", "macro_f1")
        if name in metrics
    }
    for label, class_metrics in metrics.get("per_class", {}).items():
        for name, value in class_metrics.items():
            logged[f"{prefix}/per_class/{label}/{name}"] = value
    return logged


def build_model(name: str, tokenizer: BaseTokenizer, **kwargs) -> nn.Module:
    return MODEL_REGISTRY[name](tokenizer.vocab_size, tokenizer.pad_token_id, **kwargs)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scheduler=None,
    max_grad_norm=1.0,
    log_every=200,
):
    model.train()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    n = 0
    for step, batch in tqdm.tqdm(enumerate(loader, 1), total=len(loader)):
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
        total_loss += loss.detach() * bs
        correct += (logits.argmax(-1) == batch["labels"]).sum()
        n += bs
        if step % log_every == 0:
            print(
                f"  step {step}/{len(loader)}  loss {total_loss.item() / n:.4f}  "
                f"acc {correct.item() / n:.4f}"
            )
    return {"loss": total_loss.item() / n, "accuracy": correct.item() / n}


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


def cross_validate(
    dataset: SNLIDataset,
    tokenizer,
    model_fn,
    optimizer_fn,
    criterion,
    device,
    k=5,
    epochs=3,
    batch_size=64,
    seed=42,
    scheduler_fn=None,
    num_workers=4,
    run_factory=None,
    checkpoint_dir="checkpoints",
):
    """
    model_fn:     () -> nn.Module                       (fresh model per fold)
    optimizer_fn: (params) -> torch.optim.Optimizer     (fresh optimizer per fold)
    scheduler_fn: (optimizer, steps_per_epoch*epochs) -> scheduler | None  (optional)
    run_factory:  (fold) -> wandb run | None            (one tracked run per fold, optional)
    """
    fold_results = []
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for fold, (tr_idx, va_idx) in tqdm.tqdm(
        enumerate(group_kfold_indices(dataset.premises, k, seed), start=1)
    ):
        print(f"\n=== Fold {fold}/{k}  (train {len(tr_idx)}, val {len(va_idx)}) ===")
        set_seed(seed + fold)
        run = run_factory(fold) if run_factory else None

        train_loader = make_dataloader(
            Subset(dataset, tr_idx),
            tokenizer,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = make_dataloader(
            Subset(dataset, va_idx), tokenizer, batch_size, num_workers=num_workers
        )

        model = model_fn().to(device)
        optimizer = optimizer_fn(model.parameters())
        scheduler = (
            scheduler_fn(optimizer, len(train_loader) * epochs)
            if scheduler_fn
            else None
        )
        checkpoint_path = checkpoint_dir / f"cv_fold_{fold:02d}_best.pt"
        best_accuracy = float("-inf")
        best_epoch = 0
        best_metrics = None

        for epoch in tqdm.tqdm(range(1, epochs + 1)):
            tr = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scheduler,
                log_every=10**9,
            )
            va = evaluate(model, val_loader, criterion, device)
            print(
                f"  epoch {epoch}: train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
                f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} F1 {va['macro_f1']:.4f}"
            )
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        **_wandb_metrics("train", tr),
                        **_wandb_metrics("validation", va),
                    }
                )

            if va["accuracy"] > best_accuracy:
                best_accuracy = va["accuracy"]
                best_epoch = epoch
                best_metrics = va
                torch.save(model.state_dict(), checkpoint_path)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        fold_results.append(best_metrics)
        print(
            f"  best fold checkpoint: {checkpoint_path} "
            f"(epoch {best_epoch}, val acc {best_accuracy:.4f})"
        )
        if run is not None:
            run.summary.update(
                {
                    "best/epoch": best_epoch,
                    "best/checkpoint": str(checkpoint_path),
                    **_wandb_metrics("best", best_metrics),
                }
            )
            run.finish()

    summary = {}
    for metric in ["loss", "accuracy", "macro_f1"]:
        vals = np.array([r[metric] for r in fold_results])
        summary[metric] = {
            "mean": vals.mean(),
            "std": vals.std(ddof=1) if k > 1 else 0.0,
            "folds": vals.tolist(),
        }

    print(f"\n=== {k}-fold CV summary ===")
    for m, s in summary.items():
        print(f"  {m:<9} {s['mean']:.4f} ± {s['std']:.4f}")
    return summary, fold_results


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    scheduler=None,
    ckpt_path="best_model.pt",
    run=None,
):
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = float("-inf")
    best_epoch = 0
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        print(f"Epoch {epoch}/{epochs}")
        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        va = evaluate(model, val_loader, criterion, device)
        print(
            f"  train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} macro-F1 {va['macro_f1']:.4f}"
        )
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    **_wandb_metrics("train", tr),
                    **_wandb_metrics("validation", va),
                }
            )
        if va["accuracy"] > best_acc:
            best_acc = va["accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(
        f"Best final-fit checkpoint: {ckpt_path} "
        f"(epoch {best_epoch}, val acc {best_acc:.4f})"
    )
    if run is not None:
        run.summary.update(
            {
                "best/epoch": best_epoch,
                "best/validation_accuracy": best_acc,
                "best/checkpoint": str(ckpt_path),
            }
        )
    return model


def train(cfg: Config = Config()):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group = f"{cfg.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"

    def make_run(job_type, name, **extra_config):
        return wandb.init(
            project=cfg.wandb_project,
            group=group,
            job_type=job_type,
            name=name,
            config={**asdict(cfg), **extra_config},
            mode=cfg.wandb_mode,
            reinit="create_new",
        )

    train_ds = SNLIDataset("train", cfg.max_train_samples)
    val_ds, test_ds = SNLIDataset("validation"), SNLIDataset("test")
    tokenizer = SentenceTransformerTokenizer(cfg.tokenizer_name, cfg.max_length)
    for dataset in (train_ds, val_ds, test_ds):
        cache_tokenized_dataset(dataset, tokenizer)
    criterion = nn.CrossEntropyLoss()

    def model_fn():
        return build_model(cfg.model_name, tokenizer)

    def optimizer_fn(params):
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- optional k-fold cross-validation on the training split ---
    cv_summary = None
    if cfg.cv_folds > 1:
        cv_summary, _ = cross_validate(
            train_ds,
            tokenizer,
            model_fn,
            optimizer_fn,
            criterion,
            device,
            k=cfg.cv_folds,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            num_workers=cfg.num_workers,
            run_factory=lambda fold: make_run("cv_fold", f"cv-fold-{fold}", fold=fold),
            checkpoint_dir=cfg.checkpoint_dir,
        )

    # --- final model: train on full train split, select on validation, report on test ---
    run = make_run("final_fit", "final-fit")
    try:
        if cv_summary is not None:
            run.summary.update(
                {
                    f"cv/{metric}_{stat}": result[stat]
                    for metric, result in cv_summary.items()
                    for stat in ("mean", "std")
                }
            )
        train_loader = make_dataloader(
            train_ds,
            tokenizer,
            cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = make_dataloader(
            val_ds, tokenizer, cfg.batch_size, num_workers=cfg.num_workers
        )
        test_loader = make_dataloader(
            test_ds, tokenizer, cfg.batch_size, num_workers=cfg.num_workers
        )

        model = model_fn().to(device)
        optimizer = optimizer_fn(model.parameters())
        model = fit(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            cfg.epochs,
            run=run,
            ckpt_path=Path(cfg.checkpoint_dir) / "final_best.pt",
        )

        test = evaluate(model, test_loader, criterion, device)
        print(f"\nTest: acc {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}")
        run.log(_wandb_metrics("test", test))
    finally:
        run.finish()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model_name", type=str, default="mlp", help="key in MODEL_REGISTRY"
    )
    argparser.add_argument(
        "--tokenizer_name", type=str, default="sentence-transformers/nli-mpnet-base-v2"
    )
    argparser.add_argument("--max_length", type=int, default=64)
    argparser.add_argument("--batch_size", type=int, default=2048)
    argparser.add_argument("--num_workers", type=int, default=16)
    argparser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    argparser.add_argument("--wandb_project", type=str, default="nlp-group21-project")
    argparser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="set offline to sync runs later or disabled to turn tracking off",
    )
    argparser.add_argument("--epochs", type=int, default=10)
    argparser.add_argument("--lr", type=float, default=3e-4)
    argparser.add_argument("--weight_decay", type=float, default=1e-6)
    argparser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="set e.g. 50_000 for quick runs",
    )
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument(
        "--cv_folds",
        type=int,
        default=1,
        help="set >1 for k-fold cross-validation on the training split",
    )
    args = argparser.parse_args()
    arg_dict = vars(args)

    cfg = Config(**arg_dict)

    print(f"Training with config: {cfg}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    train(cfg)
