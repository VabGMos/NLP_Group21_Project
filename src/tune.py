"""Bayesian hyperparameter tuning (Optuna TPE) for the NLI model, logged to wandb.

Each trial trains on a (sub)sample of SNLI, reports validation accuracy every epoch and
is pruned early if it lags behind earlier trials. Every trial is its own wandb run, grouped
by study name. Studies are stored in SQLite, so an interrupted study can be resumed with the
same --study-name, and several processes can share one study.

Examples:
    uv run python -m src.tune --n-trials 40 --max-train-samples 100000
    uv run python -m src.tune --embeddings nli-mpnet-base-v2 --rnns gru lstm --sampler random
Then train the winner on the full data:
    uv run python -m src.train_nli --params-file output/tuning/<study>_best.json
"""

import argparse
import json
from pathlib import Path

import lightning as L
import optuna
from lightning.pytorch.callbacks import Callback

import wandb
from src.embeddings import EMBEDDINGS
from src.lit_data import SNLIDataModule
from src.models import RNN_TYPES
from src.runner import make_wandb_logger, train
from src.search_space import suggest_params

MONITOR = "val/accuracy"


class OptunaPruningCallback(Callback):
    """Report the monitored validation metric each epoch and stop pruned trials."""

    def __init__(self, trial: optuna.Trial, monitor: str = MONITOR):
        self.trial, self.monitor, self.best = trial, monitor, float("-inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        score = float(trainer.callback_metrics[self.monitor])
        self.best = max(self.best, score)
        self.trial.report(score, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned at epoch {trainer.current_epoch}")


def make_sampler(name: str, seed: int):
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=10)


def make_pruner(name: str, max_epochs: int):
    if name == "none":
        return optuna.pruners.NopPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max_epochs)
    return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-name", default="nli-tuning")
    p.add_argument("--storage", default="sqlite:///output/tuning/optuna.db")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--timeout", type=int, default=None, help="seconds")
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument(
        "--pruner", choices=["median", "hyperband", "none"], default="median"
    )
    p.add_argument("--embeddings", nargs="+", choices=sorted(EMBEDDINGS), default=None)
    p.add_argument("--rnns", nargs="+", choices=sorted(RNN_TYPES), default=["gru"])
    p.add_argument("--max-epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-train-samples", type=int, default=100_000)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="nlp-group21-nli")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    p.add_argument("--output-dir", default="output/tuning")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.storage.startswith("sqlite:///"):
        Path(args.storage.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )

    datamodule = SNLIDataModule(
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    datamodule.setup()  # tokenize once; every trial reuses the data

    def objective(trial: optuna.Trial) -> float:
        L.seed_everything(args.seed + trial.number, workers=True)
        params = suggest_params(trial, args.embeddings, args.rnns)
        logger = make_wandb_logger(
            args.wandb_project,
            name=f"{args.study_name}-trial{trial.number}",
            entity=args.wandb_entity,
            group=args.study_name,
            mode=args.wandb_mode,
        )
        pruning = OptunaPruningCallback(trial)
        try:
            train(
                params,
                datamodule,
                max_epochs=args.max_epochs,
                patience=args.patience,
                monitor=MONITOR,
                logger=logger,
                callbacks=[pruning],
                enable_progress_bar=False,
            )
        finally:
            wandb.finish()
        return pruning.best

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=make_sampler(args.sampler, args.seed),
        pruner=make_pruner(args.pruner, args.max_epochs),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best = study.best_trial
    print(f"\nbest {MONITOR}: {best.value:.4f} (trial {best.number})")
    print(json.dumps(best.params, indent=2))
    best_file = out_dir / f"{args.study_name}_best.json"
    best_file.write_text(json.dumps(best.params, indent=2))
    print(f"saved to {best_file}")
    print(
        f"train on full data: uv run python -m src.train_nli --params-file {best_file}"
    )


if __name__ == "__main__":
    main()
