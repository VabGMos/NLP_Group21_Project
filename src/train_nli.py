"""Train one NLI model on SNLI with Lightning and Weights & Biases.

MPNet tokenizer -> chosen embedding -> recurrent or pooled sentence encoder -> MLP.

Examples:
    uv run python -m src.train_nli --embedding nli-mpnet-base-v2 --rnn gru
    uv run python -m src.train_nli --pooling mean --wandb-mode disabled
    uv run python -m src.train_nli --params-file output/tuning/<study>_best.json
"""

import argparse
import json
from pathlib import Path

import lightning as L

from src.dataset import LABEL_NAMES
from src.embeddings import EMBEDDINGS
from src.lit_data import SNLIDataModule
from src.models import RNN_TYPES
from src.pooling_sentence_encoder import POOLING_TYPES
from src.runner import DEFAULT_PARAMS, make_wandb_logger, train


def parse_args(argv=None):
    d = DEFAULT_PARAMS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--params-file",
        help="JSON of hyperparameters (e.g. tuning output); overrides flags",
    )

    data = p.add_argument_group("data")
    data.add_argument("--max-length", type=int, default=128)
    data.add_argument("--max-train-samples", type=int, default=None)
    data.add_argument("--max-eval-samples", type=int, default=None)
    data.add_argument("--num-workers", type=int, default=0)

    emb = p.add_argument_group("embedding")
    emb.add_argument("--embedding", choices=sorted(EMBEDDINGS), default=d["embedding"])
    emb.add_argument(
        "--train-embeddings", action="store_true", help="fine-tune the embedding table"
    )
    emb.add_argument(
        "--embedding-lr-scale", type=float, default=d["embedding_lr_scale"]
    )
    emb.add_argument("--embedding-dropout", type=float, default=d["embedding_dropout"])

    p.add_argument(
        "--pooling", choices=POOLING_TYPES, default=d["pooling"],
        help="Use mean/sum/max pooling instead of --rnn (default: recurrent encoder)",
    )

    rnn = p.add_argument_group("recurrent encoder")
    rnn.add_argument("--rnn", choices=sorted(RNN_TYPES), default=d["rnn"])
    rnn.add_argument("--hidden-size", type=int, default=d["hidden_size"])
    rnn.add_argument("--num-layers", type=int, default=d["num_layers"])
    rnn.add_argument("--unidirectional", action="store_true")
    rnn.add_argument("--rnn-dropout", type=float, default=d["rnn_dropout"])

    mlp = p.add_argument_group("mlp")
    mlp.add_argument("--mlp-layers", type=int, default=d["mlp_layers"])
    mlp.add_argument("--mlp-width", type=int, default=d["mlp_width"])
    mlp.add_argument("--mlp-dropout", type=float, default=d["mlp_dropout"])

    opt = p.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=10)
    opt.add_argument("--patience", type=int, default=3)
    opt.add_argument("--batch-size", type=int, default=d["batch_size"])
    opt.add_argument("--lr", type=float, default=d["lr"])
    opt.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    opt.add_argument("--clip-norm", type=float, default=d["clip_norm"])
    opt.add_argument("--seed", type=int, default=42)

    log = p.add_argument_group("logging")
    log.add_argument("--wandb-project", default="nlp-group21-nli")
    log.add_argument("--wandb-entity", default=None)
    log.add_argument("--run-name", default=None)
    log.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    log.add_argument("--output-dir", default="output/checkpoints")
    return p.parse_args(argv)


def params_from_args(args) -> dict:
    params = {k: getattr(args, k) for k in DEFAULT_PARAMS if k != "bidirectional"}
    params["bidirectional"] = not args.unidirectional
    if args.params_file:
        params.update(json.loads(Path(args.params_file).read_text()))
    return params


def main(argv=None):
    args = parse_args(argv)
    L.seed_everything(args.seed, workers=True)
    params = params_from_args(args)

    encoder_name = params["pooling"] or params["rnn"]
    run_name = args.run_name or f"{params['embedding']}-{encoder_name}"
    logger = make_wandb_logger(
        args.wandb_project, run_name, args.wandb_entity, mode=args.wandb_mode
    )
    datamodule = SNLIDataModule(
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    trainer, module = train(
        params,
        datamodule,
        max_epochs=args.epochs,
        patience=args.patience,
        logger=logger,
        checkpoint_dir=str(Path(args.output_dir) / run_name),
    )
    results = trainer.test(module, datamodule=datamodule, ckpt_path="best")[0]
    print(f"test results ({', '.join(LABEL_NAMES)}): {results}")
    print(f"best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
