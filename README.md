# Training and experiment tracking

Training runs are logged to Weights & Biases (W&B). Authenticate once with
`uv run wandb login`, then run training as usual:

```powershell
uv run python src\nlp_group21_project\train.py --cv_folds 5
```

By default, runs are sent online to the `nlp-group21-project` project. Use
`--wandb_project <name>` to choose another project, or `--wandb_mode offline`
to save runs locally for later syncing. Set `--wandb_mode disabled` to turn
tracking off.

Each invocation creates one W&B group containing one run per CV fold
(`job_type=cv_fold`, name `cv-fold-N`, with `fold` in the run config) plus a
`final_fit` run. Group or filter by `fold` / `job_type` in the W&B UI. Fold runs
log per-epoch `train/*` and `validation/*` metrics and `best/*` summary values;
the `final_fit` run also holds the aggregate `cv/<metric>_mean|std` values and
the final `test/*` metrics. Per-class precision, recall, and F1 are included for validation and
test evaluations. The best-validation-accuracy checkpoint for each CV fold
(`cv_fold_01_best.pt`, etc.) and the final fit (`final_best.pt`) are saved under
`checkpoints/` by default; change this with `--checkpoint_dir`.