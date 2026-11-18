"""Pretrained MPNet token-embedding tables, selectable by short name.

All registered models share the MPNet vocabulary, so one tokenizer (see
``src.tokenizer.load_mpnet_tokenizer``) can be combined with any of them.
"""

from functools import cache

import torch
from transformers import AutoModel

EMBEDDINGS = {
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "nli-mpnet-base-v2": "sentence-transformers/nli-mpnet-base-v2",
}


def register_embedding(name: str, hf_model_name: str) -> None:
    """Make another MPNet-vocabulary checkpoint selectable by ``name``."""
    EMBEDDINGS[name] = hf_model_name


def load_embedding_weights(name: str) -> torch.Tensor:
    """Return the (vocab_size, dim) input-embedding matrix of a registered model.

    Results are cached, so treat the returned tensor as read-only (models clone it).
    """
    return _load(name)


@cache
def _load(name: str) -> torch.Tensor:
    if name not in EMBEDDINGS:
        raise KeyError(f"Unknown embedding {name!r}; choose from {sorted(EMBEDDINGS)}")
    model = AutoModel.from_pretrained(EMBEDDINGS[name])
    weights = model.get_input_embeddings().weight.detach().cpu().clone()
    del model
    return weights
