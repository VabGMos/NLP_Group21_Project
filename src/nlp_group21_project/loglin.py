from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
from joblib import Parallel, delayed


def encode_batch(batch, encoder):
    return [encoder(text) for text in batch]


def parallel_encode(texts, encoder, n_jobs=-1, batch_size=1000):
    batches = [
        texts[i:i + batch_size]
        for i in range(0, len(texts), batch_size)
    ]

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
    )(
        delayed(encode_batch)(batch, encoder)
        for batch in batches
    )

    return [
        vector
        for batch in results
        for vector in batch
    ]


def encode_sparse(texts, encoder, batch_size=1000):
    """Encode one small dense batch at a time, keeping completed batches sparse."""
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors = np.asarray(
            [encoder(text) for text in batch],
            dtype=np.float32,
        )
        chunks.append(csr_matrix(vectors))
    return vstack(chunks, format="csr")


def load(path):
    """
    Load a logistic model from {path}.pickle and return it.
    """
    with open(f"{path}.pickle", "rb") as f:
        logistic_model = pickle.load(f)
    return logistic_model


def fit(training_set, encoder: "Encoder", save_to=None, max_iter=100, n_jobs=-1):
    """
    Constructs, trains and returns a softmax regressor.
    Uses GPU (cuML) when available, otherwise CPU (scikit-learn).
    """

    # Convert each small encoding batch to sparse form immediately.
    X_train_sparse = encode_sparse(
        [text for text, label in training_set],
        encoder,
    )
    training_labels = np.array([label for text, label in training_set])

    # Check GPU availability
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sparse
        from cuml.linear_model import LogisticRegression as GPULogReg

        gpu_available = cp.cuda.runtime.getDeviceCount() > 0

    except (ImportError, RuntimeError):
        gpu_available = False

    if gpu_available:
        print("Training on GPU")

        X_gpu = cpx_sparse.csr_matrix(X_train_sparse)
        y_gpu = cp.asarray(training_labels)

        logistic_model = GPULogReg(max_iter=max_iter)
        logistic_model.fit(X_gpu, y_gpu)

    else:
        print("Training on CPU")

        logistic_model = LogisticRegression(
            solver="saga",
            max_iter=max_iter,
        )
        logistic_model.fit(X_train_sparse, training_labels)

    # Save model if requested
    if save_to is not None:
        with open(f"{save_to}.pickle", "wb") as f:
            pickle.dump(
                logistic_model,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    return logistic_model



def predict(texts, model, encoder: "Encoder"):
    """
    Predicts labels using a trained logistic regression model.
    Supports both GPU (cuML) and CPU (scikit-learn).
    """
    # Encode validation data
    X_val_sparse = encode_sparse(texts, encoder)

    # Check whether the trained model uses GPU
    if type(model).__module__.startswith("cuml"):
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sparse

        # Transfer validation data to GPU
        X_gpu = cpx_sparse.csr_matrix(X_val_sparse)

        # Predict on GPU
        predictions = model.predict(X_gpu)

        # Convert predictions back to NumPy
        return cp.asarray(predictions).get()

    # Predict on CPU
    return model.predict(X_val_sparse)


def predict_probs(texts, model, encoder: "Encoder"):
    """
    Predicts labels (via the most probable class decision rule) for a collection of texts.
    (scikit-learn hard-codes the most probable class as the decision rule of choice)

    texts: a list of texts, each a string
    model: a trained instance of sklearn's LogisticRegression
    encoder: the Encoder used for in the model's training
    """
    validation_vectors = [encoder(text) for text in texts]
    X_val_sparse = csr_matrix(validation_vectors)
    probs = model.predict_proba(X_val_sparse)
    return probs
