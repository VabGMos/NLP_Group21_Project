class MeanPooling:
    """Average a nonempty [tokens, dimensions] NumPy array over tokens."""

    def __call__(self, vectors):
        return vectors.mean(axis=0)
