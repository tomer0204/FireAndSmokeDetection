import numpy as np


def fuse_probabilities(method, probs, weights=None, eps=1e-6):
    stacked = np.stack(probs, axis=-1).astype(np.float32)

    if method == "mean":
        return np.mean(stacked, axis=-1)

    elif method == "product":
        stacked = np.clip(stacked, eps, 1.0)
        return np.prod(stacked, axis=-1)

    elif method == "weighted":
        if weights is None:
            weights = np.ones(len(probs), dtype=np.float32) / float(len(probs))
        else:
            weights = np.array(weights, dtype=np.float32)
            weights = weights / (np.sum(weights) + eps)
        return np.tensordot(stacked, weights, axes=([-1], [0]))

    else:
        raise ValueError("Unknown fusion method")
