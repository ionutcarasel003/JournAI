import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply sigmoid then threshold at 0.5
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    # Convert to shapes (N, C)
    labels = np.array(labels)

    # Example-based accuracy (subset accuracy): exact match of all labels
    # For single-label encoded as one-hot, this behaves like standard accuracy
    subset_acc = (preds == labels).all(axis=1).mean()

    # Micro F1 over all labels
    f1_micro = f1_score(labels.flatten(), preds.flatten(), average='micro', zero_division=0)

    return {
        'accuracy': subset_acc,
        'f1_micro': f1_micro,
    }
