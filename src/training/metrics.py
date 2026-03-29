# src/training/metrics.py

from __future__ import annotations

import torch

def accuracy(
        logits : torch.Tensor,
        labels : torch.Tensor,
) -> float:
    """
    Top-1 classification accuracy for a single batch.

    Args:
        logits: raw model output, shape [batch_size, num_classes]
                These are the numbers that come directly out of
                the model's final Linear layer — NOT softmax,
                NOT probabilities.

        labels: ground truth class indices, shape [batch_size]
                Integer values in range [0, num_classes - 1]

    Returns:
        float between 0.0 and 1.0
        e.g. 0.91 means 91% of samples in this batch
        were classified correctly

    How it works:
        logits.argmax(dim=-1) finds the index of the largest
        logit for each sample — that index is the predicted class.
        dim=-1 means "along the last dimension" which is the
        class dimension for shape [batch_size, num_classes].

        == labels compares predicted class to true class
        element-wise, producing a bool tensor.

        .float() converts True/False to 1.0/0.0

        .mean() averages them — fraction that were correct.

        .item() extracts the Python float from the tensor.

    Example:
        logits = [[2.1, 0.3, -1.2],   → argmax → [0, 2]
                  [-0.4, 0.1, 3.8]]
        labels = [0, 2]               → [correct, correct]
        accuracy = 1.0
    """
    predicted = logits.argmax(dim=-1)
    return (predicted == labels).float().mean().item()

def top_k_accuracy(
        logits : torch.Tensor,
        labels : torch.Tensor,
        k : int = 5,
) -> float:
    """
    Top-k classification accuracy for a single batch.

    A prediction is "correct" if the true label appears
    anywhere in the model's top-k predictions.

    Why top-5 matters:
        ImageNet has 1000 classes. Many classes are very similar
        (150 breeds of dogs). Top-5 accuracy measures whether
        the model has the right general idea even if it picks
        the slightly wrong variant.
        For CIFAR-10 with only 10 classes, top-5 is less useful
        but good to have for projects with many classes.

    Args:
        logits: shape [batch_size, num_classes]
        labels: shape [batch_size]
        k:      how many top predictions to consider

    Returns:
        float between 0.0 and 1.0

    Example with k=3:
        logits for one sample: [0.1, 3.2, 0.8, -0.5, 2.1]
        top-3 indices:         [1, 4, 2]
        true label:            2
        correct? Yes — 2 is in the top-3.
    """

    # topk returns (values, indices) — we only need indices
    # largest=True means we want the highest logits
    top_k_indices = logits.topk(k=k, dim=-1, largest=True).indices
    # labels.unsqueeze(-1) reshapes [B] → [B, 1] for comparison
    # with top_k_indices of shape [B, k]
    correct = labels.unsqueeze(-1).eq(top_k_indices).any(dim=-1)
    return correct.float().mean().item()

class MetricTracker:
    """
    Accumulates metrics across all batches in an epoch and
    computes the correct epoch-level average at the end.

    Why not just average batch metrics?
        If your last batch has 37 samples instead of 128,
        averaging batch accuracies weights that batch equally
        to full batches — giving the wrong epoch accuracy.
        MetricTracker accumulates the raw counts (correct predictions
        and total samples) and divides at the end, which is correct
        regardless of batch sizes.

    Usage in train_epoch():
        tracker = MetricTracker()
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            tracker.update(loss=loss.item(), logits=logits, labels=y)
        metrics = tracker.compute()
        # metrics = {"loss": 0.32, "acc": 0.91, "top5_acc": 0.99}
    """

    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        """
        Clear all accumulated values.
        Call this at the start of each epoch.
        """
        self._total_loss : float = 0.0
        self._n_correct : int = 0
        self._n_correct_top5 : int  =0
        self._n_total : int = 0
        self._n_batches : int = 0

    def update(
            self,
            loss : float,
            logits : torch.Tensor,
            labels : torch.Tensor,
    ) -> None:
        """
        Accumulate one batch's worth of results.

        Args:
            loss:   scalar loss value for this batch (.item() already called)
            logits: model output, shape [batch_size, num_classes]
            labels: ground truth, shape [batch_size]

        Why accumulate counts instead of averaging?
            Correct: n_correct=91, n_total=100 → acc=0.91
            Wrong:   avg([0.90, 0.92]) = 0.91 only if batches are equal size

            The correct approach always works.
            The wrong approach silently gives wrong numbers
            when batch sizes differ.
        """
        batch_size = labels.size(0)
        num_classes = logits.size(-1)

        # Accumulate loss total — divide by n_batches at the end
        self._total_loss += loss
        self._n_batches += 1

        # Accumulate correct predictions — numerator
        predicted  = logits.argmax(dim=-1)
        self._n_correct += (predicted == labels).sum().item()

        # Top-5 only makes sense when num_classes >= 5
        if num_classes > 5:
            k = min(5, num_classes)
            top_k_index = logits.topk(k=k, dim=-1).indices
            self._n_correct_top5 += (
                labels.unsqueeze(-1).eq(top_k_index).any(dim=-1).sum().item()
            )
        
        # Accumulate total samples — denominator
        self._n_total += batch_size

    def compute(self) -> dict[str, float]:
        """
        Compute final epoch metrics from accumulated counts.

        Returns:
            dict with keys: "loss", "acc", "top5_acc"
            All values are floats.

        Call this once at the end of the epoch, after all
        batches have been passed to update().
        """
        if self._n_total == 0:
            raise RuntimeError(
                "No batches were accumulated. "
                "Did you forget to call update()?"
            )
        
        return {
            # Average loss over batches
            "loss" : self._total_loss / self._n_batches,

            # Fraction of samples correctly classified (top-1)
            "acc" : self._n_correct / self._n_total,

            # Fraction where true label is in top-5 predictions

            "top5_acc" : self._n_correct_top5 / self._n_total,
        }
