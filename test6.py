import torch
from src.training.metrics import accuracy, top_k_accuracy, MetricTracker

# ── Test accuracy() ───────────────────────────────────────────

# Perfect accuracy: model predicts correctly for all samples
logits = torch.Tensor([
    [10.0, 0.0, 0.0],   # predicts class 0
    [0.0, 10.0, 0.0],   # predicts class 1
    [0.0, 0.0, 10.0],   # predicts class 2
])

labels = torch.tensor([0, 1, 2])
print(f"Perfect accuracy:  {accuracy(logits, labels):.2f}")  # 1.00

# Zero accuracy: model is wrong on every sample
labels_wrong = torch.tensor([1, 2, 0])
print(f"Zero accuracy:     {accuracy(logits, labels_wrong):.2f}")  # 0.00

# Partial accuracy: 2 out of 4 correct
logits_partial = torch.tensor([
    [10.0, 0.0],   # predicts 0 — correct
    [10.0, 0.0],   # predicts 0 — wrong (label is 1)
    [0.0, 10.0],   # predicts 1 — correct
    [10.0, 0.0],   # predicts 0 — wrong (label is 1)
])
labels_partial = torch.tensor([0, 1, 1, 1])
print(f"Partial accuracy:  {accuracy(logits_partial, labels_partial):.2f}")  # 0.50

# ── Test top_k_accuracy() ─────────────────────────────────────

# 5 classes, true label is 2
# Model gives highest score to class 0, but class 2 is in top-3
logits_5class = torch.tensor([
    [3.0, 0.1, 2.5, 0.2, 1.8],   # top-3: [0, 2, 4]
])
label_5class = torch.tensor([2])
print(f"\nTop-3 acc (label in top-3): {top_k_accuracy(logits_5class, label_5class, k=3):.2f}")  # 1.00
print(f"Top-1 acc (label not #1):   {accuracy(logits_5class, label_5class):.2f}")              # 0.00

# ── Test MetricTracker ────────────────────────────────────────
print("\n── MetricTracker across 3 unequal batches ───────────────")
tracker = MetricTracker()

# Batch 1: 4 samples, 3 correct, loss=0.5
logits_b1 = torch.tensor([
    [10.0, 0.0], [10.0, 0.0],
    [0.0, 10.0], [10.0, 0.0],
])
labels_b1 = torch.tensor([0, 0, 1, 1])   # 3 correct, 1 wrong
tracker.update(loss=0.5, logits=logits_b1, labels=labels_b1)

# Batch 2: 4 samples, 4 correct, loss=0.3
logits_b2 = torch.tensor([
    [10.0, 0.0], [0.0, 10.0],
    [10.0, 0.0], [0.0, 10.0],
])
labels_b2 = torch.tensor([0, 1, 0, 1])   # 4 correct
tracker.update(loss=0.3, logits=logits_b2, labels=labels_b2)

# Batch 3: 2 samples (smaller last batch), 1 correct, loss=0.6
logits_b3 = torch.tensor([
    [10.0, 0.0], [10.0, 0.0],
])
labels_b3 = torch.tensor([0, 1])          # 1 correct, 1 wrong
tracker.update(loss=0.6, logits=logits_b3, labels=labels_b3)

metrics = tracker.compute()
print(f"Loss:     {metrics['loss']:.4f}")       # (0.5+0.3+0.6)/3 = 0.4667
print(f"Accuracy: {metrics['acc']:.4f}")        # 8/10 = 0.8000
print(f"Top5 acc: {metrics['top5_acc']:.4f}")   # N/A for 2 classes

# ── Verify reset works ────────────────────────────────────────
tracker.reset()
try:
    tracker.compute()
except RuntimeError as e:
    print(f"\nreset() works — compute() on empty tracker raises: {e}")

