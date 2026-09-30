from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


class MaskedSyndromeAutoencoder(nn.Module):
    """Learns the baseline syndrome distribution without logical labels."""

    def __init__(self, n_detectors: int, hidden: int = 256, bottleneck: int = 64):
        super().__init__()
        hidden = max(32, min(hidden, max(32, n_detectors * 2)))
        bottleneck = max(8, min(bottleneck, hidden // 2))
        self.net = nn.Sequential(
            nn.Linear(n_detectors, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_detectors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DriftModel:
    model: MaskedSyndromeAutoencoder
    threshold: float
    baseline_mean_loss: float
    baseline_std_loss: float
    mask_probability: float


def _masked_batch(x: torch.Tensor, mask_probability: float) -> torch.Tensor:
    mask = torch.rand_like(x) < mask_probability
    out = x.clone()
    out[mask] = 0.0
    return out


def train_drift_model(
    detectors: np.ndarray,
    *,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    mask_probability: float = 0.15,
    seed: int = 7,
    z_threshold: float = 4.0,
) -> DriftModel:
    torch.manual_seed(seed)
    x = torch.as_tensor(detectors, dtype=torch.float32)
    if x.ndim != 2:
        raise ValueError("detectors must have shape [shots, detectors]")

    model = MaskedSyndromeAutoencoder(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    model.train()
    for _ in range(epochs):
        order = torch.randperm(x.shape[0])
        for start in range(0, x.shape[0], batch_size):
            batch = x[order[start : start + batch_size]]
            corrupted = _masked_batch(batch, mask_probability)
            logits = model(corrupted)
            loss = criterion(logits, batch).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    losses = reconstruction_losses(model, detectors, mask_probability=0.0)
    return DriftModel(
        model=model,
        threshold=float(z_threshold),
        baseline_mean_loss=float(losses.mean()),
        baseline_std_loss=float(max(losses.std(ddof=1), 1e-6)),
        mask_probability=mask_probability,
    )


@torch.no_grad()
def reconstruction_losses(
    model: MaskedSyndromeAutoencoder,
    detectors: np.ndarray,
    *,
    mask_probability: float = 0.0,
) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(detectors, dtype=torch.float32)
    if mask_probability > 0:
        x_in = _masked_batch(x, mask_probability)
    else:
        x_in = x
    logits = model(x_in)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        x,
        reduction="none",
    ).mean(dim=1)
    return loss.cpu().numpy()


def drift_score(drift_model: DriftModel, detectors: np.ndarray) -> float:
    """Return a window-level z-score for self-supervised reconstruction loss.

    The model is trained only on baseline syndromes. During adaptation, no
    logical-state labels are required. A positive score means the current
    window is harder for the baseline model to reconstruct than expected.

    We compare the mean reconstruction loss of a window with the baseline mean
    using the standard error of that window. This fixes the overly conservative
    v0.1 behavior that compared a window average against a single-shot 4-sigma
    threshold.
    """
    losses = reconstruction_losses(drift_model.model, detectors)
    n = max(1, int(losses.size))
    standard_error = drift_model.baseline_std_loss / math.sqrt(n)
    return float((losses.mean() - drift_model.baseline_mean_loss) / max(standard_error, 1e-9))
