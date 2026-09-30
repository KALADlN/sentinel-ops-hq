from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import stim


@dataclass
class SyndromeWindow:
    detectors: np.ndarray
    observables: np.ndarray
    physical_error_rate: float


def make_surface_code_circuit(
    *,
    distance: int,
    rounds: int,
    physical_error_rate: float,
) -> stim.Circuit:
    """Build a noisy rotated surface-code memory circuit."""
    p = float(physical_error_rate)
    if not (0.0 <= p < 0.5):
        raise ValueError("physical_error_rate must be in [0, 0.5).")

    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def sample_window(
    *,
    distance: int,
    rounds: int,
    physical_error_rate: float,
    shots: int,
) -> SyndromeWindow:
    circuit = make_surface_code_circuit(
        distance=distance,
        rounds=rounds,
        physical_error_rate=physical_error_rate,
    )
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(
        shots=shots,
        separate_observables=True,
    )
    return SyndromeWindow(
        detectors=np.asarray(detectors, dtype=np.float32),
        observables=np.asarray(observables[:, 0], dtype=np.uint8),
        physical_error_rate=float(physical_error_rate),
    )


def make_schedule(
    *,
    kind: str,
    windows: int,
    baseline_p: float,
    peak_p: float,
) -> list[float]:
    if windows < 2:
        raise ValueError("windows must be >= 2")
    if peak_p < baseline_p:
        raise ValueError("peak_p must be >= baseline_p")

    kind = kind.lower()
    if kind == "step":
        split = windows // 2
        return [baseline_p] * split + [peak_p] * (windows - split)

    if kind == "ramp":
        return np.linspace(baseline_p, peak_p, windows).astype(float).tolist()

    if kind == "sinusoid":
        midpoint = (baseline_p + peak_p) / 2.0
        amplitude = (peak_p - baseline_p) / 2.0
        return [
            float(midpoint - amplitude * math.cos(2.0 * math.pi * i / (windows - 1)))
            for i in range(windows)
        ]

    raise ValueError(f"unknown schedule: {kind}")
