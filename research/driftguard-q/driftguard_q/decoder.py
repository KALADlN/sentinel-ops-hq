from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymatching

from .sim import make_surface_code_circuit, sample_window


def build_matching(*, distance: int, rounds: int, physical_error_rate: float) -> pymatching.Matching:
    circuit = make_surface_code_circuit(
        distance=distance,
        rounds=rounds,
        physical_error_rate=physical_error_rate,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def decode_logical(matching: pymatching.Matching, detectors: np.ndarray) -> np.ndarray:
    predictions = np.asarray(matching.decode_batch(detectors.astype(np.uint8)))
    if predictions.ndim == 1:
        return predictions.astype(np.uint8)
    return predictions[:, 0].astype(np.uint8)


@dataclass
class RateCalibration:
    event_rates: np.ndarray
    physical_error_rates: np.ndarray

    def estimate(self, detectors: np.ndarray) -> float:
        observed_rate = float(np.mean(detectors))
        return float(
            np.interp(
                observed_rate,
                self.event_rates,
                self.physical_error_rates,
                left=self.physical_error_rates[0],
                right=self.physical_error_rates[-1],
            )
        )


def calibrate_detector_rate(
    *,
    distance: int,
    rounds: int,
    p_values: list[float],
    shots_per_point: int,
) -> RateCalibration:
    pairs: list[tuple[float, float]] = []
    for p in p_values:
        window = sample_window(
            distance=distance,
            rounds=rounds,
            physical_error_rate=p,
            shots=shots_per_point,
        )
        pairs.append((float(np.mean(window.detectors)), float(p)))

    # Detector density is expected to grow with global noise over this range.
    # Sort by observed density so interpolation stays well-defined even with
    # finite-shot fluctuations.
    pairs.sort(key=lambda item: item[0])
    return RateCalibration(
        event_rates=np.asarray([x[0] for x in pairs], dtype=float),
        physical_error_rates=np.asarray([x[1] for x in pairs], dtype=float),
    )
