from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .decoder import (
    build_matching,
    calibrate_detector_rate,
    decode_logical,
)
from .model import drift_score, train_drift_model
from .sim import make_schedule, sample_window


def run_experiment(
    *,
    distance: int = 3,
    rounds: int = 5,
    baseline_p: float = 0.002,
    peak_p: float = 0.012,
    schedule: str = "step",
    windows: int = 12,
    train_shots: int = 4000,
    shots_per_window: int = 1200,
    calibration_shots: int = 1500,
    epochs: int = 8,
    seed: int = 7,
) -> dict:
    np.random.seed(seed)

    baseline = sample_window(
        distance=distance,
        rounds=rounds,
        physical_error_rate=baseline_p,
        shots=train_shots,
    )
    drift_model = train_drift_model(
        baseline.detectors,
        epochs=epochs,
        seed=seed,
    )

    grid = np.linspace(
        max(0.0002, baseline_p * 0.25),
        max(peak_p * 1.25, baseline_p * 2.0),
        10,
    ).astype(float).tolist()

    calibration = calibrate_detector_rate(
        distance=distance,
        rounds=rounds,
        p_values=grid,
        shots_per_point=calibration_shots,
    )

    static_matching = build_matching(
        distance=distance,
        rounds=rounds,
        physical_error_rate=baseline_p,
    )
    adaptive_matching = static_matching
    active_p = baseline_p

    p_schedule = make_schedule(
        kind=schedule,
        windows=windows,
        baseline_p=baseline_p,
        peak_p=peak_p,
    )

    rows = []
    static_failures = 0
    adaptive_failures = 0
    total_shots = 0

    for index, true_p in enumerate(p_schedule):
        window = sample_window(
            distance=distance,
            rounds=rounds,
            physical_error_rate=true_p,
            shots=shots_per_window,
        )

        score = drift_score(drift_model, window.detectors)
        drifted = bool(score > drift_model.threshold)
        estimated_p = calibration.estimate(window.detectors)

        if drifted:
            # Critical research constraint:
            # this update uses syndrome data only. No logical observable is fed
            # into the drift detector, p estimator, or decoder selection.
            adaptive_matching = build_matching(
                distance=distance,
                rounds=rounds,
                physical_error_rate=estimated_p,
            )
            active_p = estimated_p

        static_pred = decode_logical(static_matching, window.detectors)
        adaptive_pred = decode_logical(adaptive_matching, window.detectors)

        static_errors = int(np.count_nonzero(static_pred != window.observables))
        adaptive_errors = int(np.count_nonzero(adaptive_pred != window.observables))

        static_failures += static_errors
        adaptive_failures += adaptive_errors
        total_shots += shots_per_window

        rows.append(
            {
                "window": index,
                "true_p": float(true_p),
                "detector_event_rate": float(np.mean(window.detectors)),
                "drift_score": score,
                "drift_threshold": drift_model.threshold,
                "drift_triggered": drifted,
                "estimated_p_from_syndrome": estimated_p,
                "adaptive_decoder_p": float(active_p),
                "static_logical_error_rate": static_errors / shots_per_window,
                "adaptive_logical_error_rate": adaptive_errors / shots_per_window,
            }
        )

    static_ler = static_failures / total_shots
    adaptive_ler = adaptive_failures / total_shots
    relative_change = (
        (adaptive_ler - static_ler) / static_ler if static_ler > 0 else 0.0
    )

    return {
        "config": {
            "distance": distance,
            "rounds": rounds,
            "baseline_p": baseline_p,
            "peak_p": peak_p,
            "schedule": schedule,
            "windows": windows,
            "train_shots": train_shots,
            "shots_per_window": shots_per_window,
            "calibration_shots": calibration_shots,
            "epochs": epochs,
            "seed": seed,
        },
        "adaptation_constraint": (
            "Syndrome data only is used for drift detection, noise estimation, "
            "and adaptive decoder updates. Logical observables are evaluation-only."
        ),
        "baseline_drift_threshold": drift_model.threshold,
        "static_logical_error_rate": static_ler,
        "adaptive_logical_error_rate": adaptive_ler,
        "relative_error_rate_change": relative_change,
        "windows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriftGuard-Q v0.1")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--baseline-p", type=float, default=0.002)
    parser.add_argument("--peak-p", type=float, default=0.012)
    parser.add_argument("--schedule", choices=["step", "ramp", "sinusoid"], default="step")
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--train-shots", type=int, default=4000)
    parser.add_argument("--shots-per-window", type=int, default=1200)
    parser.add_argument("--calibration-shots", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = run_experiment(
        distance=args.distance,
        rounds=args.rounds,
        baseline_p=args.baseline_p,
        peak_p=args.peak_p,
        schedule=args.schedule,
        windows=args.windows,
        train_shots=args.train_shots,
        shots_per_window=args.shots_per_window,
        calibration_shots=args.calibration_shots,
        epochs=args.epochs,
        seed=args.seed,
    )

    summary = {
        "static_logical_error_rate": result["static_logical_error_rate"],
        "adaptive_logical_error_rate": result["adaptive_logical_error_rate"],
        "relative_error_rate_change": result["relative_error_rate_change"],
        "drift_triggers": sum(w["drift_triggered"] for w in result["windows"]),
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
