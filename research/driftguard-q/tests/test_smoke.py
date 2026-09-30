import numpy as np

from driftguard_q.decoder import build_matching, decode_logical
from driftguard_q.sim import make_schedule, sample_window


def test_surface_code_sample_and_decode():
    window = sample_window(
        distance=3,
        rounds=3,
        physical_error_rate=0.003,
        shots=32,
    )
    matching = build_matching(
        distance=3,
        rounds=3,
        physical_error_rate=0.003,
    )
    pred = decode_logical(matching, window.detectors)

    assert window.detectors.shape[0] == 32
    assert window.observables.shape == (32,)
    assert pred.shape == (32,)
    assert np.isin(pred, [0, 1]).all()


def test_step_schedule_changes_noise():
    schedule = make_schedule(
        kind="step",
        windows=6,
        baseline_p=0.002,
        peak_p=0.01,
    )
    assert schedule[:3] == [0.002, 0.002, 0.002]
    assert schedule[3:] == [0.01, 0.01, 0.01]
