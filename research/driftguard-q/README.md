# DriftGuard-Q

DriftGuard-Q is an experimental AI layer for quantum error correction under non-stationary noise.

The v0.1 research question is narrow:

> Can a self-supervised model detect that a surface-code syndrome distribution has drifted, estimate the new global noise level from syndrome statistics alone, and adapt a conventional PyMatching decoder without using the true logical state during adaptation?

## What v0.1 does

1. Generates rotated surface-code memory experiments with Stim.
2. Trains a masked syndrome autoencoder on a stable baseline noise regime.
3. Measures reconstruction loss as a label-free drift score.
4. Calibrates detector-event density against simulated physical error rates.
5. Runs a time-varying noise schedule.
6. Uses only syndrome data to decide when to rebuild the PyMatching decoder.
7. Uses logical observables only after decoding, for evaluation.

That separation matters: the adaptive loop never receives the correct logical answer.

## Architecture

```text
Stim surface-code simulator
          |
          v
   detector syndromes
          |
          +----------------------+
          |                      |
          v                      v
masked autoencoder        detector-event rate
(self-supervised)          noise calibration
          |                      |
          v                      v
     drift score            estimated p
          |                      |
          +----------+-----------+
                     |
                     v
             rebuild PyMatching
                     |
                     v
                  decode
                     |
                     v
      logical observable used only
           for final evaluation
```

## Install

Python 3.11+ is recommended.

```bash
cd research/driftguard-q
python -m pip install -e .
```

## Run

```bash
driftguard-q --schedule step --distance 3 --rounds 5
```

Other schedules:

```bash
driftguard-q --schedule sinusoid
driftguard-q --schedule ramp
```

Write a machine-readable result:

```bash
driftguard-q --schedule step --out result.json
```

## Success criteria for v0.1

The first useful result is not "AI beats every decoder." It is evidence that:

- self-supervised syndrome reconstruction loss rises when the noise distribution changes;
- adaptation is triggered without logical-state labels;
- the syndrome-only noise estimate tracks global drift well enough to produce a useful decoder update;
- under at least one drift schedule, adaptive decoding reduces logical failures versus a decoder frozen at the baseline noise model.

## Important limitation

v0.1 estimates one global physical-error parameter. Real hardware has per-qubit, per-gate, correlated, leakage, measurement, and time-dependent error processes. A successful v0.1 only justifies the next step: spatial noise maps and graph-edge reweighting.

## Next research steps

- Localize drift by detector / stabilizer region.
- Replace scalar noise estimation with a learned spatial noise map.
- Reweight matching-graph edges instead of rebuilding from a single global p.
- Add temporal models over continuous QEC rounds.
- Test abrupt, periodic, localized, correlated, and adversarial drift.
- Benchmark latency and sample efficiency.
- Compare against fixed MWPM, periodic recalibration, and learned decoders.
- Ingest public experimental syndrome datasets when available.

## Research integrity

Logical observables are allowed for evaluation and benchmarking. They are intentionally excluded from the adaptive decision path.
