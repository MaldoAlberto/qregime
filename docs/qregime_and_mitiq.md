# qregime and Mitiq: decide, then mitigate

qregime and [Mitiq](https://github.com/unitaryfoundation/mitiq) work at
different layers:

- **qregime** answers the question before mitigation: should this circuit be
  mitigated (QEM), partially error-corrected, or run fault-tolerantly, and at
  what cost?
- **Mitiq** answers the next question: how to mitigate it (ZNE, PEC, CDR,
  DDD, readout mitigation).

```
circuit -> qregime -> QEM   -> Mitiq
                   -> QEC   -> Deltakit / Stim
                   -> FTQC  -> Azure QRE / QREChem
```

[`examples/qregime_to_mitiq.py`](../examples/qregime_to_mitiq.py) hands
circuits from one to the other and compares qregime's QEM boundary with what
Mitiq's zero-noise extrapolation actually recovers.

## Setup

- **Circuits:** 6-qubit layered mirror circuits U U† (RY rotations and a CX
  brickwork), 6 to 640 CX gates. The ideal output is |000000⟩, so the ideal
  success probability is 1.
- **Noise:** depolarizing two-qubit noise at the IBM heavy-hex profile's
  p_2q = 0.32%. This is the same noise qregime simulates.
- **qregime:** `decide_regime` with its default 50% reliability target.
- **Mitiq:** ZNE with scale factors 1, 2, 3 and random gate folding. Two
  extrapolations: Richardson, and exponential (`ExpFactory`, asymptote 1/2⁶).
  Each executed circuit gets 10,000 shots, and every point is repeated 20
  times.

![qregime's QEM boundary vs. Mitiq ZNE](qregime_to_mitiq.png)

## Results

| CX | qregime | qregime reliability | Raw (exact) | ZNE Richardson | ZNE exponential |
|---|---|---|---|---|---|
| 40 | QEM | 0.904 | 0.903 | 0.999 ± 0.042 | 0.995 ± 0.007 |
| 120 | QEM | 0.693 | 0.715 | 0.983 ± 0.055 | 0.996 ± 0.011 |
| 200 | QEM | 0.545 | 0.564 | 0.918 ± 0.040 | 0.994 ± 0.017 |
| 240 | GAP | 0.481 | 0.504 | 0.864 ± 0.029 | 0.999 ± 0.019 |
| 400 | GAP | 0.295 | 0.322 | 0.674 ± 0.020 | 0.997 ± 0.032 |
| 640 | GAP | 0.141 | 0.164 | 0.394 ± 0.009 | 1.002 ± 0.129 |

(± is the standard deviation over 20 repeats; ideal = 1.)

1. **qregime's reliability prediction is accurate.** Its noisy simulation
   tracks the exact unmitigated success within 0.03 at every depth.
2. **Richardson ZNE fails where qregime leaves QEM.** Its error stays at or
   below 0.09 while qregime says QEM, and grows to 0.14 at the boundary and
   0.61 at 640 CX. The failure is bias, not noise: the spread stays at
   0.01-0.06.
3. **Exponential ZNE keeps working far past qregime's boundary.** Under pure
   depolarizing noise, decay is exactly exponential, which is the best case
   for this extrapolation. Its mean stays within 0.01 of the ideal up to
   640 CX, while its spread grows from 0.002 to 0.13.
4. **The real limit is shots.** For ±0.01 precision with exponential ZNE,
   each executed circuit needs about 5,500 shots at 40 CX, 38,000 at the
   qregime boundary (240 CX), and 1.7 million at 640 CX.

## What this means

- qregime's QEM column is the **unmitigated** reliability. On this benchmark
  it is conservative: a well-chosen ZNE recovers circuits that qregime sends
  to GAP.
- That benchmark is ZNE's best case. Real devices add noise that does not
  decay exponentially (coherent errors, leakage, drift; see the IBM
  Nighthawk case study in `research/case_studies/`, where measured Λ was
  four times below the calibration model). There, extrapolation bias returns.
- **qregime now reports that cost.** `decide_regime(...)["qem"]["mitigation"]`
  gives the shots exponential ZNE needs for ±0.01 precision, and
  `qregime.estimate_zne_cost(reliability, precision)` computes it directly.
  The model propagates binomial shot noise through the least-squares
  exponential fit (`src/qregime/mitigation.py`). On this benchmark its
  predicted spread is within about 30% of Mitiq's measured spread at every
  depth (`tests/test_mitigation.py`). It assumes no extrapolation bias, so it
  is a lower bound on real hardware; see the next section for the bias. The decision becomes "mitigate at
  N shots, or correct at M qubits", which is the comparison a user actually
  faces.

## Under a real device calibration

The benchmark above is ZNE's best case. `qregime.mitigation.simulate_zne`
(also `decide_regime(..., simulate_mitigation=True)` and `qregime decide
--zne`) runs ZNE on the FakeTorino calibration instead. That calibration has
per-qubit CZ and one-qubit errors, T1/T2 relaxation during gates and idle
times, readout errors, and a noise-aware layout. Every gate is folded, as
Mitiq does by default. Probabilities are exact (density matrix), so the bias
below contains no sampling noise.
[`examples/zne_calibrated.py`](../examples/zne_calibrated.py) runs the same
kind of 6-qubit mirror circuits:

![ZNE on the FakeTorino calibration](zne_calibrated.png)

| CX | Calibrated, raw | ZNE exp. | ZNE exp. + readout mit. | ZNE Richardson + readout mit. | Shots for ±0.01 (exp. + readout mit.) |
|---|---|---|---|---|---|
| 6 | 0.92 | 0.943 | 1.000 | 1.003 | 2,889 |
| 20 | 0.75 | 0.912 | 0.994 | 0.996 | 25,428 |
| 40 | 0.61 | 0.901 | 0.983 | 0.972 | 62,598 |
| 60 | 0.50 | 0.885 | 0.966 | 0.906 | 118,731 |
| 120 | 0.26 | 0.829 | 0.906 | 0.617 | 960,606 |
| 160 | 0.17 | 0.690 | 0.756 | 0.443 | 2,695,800 |
| 240 | 0.08 | 0.912 | 0.988 | 0.208 | 1.6 × 10⁹ |
| 320 | 0.04 | 0.218 | 0.242 | 0.090 | 4.3 × 10⁸ |

(All rows, including 10 and 80 CX, are in `zne_calibrated.json`.)

1. **ZNE without readout mitigation never reaches the ideal value.** Folding
   does not amplify readout error, so the extrapolation lands on the readout
   floor: 0.94 even at 6 CX. ZNE has to be paired with readout mitigation.
2. **With readout mitigation, bias sets the limit before shots do.**
   Exponential ZNE is within ±0.01 only up to 20 CX. The bias is 0.034 at
   60 CX and 0.094 at 120 CX. Under depolarizing noise the bias stayed below
   0.01 up to 640 CX. Richardson degrades sooner.
3. **The depolarizing model is optimistic for this device.** It has
   two-qubit noise only. The calibration adds one-qubit, idle and readout
   errors. The calibrated success fraction falls to 0.50 at 60 CX, where the
   depolarizing model predicts 0.84. Because of this, `decide_regime` now
   uses the calibration for its QEM reliability whenever it can (see below).
4. **An unbiased-looking point can be a fit artifact.** At 240 CX the
   scaled values sit at the asymptote 1/2⁶. The exponential fit lands near 1
   by chance, and the shot cost (1.6 × 10⁹) shows it is unusable.
   `simulate_zne` counts ZNE as viable only if the bias is within the
   precision and the shots stay within `max_shots` (10⁷ by default).

### qregime's decision now uses the calibration

For circuits up to 10 qubits on the `superconductor_ibm` profile,
`decide_regime` takes the QEM reliability from an exact simulation on the
FakeTorino calibration (`reliability_source` starts with
`calibrated_simulation`). It needs `qregime[realistic]`. Use
`use_calibration=False` or `qregime decide --no-calibration` for the
depolarizing model. On the mirror circuits, with the default 50% target:

| CX | Depolarizing model | FakeTorino calibration (default) |
|---|---|---|
| 6 | QEM (0.98) | QEM (0.92) |
| 20 | QEM (0.95) | QEM (0.75) |
| 40 | QEM (0.90) | QEM (0.61) |
| 60 | QEM (0.84) | QEM (0.50) |
| 80 | QEM (0.79) | GAP (0.40) |
| 120 | QEM (0.70) | GAP (0.26) |
| 160 | QEM (0.62) | GAP (0.17) |
| 240 | GAP (0.48) | GAP (0.08) |

The QEM boundary moves from 240 CX to 80 CX. The reliability is the
unmitigated one, readout error included. Larger circuits, other profiles and
circuits whose output support covers most bitstrings keep the previous
sources; `qem["calibration"]["reason"]` says why. The example circuits in
`examples/circuits/` are all in that group, so their decisions do not change.

Not modelled: coherent over-rotations, crosstalk, leakage and drift. The
fake-backend calibration does not contain them, and on hardware they add
more bias. The shot counts with readout mitigation ignore the extra variance
that readout inversion adds.

Run (about 12 minutes):

```bash
pip install -e ".[compare]"
python examples/qregime_to_mitiq.py   # writes docs/qregime_to_mitiq.png and .json

pip install -e ".[realistic]"
python examples/zne_calibrated.py     # writes docs/zne_calibrated.png and .json (~5 min)
```
