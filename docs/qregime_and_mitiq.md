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
- **Next step for qregime:** report a mitigated reliability together with
  its sampling cost, i.e. the shots needed for a target precision. Then the
  QEM-vs-QEC decision becomes "mitigate at N times the shots, or correct at
  M times the qubits", which is the comparison a user actually faces.

Run (about 12 minutes):

```bash
pip install -e ".[compare]"
python examples/qregime_to_mitiq.py   # writes docs/qregime_to_mitiq.png and .json
```
