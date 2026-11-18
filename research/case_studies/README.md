# Case studies: two 2026 hardware results through qregime and Deltakit

Two experiments published in October 2026, rebuilt with Deltakit and fed
into qregime's regime decision:

1. **IBM Nighthawk r2 surface-code memory** (Kan et al., [arXiv:2610.23840](https://arxiv.org/abs/2610.23840)).
2. **Magic state cultivation on Google Willow** (Rosenfeld, Gidney, Satzinger et al.,
   [arXiv:2512.13908](https://arxiv.org/abs/2512.13908); *Science*, 15 Oct 2026).

Neither paper releases its circuits or raw data. Both are rebuilt here from
the numbers the papers state; each input below says where it comes from.

| Script | Writes |
|---|---|
| [`ibm_nighthawk_2026.py`](ibm_nighthawk_2026.py) | `ibm_nighthawk_2026_results.json` (Deltakit memories, about 6 min) |
| [`ibm_nighthawk_2026_projection.py`](ibm_nighthawk_2026_projection.py) | `ibm_nighthawk_2026_projection.json` (qregime, seconds) |
| [`google_cultivation_2026.py`](google_cultivation_2026.py) | `google_cultivation_2026_results.json` (qregime + Deltakit, under 1 min) |

The circuit used for the projections is qregime's flagship circuit:
16 logical qubits, T-count 3,930, 1% error budget.

## 1. IBM Nighthawk r2: the model does not reproduce the hardware

### What the paper measured

Rotated surface-code memories on `ibm_phoenix` (120 qubits, square lattice),
other distance fixed at 3, fit over 3-13 rounds, exact maximum-likelihood
decoding:

| | d=3 | d=6 | Λ |
|---|---|---|---|
| Z memory, error per round | 0.526% | 0.244% | 1.68 [1.57, 1.81] |
| X memory, error per round | 0.523% | 0.220% | 1.85 [1.71, 2.00] |

### The rebuilt experiment

Deltakit `PhysicalNoise` with the paper's median calibration for the
distance-scaling day (Appendix B1, Table 1): CZ error 1.6×10⁻³, √X error
2.1×10⁻⁴, `measure_2` error 0.46%, T1 = 184 µs, T2 = 162 µs. The paper's
timings (Methods .1) are used: CZ 68 ns, 1q gates 40 ns, readout 1.14 µs
with fused reset. Gate noise is depolarizing, matched to the average gate
error. Idles get Pauli-twirled T1/T2 noise. The same rounds (3, 5, 9, 13)
and the same per-round fit are used. Decoding is MWPM (PyMatching).

| | d=3 | d=6 | Λ |
|---|---|---|---|
| Simulated Z memory, echo | 0.266% | 0.0142% | 7.1 |
| Simulated X memory, echo | 0.291% | 0.0170% | 6.6 |
| **Measured, Z** | **0.526%** | **0.244%** | **1.68** |

The calibration-derived model is about 2 times too optimistic at d=3 and
about 4 times too optimistic in Λ.

### No uniformly worse device explains it

Every error probability is scaled by k, and T1 and T2 are divided by k
(Z memory, echo):

| k | ε, d=3 | ε, d=6 | Λ |
|---|---|---|---|
| 1.0 | 0.256% | 0.0134% | 7.13 |
| 1.5 | 0.583% | 0.0548% | 4.84 |
| 2.0 | 0.960% | 0.142% | 3.58 |
| 2.5 | 1.443% | 0.287% | 2.94 |
| 3.0 | 1.949% | 0.501% | 2.47 |
| **measured** | **0.526%** | **0.244%** | **1.68** |

- Matching the measured d=3 error takes k ≈ 1.45, which still predicts Λ ≈ 5.
- Bringing Λ down toward 1.7 takes k > 3, which predicts a d=3 error about
  4 times higher than measured.
- Both cannot hold at once, so the gap is not a worse average device. It is
  errors that do not shrink with distance: correlated errors, leakage,
  drift, or a few bad qubits. The paper reports detector correlations that
  build up over long memories and calibration tails up to 9.4% CZ error.
- The paper decodes with exact ML, this study with MWPM. The paper reports
  ML failing 0.69-0.86 times as often as MWPM. Correcting for that moves
  the measured d=3 point to about 0.61-0.76%, and the conclusion stands.

### What it means for a real circuit

Flagship circuit at p = 1.6×10⁻³, under three surface-code models
(`ibm_nighthawk_2026_projection.py`):

| Model | p/p_th | Full fault tolerance | Best within 1M qubits |
|---|---|---|---|
| qregime SI1000 (default superconducting model) | 0.40 | d=27, 39,366 qubits | reliability 1.00 |
| Deltakit, median calibration | 0.15 | d=17, 15,606 qubits | reliability 1.00 |
| **Measured Λ (ML decoding)** | **0.66** | **infeasible up to d=51** | **reliability 0.91 (d=51)** |

- With the measured Λ, a 100,000-qubit budget reaches 0.61 reliability (d=43)
  and a 10,000-qubit budget essentially zero.
- The measured-Λ row extrapolates Λ from d = 3-6 to d = 51. The paper
  reports that Λ weakens over long memories, so even this row is
  optimistic.
- **Lesson for qregime:** a threshold fitted to a standard noise model
  (SI1000) or to median calibration can understate the cost on real
  hardware by orders of magnitude. Where a measured Λ exists, it should
  drive the estimate.

## 2. Google Willow: cultivated T states for a real circuit

### What the paper measured

A |T⟩ state grown in a distance-3 color code on Willow, using 12-14 qubits:

| Quantity | Value |
|---|---|
| T-state error, with cultivation | 1(1)×10⁻⁴ (upper bound) |
| Injection only, no cultivation | 4.4(2)×10⁻³ |
| Attempts kept | 8% |
| Simulation noise in the paper | SI1000, p = 2.3×10⁻³ |
| Expected scaling | error ∝ p³ |

### Is 10⁻⁴ enough for the flagship circuit?

| Question | Answer |
|---|---|
| T error needed (1% budget, all spent on T states) | 2.5×10⁻⁶, 39 times below the measured bound |
| Circuit success from T errors alone at 10⁻⁴ | 67.5%: passes qregime's default 50% reliability target, not the 1% budget |
| p for cultivation alone to reach 2.5×10⁻⁶ (p³ scaling) | 6.8×10⁻⁴, 3.4 times below today's 2.3×10⁻³ |
| One 15-to-1 round on cultivated inputs (35 ε³) | 3.5×10⁻¹¹, far below the need |
| Expected attempts per accepted state | 12.5 |
| qregime FTQC at p = 2.3×10⁻³ (SI1000 fit) | d=45, 109,350 qubits, of which 44,550 are the Litinski factory |

- At p = 2.3×10⁻³ the surface code itself is the bottleneck (p/p_th = 0.57),
  not the T states. qregime needs d=45, and the factory alone is 44,550
  qubits, against about 14 qubits per cultivation attempt.
- Cultivation followed by one distillation round already meets the T
  requirement. The open question for a cost model is the time per accepted
  state (12.5 attempts on average), which the paper does not give as a
  wall-clock number.

### Deltakit: how much can leakage cost?

Distance-5 surface-code memory, 5 rounds, Deltakit `SI1000Noise` at the
paper's p = 2.3×10⁻³, with the leakage parameter `pL`:

| pL | P_L (5 rounds) | Factor vs. no leakage |
|---|---|---|
| 0 | 1.09% | 1.0 |
| 1×10⁻⁴ | 1.42% | 1.3 |
| 3×10⁻⁴ | 2.04% | 1.9 |
| 1×10⁻³ | 4.49% | 4.1 |
| 2.3×10⁻³ | 10.31% | 9.5 |

The paper says the distance-5 graft performed worse than simulated and that
leakage, absent from its model, may explain it. A secondary summary puts the
excess at about 7 times (not verified against the paper's text). In
Deltakit, a leakage rate of about 1.5×10⁻³, comparable to p, produces a
factor of that size. Leakage is a plausible cause, not a demonstrated one.

## Caveats

- Both experiments are rebuilt from published numbers, not from released
  circuits. The IBM schedule follows the paper's description, but the exact
  gate order is only in a figure that was not read.
- IBM calibration values are medians. The device has wide per-qubit tails,
  and a uniform noise model cannot represent them.
- Deltakit sampling is unseeded; reruns agree within the sampling error.
- The Google numbers are upper bounds with large relative uncertainty
  (1(1)×10⁻⁴).

## Sources

- S. Kan et al., "Demonstration of Logical Error Suppression in a Surface Code Memory on a Cloud-Accessible IBM Superconducting Processor", [arXiv:2610.23840](https://arxiv.org/abs/2610.23840) (16 Oct 2026).
- Rosenfeld, Gidney, Satzinger et al., "Magic state cultivation on a superconducting quantum processor", [arXiv:2512.13908](https://arxiv.org/abs/2512.13908); *Science* (15 Oct 2026).
- D. Litinski, "A Game of Surface Codes", [arXiv:1808.02892](https://arxiv.org/abs/1808.02892) (15-to-1 factory cost).
- Deltakit leakage model: [arXiv:2411.10343](https://arxiv.org/abs/2411.10343).
