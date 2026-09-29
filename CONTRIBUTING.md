# Contributing to qregime

Thanks for considering it. Contributions of any size are welcome.

## Setup

```bash
git clone https://github.com/YOUR-GITHUB-USER/qregime.git
cd qregime
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,qec]"
pytest
```

## The one rule of this project

**Every claim needs a number you can rerun, and negative results count.**
This project exists because several of its own early results turned out to be
wrong under stricter validation, and were retracted rather than kept. In
practice:

- If a pull request changes a reported number, include the command that
  reproduces it (ideally as a test).
- If you find that something in the paper or the tool is wrong, that is a
  valuable contribution: open an issue with the evidence.
- Declare limitations in the docstring instead of hiding them.
- Leave-one-family-out, not random k-fold, is the standard for any predictor
  that claims to generalize to new kinds of circuits.

## Good first contributions

| Task | Where |
|---|---|
| Translate recommendation `reason` strings to English | `src/qregime/advisor.py` (`recommend_techniques`) |
| Translate code comments and docstrings to English | `src/qregime/*.py` |
| Add a hardware profile, with its published source | `HARDWARE_PROFILES` in `src/qregime/tool.py` |
| Replace deprecated Qiskit classes (`QFT`, `CDKMRippleCarryAdder`) in tests | `tests/` |

## Bigger open problems

- **A mechanism-based ZX-calculus predictor.** Count repeated parities in the
  phase polynomial of each Hadamard-free segment (phase folding, Amy-Maslov-Mosca
  2014). If ZX reduction subsumes phase folding, this is a provable lower bound
  on ZX gain, with Proposition 1 as the zero case. The statistical predictors in
  the paper do not generalize to new circuit families; this one might.
- **Validate partial QEC at small code distances** (d = 3-11) against direct
  Stim simulation; today it is an extrapolation of the full-fault-tolerance fit.
- **Robust decisions under fit uncertainty:** propagate the uncertainty of the
  threshold fit into the reliability, reporting intervals instead of points
  (Proposition 3 makes the worst case easy to find).
- **Dynamic circuits** with classically controlled corrections.
- **Faster QASM ingestion:** pyzx's parser dominates runtime on circuits with
  many distinct rotation angles.

## Pull requests

Keep them focused, run `pytest`, and describe what changed and how you verified
it. By contributing you agree that your contribution is licensed under Apache 2.0.
