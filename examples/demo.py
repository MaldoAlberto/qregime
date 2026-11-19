"""qregime demo: one decision per example circuit, on one hardware profile.

    python examples/demo.py                       # IBM-like superconductor
    python examples/demo.py trapped_ion_quantinuum_h2
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import qregime

HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUITS = ["vqe_ansatz", "qft10", "grover", "qaoa_farm", "adder8", "shor_like_sequential"]


def main(hardware="superconductor_ibm"):
    hw = qregime.HARDWARE_PROFILES[hardware]
    print(f"Hardware: {hw.name} (p_2q = {hw.p_2q}), budget 1,121 physical qubits, "
          f"reliability target 50%\n")
    print(f"{'circuit':22s} {'qubits':>6s} {'T-count':>8s} {'QEM rel.':>9s} "
          f"{'FTQC qubits':>12s}  regime")
    print("-" * 78)
    for name in CIRCUITS:
        path = os.path.join(HERE, "circuits", f"{name}.qasm")
        t0 = time.time()
        r = qregime.decide_regime(path, hardware=hardware)
        p = r["profile"]
        ftqc = r["ftqc"]["cost_qubits"]
        ftqc = f"{ftqc:>12,}" if ftqc is not None else f"{'infeasible':>12s}"
        print(f"{name:22s} {p['num_qubits']:>6d} {p['tcount']:>8d} "
              f"{r['qem']['reliability']:>8.1%} {ftqc}  "
              f"{r['regime']}  ({time.time() - t0:.0f}s)")
    print("\nRegime GAP = neither mitigation nor fault tolerance meets the target within "
          "budget;\nrun `qregime decide <circuit>` to see how large the gap is.")


if __name__ == "__main__":
    main(*sys.argv[1:2])
