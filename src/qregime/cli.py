"""Command-line interface: ``qregime decide circuit.qasm``."""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import api


def _fmt_int(x):
    if x is None:
        return "n/a"
    return f"{x:,}" if isinstance(x, int) else str(x)


def _unit_interval(text):
    value = float(text)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {value}")
    return value


def _cmd_hardware(_args):
    for key, hw in api.HARDWARE_PROFILES.items():
        print(f"{key:28s} p_2q={hw.p_2q:<8g} connectivity={hw.connectivity:<11s} {hw.name}")
    return 0


def _cmd_profile(args):
    prof = api.profile(args.qasm)
    if args.json:
        print(json.dumps(prof, indent=2, default=str))
        return 0
    for k, v in prof.items():
        print(f"  {k:36s} {v}")
    return 0


def _cmd_decide(args):
    t0 = time.time()
    r = api.decide_regime(args.qasm, hardware=args.hardware,
                          physical_qubit_budget=args.budget,
                          reliability_target=args.target)
    recs = api.recommend(args.qasm, circuit_profile=r["profile"])
    elapsed = time.time() - t0
    if args.json:
        print(json.dumps({"decision": r, "recommendations": recs}, indent=2, default=str))
        return 0

    p = r["profile"]
    print(f"\nCircuit   {args.qasm}")
    print(f"          {p['num_qubits']} qubits, {p['num_gates']} gates, T-count {p['tcount']}")
    print(f"Hardware  {r['hardware']}  (p_2q = {r['p_2q']})")
    print(f"Budget    {_fmt_int(r['physical_qubit_budget'])} physical qubits, "
          f"reliability target {r['reliability_target']:.0%}\n")

    print("Techniques worth trying:")
    for name, rec in recs.items():
        mark = {True: "yes ", False: "no  "}.get(rec["recommended"], "WARN")
        print(f"  [{mark}] {name}")

    q, f = r["qem"], r["ftqc"]
    print("\nRegime comparison:")
    print(f"  QEM   {_fmt_int(q['cost_qubits']):>10s} qubits   reliability {q['reliability']:.1%}"
          f"   ({q['reliability_source']})")
    print(f"  FTQC  {_fmt_int(f['cost_qubits']):>10s} qubits   fits budget: {f['fits_budget']}"
          f"   reliability {f['reliability']:.1%}   T-count incl. rotation synthesis "
          f"{_fmt_int(f['tcount_effective'])}")
    if f["infeasible_reason"]:
        print(f"        not estimable: {f['infeasible_reason']}")

    print(f"\n=> Regime: {r['regime']}", end="")
    if r["cost_physical_qubits"] is not None:
        print(f"  ({_fmt_int(r['cost_physical_qubits'])} physical qubits)")
    else:
        print()
    if r["gap"]:
        g = r["gap"]
        overrun = g["ftqc_qubit_overrun"]
        ftqc_msg = ("FTQC is not estimable at this error rate." if overrun is None else
                    f"FTQC overruns the budget by {_fmt_int(overrun)} qubits." if overrun > 0 else
                    "FTQC fits the budget but not the reliability target.")
        print(f"   QEM falls short by {g['qem_reliability_shortfall']:.1%} reliability; {ftqc_msg}")
        best = api.best_within_budget(p["num_qubits"], f["tcount_effective"], r["p_2q"],
                                      r["physical_qubit_budget"])
        if best.get("realizable"):
            print(f"   Best reachable within budget: d={best['code_distance']}, "
                  f"reliability {best['reliability']:.1%} "
                  f"(quartile {best['confidence_quartile']}, {best['confidence_label']}).")
        else:
            print(f"   {best['reason']}")
    if r["classical_shortcut_advisory"]:
        print("   Advisory: low treewidth and low entanglement -- check for a classical "
              "tensor-network shortcut before spending quantum resources.")
    print(f"\n({elapsed:.1f} s)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="qregime",
        description="QEM, QEC or FTQC? Decide the error-handling regime for a quantum circuit.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_hw = sub.add_parser("hardware", help="list built-in hardware profiles")
    p_hw.set_defaults(func=_cmd_hardware)

    p_prof = sub.add_parser("profile", help="structural profile of a circuit")
    p_prof.add_argument("qasm")
    p_prof.add_argument("--json", action="store_true")
    p_prof.set_defaults(func=_cmd_profile)

    p_dec = sub.add_parser("decide", help="recommend techniques and choose a regime")
    p_dec.add_argument("qasm")
    p_dec.add_argument("--hardware", default="superconductor_ibm",
                       choices=sorted(api.HARDWARE_PROFILES))
    p_dec.add_argument("--budget", type=int, default=api.DEFAULT_PHYSICAL_QUBIT_BUDGET,
                       help="physical-qubit budget (default: %(default)s, IBM Condor)")
    p_dec.add_argument("--target", type=_unit_interval, default=api.DEFAULT_RELIABILITY_TARGET,
                       help="reliability target in (0,1] (default: %(default)s)")
    p_dec.add_argument("--json", action="store_true")
    p_dec.set_defaults(func=_cmd_decide)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
