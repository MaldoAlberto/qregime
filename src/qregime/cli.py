"""Command-line interface: ``qregime decide circuit.qasm``."""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import api, calibration


def _fmt_int(x):
    return f"{x:,}" if isinstance(x, int) else str(x)


def _cmd_hardware(_args):
    for key, hw in api.HARDWARE_PROFILES.items():
        print(f"{key:28s} p_2q={hw.p_2q:<8g} noise={hw.noise_model:<7s} "
              f"connectivity={hw.connectivity:<11s} calibration={hw.calibration or '-':<16s} "
              f"{hw.name}")
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
                          reliability_target=args.target,
                          simulate_mitigation=args.zne,
                          use_calibration=not args.no_calibration,
                          calibration=args.calibration)
    recs = api.recommend(args.qasm, circuit_profile=r["profile"])
    elapsed = time.time() - t0
    if args.json:
        print(json.dumps({"decision": r, "recommendations": recs}, indent=2, default=str))
        return 0

    p = r["profile"]
    print(f"\nCircuit   {args.qasm}")
    print(f"          {p['num_qubits']} qubits, {p['num_gates']} gates, T-count {p['tcount']}")
    print(f"Hardware  {r['hardware']}  (p_2q = {r['p_2q']}, noise model {r['noise_model']})")
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
    m = q["mitigation"]
    if m["feasible"]:
        print(f"        mitigated with ZNE (+/-{m['precision']:g}): "
              f"{_fmt_int(m['total_shots'])} shots ({_fmt_int(m['shots_per_circuit'])} "
              f"x {len(m['scale_factors'])} noise scales; best case, bias not modelled)")
    sim = m.get("simulated")
    if sim is not None and not sim["available"]:
        print(f"        ZNE on device calibration: not run ({sim['reason']})")
    elif sim is not None:
        shots = sim["total_shots"]
        print(f"        ZNE on {sim['device_model'].split(' ')[0]} calibration: "
              f"{sim['best_method']}, bias {sim['best_abs_bias']:.3f} "
              f"-> {'reaches' if sim['viable'] else 'does NOT reach'} +/-{sim['precision']:g}"
              + (f", {_fmt_int(shots)} shots" if shots else ""))
    if f["cost_qubits"] is None:
        print(f"  FTQC  not feasible: {f['infeasible_reason']}")
    else:
        print(f"  FTQC  {_fmt_int(f['cost_qubits']):>10s} qubits   fits budget: {f['fits_budget']}"
              + ("" if f["meets_target"] else "   (design reliability below target)"))

    print(f"\n=> Regime: {r['regime']}", end="")
    if r["cost_physical_qubits"] is not None:
        print(f"  ({_fmt_int(r['cost_physical_qubits'])} physical qubits)")
    else:
        print()
    if r["gap"]:
        g = r["gap"]
        ftqc_msg = (f"FTQC overruns the budget by {_fmt_int(g['ftqc_qubit_overrun'])} qubits."
                    if g["ftqc_qubit_overrun"] is not None else "FTQC is not feasible.")
        print(f"   QEM falls short by {g['qem_reliability_shortfall']:.1%} reliability; {ftqc_msg}")
        best = api.best_within_budget(p["num_qubits"], max(p["tcount"], 1), r["p_2q"],
                                      r["physical_qubit_budget"],
                                      noise_model=r["noise_model"])
        if best.get("realizable"):
            print(f"   Best reachable within budget: d={best['code_distance']}, "
                  f"reliability {best['reliability']:.1%} "
                  f"(quartile {best['confidence_quartile']}, {best['confidence_label']}).")
        else:
            print(f"   {best['reason']}")
    band = r["ftqc_noise_band"]
    parts = ", ".join(
        f"{m} {_fmt_int(v['physical_qubits']) if v['feasible'] else 'infeasible'}"
        for m, v in band["per_model"].items())
    print(f"   FTQC across noise models: {parts}.")
    if band["near_threshold"]:
        print("   Warning: p_2q is above half of at least one fitted threshold; "
              "the noise model, not the fit, dominates the uncertainty.")
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
    p_dec.add_argument("--budget", type=int, default=1121,
                       help="physical-qubit budget (default: 1121, IBM Condor)")
    p_dec.add_argument("--calibration", default=None,
                       help="device calibration: a bundled name "
                            f"({', '.join(calibration.available())}) or a JSON file "
                            "(default: the hardware profile's own)")
    p_dec.add_argument("--no-calibration", action="store_true",
                       help="use the depolarizing model even when a device calibration "
                            "is available")
    p_dec.add_argument("--zne", action="store_true",
                       help="also simulate ZNE on the device calibration (<= 10 qubits)")
    p_dec.add_argument("--target", type=float, default=0.5,
                       help="reliability target in [0,1] (default: 0.5)")
    p_dec.add_argument("--json", action="store_true")
    p_dec.set_defaults(func=_cmd_decide)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
