"""Vendor-neutral device calibrations and a calibrated simulator (exact density
matrix up to 10 qubits, quantum trajectories up to 20).

A calibration is a small JSON document (schema ``qregime-calibration/1``)
that any vendor's published numbers can fill in:

    {
      "schema": "qregime-calibration/1",
      "name": "my_device", "source": "where the numbers come from", "date": "2026-01-01",
      "num_qubits": 56,
      "connectivity": "all_to_all"            # or [[0, 1], [1, 2], ...]
      "single_qubit": {"error": 8e-5,          # per pulse; number or per-qubit list
                       "duration": 0.0,        # seconds (or layers, see "idle")
                       "decomposition": "arbitrary"},   # or "zsx" (IBM: 0-2 sx pulses)
      "two_qubit": {"native": "rzz",           # "cz", "cx" or "rzz"
                    "error": 1.4e-3, "duration": 1.0,
                    "pairs": {"0,1": {"error": ..., "duration": ...,   # optional overrides
                                      # coherent fSim error after each gate, radians
                                      "swap_angle": 0.0, "conditional_phase": 0.0}},
                    "zz_overrotation": 0.0},    # optional coherent error (radians)
      "readout": {"p01": 9e-4, "p10": 1.8e-3},  # P(read 1 | 0), P(read 0 | 1); numbers or lists
      "idle": {"t1": [...], "t2": [...]}        # seconds, per qubit or one number
           or {"memory_error_per_layer": 5.3e-4}   # dephasing per qubit per two-qubit layer
    }

Missing sections mean "no error of that kind". Error values are average
gate infidelities; as in Qiskit Aer, the part of a gate's error explained by
relaxation during the gate is not counted twice.

``success_fraction`` maps a circuit onto the device (embedding its
interaction graph when possible, otherwise routing with SWAPs), compiles it
to the native gates, schedules it (idle times included) and simulates it
exactly with a density matrix. The metric is qregime's: the probability of
landing in the ideal output support.

Not modelled: leakage, crosstalk, drift, non-Markovian noise.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Optional, Union

import numpy as np

from . import circuit as _circuit
from .simulate import DensityMatrix, StateVector, apply_readout_error, statevector, u_matrix

SCHEMA = "qregime-calibration/1"
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibrations")
MAX_QUBITS = 10          # exact density matrix: 4**n complex entries
MAX_QUBITS_TRAJECTORIES = 20   # pure-state trajectories: 2**n amplitudes each
MAX_TRAJECTORY_WORK = 6e9      # gate applications x 2**n, over all trajectories
MIN_TRAJECTORIES, MAX_TRAJECTORIES = 16, 400
TARGET_STANDARD_ERROR = 0.01
CHECKPOINT_BYTES = 2 ** 28     # memory for saved ideal states (256 MB)

_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_CX = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)  # |q1 q0>, control q0


def fsim_matrix(theta: float, phi: float) -> np.ndarray:
    """fSim(theta, phi), as cirq's PhasedFSimGate(theta, phi) with zero
    single-qubit phases: swap angle theta, conditional phase phi."""
    c, s = math.cos(theta), -1j * math.sin(theta)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, s, c, 0],
                     [0, 0, 0, np.exp(-1j * phi)]], dtype=complex)


def rzz_matrix(theta: float) -> np.ndarray:
    """exp(-i theta/2 Z(x)Z), OpenQASM's rzz."""
    a, b = np.exp(-0.5j * theta), np.exp(0.5j * theta)
    return np.diag([a, b, b, a])


# --------------------------------------------------------------------------
# loading and validation
# --------------------------------------------------------------------------

def available() -> list:
    """Names of the calibrations bundled with qregime."""
    if not os.path.isdir(_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(_DIR) if f.endswith(".json"))


@lru_cache(maxsize=None)
def _load_file(path: str) -> "Calibration":
    with open(path) as f:
        return Calibration(json.load(f))


def load(spec: Union[str, dict, "Calibration"]) -> "Calibration":
    """A bundled calibration by name, a JSON file path, a dict, or a Calibration."""
    if isinstance(spec, Calibration):
        return spec
    if isinstance(spec, dict):
        return Calibration(spec)
    path = spec if os.path.exists(spec) else os.path.join(_DIR, f"{spec}.json")
    if not os.path.exists(path):
        raise ValueError(f"unknown calibration {spec!r}; bundled: {available()}")
    return _load_file(os.path.abspath(path))


def _per_qubit(value, n, name, default=0.0):
    if value is None:
        return [default] * n
    if isinstance(value, (int, float)):
        return [float(value)] * n
    if len(value) != n:
        raise ValueError(f"calibration {name}: expected {n} values, got {len(value)}")
    return [float(v) if v is not None else default for v in value]


class Calibration:
    """A validated ``qregime-calibration/1`` document."""

    def __init__(self, data: dict):
        if data.get("schema") != SCHEMA:
            raise ValueError(f"calibration schema must be {SCHEMA!r}")
        self.data = data
        self.name = data.get("name", "unnamed")
        self.source = data.get("source", "")
        n = self.num_qubits = int(data["num_qubits"])
        conn = data.get("connectivity", "all_to_all")
        self.all_to_all = conn == "all_to_all"
        self.edges = set() if self.all_to_all else {tuple(sorted(map(int, e))) for e in conn}

        sq = data.get("single_qubit", {})
        self.sq_error = _per_qubit(sq.get("error"), n, "single_qubit.error")
        self.sq_duration = _per_qubit(sq.get("duration"), n, "single_qubit.duration")
        self.decomposition = sq.get("decomposition", "arbitrary")
        if self.decomposition not in ("arbitrary", "zsx"):
            raise ValueError("single_qubit.decomposition must be 'arbitrary' or 'zsx'")

        tq = data.get("two_qubit", {})
        self.native = tq.get("native", "cx")
        if self.native not in ("cx", "cz", "rzz"):
            raise ValueError("two_qubit.native must be 'cx', 'cz' or 'rzz'")
        self.tq_default = (float(tq.get("error", 0.0)), float(tq.get("duration", 0.0)))
        self.tq_pairs = {}
        self.coherent_2q = {}   # (a, b) -> 4x4 coherent error after each native gate
        for key, v in tq.get("pairs", {}).items():
            a, b = sorted(int(x) for x in key.split(","))
            self.tq_pairs[(a, b)] = (float(v.get("error", self.tq_default[0])),
                                     float(v.get("duration", self.tq_default[1])))
            theta = float(v.get("swap_angle", 0.0))
            phi = float(v.get("conditional_phase", 0.0))
            if theta or phi:
                self.coherent_2q[(a, b)] = fsim_matrix(theta, phi)
        self.zz_overrotation = float(tq.get("zz_overrotation", 0.0))

        ro = data.get("readout", {})
        self.p01 = _per_qubit(ro.get("p01"), n, "readout.p01")
        self.p10 = _per_qubit(ro.get("p10"), n, "readout.p10")

        idle = data.get("idle", {})
        if "memory_error_per_layer" in idle:
            m = float(idle["memory_error_per_layer"])
            # dephasing with probability m per layer: coherence x (1 - 2m) per layer
            t2 = -1.0 / math.log(1 - 2 * m) if m > 0 else math.inf
            self.t1 = [math.inf] * n
            self.t2 = [t2] * n
        else:
            self.t1 = _per_qubit(idle.get("t1"), n, "idle.t1", default=math.inf)
            self.t2 = _per_qubit(idle.get("t2"), n, "idle.t2", default=math.inf)

    def two_qubit(self, a: int, b: int):
        pair = (min(a, b), max(a, b))
        if not self.all_to_all and pair not in self.edges:
            raise ValueError(f"qubits {pair} are not coupled on {self.name}")
        return self.tq_pairs.get(pair, self.tq_default)

    def coupling_graph(self):
        import rustworkx as rx
        g = rx.PyGraph()
        g.add_nodes_from(range(self.num_qubits))
        edges = self.edges if not self.all_to_all else {
            (a, b) for a in range(self.num_qubits) for b in range(a + 1, self.num_qubits)}
        g.add_edges_from_no_data(sorted(edges))
        return g


# --------------------------------------------------------------------------
# compilation: logical circuit -> native ops on a set of physical qubits
# --------------------------------------------------------------------------

def _logical_ops(circ, native: str) -> list:
    """("1q", (q,), U) and ("2q", (a, b), M, name) ops on logical qubits.

    CX is rewritten into the native entangler (CZ: H CZ H; RZZ: one
    rzz(pi/2) plus single-qubit rotations); top-level ``rzz`` gates stay a
    single native gate when the device's native gate is rzz.
    """
    ops = []
    for inst in circ.gates():
        if native == "rzz" and inst.name == "rzz":
            ops.append(("2q", inst.qubits, rzz_matrix(inst.params[0]), "rzz"))
            continue
        for kind, qubits, params in circ.expand(inst):
            if kind == "U":
                ops.append(("1q", qubits, u_matrix(*params)))
            else:
                ops.extend(_native_cx(qubits[0], qubits[1], native))
    return ops


def _native_cx(c: int, t: int, native: str) -> list:
    if native == "cx":
        return [("2q", (c, t), _CX, "cx")]
    if native == "cz":
        return [("1q", (t,), _H), ("2q", (c, t), _CZ, "cz"), ("1q", (t,), _H)]
    # CZ = e^{i pi/4} (Rz(pi/2) (x) Rz(pi/2)) Rzz(-pi/2), up to global phase;
    # the -pi/2 angle has the same cost as +pi/2 on ZZ-native hardware
    rz = np.diag([np.exp(-0.25j * math.pi), np.exp(0.25j * math.pi)])
    return [("1q", (t,), _H), ("2q", (c, t), rzz_matrix(-math.pi / 2), "rzz"),
            ("1q", (c,), rz), ("1q", (t,), rz), ("1q", (t,), _H)]


def _fuse(ops) -> list:
    pending: dict = {}
    out = []

    def flush(q):
        if q in pending:
            out.append(("1q", (q,), pending.pop(q)))

    for op in ops:
        if op[0] == "1q":
            q = op[1][0]
            pending[q] = op[2] @ pending.get(q, np.eye(2, dtype=complex))
        else:
            for q in op[1]:
                flush(q)
            out.append(op)
    for q in sorted(pending):
        flush(q)
    return out


def _choose_layout(cal: Calibration, n: int, pairs: dict, max_mappings: int = 20000,
                   ops_per_qubit=None):
    """Physical qubit for each logical qubit. Embeds the interaction graph
    (weighted by gate counts) into the coupling graph when possible,
    scoring by expected error; otherwise returns a connected region and
    lets the router insert SWAPs. Returns (layout, needs_routing).

    The score adds, per qubit, readout error, single-qubit error times the
    qubit's gate count, and decoherence over an estimated circuit duration
    (two-qubit gate count x two-qubit duration / T2); per coupler, the
    two-qubit error times its gate count."""
    import rustworkx as rx

    ops_per_qubit = ops_per_qubit or [1] * n
    n_2q = sum(pairs.values())
    est_duration = n_2q * cal.tq_default[1] * 2 / max(1, n)

    def cost_q_for(q, k=0):
        t2 = cal.t2[q]
        idle = est_duration / t2 if math.isfinite(t2) and t2 > 0 else 0.0
        return cal.p01[q] + cal.p10[q] + 2 * cal.sq_error[q] * ops_per_qubit[k] + idle

    cost_q = [cost_q_for(q) for q in range(cal.num_qubits)]
    if cal.all_to_all:
        best = sorted(range(cal.num_qubits), key=lambda q: cost_q[q])[:n]
        return list(best), False

    coupling = cal.coupling_graph()
    logical = rx.PyGraph()
    logical.add_nodes_from(range(n))
    logical.add_edges_from_no_data(sorted(pairs))

    def component_score(nodes, assignment):
        s = sum(cost_q_for(assignment[q], q) for q in nodes)
        for (a, b), count in pairs.items():
            if a in assignment and b in assignment and a in nodes:
                pa, pb = assignment[a], assignment[b]
                if (min(pa, pb), max(pa, pb)) not in cal.edges:
                    return math.inf
                s += count * -math.log(max(1e-12, 1 - cal.two_qubit(pa, pb)[0]))
        return s

    # embed each connected component separately (largest first), then
    # assign greedily without overlaps: enumerating whole-circuit mappings
    # never leaves one corner of the device when components are disjoint
    components = sorted((sorted(c) for c in rx.connected_components(logical)),
                        key=len, reverse=True)
    used, layout_map = set(), {}
    for nodes in components:
        if len(nodes) == 1:
            free = [q for q in range(cal.num_qubits) if q not in used and coupling.degree(q)]
            choice = {nodes[0]: min(free, key=lambda q: cost_q_for(q, nodes[0]))}
        else:
            sub = rx.PyGraph()
            sub.add_nodes_from(range(len(nodes)))
            index = {q: k for k, q in enumerate(nodes)}
            sub.add_edges_from_no_data([(index[a], index[b]) for (a, b) in pairs
                                        if a in index and b in index])
            candidates = []
            for k, mapping in enumerate(rx.vf2_mapping(coupling, sub, subgraph=True,
                                                       induced=False, id_order=False)):
                if k >= max_mappings:
                    break
                assignment = {nodes[log]: phys for phys, log in mapping.items()}
                if used.isdisjoint(assignment.values()):
                    candidates.append((component_score(nodes, assignment), assignment))
            if not candidates:
                layout_map = None
                break
            choice = min(candidates, key=lambda c: c[0])[1]
        layout_map.update(choice)
        used.update(choice.values())
    if layout_map is not None and len(layout_map) == n:
        return [layout_map[q] for q in range(n)], False
    # no embedding: grow a connected region from the best qubit
    seed = min(range(cal.num_qubits), key=lambda q: cost_q[q] if coupling.degree(q) else math.inf)
    region, frontier = [seed], [seed]
    while len(region) < n and frontier:
        nxt = []
        for q in frontier:
            for nb in sorted(coupling.neighbors(q), key=lambda x: cost_q[x]):
                if nb not in region and len(region) < n:
                    region.append(nb)
                    nxt.append(nb)
        frontier = nxt
    if len(region) < n:
        raise ValueError(f"{cal.name}: no connected region of {n} qubits")
    return region, True


def compile_to_device(circ, cal: Calibration, layout: Optional[list] = None) -> dict:
    """Native, fused ops on local slots 0..n-1 (slot k = physical qubit
    ``physical[k]``), plus the final position of every logical qubit."""
    import rustworkx as rx

    n = circ.num_qubits
    if n > cal.num_qubits:
        raise ValueError(f"{n} qubits do not fit on {cal.name} ({cal.num_qubits} qubits)")
    ops = _logical_ops(circ, cal.native)
    pairs: dict = {}
    for op in ops:
        if op[0] == "2q":
            key = (min(op[1]), max(op[1]))
            pairs[key] = pairs.get(key, 0) + 1
    if layout is None:
        per_qubit = [0] * n
        for op in ops:
            for q in op[1]:
                per_qubit[q] += 1
        layout, _ = _choose_layout(cal, n, pairs, ops_per_qubit=per_qubit)
    physical = list(layout)
    pos = list(range(n))                  # logical -> slot
    occupant = list(range(n))             # slot -> logical
    if cal.all_to_all:
        adjacent = lambda a, b: True      # noqa: E731
    else:
        slots = rx.PyGraph()
        slots.add_nodes_from(range(n))
        slots.add_edges_from_no_data([(k, l) for k in range(n) for l in range(k + 1, n)
                                      if (min(physical[k], physical[l]),
                                          max(physical[k], physical[l])) in cal.edges])
        dist = rx.distance_matrix(slots, null_value=math.inf)
        adjacent = lambda a, b: dist[a][b] == 1         # noqa: E731

    routed = []
    for op in ops:
        if op[0] == "1q":
            routed.append(("1q", (pos[op[1][0]],), op[2]))
            continue
        a, b = op[1]
        while not adjacent(pos[a], pos[b]):
            # move a one step towards b along a shortest path
            sa, sb = pos[a], pos[b]
            if not math.isfinite(dist[sa][sb]):
                raise ValueError(f"layout {physical}: qubits {physical[sa]} and {physical[sb]} "
                                 f"are not connected within the layout on {cal.name}")
            step = min((s for s in range(n) if dist[sa][s] == 1),
                       key=lambda s: dist[s][sb])
            for c, t in ((sa, step), (step, sa), (sa, step)):
                routed.extend(_native_cx(c, t, cal.native))
            la, lb = occupant[sa], occupant[step]
            occupant[sa], occupant[step] = lb, la
            pos[la], pos[lb] = step, sa
        routed.append((op[0], (pos[a], pos[b])) + tuple(op[2:]))
    return {"ops": _fuse(routed), "physical": physical, "final_pos": pos}


# --------------------------------------------------------------------------
# noisy simulation
# --------------------------------------------------------------------------

def _pulses(cal: Calibration, m: np.ndarray) -> int:
    """Physical single-qubit pulses needed for the 2x2 unitary ``m``."""
    theta = 2 * math.acos(min(1.0, abs(m[0, 0])))
    if theta < 1e-9:
        return 0                          # diagonal: virtual Z
    if cal.decomposition == "arbitrary":
        return 1
    if abs(theta - math.pi / 2) < 1e-9 or abs(theta - math.pi) < 1e-9:
        return 1                          # sx, or x
    return 2


def _relax_fidelity(duration, t1, t2):
    """Process fidelity of T1/T2 relaxation over ``duration``."""
    if duration <= 0:
        return 1.0
    t2 = min(t2, 2 * t1)
    gamma = 1 - math.exp(-duration / t1) if math.isfinite(t1) else 0.0
    coh = math.exp(-duration / t2) if math.isfinite(t2) else 1.0
    return (1 + (1 - gamma) + 2 * coh) / 4


def _depolarizing_param(error, dim, relax_process_fidelity):
    """Depolarizing parameter giving average infidelity ``error`` once the
    relaxation during the gate is accounted for (Aer's convention)."""
    if error <= 0:
        return 0.0
    f_total = ((dim + 1) * (1 - error) - 1) / dim          # process fidelity
    f_dep = min(1.0, f_total / relax_process_fidelity)
    return max(0.0, (1 - f_dep) * dim * dim / (dim * dim - 1))


def _fold(ops, scale: float):
    """Local folding G -> G G^dag G of a share of the ops, spread evenly."""
    m = len(ops)
    n_fold = min(m, round((scale - 1) / 2 * m)) if m else 0
    chosen = {int(j * m / n_fold) for j in range(n_fold)} if n_fold else set()
    out = []
    for k, op in enumerate(ops):
        out.append(op)
        if k in chosen:
            inv = (op[0], op[1], op[2].conj().T) + tuple(op[3:])
            out.extend([inv, op])
    return out, (1 + 2 * n_fold / m) if m else 1.0


def simulate_compiled(compiled: dict, cal: Calibration, n: int) -> np.ndarray:
    """Probabilities over logical bitstrings (no readout error) for compiled
    ops, exact (density matrix)."""
    dm = DensityMatrix(n)
    _run(compiled, cal, n, dm)
    return _to_logical(dm.probabilities(), compiled, n)


def _to_logical(probs: np.ndarray, compiled: dict, n: int) -> np.ndarray:
    """Reorder probabilities (bit k = slot k) so that bit i = logical qubit i."""
    pos = compiled["final_pos"]
    tensor = probs.reshape((2,) * n)                  # axis n-1-k = slot k
    tensor = np.transpose(tensor, [n - 1 - pos[n - 1 - axis] for axis in range(n)])
    return tensor.reshape(-1)


def _run(compiled: dict, cal: Calibration, n: int, dm) -> None:
    """Drive ``dm`` (anything with apply_1q, apply_2q, depolarize, relax)
    through the scheduled, noisy compiled circuit."""
    phys = compiled["physical"]
    t = [0.0] * n

    # relaxation is applied lazily: T1/T2 channels compose, so the time a
    # qubit spends after its last gate is accumulated and applied once,
    # right before the qubit is used again (or at the end)
    pending = [0.0] * n

    def flush(slot):
        p = phys[slot]
        duration, pending[slot] = pending[slot], 0.0
        if duration > 0 and (math.isfinite(cal.t1[p]) or math.isfinite(cal.t2[p])):
            dm.relax(slot, duration, cal.t1[p], cal.t2[p])

    def relax(slot, duration):
        pending[slot] += duration

    for op in compiled["ops"]:
        slots = op[1]
        if op[0] == "1q":
            p = phys[slots[0]]
            pulses = _pulses(cal, op[2])
            duration = pulses * cal.sq_duration[p]
            error = 1 - (1 - cal.sq_error[p]) ** pulses
            dims = 2
        else:
            pa, pb = phys[slots[0]], phys[slots[1]]
            error, duration = cal.two_qubit(pa, pb)
            dims = 4
        start = max(t[s] for s in slots)
        for s in slots:
            relax(s, start - t[s])
            flush(s)
        if op[0] == "1q":
            dm.apply_1q(op[2], slots[0])
        else:
            # 4x4 matrices are in the basis |slot1 slot0>
            dm.apply_2q(op[2], slots[0], slots[1])
            if cal.zz_overrotation:
                dm.apply_2q(rzz_matrix(cal.zz_overrotation), slots[0], slots[1])
            coherent = cal.coherent_2q.get((min(pa, pb), max(pa, pb)))
            if coherent is not None:
                dm.apply_2q(coherent, slots[0], slots[1])
        f_relax = 1.0
        for s in slots:
            f_relax *= _relax_fidelity(duration, cal.t1[phys[s]], cal.t2[phys[s]])
        lam = _depolarizing_param(error, dims, f_relax)
        if lam > 0:
            dm.depolarize(slots, lam)
        for s in slots:
            relax(s, duration)
            t[s] = start + duration
    end = max(t) if t else 0.0
    for s in range(n):
        relax(s, end - t[s])
        flush(s)


class _Program:
    """Records the noisy circuit as unitaries and Pauli channels: the
    trajectory form of what ``_run`` does to a density matrix. Depolarizing
    noise is exact; T1/T2 relaxation is Pauli-twirled (px = py = gamma/4,
    pz = (1 - exp(-t/T2))/2 - gamma/4), as in Stim-style circuit noise."""

    def __init__(self):
        self.steps = []        # ("u", qubits, matrix) or ("p", qubits, p_event, weights)

    def apply_1q(self, m, q):
        self.steps.append(("u", (q,), m))

    def apply_2q(self, m, q0, q1):
        self.steps.append(("u", (q0, q1), m))

    def depolarize(self, qubits, lam):
        d2 = 4 ** len(qubits)
        if lam > 0:
            self.steps.append(("p", tuple(qubits), lam * (d2 - 1) / d2, None))

    def relax(self, q, duration, t1, t2):
        if duration <= 0:
            return
        t2 = min(t2, 2 * t1)
        gamma = 1 - math.exp(-duration / t1) if math.isfinite(t1) else 0.0
        pz = (1 - math.exp(-duration / t2)) / 2 - gamma / 4 if math.isfinite(t2) else 0.0
        weights = np.array([gamma / 4, gamma / 4, max(0.0, pz)])
        if weights.sum() > 0:
            self.steps.append(("p", (q,), float(weights.sum()), weights / weights.sum()))


def _apply_event(state: StateVector, qubits, weights, rng):
    """A non-identity Pauli on ``qubits``: uniform for depolarizing noise
    (weights None), else X/Y/Z with the given weights."""
    if weights is None:
        k = int(rng.integers(1, 4 ** len(qubits)))
        for i, q in enumerate(qubits):
            state.pauli((k >> (2 * i)) & 3, q)
    else:
        state.pauli(int(rng.choice(3, p=weights)) + 1, qubits[0])


def _trajectories(compiled: dict, cal: Calibration, n: int, support: np.ndarray, seed: int = 7):
    """Mean probabilities by quantum trajectories with no-error splitting.

    The probability P0 that no noise event happens is computed exactly; its
    contribution is the ideal output. Only trajectories with at least one
    event are sampled: the first event is drawn from its exact distribution,
    and the run restarts from a saved copy of the ideal state just before it.
    """
    prog = _Program()
    _run(compiled, cal, n, prog)
    steps = prog.steps
    p_event = np.array([st[2] if st[0] == "p" else 0.0 for st in steps])
    log_keep = np.log1p(-np.minimum(p_event, 1 - 1e-15))
    before = np.concatenate([[0.0], np.cumsum(log_keep)[:-1]])   # log P(no event before i)
    p0 = float(np.exp(log_keep.sum()))
    first = np.where(p_event > 0, p_event * np.exp(before), 0.0)
    n_unitary = sum(1 for st in steps if st[0] == "u")

    # ideal run with checkpoints
    stride = max(1, math.ceil(len(steps) / max(1, CHECKPOINT_BYTES // (16 * 2 ** n))))
    state, checkpoints = StateVector(n), {}
    for i, st in enumerate(steps):
        if i % stride == 0:
            checkpoints[i] = state.copy()
        if st[0] == "u":
            (state.apply_1q if len(st[1]) == 1 else state.apply_2q)(st[2], *st[1])
    ideal = _to_logical(state.probabilities(), compiled, n)
    if p0 >= 1 - 1e-12 or first.sum() <= 0:
        return ideal, {"engine": "trajectories (no noise events)", "trajectories": 0,
                       "standard_error": 0.0}

    rng = np.random.default_rng(seed)
    first = first / first.sum()
    max_traj = int(min(MAX_TRAJECTORIES,
                       max(MIN_TRAJECTORIES, MAX_TRAJECTORY_WORK / (max(1, n_unitary) * 2 ** n))))
    total, values = None, []
    while len(values) < max_traj:
        i0 = int(rng.choice(len(steps), p=first))
        start = (i0 // stride) * stride
        sv = checkpoints[start].copy()
        for i in range(start, len(steps)):
            st = steps[i]
            if st[0] == "u":
                (sv.apply_1q if len(st[1]) == 1 else sv.apply_2q)(st[2], *st[1])
            elif i == i0 or (i > i0 and rng.random() < st[2]):
                _apply_event(sv, st[1], st[3], rng)
        probs = _to_logical(sv.probabilities(), compiled, n)
        total = probs if total is None else total + probs
        values.append(float(probs[support].sum()))
        k = len(values)
        if k >= MIN_TRAJECTORIES and \
                (1 - p0) * np.std(values, ddof=1) / math.sqrt(k) <= TARGET_STANDARD_ERROR:
            break
    k = len(values)
    se = float((1 - p0) * np.std(values, ddof=1) / math.sqrt(k)) if k > 1 else math.nan
    mean = p0 * ideal + (1 - p0) * total / k
    return mean, {"engine": f"{k} trajectories", "trajectories": k, "standard_error": se,
                  "p_no_error": p0}


def _readout(compiled, cal):
    phys, pos = compiled["physical"], compiled["final_pos"]
    return ([cal.p01[phys[pos[i]]] for i in range(len(pos))],
            [cal.p10[phys[pos[i]]] for i in range(len(pos))])


def _mirror(compiled: dict) -> dict:
    """The compiled circuit followed by its inverse (same native ops,
    reversed and conjugated). Ideally it returns every qubit to |0>, so its
    success is measurable whatever the original output looks like."""
    inverse = []
    for op in reversed(compiled["ops"]):
        inverse.append((op[0], op[1], op[2].conj().T) + tuple(op[3:]))
    n = len(compiled["final_pos"])
    return {**compiled, "ops": list(compiled["ops"]) + inverse, "final_pos": list(range(n))}


def _prepare(qasm_path_or_circuit, calibration, exact_only: bool = False):
    circ = (qasm_path_or_circuit if isinstance(qasm_path_or_circuit, _circuit.Circuit)
            else _circuit.load(qasm_path_or_circuit))
    cal = load(calibration)
    n = circ.num_qubits
    limit = MAX_QUBITS if exact_only else MAX_QUBITS_TRAJECTORIES
    if n > limit:
        kind = "exact density-matrix" if exact_only else "trajectory"
        return None, f"{n} qubits > {limit} ({kind} limit)"
    support = np.abs(statevector(circ)) ** 2 > 1e-9
    fraction = float(support.sum()) / 2 ** n
    return (circ, cal, support, fraction), None


def _simulate(compiled: dict, cal: Calibration, n: int, support: np.ndarray, seed: int = 7):
    """Mean probabilities and engine info: exact up to MAX_QUBITS, else
    quantum trajectories (see ``_trajectories``)."""
    if n <= MAX_QUBITS:
        return simulate_compiled(compiled, cal, n), {"engine": "exact density matrix"}
    return _trajectories(compiled, cal, n, support, seed)


def success_fraction(qasm_path_or_circuit, calibration, layout: Optional[list] = None) -> dict:
    """Unmitigated reliability on a calibrated device.

    Two metrics, chosen from the ideal output:

    - ``output_support`` (ideal output on at most half of the bitstrings):
      the probability of landing in the ideal output support;
    - ``mirror`` (broad outputs, e.g. variational circuits): the circuit
      followed by its inverse is simulated on the device; its probability of
      returning to |0...0> without readout error, F_mirror, gives the
      circuit's estimated fidelity sqrt(F_mirror), times the probability
      that no measured qubit is misread. An estimate: errors in the two
      halves are assumed independent; coherent errors can partly cancel.

    Exact (density matrix) up to 10 qubits, quantum trajectories up to 20
    (``standard_error`` then reports the sampling uncertainty).
    """
    prepared, reason = _prepare(qasm_path_or_circuit, calibration)
    if prepared is None:
        return {"available": False, "reason": reason}
    circ, cal, support, fraction = prepared
    n = circ.num_qubits
    compiled = compile_to_device(circ, cal, layout)
    p01, p10 = _readout(compiled, cal)
    result = {"available": True,
              "device_model": f"{cal.name} calibration",
              "calibration_source": cal.source,
              "physical_qubits": compiled["physical"],
              "two_qubit_gates": sum(1 for op in compiled["ops"] if op[0] == "2q")}
    if fraction <= 0.5:
        probs, engine = _simulate(compiled, cal, n, support)
        result.update(metric="output_support", **engine,
                      success_fraction=float(apply_readout_error(probs, p01, p10)[support].sum()),
                      success_fraction_readout_mitigated=float(probs[support].sum()))
    else:
        zero = np.zeros(2 ** n, dtype=bool)
        zero[0] = True
        probs, engine = _simulate(_mirror(compiled), cal, n, zero)
        f_mirror = float(probs[0])
        fidelity = math.sqrt(max(0.0, f_mirror))
        readout = float(np.prod([1 - (a + b) / 2 for a, b in zip(p01, p10)]))
        if "standard_error" in engine:          # propagate to sqrt
            engine["standard_error"] = engine["standard_error"] / (2 * max(fidelity, 1e-12))
        result.update(metric="mirror", **engine, mirror_success=f_mirror,
                      output_support_fraction=fraction,
                      success_fraction=fidelity * readout,
                      success_fraction_readout_mitigated=fidelity)
    return result


def zne_values(qasm_path_or_circuit, calibration, scale_factors=(1, 2, 3),
               layout: Optional[list] = None):
    """Success fractions at each noise scale (local folding of the native
    ops). Returns (prepared info dict, scales reached, raw values, values
    with ideal readout mitigation) or (None, reason)."""
    prepared, reason = _prepare(qasm_path_or_circuit, calibration, exact_only=True)
    if prepared is None:
        return None, reason
    circ, cal, support, fraction = prepared
    if fraction > 0.5:
        return None, (f"ideal output support covers {fraction:.0%} of bitstrings: "
                      "the ZNE bias needs a concentrated output")
    compiled = compile_to_device(circ, cal, layout)
    if not any(op[0] == "2q" for op in compiled["ops"]):
        return None, "no two-qubit gates to fold"
    p01, p10 = _readout(compiled, cal)
    scales, raw, rem = [], [], []
    for scale in scale_factors:
        ops, reached = _fold(compiled["ops"], float(scale)) if scale != 1 else (compiled["ops"], 1.0)
        probs = simulate_compiled({**compiled, "ops": ops}, cal, circ.num_qubits)
        scales.append(reached)
        rem.append(float(probs[support].sum()))
        raw.append(float(apply_readout_error(probs, p01, p10)[support].sum()))
    info = {"device_model": f"{cal.name} calibration", "physical_qubits": compiled["physical"],
            "asymptote": fraction}
    return (info, scales, raw, rem), None


# --------------------------------------------------------------------------
# export from a Qiskit backend (optional)
# --------------------------------------------------------------------------

def from_qiskit_backend(backend, name: Optional[str] = None, source: str = "") -> dict:
    """Build a calibration dict from a Qiskit ``BackendV2`` (a real IBM
    backend or a fake one). Needs Qiskit; qregime itself does not."""
    target = backend.target
    n = backend.num_qubits
    props = target.qubit_properties or [None] * n

    def first_2q():
        for gate in ("cz", "ecr", "cx"):
            if gate in target.operation_names:
                return gate
        raise ValueError("backend has no cz, ecr or cx gate")

    gate2 = first_2q()
    sx = target["sx"]
    measure = target["measure"]
    pairs, edges = {}, set()
    for qargs, inst in target[gate2].items():
        if inst is None:
            continue
        a, b = sorted(qargs)
        edges.add((a, b))
        key = f"{a},{b}"
        error = inst.error or 0.0
        if key in pairs:      # directed gates: keep the better direction
            error = min(error, pairs[key]["error"])
        pairs[key] = {"error": error, "duration": inst.duration or 0.0}
    return {
        "schema": SCHEMA,
        "name": name or backend.name,
        "source": source or f"exported from Qiskit backend {backend.name}",
        "num_qubits": n,
        "connectivity": sorted([list(e) for e in edges]),
        "single_qubit": {"error": [(sx[(q,)].error or 0.0) if (q,) in sx else None for q in range(n)],
                         "duration": [(sx[(q,)].duration or 0.0) if (q,) in sx else None
                                      for q in range(n)],
                         "decomposition": "zsx"},
        "two_qubit": {"native": "cx" if gate2 == "cx" else "cz",
                      "error": float(np.median([p["error"] for p in pairs.values()])),
                      "duration": float(np.median([p["duration"] for p in pairs.values()])),
                      "pairs": pairs},
        "readout": {"p01": [(measure[(q,)].error or 0.0) if (q,) in measure else None
                            for q in range(n)],
                    "p10": [(measure[(q,)].error or 0.0) if (q,) in measure else None
                            for q in range(n)]},
        "idle": {"t1": [getattr(p, "t1", None) for p in props],
                 "t2": [getattr(p, "t2", None) for p in props]},
    }
