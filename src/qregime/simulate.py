"""Small numpy simulators for qregime's circuits (no quantum SDK needed).

- ``statevector``: exact pure state, used for the ideal output support and
  the entanglement probe (up to ~20 qubits).
- ``DensityMatrix``: exact mixed state with the noise channels qregime
  needs (depolarizing, T1/T2 relaxation), used for noisy and calibrated
  simulations up to ~10 qubits.

Conventions: qubit 0 is the least significant bit of a basis-state index
(as in Qiskit), so probabilities can be compared directly across tools.
"""
from __future__ import annotations

import cmath
import math

import numpy as np

from .circuit import Circuit

_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def u_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """OpenQASM U(theta, phi, lambda)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -cmath.exp(1j * lam) * s],
                     [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c]], dtype=complex)


def fused_ops(ops, marks=()) -> list:
    """Merge consecutive one-qubit gates on the same qubit into one 2x2
    matrix. Input: (kind, qubits, params, tag) from ``Circuit.basis_ops``;
    output: ("1q", (q,), matrix, tags) and ("CX", (c, t), None, tags), where
    tags lists the top-level instructions each op came from.

    For every tag in ``marks``, pending gates on that instruction's qubits
    are flushed right after its last op and a ("mark", qubits, None, [tag])
    entry is emitted there, so noise can be attached to the instruction
    without moving it past later gates.
    """
    marks = set(marks)
    pending: dict = {}
    out = []

    def flush(q):
        if q in pending:
            m, tags = pending.pop(q)
            out.append(("1q", (q,), m, tags))

    def close(tag, qubits):
        for q in qubits:
            flush(q)
        out.append(("mark", tuple(qubits), None, [tag]))

    current, current_qubits = None, set()
    for kind, qubits, params, tag in ops:
        if tag != current:
            if current in marks:
                close(current, sorted(current_qubits))
            current, current_qubits = tag, set()
        current_qubits.update(qubits)
        if kind == "U":
            q = qubits[0]
            m, tags = pending.get(q, (np.eye(2, dtype=complex), []))
            pending[q] = (u_matrix(*params) @ m, tags + [tag])
        else:
            for q in qubits:
                flush(q)
            out.append(("CX", qubits, None, [tag]))
    if current in marks:
        close(current, sorted(current_qubits))
    for q in sorted(pending):
        flush(q)
    return out


# --------------------------------------------------------------------------
# pure states
# --------------------------------------------------------------------------

def _apply_1q_sv(psi: np.ndarray, m: np.ndarray, q: int, n: int) -> np.ndarray:
    psi = psi.reshape(2 ** (n - 1 - q), 2, 2 ** q)
    return np.einsum("ij,ajb->aib", m, psi).reshape(-1)


def _apply_cx_sv(psi: np.ndarray, c: int, t: int, n: int) -> np.ndarray:
    tensor = psi.reshape((2,) * n)
    ax_c, ax_t = n - 1 - c, n - 1 - t
    index = [slice(None)] * n
    index[ax_c] = 1
    sub = tensor[tuple(index)]
    axis = ax_t if ax_t < ax_c else ax_t - 1
    sub[...] = np.flip(sub, axis=axis).copy()
    return tensor.reshape(-1)


def statevector(circ: Circuit) -> np.ndarray:
    """Final state of ``circ`` from |0...0>, ignoring non-unitary instructions."""
    n = circ.num_qubits
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for kind, qubits, m, _ in fused_ops(circ.basis_ops()):
        if kind == "1q":
            psi = _apply_1q_sv(psi, m, qubits[0], n)
        else:
            psi = _apply_cx_sv(psi, qubits[0], qubits[1], n)
    return psi


def bipartite_entropy(psi: np.ndarray, n: int, n_a: int) -> float:
    """Von Neumann entropy (bits) of qubits 0..n_a-1 of the pure state psi."""
    s = np.linalg.svd(psi.reshape(2 ** (n - n_a), 2 ** n_a), compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    return max(0.0, float(-np.sum(p * np.log2(p))))


# --------------------------------------------------------------------------
# mixed states
# --------------------------------------------------------------------------

class DensityMatrix:
    """Exact density matrix on n qubits (practical up to ~10-11 qubits)."""

    def __init__(self, n: int):
        self.n = n
        self.rho = np.zeros((2,) * (2 * n), dtype=complex)
        self.rho[(0,) * (2 * n)] = 1.0

    def _axes(self, q):
        return self.n - 1 - q, 2 * self.n - 1 - q

    def apply_1q(self, m: np.ndarray, q: int):
        r, c = self._axes(q)
        rho = np.moveaxis(np.tensordot(m, self.rho, axes=([1], [r])), 0, r)
        self.rho = np.moveaxis(np.tensordot(rho, m.conj(), axes=([c], [1])), -1, c)

    def apply_2q(self, m: np.ndarray, q0: int, q1: int):
        """``m`` is 4x4 in the basis |q1 q0> (q0 least significant)."""
        g = m.reshape(2, 2, 2, 2)                     # (q1', q0', q1, q0)
        (r0, c0), (r1, c1) = self._axes(q0), self._axes(q1)
        rho = np.tensordot(g, self.rho, axes=([2, 3], [r1, r0]))
        rho = np.moveaxis(rho, [0, 1], [r1, r0])
        rho = np.tensordot(rho, g.conj(), axes=([c1, c0], [2, 3]))
        self.rho = np.moveaxis(rho, [-2, -1], [c1, c0])

    def apply_cx(self, c: int, t: int):
        n = self.n
        tensor = self.rho
        for offset in (0, n):                          # rows, then columns
            ax_c, ax_t = offset + n - 1 - c, offset + n - 1 - t
            index = [slice(None)] * (2 * n)
            index[ax_c] = 1
            sub = tensor[tuple(index)]
            axis = ax_t if ax_t < ax_c else ax_t - 1
            sub[...] = np.flip(sub, axis=axis).copy()

    def depolarize(self, qubits, lam: float):
        """rho -> (1 - lam) rho + lam Tr_S(rho) (x) I/d on the qubits S."""
        if lam <= 0:
            return
        traced = self.rho
        axes = sorted((ax for q in qubits for ax in self._axes(q)))
        # partial trace over S, keeping the axis positions with size 1
        for q in qubits:
            r, c = self._axes(q)
            diag = np.take(traced, 0, axis=r).take(0, axis=c - 1) + \
                np.take(traced, 1, axis=r).take(1, axis=c - 1)
            traced = np.expand_dims(np.expand_dims(diag, r), c)
        d = 2 ** len(qubits)
        mixed = np.broadcast_to(traced, self.rho.shape).copy()
        # keep only the diagonal in S (identity), divided by d
        mask = np.ones(self.rho.shape, dtype=bool)
        for q in qubits:
            r, c = self._axes(q)
            shape = [1] * (2 * self.n)
            shape[r], shape[c] = 2, 2
            mask &= np.eye(2, dtype=bool).reshape(shape)
        del axes
        self.rho = (1 - lam) * self.rho + lam * np.where(mask, mixed, 0) / d

    def relax(self, q: int, duration: float, t1: float, t2: float):
        """Amplitude damping (T1) and dephasing (T2) for ``duration`` seconds."""
        if duration <= 0:
            return
        t2 = min(t2, 2 * t1)
        a = math.exp(-duration / t1)
        coh = math.exp(-duration / t2)
        r, c = self._axes(q)

        def part(i, j):
            index = [slice(None)] * (2 * self.n)
            index[r], index[c] = i, j
            return tuple(index)

        p11 = self.rho[part(1, 1)].copy()
        self.rho[part(0, 0)] += (1 - a) * p11
        self.rho[part(1, 1)] = a * p11
        self.rho[part(0, 1)] *= coh
        self.rho[part(1, 0)] *= coh

    def probabilities(self) -> np.ndarray:
        size = 2 ** self.n
        return np.real(np.diagonal(self.rho.reshape(size, size))).clip(min=0)


class StateVector:
    """A pure state on n qubits with in-place gate and Pauli kernels, for
    quantum-trajectory simulation (``qregime.calibration`` averages many).
    Bit q of the amplitude index is qubit q. Views use a few large axes
    rather than n axes of size 2, which keeps numpy's inner loops long."""

    def __init__(self, n: int, psi: np.ndarray = None):
        self.n = n
        if psi is None:
            psi = np.zeros(2 ** n, dtype=complex)
            psi[0] = 1.0
        self.psi = psi

    def copy(self) -> "StateVector":
        return StateVector(self.n, self.psi.copy())

    def _view(self, fixed: dict):
        qs = sorted(fixed, reverse=True)
        shape, index, prev = [], [], self.n
        for q in qs:
            shape += [2 ** (prev - q - 1), 2]
            index += [slice(None), fixed[q]]
            prev = q
        shape.append(2 ** prev)
        index.append(slice(None))
        return self.psi.reshape(shape)[tuple(index)]

    def apply_1q(self, m: np.ndarray, q: int):
        a, b = self._view({q: 0}), self._view({q: 1})
        a_old = a.copy()
        a *= m[0, 0]
        a += m[0, 1] * b
        b *= m[1, 1]
        b += m[1, 0] * a_old

    def apply_2q(self, m: np.ndarray, q0: int, q1: int):
        """``m`` is 4x4 in the basis |q1 q0> (q0 least significant)."""
        views = [self._view({q1: k >> 1, q0: k & 1}) for k in range(4)]
        old = [v.copy() for v in views]
        for i, v in enumerate(views):
            terms = [(m[i, j], old[j]) for j in range(4) if m[i, j] != 0]
            v[...] = terms[0][0] * terms[0][1]
            for c, o in terms[1:]:
                v += c * o

    def pauli(self, k: int, q: int):
        """Apply Pauli k (0 = I, 1 = X, 2 = Y, 3 = Z) to qubit q."""
        if k == 1 or k == 2:
            a, b = self._view({q: 0}), self._view({q: 1})
            tmp = a.copy()
            a[...] = b
            b[...] = tmp
            if k == 2:                  # Y = i X Z: after the swap, a holds old b
                a *= -1j
                b *= 1j
        elif k == 3:
            self._view({q: 1})[...] *= -1

    def probabilities(self) -> np.ndarray:
        return self.psi.real ** 2 + self.psi.imag ** 2


def apply_readout_error(probs: np.ndarray, p01, p10) -> np.ndarray:
    """Per-qubit readout flips: p01[i] = P(read 1 | 0), p10[i] = P(read 0 | 1)."""
    n = len(p01)
    t = np.asarray(probs, dtype=float).reshape((2,) * n)
    for i in range(n):
        axis = n - 1 - i
        m = np.array([[1 - p01[i], p10[i]], [p01[i], 1 - p10[i]]])
        t = np.moveaxis(np.tensordot(m, t, axes=([1], [axis])), 0, axis)
    return t.reshape(-1)


# --------------------------------------------------------------------------
# qregime's depolarizing model (two-qubit gates only)
# --------------------------------------------------------------------------

#: top-level gate names that receive the two-qubit depolarizing error
DEPOLARIZED_GATES = ("cx", "cz", "ecr", "rzz", "cp")


def depolarizing_success(circ: Circuit, support: np.ndarray, p_2q: float) -> float:
    """Exact probability of landing in ``support`` when every top-level
    two-qubit gate named in ``DEPOLARIZED_GATES`` is followed by a two-qubit
    depolarizing channel with parameter ``p_2q``."""
    dm = DensityMatrix(circ.num_qubits)
    noisy = {k for k, inst in enumerate(circ.instructions)
             if inst.name in DEPOLARIZED_GATES and len(inst.qubits) == 2}
    for kind, qubits, m, tags in fused_ops(circ.basis_ops(), marks=noisy):
        if kind == "1q":
            dm.apply_1q(m, qubits[0])
        elif kind == "CX":
            dm.apply_cx(*qubits)
        else:
            dm.depolarize(circ.instructions[tags[0]].qubits, p_2q)
    return float(dm.probabilities()[support].sum())


_PAULIS = (np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], dtype=complex),
           np.array([[0, -1j], [1j, 0]], dtype=complex), np.diag([1, -1]).astype(complex))
#: cap on trajectory work (amplitude updates) before giving up
MAX_TRAJECTORY_WORK = 4e10


def depolarizing_trajectories(circ: Circuit, support: np.ndarray, p_2q: float,
                              shots: int = 150, seed: int = 42):
    """Monte Carlo version of :func:`depolarizing_success` for circuits too
    large for a density matrix. Each trajectory draws the depolarizing
    events (a uniformly random two-qubit Pauli with probability ``p_2q``
    after each noisy gate); trajectories without events land in the support
    with probability 1, the others are simulated and contribute their exact
    probability of landing in the support. Returns (estimate, trajectories
    simulated), or None if the work would exceed ``MAX_TRAJECTORY_WORK``.
    """
    rng = np.random.default_rng(seed)
    noisy = [k for k, inst in enumerate(circ.instructions)
             if inst.name in DEPOLARIZED_GATES and len(inst.qubits) == 2]
    ops = fused_ops(circ.basis_ops(), marks=noisy)
    hits = rng.random((shots, len(noisy))) < p_2q
    paulis = rng.integers(0, 4, size=(shots, len(noisy), 2))
    to_simulate = int(hits.any(axis=1).sum())
    n = circ.num_qubits
    if to_simulate * len(ops) * 2 ** n > MAX_TRAJECTORY_WORK:
        return None
    slot = {tag: j for j, tag in enumerate(noisy)}
    total = 0.0
    for shot in range(shots):
        if not hits[shot].any():
            total += 1.0
            continue
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1.0
        for kind, qubits, m, tags in ops:
            if kind == "1q":
                psi = _apply_1q_sv(psi, m, qubits[0], n)
            elif kind == "CX":
                psi = _apply_cx_sv(psi, qubits[0], qubits[1], n)
            else:
                j = slot[tags[0]]
                if hits[shot, j]:
                    for q, k in zip(circ.instructions[tags[0]].qubits, paulis[shot, j]):
                        if k:
                            psi = _apply_1q_sv(psi, _PAULIS[k], q, n)
        total += float(np.sum(np.abs(psi[support]) ** 2))
    return total / shots, to_simulate
