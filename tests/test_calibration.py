"""Vendor-neutral calibrations and the calibrated simulator."""
import math

import numpy as np
import pytest

from qregime import calibration as K
from qregime import circuit as C

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _mirror(n=4, layers=4, seed=1):
    rng = np.random.default_rng(seed)
    ops = []
    for layer in range(layers):
        ops += [f"ry({rng.uniform(0.1, 3.0)!r}) q[{q}];" for q in range(n)]
        ops += [f"cx q[{q}],q[{q + 1}];" for q in range(layer % 2, n - 1, 2)]
    inverse = []
    for op in reversed(ops):
        if op.startswith("ry("):
            angle, rest = op[3:].split(")", 1)
            inverse.append(f"ry({-float(angle)!r}){rest}")
        else:
            inverse.append(op)
    return C.parse(HEADER + f"qreg q[{n}];\n" + "\n".join(ops + inverse) + "\n")


def _noiseless(**overrides):
    data = {"schema": K.SCHEMA, "name": "ideal", "num_qubits": 8,
            "connectivity": [[i, i + 1] for i in range(7)],
            "two_qubit": {"native": "cz"}, "single_qubit": {"decomposition": "zsx"}}
    data.update(overrides)
    return K.load(data)


@pytest.mark.parametrize("native", ["cx", "cz", "rzz"])
def test_native_rewrites_equal_cx(native):
    m = np.eye(4, dtype=complex)
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    for op in K._native_cx(0, 1, native):
        if op[0] == "1q":
            g = np.kron(np.eye(2), op[2]) if op[1] == (0,) else np.kron(op[2], np.eye(2))
        else:
            g = op[2] if op[1] == (0, 1) else swap @ op[2] @ swap
        m = g @ m
    assert np.allclose(m / m[0, 0], K._CX)


def test_routing_on_a_line_preserves_the_output():
    text = HEADER + "qreg q[6];\nx q[0];\nx q[3];\n" + "".join(
        f"cx q[{a}],q[{b}];\n" for a in range(6) for b in range(6) if a != b and (a + b) % 3 == 0)
    circ = C.parse(text + "ccx q[0],q[5],q[2];\n")
    r = K.success_fraction(circ, _noiseless())
    assert r["success_fraction"] == pytest.approx(1.0)
    assert r["two_qubit_gates"] > 20                      # SWAPs were inserted
    all_to_all = _noiseless(connectivity="all_to_all", two_qubit={"native": "rzz"})
    assert K.success_fraction(circ, all_to_all)["success_fraction"] == pytest.approx(1.0)


def test_noise_lowers_success_and_readout_mitigation_helps():
    circ = _mirror()
    for name in K.available():
        r = K.success_fraction(circ, name)
        assert 0 < r["success_fraction"] < r["success_fraction_readout_mitigated"] < 1, name


def test_layout_avoids_bad_qubits():
    bad = [0.5] * 8
    bad[6] = bad[7] = 0.0
    cal = _noiseless(readout={"p01": bad, "p10": bad})
    circ = C.parse(HEADER + "qreg q[2];\nx q[0];\ncx q[0],q[1];\n")
    r = K.success_fraction(circ, cal)
    assert sorted(r["physical_qubits"]) == [6, 7]
    assert r["success_fraction"] == pytest.approx(1.0)


def test_memory_error_per_layer_matches_dephasing():
    m = 0.01
    cal = K.load({"schema": K.SCHEMA, "name": "mem", "num_qubits": 1,
                  "idle": {"memory_error_per_layer": m},
                  "two_qubit": {"duration": 1.0}})
    assert math.exp(-1 / cal.t2[0]) == pytest.approx(1 - 2 * m)


def test_schema_is_checked():
    with pytest.raises(ValueError, match="schema"):
        K.load({"name": "x", "num_qubits": 1})
    with pytest.raises(ValueError, match="expected 2 values"):
        K.load({"schema": K.SCHEMA, "num_qubits": 2, "readout": {"p01": [0.1]}})
    with pytest.raises(ValueError, match="unknown calibration"):
        K.load("no_such_device")


def test_bundled_calibrations_load():
    assert {"ibm_torino", "quantinuum_h2_2"} <= set(K.available())
    torino = K.load("ibm_torino")
    assert torino.num_qubits == 133 and torino.native == "cz" and not torino.all_to_all
    h2 = K.load("quantinuum_h2_2")
    assert h2.all_to_all and h2.native == "rzz"


@pytest.mark.parametrize("fake, name", [("FakeTorino", "ibm_torino"),
                                        ("FakeBoston", "ibm_boston"),
                                        ("FakeMiami", "ibm_miami")])
def test_agrees_with_qiskit_aer_on_fake_backend(tmp_path, fake, name):
    """Same layout, Aer density matrix with the fake backend's noise model and
    single-qubit fusion: success fractions agree within 0.005."""
    pytest.importorskip("qiskit_ibm_runtime")
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import PassManager, generate_preset_pass_manager
    from qiskit.transpiler.passes import (ALAPScheduleAnalysis, Optimize1qGatesDecomposition,
                                          PadDelay)
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider

    circ = _mirror(n=4, layers=6)
    qc = QuantumCircuit.from_qasm_str(C.to_qasm(circ))
    if not hasattr(fake_provider, fake):
        pytest.skip(f"{fake} needs a newer qiskit-ibm-runtime")
    backend = getattr(fake_provider, fake)()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)
    pm.init = generate_preset_pass_manager(optimization_level=0, backend=backend).init
    pm.optimization = PassManager([Optimize1qGatesDecomposition(target=backend.target)])
    tqc = pm.run(qc)
    phys = tqc.layout.final_index_layout()
    timed = PassManager([ALAPScheduleAnalysis(target=backend.target),
                         PadDelay(target=backend.target)]).run(tqc)
    timed.save_probabilities(qubits=phys)
    probs = np.asarray(AerSimulator(method="density_matrix",
                                    noise_model=NoiseModel.from_backend(backend))
                       .run(timed).result().data()["probabilities"])
    support = np.abs(Statevector(qc).data) ** 2 > 1e-9
    ours = K.success_fraction(circ, name, layout=list(phys))
    assert ours["success_fraction_readout_mitigated"] == pytest.approx(probs[support].sum(), abs=5e-3)


def test_bundled_vendor_calibrations():
    expected = {"google_willow": (105, "cz", False), "iqm_garnet": (20, "cz", False),
                "ibm_boston": (156, "cz", False), "ibm_miami": (120, "cz", False),
                "iqm_emerald": (54, "cz", False), "quantinuum_helios": (98, "rzz", True),
                "ionq_forte": (30, "rzz", True), "quantinuum_h2_2": (56, "rzz", True),
                "ibm_torino": (133, "cz", False)}
    assert set(expected) <= set(K.available())
    for name, (n, native, all_to_all) in expected.items():
        cal = K.load(name)
        assert (cal.num_qubits, cal.native, cal.all_to_all) == (n, native, all_to_all), name
    assert K.load("google_willow").coherent_2q          # fSim errors per coupler


def test_fsim_coherent_error_is_exact():
    theta, phi = 0.3, 0.5
    cal = K.load({"schema": K.SCHEMA, "name": "fsim", "num_qubits": 2,
                  "connectivity": [[0, 1]], "single_qubit": {"decomposition": "arbitrary"},
                  "two_qubit": {"native": "cz", "pairs": {"0,1": {
                      "swap_angle": theta, "conditional_phase": phi}}}})
    # X on q0, CZ: fSim moves sin^2(theta) of the population to |01>
    circ = C.parse(HEADER + "qreg q[2];\nx q[0];\ncz q[0],q[1];\n")
    r = K.success_fraction(circ, cal, layout=[0, 1])
    assert r["success_fraction"] == pytest.approx(math.cos(theta) ** 2)


def _vendor_mirror(n, layers):
    return _mirror(n=n, layers=layers, seed=1)


def test_agrees_with_cirq_google_noise_model():
    """google_willow vs cirq's own noise model built from the same Google
    calibration, on the same compiled circuit: within 0.003."""
    cirq = pytest.importorskip("cirq")
    cg = pytest.importorskip("cirq_google")
    import json
    import os
    import warnings

    warnings.filterwarnings("ignore")
    from qregime import simulate as S

    pkg = os.path.dirname(cg.__file__)
    calibration = cirq.read_json(
        pkg + "/devices/calibrations/willow_pink_d7v1-2024_08_16_calibration.json")
    times = {cirq.PhasedXZGate: 25, cirq.CZPowGate: 42, cirq.MeasurementGate: 600,
             cirq.ZPowGate: 0, cirq.ResetChannel: 160, cirq.WaitGate: 0, cirq.FSimGate: 42,
             cirq.ISwapPowGate: 42, cirq.PhasedFSimGate: 42, cirq.GlobalPhaseGate: 0,
             cirq.IdentityGate: 0, cirq.PhasedISwapPowGate: 42}
    noise = cg.NoiseModelFromGoogleNoiseProperties(
        cg.engine.noise_properties_from_calibration(calibration, gate_times_ns=times))
    cal = K.load("google_willow")
    labels = json.load(open(os.path.join(K._DIR, "google_willow.json")))["qubit_labels"]
    grid = [cirq.GridQubit(*map(int, label.split("_"))) for label in labels]
    for n, layers in ((3, 2), (5, 8)):
        circ = _vendor_mirror(n, layers)
        ours = K.success_fraction(circ, cal)
        comp = K.compile_to_device(circ, cal, ours["physical_qubits"])
        qs = [grid[p] for p in comp["physical"]]
        ops = []
        for op in comp["ops"]:
            if op[0] == "1q":
                gate = cirq.single_qubit_matrix_to_phxz(op[2])
                if gate is not None:
                    ops.append(gate.on(qs[op[1][0]]))
            else:
                ops.append(cirq.CZ(qs[op[1][0]], qs[op[1][1]]))
        rho = cirq.DensityMatrixSimulator(noise=noise, dtype=np.complex128).simulate(
            cirq.Circuit(ops), qubit_order=qs).final_density_matrix
        support = np.abs(S.statevector(circ)) ** 2 > 1e-9
        m = len(qs)
        theirs = sum(float(np.real(rho[i, i])) * support[sum(
            ((i >> (m - 1 - comp["final_pos"][q])) & 1) << q for q in range(n))]
            for i in range(2 ** m))
        assert ours["success_fraction_readout_mitigated"] == pytest.approx(theirs, abs=3e-3)


def test_agrees_with_iqm_garnet_noise_model():
    """iqm_garnet vs Aer with IQM's own IQMFakeGarnet noise model, same
    compiled circuit and the same idle times: within 0.003."""
    pytest.importorskip("iqm.qiskit_iqm")
    pytest.importorskip("qiskit_aer")
    import warnings

    warnings.filterwarnings("ignore")
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    from qiskit import QuantumCircuit
    from qiskit.circuit import Delay
    from qiskit.synthesis import OneQubitEulerDecomposer
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import RelaxationNoisePass

    from qregime import simulate as S

    backend = IQMFakeGarnet()
    prof = backend.error_profile
    labels = sorted(prof.t1s, key=lambda q: int(q[2:]))
    t1s = [prof.t1s[q] * 1e-9 for q in labels]
    t2s = [prof.t2s[q] * 1e-9 for q in labels]
    cal = K.load("iqm_garnet")
    zxz = OneQubitEulerDecomposer("ZXZ")
    sim = AerSimulator(method="density_matrix", noise_model=backend.noise_model)
    for n, layers in ((3, 2), (6, 8)):
        circ = _vendor_mirror(n, layers)
        ours = K.success_fraction(circ, cal)
        comp = K.compile_to_device(circ, cal, ours["physical_qubits"])
        phys = comp["physical"]
        qc = QuantumCircuit(cal.num_qubits)
        t = {s: 0.0 for s in range(len(phys))}

        def wait(s, until):
            if until - t[s] > 0:
                qc.delay(until - t[s], phys[s], unit="s")

        for op in comp["ops"]:
            slots = op[1]
            if op[0] == "1q":
                pulses = K._pulses(cal, op[2])
                duration = pulses * cal.sq_duration[phys[slots[0]]]
            else:
                duration = cal.two_qubit(phys[slots[0]], phys[slots[1]])[1]
            start = max(t[s] for s in slots)
            for s in slots:
                wait(s, start)
            if op[0] == "1q":
                theta, phi, lam = zxz.angles(op[2])
                if pulses:
                    qc.r(theta, -lam, phys[slots[0]])
                qc.rz(phi + lam, phys[slots[0]])     # virtual Z: noiseless
            else:
                qc.cz(phys[slots[0]], phys[slots[1]])
            for s in slots:
                t[s] = start + duration
        end = max(t.values())
        for s in t:
            wait(s, end)
        qc = RelaxationNoisePass(t1s, t2s, dt=None, op_types=[Delay])(qc)
        qc.save_density_matrix(qubits=phys)
        rho = np.asarray(sim.run(qc).result().data()["density_matrix"])
        support = np.abs(S.statevector(circ)) ** 2 > 1e-9
        theirs = sum(float(np.real(rho[i, i])) * support[sum(
            ((i >> comp["final_pos"][q]) & 1) << q for q in range(n))]
            for i in range(2 ** len(phys)))
        assert ours["success_fraction_readout_mitigated"] == pytest.approx(theirs, abs=3e-3)


@pytest.mark.parametrize("hardware", ["superconductor_google_willow",
                                      "superconductor_iqm_garnet", "trapped_ion_ionq_forte",
                                      "superconductor_ibm_heron", "superconductor_ibm_nighthawk",
                                      "superconductor_iqm_emerald",
                                      "trapped_ion_quantinuum_helios"])
def test_vendor_profiles_use_their_calibration(tmp_path, hardware):
    import qregime

    path = tmp_path / "m.qasm"
    path.write_text(C.to_qasm(_mirror(n=4, layers=4)))
    r = qregime.decide_regime(str(path), hardware=hardware, simulate_mitigation=True)
    cal = qregime.HARDWARE_PROFILES[hardware].calibration
    assert r["qem"]["reliability_source"].startswith(f"calibrated_simulation ({cal}")
    assert r["qem"]["mitigation"]["simulated"]["available"]


# --------------------------------------------------------------------------
# quantum trajectories and the mirror metric
# --------------------------------------------------------------------------

def test_state_vector_kernels_match_matrices():
    from qregime.simulate import StateVector

    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
              np.diag([1, -1])]
    u2 = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]

    def full(m, qubits):           # dense operator, bit q of the index = qubit q
        out = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            for j in range(8):
                if any((i >> q & 1) != (j >> q & 1) for q in range(3) if q not in qubits):
                    continue
                a = sum((i >> q & 1) << k for k, q in enumerate(qubits))
                b = sum((j >> q & 1) << k for k, q in enumerate(qubits))
                out[i, j] = m[a, b]
        return out

    for k in range(4):
        for q in range(3):
            sv = StateVector(3, psi.copy())
            sv.pauli(k, q)
            assert np.allclose(sv.psi, full(paulis[k], (q,)) @ psi)
    sv = StateVector(3, psi.copy())
    sv.apply_2q(u2, 2, 0)
    assert np.allclose(sv.psi, full(u2, (2, 0)) @ psi)


@pytest.mark.parametrize("name", ["ibm_boston", "google_willow", "iqm_garnet"])
def test_trajectories_agree_with_density_matrix(name, monkeypatch):
    monkeypatch.setattr(K, "TARGET_STANDARD_ERROR", 0.002)
    monkeypatch.setattr(K, "MIN_TRAJECTORIES", 200)
    circ, cal = _mirror(n=5, layers=8, seed=3), K.load(name)
    compiled = K.compile_to_device(circ, cal)
    support = np.abs(K.statevector(circ)) ** 2 > 1e-9
    exact = K.simulate_compiled(compiled, cal, 5)[support].sum()
    probs, info = K._trajectories(compiled, cal, 5, support)
    # T1 is Pauli-twirled in trajectories: a small pessimistic bias is expected
    assert abs(probs[support].sum() - exact) < 4 * info["standard_error"] + 0.006


def test_large_circuit_uses_trajectories(tmp_path):
    n = 12
    text = HEADER + f"qreg q[{n}];\nh q[0];\n" + "".join(
        f"cx q[{q}],q[{q + 1}];\n" for q in range(n - 1))
    r = K.success_fraction(C.parse(text), "quantinuum_helios")
    assert r["available"] and r["metric"] == "output_support"
    assert r["engine"].endswith("trajectories") and r["standard_error"] < 0.02
    # 11 ZZ gates at ~8e-4 plus readout: high but below 1
    assert 0.9 < r["success_fraction"] < 1.0


def test_mirror_estimate_matches_direct_metric_without_coherent_error():
    circ, cal = _mirror(n=5, layers=8, seed=3), K.load("ibm_boston")
    compiled = K.compile_to_device(circ, cal)
    support = np.abs(K.statevector(circ)) ** 2 > 1e-9
    direct = K.simulate_compiled(compiled, cal, 5)[support].sum()
    mirror = K.simulate_compiled(K._mirror(compiled), cal, 5)[0]
    assert math.sqrt(mirror) == pytest.approx(direct, abs=0.005)


def test_broad_output_uses_mirror_metric():
    text = HEADER + "qreg q[4];\n" + "".join(f"ry(0.7) q[{q}];\n" for q in range(4)) + \
        "cx q[0],q[1];\ncx q[2],q[3];\ncx q[1],q[2];\n" + \
        "".join(f"ry(1.1) q[{q}];\n" for q in range(4))
    r = K.success_fraction(C.parse(text), "ibm_boston")
    assert r["available"] and r["metric"] == "mirror"
    assert r["output_support_fraction"] > 0.5
    assert 0 < r["success_fraction"] < r["success_fraction_readout_mitigated"] < 1
    assert r["success_fraction_readout_mitigated"] == pytest.approx(math.sqrt(r["mirror_success"]))
    ideal = K.success_fraction(C.parse(text), _noiseless())
    assert ideal["success_fraction"] == pytest.approx(1.0)
