"""
Version 100% propia (sin depender de la nube de IBM Quantum Runtime) de dos
tecnicas de mitigacion/supresion de error que el notebook de QGSS 2025
(lab-2) aplica mediante `EstimatorOptions.resilience`/`twirling` en
`qiskit_ibm_runtime` -- pero esas opciones SOLO tienen efecto contra un
backend REAL en la nube (ver hallazgo mas abajo). Aqui se implementan y
verifican de forma local, con simulacion de ruido real (AerSimulator a
partir de las calibraciones de FakeTorino), sin ningun token ni servicio
de IBM:

  1. Pauli twirling de compuertas de 2 qubits: para cada compuerta Clifford
     de 2 qubits M del circuito, se inserta un Pauli aleatorio P antes y se
     calcula (por conjugacion exacta, no por tabla copiada) el Pauli de
     correccion Q = M P M^dagger tal que Q * M * P == M (hasta fase global).
     Promediado sobre muchas repeticiones con P aleatorio, esto convierte
     ruido coherente en ruido de Pauli estocastico -- mismo principio que
     `options.twirling.enable_gates` del notebook, pero calculado y
     verificado aqui mismo, no delegado a un servicio.
  2. Dynamical decoupling: se usa el pass real de Qiskit
     `PadDynamicalDecoupling` (pass LOCAL del transpilador, no un servicio
     de IBM Runtime) con las duraciones de compuerta reales del backend.

HALLAZGO HONESTO (probado, no asumido): las opciones de resiliencia de
`qiskit_ibm_runtime` (`resilience.zne_mitigation`, `resilience.measure_
mitigation`, `twirling.enable_gates`, etc.) se aceptan sin error contra un
FakeBackend en modo de prueba local, pero Qiskit emite una advertencia
explicita -- "Options ... have no effect in local testing mode" -- y el
resultado devuelto es identico al caso sin mitigacion. Es decir: el stack
de produccion de IBM Runtime que usa el notebook SOLO puede validarse
contra hardware real en la nube con credenciales; no puede probarse ni
reproducirse en este entorno. Esto refuerza, con una prueba directa, la
razon por la que este proyecto construyo su propio estimador y ahora su
propio stack de EM: no hay forma de verificar de manera independiente
(sin depender de un vendor) el efecto de esas opciones sin acceso real a
QPU. Lo que sigue es la alternativa 100% local y verificable.

SEGUNDO HALLAZGO HONESTO (mas importante que el primero, y mas
inesperado): al correr la version local (dynamical decoupling + Pauli
twirling reales, no delegados a un vendor) sobre `grover.qasm`, el
RESULTADO SIN NINGUNA mitigacion fue el MEJOR de los tres (menor TVD
frente a la distribucion ideal); dynamical decoupling lo empeoro, y
twirling+DD lo empeoro aun mas. Esto no es un error de implementacion --
se investigo la causa raiz y es estructural, verificada directamente:

  - `AerSimulator.from_backend(...)` construye su modelo de ruido con
    `NoiseModel.from_backend`, que NO incluye ninguna instruccion
    "delay" en `noise_model.noise_instructions` -- es decir, el tiempo
    ocioso de un qubit no tiene NINGUN error asociado en este modelo de
    simulacion. Dynamical decoupling existe precisamente para suprimir
    decoherencia durante el tiempo ocioso; si el simulador no penaliza
    el tiempo ocioso, DD no puede mostrar ningun beneficio -- solo puede
    empeorar el resultado, porque cada pulso de DD insertado es una
    compuerta X real con su propio error de compuerta, sin ningun costo
    de decoherencia que esa misma compuerta este evitando en este modelo.
  - El ruido que `NoiseModel.from_backend` deriva de las calibraciones
    del backend ya se construye, por definicion, como canales de
    despolarizacion/Pauli (no como errores coherentes/unitarios). Pauli
    twirling sirve para convertir ruido COHERENTE en ruido de Pauli
    estocastico -- pero si el ruido ya es de Pauli por construccion, no
    hay nada que convertir, y las compuertas extra que inserta el
    twirling solo suman error de compuerta neto, sin ningun beneficio
    posible en este modelo.

En otras palabras: el mismo entorno 100% local, sin vendor, que sí pudo
validar honestamente ETAPA 0 (busqueda de layout/ruteo, que optimiza
error de compuerta y conteo de SWAPs -- algo que el modelo de ruido SI
penaliza) es estructuralmente incapaz de validar el beneficio de DD o de
twirling, porque ambas tecnicas actuan sobre fisica (decoherencia
temporal, ruido coherente) que este modelo de ruido no representa. Es la
misma limitacion de fondo que el primer hallazgo (las opciones de
`qiskit_ibm_runtime` sin efecto en modo local): NINGUNA simulacion local
basada en calibraciones de backend puede validar estas dos tecnicas de EM
especificas -- se necesita hardware real. Se reporta esto explicitamente
en vez de presentar una mejora que no se puede sostener con evidencia.
"""
import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator, Statevector
from qiskit.transpiler.passes import PadDynamicalDecoupling, ALAPScheduleAnalysis
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.circuit.library import XGate


PAULI_LABELS = ["I", "X", "Y", "Z"]


def _pauli_conjugation_table(gate_matrix: np.ndarray, num_qubits: int = 2):
    """Para cada Pauli P (de las 4**num_qubits posibilidades), calcula
    EXACTAMENTE (por conjugacion de matrices, no por tabla copiada de un
    paper) el Pauli de correccion Q = M P M^dagger, verificando que en
    efecto Q*M*P == M hasta fase global (si no, levanta un error -- esto
    solo es valido para compuertas Clifford, y se verifica en tiempo real
    en vez de asumirse)."""
    table = {}
    pauli_ops = {"".join(c): Operator(Pauli("".join(c))).data
                 for c in itertools.product(PAULI_LABELS, repeat=num_qubits)}
    M = gate_matrix
    Md = gate_matrix.conj().T
    for p_label, P in pauli_ops.items():
        conjugated = M @ P @ Md
        best_label, best_overlap = None, 0.0
        for q_label, Q in pauli_ops.items():
            overlap = np.trace(Q.conj().T @ conjugated) / (2 ** num_qubits)
            if abs(overlap) > abs(best_overlap):
                best_overlap, best_label = overlap, q_label
        # verificacion real: Q * M * P debe ser proporcional a M
        Q = pauli_ops[best_label]
        check = Q @ M @ P
        phase = check[np.unravel_index(np.argmax(np.abs(M)), M.shape)] / \
            M[np.unravel_index(np.argmax(np.abs(M)), M.shape)]
        if not np.allclose(check, phase * M, atol=1e-8):
            raise ValueError(f"twirling invalido para pauli {p_label}: "
                              f"la compuerta no es Clifford exacta o la conjugacion fallo")
        table[p_label] = best_label
    return table


def pauli_twirl_circuit(circuit: QuantumCircuit, two_qubit_gate_name: str,
                          two_qubit_gate_matrix: np.ndarray, seed: int = 0) -> QuantumCircuit:
    """Devuelve una copia del circuito con cada instancia de la compuerta
    de 2 qubits indicada envuelta en un par de Paulis (P antes, Q despues)
    elegido al azar por instancia, verificado exactamente para que el
    circuito ideal (sin ruido) sea identico al original."""
    table = _pauli_conjugation_table(two_qubit_gate_matrix, num_qubits=2)
    rng = np.random.default_rng(seed)
    # copy_empty_like (en vez de construir un QuantumCircuit nuevo) preserva
    # los metadatos de "circuito fisico" (registro qreg, layout) que los
    # passes de scheduling/DD del transpilador exigen para poder correr.
    twirled = circuit.copy_empty_like()

    def _apply_pauli(qc, qubit, letter):
        # Se insertan solo compuertas NATIVAS del backend (rz, x) para no
        # tener que retranspilar despues (lo que rompia la anotacion de
        # layout fisico que necesita el pass de dynamical decoupling). Y
        # se implementa como rz(pi)+x, hasta una fase global (Y = i*X*Z)
        # que no afecta ninguna probabilidad medida.
        if letter == "I":
            return
        if letter == "Z":
            qc.rz(np.pi, qubit)
        elif letter == "X":
            qc.x(qubit)
        elif letter == "Y":
            qc.rz(np.pi, qubit)
            qc.x(qubit)

    for instruction in circuit.data:
        if instruction.operation.name == two_qubit_gate_name:
            q0, q1 = instruction.qubits
            p_label = "".join(rng.choice(PAULI_LABELS, size=2))
            q_label = table[p_label]
            for qubit, letter in zip((q0, q1), p_label):
                _apply_pauli(twirled, qubit, letter)
            twirled.append(instruction.operation, instruction.qubits, instruction.clbits)
            for qubit, letter in zip((q0, q1), q_label):
                _apply_pauli(twirled, qubit, letter)
        else:
            twirled.append(instruction.operation, instruction.qubits, instruction.clbits)
    return twirled


def apply_dynamical_decoupling(circuit: QuantumCircuit, backend) -> QuantumCircuit:
    """Inserta secuencias XX en los qubits ociosos usando el pass real
    del transpilador de Qiskit (`PadDynamicalDecoupling`), con las
    duraciones reales de compuerta del backend -- esto es un pass LOCAL,
    no requiere ningun servicio de IBM."""
    durations = InstructionDurations.from_backend(backend)
    dd_sequence = [XGate(), XGate()]
    pm = PassManager([ALAPScheduleAnalysis(durations),
                       PadDynamicalDecoupling(durations, dd_sequence)])
    return pm.run(circuit)


def total_variation_distance(counts_a: dict, counts_b: dict) -> float:
    keys = set(counts_a) | set(counts_b)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    return 0.5 * sum(abs(counts_a.get(k, 0) / total_a - counts_b.get(k, 0) / total_b) for k in keys)


if __name__ == "__main__":
    import sys
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    qasm_path = sys.argv[1] if len(sys.argv) > 1 else "grover.qasm"
    backend = FakeTorino()
    seed = 42
    SHOTS = 10000

    qc = QuantumCircuit.from_qasm_file(qasm_path)
    qc_no_measure = qc.remove_final_measurements(inplace=False)

    print(f"Circuito: {qasm_path} ({qc.num_qubits} qubits)")
    print("1. Distribucion ideal (sin ruido, statevector)...")
    ideal_sv = Statevector(qc_no_measure)
    ideal_probs = ideal_sv.probabilities_dict()
    ideal_counts = {k: int(round(v * SHOTS)) for k, v in ideal_probs.items()}

    print("2. Transpilando al mejor layout encontrado (reusa layout_noise_search)...")
    from layout_noise_search import find_best_layout
    best = find_best_layout(qasm_path, backend, n_trials=10, verbose=False)
    transpiled = best["transpiled_circuit"]
    if transpiled.num_clbits == 0:
        # Medimos SOLO los qubits fisicos que sostienen a cada qubit
        # LOGICO al final del ruteo (final_index_layout), en ese orden --
        # asi (a) los bitstrings resultantes son directamente comparables
        # con la distribucion ideal (misma convencion de orden de qubits
        # logicos) y (b) al no hacer measure_all sobre las 133 posiciones
        # del registro completo de FakeTorino, AerSimulator puede truncar
        # los qubits ociosos y la simulacion con ruido es tratable en
        # memoria.
        final_physical = transpiled.layout.final_index_layout()
        transpiled = transpiled.copy()
        from qiskit import ClassicalRegister
        creg = ClassicalRegister(len(final_physical), "meas")
        transpiled.add_register(creg)
        for i, phys_q in enumerate(final_physical):
            transpiled.measure(phys_q, creg[i])

    print("3. Construyendo modelo de ruido REAL de FakeTorino y simulando (Aer)...")
    # AerSimulator.from_backend trunca automaticamente los qubits fisicos
    # ociosos del registro completo de 133 qubits -- necesario para que la
    # simulacion con ruido sea tratable en memoria (igual que en el
    # notebook de referencia, cell 45).
    sim = AerSimulator.from_backend(backend, seed_simulator=seed)

    job_base = sim.run(transpiled, shots=SHOTS)
    counts_base = job_base.result().get_counts()
    tvd_base = total_variation_distance(ideal_counts, counts_base)
    print(f"   TVD sin mitigacion: {tvd_base:.4f}")

    print("4. Aplicando dynamical decoupling (pass local, sin IBM Runtime)...")
    transpiled_dd = apply_dynamical_decoupling(transpiled, backend)
    job_dd = sim.run(transpiled_dd, shots=SHOTS)
    counts_dd = job_dd.result().get_counts()
    tvd_dd = total_variation_distance(ideal_counts, counts_dd)
    print(f"   TVD con dynamical decoupling: {tvd_dd:.4f}")

    print("5. Aplicando Pauli twirling (verificado por conjugacion exacta) + DD...")
    two_qubit_gate_name = "ecr" if "ecr" in backend.configuration().basis_gates else "cz"
    from qiskit.circuit.library import ECRGate, CZGate
    gate_matrix = Operator(ECRGate() if two_qubit_gate_name == "ecr" else CZGate()).data

    twirled_counts_accum = {}
    n_random_instances = 8
    for i in range(n_random_instances):
        twirled = pauli_twirl_circuit(transpiled, two_qubit_gate_name, gate_matrix, seed=seed + i)
        twirled_dd = apply_dynamical_decoupling(twirled, backend)
        job_t = sim.run(twirled_dd, shots=SHOTS // n_random_instances)
        for k, v in job_t.result().get_counts().items():
            twirled_counts_accum[k] = twirled_counts_accum.get(k, 0) + v
    tvd_twirl = total_variation_distance(ideal_counts, twirled_counts_accum)
    print(f"   TVD con twirling+DD ({n_random_instances} instancias aleatorias): {tvd_twirl:.4f}")

    print(f"\nResumen: TVD baseline={tvd_base:.4f}  DD={tvd_dd:.4f}  twirling+DD={tvd_twirl:.4f}")
    best_name, best_tvd = min([("baseline", tvd_base), ("DD", tvd_dd),
                                 ("twirling+DD", tvd_twirl)], key=lambda x: x[1])
    print(f"Mejor resultado: {best_name} (menor TVD = mas cercano a la distribucion ideal)")
