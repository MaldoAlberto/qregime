"""
Verifica NUMERICAMENTE (misma metodologia que la verificacion de RZZ por
teleportacion, mas temprano en la sesion) la identidad de "uncomputo por
medicion" para una ancilla AND (a.k.a. CCX/Toffoli), publicada por
Gidney (2018, "Halving the cost of quantum addition", Quantum 2, 74,
https://quantum-journal.org/papers/q-2018-06-18-74/) y Jones (2013,
Phys. Rev. A 87, 022328).

Afirmacion a verificar: si una ancilla t se preparo como t = a AND b via
CCX(a,b,t) (con a,b en un estado arbitrario, posiblemente en superposicion
y entrelazados con el resto del circuito), entonces:

    H(t) ; medir t en base Z -> resultado m ; si m==1: aplicar CZ(a,b)

devuelve la ancilla EXACTAMENTE a |0> y dejo (a,b) intactos -- SIN usar
ningun T-gate en el paso de "uncomputo" (H, medicion, y CZ clasicamente
controlada son todas operaciones Clifford). Esto reemplaza el Toffoli
inverso completo (que cuesta los mismos T-gates que el Toffoli directo)
por un paso de costo T-count = 0.

No se reconstruye aqui el circuito "4T" optimo de Gidney para el COMPUTO
del AND (arriesga reproducir mal un diagrama que no pude extraer del PDF
como texto) -- se usa el Toffoli estandar de Qiskit (ya verificado por la
propia libreria) para el computo, y se verifica que SOLO el paso de
uncomputo puede reemplazarse por el truco de medicion. El ahorro relativo
de T-count (uncomputo pasa de costar ~lo mismo que el computo, a costar 0)
es el mismo mecanismo que reporta el paper, incluso sin usar su circuito
optimo de computo.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, partial_trace, state_fidelity

np.random.seed(0)


def random_2q_state():
    v = np.random.randn(4) + 1j * np.random.randn(4)
    v /= np.linalg.norm(v)
    return v


def test_and_uncompute_identity(trials=20):
    all_ok = True
    for trial in range(trials):
        psi_ab = random_2q_state()

        # --- construir |a,b> (qubits 0,1) + ancilla t (qubit 2) = |0>, y
        # computar t = a AND b con un Toffoli REAL (no reinventado)
        qc = QuantumCircuit(3)
        qc.initialize(psi_ab, [0, 1])
        qc.ccx(0, 1, 2)
        state_after_compute = Statevector.from_instruction(qc)

        # estado de referencia: a,b nunca deberian cambiar por el proceso
        # de computo+uncomputo completo, asi que el "target" es psi_ab
        # tal cual, en el subespacio (a,b) trazando fuera t
        rho_ab_reference = partial_trace(state_after_compute, [2])
        # (esto de por si ya deberia dar exactamente el estado producto
        # rho(a,b) = |psi_ab><psi_ab| entrelazado con t; lo que importa es
        # el estado COMPLETO tras uncomputo, no solo la marginal)

        # --- uncomputo via medicion: H(t), medir en base Z (dos ramas),
        # aplicar CZ(a,b) clasicamente controlada si el resultado es 1
        for outcome in [0, 1]:
            qc2 = qc.copy()
            qc2.h(2)
            # proyectar la ancilla sobre |outcome> tras el Hadamard (simula
            # la medicion determinista de esa rama) y renormalizar
            sv = Statevector.from_instruction(qc2)
            proj = np.zeros((8, 8))
            for basis_idx in range(8):
                bits = [(basis_idx >> k) & 1 for k in range(3)]
                if bits[2] == outcome:  # qubit 2 = t, bit menos significativo? verificar orden
                    proj[basis_idx, basis_idx] = 1
            projected = proj @ sv.data
            norm = np.linalg.norm(projected)
            if norm < 1e-12:
                continue  # rama con probabilidad ~0, no aporta a la verificacion
            projected /= norm
            branch_state = Statevector(projected, dims=(2, 2, 2))

            # correccion clasica: la medicion deja la ancilla en |outcome>
            # de forma DETERMINISTA (esta rama fue post-seleccionada), asi
            # que el paso de "reset" real de hardware (reset clasicamente
            # controlado) se modela aplicando X a la ancilla si outcome==1
            # (para devolverla a |0>), MAS la correccion de fase real del
            # truco: CZ(a,b) si outcome==1
            qc_correction = QuantumCircuit(3)
            if outcome == 1:
                qc_correction.x(2)   # reset clasico de la ancilla a |0>
                qc_correction.cz(0, 1)  # correccion de fase real del truco
            corrected = branch_state.evolve(qc_correction)

            # la ancilla (qubit 2) deberia quedar EXACTAMENTE en |0>, y (a,b)
            # deberian quedar EXACTAMENTE en psi_ab (el estado original,
            # sin ninguna fase/entrelazamiento residual)
            rho_t = partial_trace(corrected, [0, 1])
            target_ancilla = Statevector([1, 0])
            fid_ancilla = state_fidelity(rho_t, target_ancilla)

            rho_ab = partial_trace(corrected, [2])
            target_ab = Statevector(psi_ab)
            fid_ab = state_fidelity(rho_ab, target_ab)

            ok = np.isclose(fid_ancilla, 1.0, atol=1e-8) and np.isclose(fid_ab, 1.0, atol=1e-8)
            if not ok:
                all_ok = False
                print(f"  trial {trial} outcome={outcome}: FALLA "
                      f"(fid_ancilla={fid_ancilla:.6f}, fid_ab={fid_ab:.6f})")

    return all_ok


if __name__ == "__main__":
    print("Verificando identidad de uncomputo-por-medicion de AND/Toffoli "
          f"(Gidney 2018 / Jones 2013) con {20} estados aleatorios de (a,b)...")
    ok = test_and_uncompute_identity(trials=20)
    print(f"\nResultado: {'TODOS los casos correctos (fidelidad=1 exacta)' if ok else 'FALLÓ en algún caso'}")
