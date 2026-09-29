"""
Extension de qcp_bounded_tracker.py (Parte 8 idea del usuario): en vez de
que un grupo que excede el presupuesto (n_pcm) se marque simplemente como
"desconocido" y se abandone su rastreo, hacer lo que RQAOA hace con un
problema de optimizacion -- FIJAR la variable (aqui: el qubit) mas
confiable dentro del grupo (la de mayor |P(0)-P(1)|, el analogo directo de
"mayor |<Z_i>|" en RQAOA), proyectar el grupo sobre ese resultado, reducir
su tamano en 1, y REINTENTAR encajarlo en el presupuesto -- contando
cuantas de estas "rondas de fijacion" hicieron falta en total, como el
numero de rondas recursivas de RQAOA.

Esto convierte "abandonar el rastreo" (una decision binaria: cabe o no
cabe) en "cuantos bloques temporales de medicion/fijacion hacen falta"
(una cantidad continua, medible, y comparable entre familias de circuito
-- la pregunta literal que planteo el usuario).

Limitacion declarada: fijar un qubit a su valor mas probable, cuando el
grupo NO es determinista, introduce error real (la proyeccion no es exacta
si hay superposicion genuina) -- esto es exactamente lo que RQAOA tambien
acepta (fija por REDONDEO del signo, no porque el problema sea
determinista). Se reporta tanto el numero de rondas como el error de
fidelidad acumulado por las fijaciones, para no esconder ese costo.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pyzx as zx

from qcp_bounded_tracker import UnionGroup, gate_matrix, gate_qubits


class RQAOAStyleTracker:
    def __init__(self, num_qubits, n_pcm=4):
        self.n_pcm = n_pcm
        self.groups = {q: UnionGroup([q]) for q in range(num_qubits)}
        self.rounds = 0             # cuantas fijaciones RQAOA-style ocurrieron
        self.fidelity_loss_log = [] # que tan "de golpe" fue cada fijacion

    def _marginal_prob0(self, group: UnionGroup, q):
        if not group.known:
            return None
        i = group.index_of(q)
        n = len(group.qubits)
        state = group.state.reshape([2] * n)
        idx0 = [slice(None)] * n
        idx0[i] = 0
        p0 = np.sum(np.abs(state[tuple(idx0)]) ** 2)
        return p0

    def _fix_most_confident_qubit(self, group: UnionGroup, protected=()):
        """
        Encuentra el qubit del grupo con marginal MAS cercana a 0 o 1,
        lo proyecta a ese resultado (fijandolo, al estilo RQAOA: redondear
        el signo mas probable), y devuelve un NUEVO grupo mas pequeno (sin
        ese qubit) mas el registro de cuanta "perdida de fidelidad" implico
        (0.0 si el qubit ya era determinista de verdad -- fijacion gratis).

        IMPORTANTE (bug corregido): el qubit fijado no desaparece del
        sistema -- se convierte en su PROPIO grupo singleton determinista
        (con el valor fijado), y self.groups debe apuntar ahi, no quedar
        colgando hacia el grupo viejo (eso causaba un ValueError real al
        intentar aplicar una compuerta sobre ese qubit despues).
        """
        candidates = [q for q in group.qubits if q not in protected]
        if len(candidates) == 0 or not group.known:
            return group, 0.0  # no hay ningun qubit "de sobra" que fijar sin tocar a/b

        best_q, best_conf, best_p0 = None, -1.0, 0.5
        for q in candidates:
            p0 = self._marginal_prob0(group, q)
            conf = abs(p0 - 0.5)
            if conf > best_conf:
                best_q, best_conf, best_p0 = q, conf, p0

        outcome = 0 if best_p0 >= 0.5 else 1
        fidelity_loss = 1.0 - (best_p0 if outcome == 0 else 1 - best_p0)

        i = group.index_of(best_q)
        n = len(group.qubits)
        state = group.state.reshape([2] * n)
        idx = [slice(None)] * n
        idx[i] = outcome
        projected = state[tuple(idx)]
        norm = np.linalg.norm(projected)
        if norm > 1e-12:
            projected = projected / norm

        # el qubit fijado se vuelve su propio grupo singleton determinista
        fixed_group = UnionGroup([best_q])
        fixed_group.state = np.array([1.0, 0.0]) if outcome == 0 else np.array([0.0, 1.0])
        fixed_group.known = True
        self.groups[best_q] = fixed_group

        remaining_qubits = [x for x in group.qubits if x != best_q]
        new_group = UnionGroup(remaining_qubits) if remaining_qubits else None
        if new_group is not None:
            new_group.state = projected.reshape(-1)
            new_group.known = True
            for q in remaining_qubits:
                self.groups[q] = new_group

        self.rounds += 1
        self.fidelity_loss_log.append(fidelity_loss)
        return new_group, fidelity_loss

    def _merge(self, a, b):
        ga, gb = self.groups[a], self.groups[b]
        if ga is gb:
            return ga
        merged_qubits = ga.qubits + [q for q in gb.qubits if q not in ga.qubits]

        # mientras el grupo combinado exceda el presupuesto, fijar (RQAOA-
        # style) el qubit mas confiable del grupo MAS GRANDE de los dos
        # (nunca a o b -- los necesitamos intactos para aplicar la
        # compuerta de 2 qubits al final), y reintentar
        stalled = False
        while len(merged_qubits) > self.n_pcm and not stalled:
            bigger = ga if len(ga.qubits) >= len(gb.qubits) else gb
            protected = {a, b}
            candidates = [q for q in bigger.qubits if q not in protected]
            if len(candidates) == 0:
                stalled = True  # el grupo grande ya es solo {a} o {b} -- no hay margen
                break
            new_bigger, _ = self._fix_most_confident_qubit(bigger, protected=protected)
            if bigger is ga:
                ga = new_bigger if new_bigger is not None else self.groups[a]
            else:
                gb = new_bigger if new_bigger is not None else self.groups[b]
            merged_qubits = ga.qubits + [q for q in gb.qubits if q not in ga.qubits]

        if not (ga.known and gb.known) or len(merged_qubits) > self.n_pcm:
            new_group = UnionGroup(merged_qubits)
            new_group.known = False
        else:
            new_group = UnionGroup(merged_qubits)
            new_group.state = np.kron(ga.state, gb.state) if ga.qubits and gb.qubits else \
                (ga.state if ga.qubits else gb.state)
        for q in merged_qubits:
            self.groups[q] = new_group
        return new_group

    def apply_gate(self, gate_type, qubits, phase=None, adjoint=False):
        if len(qubits) == 1:
            q = qubits[0]
            g = self.groups[q]
            if g.known:
                g.apply_1q(q, gate_matrix(gate_type, phase, adjoint))
        elif len(qubits) == 2:
            a, b = qubits
            kind = "cz" if gate_type in ("CZ",) else "cnot"
            g = self._merge(a, b)
            if g.known and g.qubits:
                g.apply_2q(a, b, kind=kind)
        else:
            for q in qubits:
                self.groups[q].known = False


def run_rqaoa_style(qasm_path, n_pcm=4, superpose_inputs=True):
    circuit = zx.Circuit.from_qasm_file(qasm_path).to_basic_gates()
    n = circuit.qubits
    tracker = RQAOAStyleTracker(n, n_pcm=n_pcm)

    if superpose_inputs:
        for q in range(n):
            tracker.apply_gate("HAD", [q])
            tracker.apply_gate("S", [q])

    for gate in circuit.gates:
        name = type(gate).__name__
        qs = gate_qubits(gate)
        phase = getattr(gate, "phase", None)
        adjoint = getattr(gate, "adjoint", False)
        tracker.apply_gate(name, qs, phase=phase, adjoint=adjoint)

    avg_fid_loss = np.mean(tracker.fidelity_loss_log) if tracker.fidelity_loss_log else 0.0
    return {
        "num_qubits": n,
        "rondas_rqaoa_style": tracker.rounds,
        "perdida_fidelidad_promedio_por_ronda": avg_fid_loss,
        "perdida_fidelidad_total_aprox": sum(tracker.fidelity_loss_log),
    }


if __name__ == "__main__":
    circuitos = {
        "QFT10": "qft10.qasm",
        "Sumador8": "adder8.qasm",
        "QAOA/farm": "qaoa_farm.qasm",
        "random8": "random_circuit8.qasm",
    }
    for n_pcm in [4, 6, 8]:
        print(f"\n=== n_pcm = {n_pcm} ===")
        for name, path in circuitos.items():
            r = run_rqaoa_style(path, n_pcm=n_pcm)
            print(f"  {name:12s} n_qubits={r['num_qubits']:3d}  "
                  f"rondas_RQAOA-style={r['rondas_rqaoa_style']:3d}  "
                  f"perdida_fidelidad_prom/ronda={r['perdida_fidelidad_promedio_por_ronda']:.4f}  "
                  f"perdida_total_aprox={r['perdida_fidelidad_total_aprox']:.3f}")
