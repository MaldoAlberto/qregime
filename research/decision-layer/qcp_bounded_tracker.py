"""
Version acotada (no el framework completo) de "Quantum Constant Propagation"
(QCP), la tecnica publicada por Chen, Fulginiti, Mendl (ICCS 2025,
"Optimization Framework for Reducing Mid-circuit Measurements and Resets")
para detectar, recorriendo el DAG del circuito en orden, en que puntos el
estado de un qubit (o de un grupo pequeno de qubits entrelazados) sigue
siendo CONOCIDO en tiempo de "compilacion" -- lo cual habilita reemplazar
mediciones/resets, o (la direccion que nos interesa aqui) insertar una
medicion en vez de seguir aplicando compuertas, siempre que el resultado
de esa medicion ya este determinado o acotado.

Lo que se implementa (una version simplificada, no el algoritmo completo
del paper):
  - Se mantiene una particion de los qubits en grupos (union-find), igual
    que la "union-table" que describe el paper.
  - Cada grupo de tamano <= n_pcm se simula EXACTO con un vector de estado
    denso (cualquier compuerta, no solo Clifford -- a diferencia de un
    tracker de estabilizadores puro, esto SI puede seguir T-gates dentro
    de un grupo pequeno).
  - Cuando una compuerta de 2 qubits fusiona dos grupos y el grupo
    resultante excede n_pcm, se marca TODO el grupo como desconocido
    (throw-away, equivalente al simbolo "T-arriba" (top) del paper) y deja
    de simularse -- el mismo criterio de "abandonar el rastreo" que usa
    QCP para mantenerse polinomial, solo que aqui el limite de tamano es
    el UNICO mecanismo de abandono (QCP real tiene ademas heuristicas
    adicionales que no reproducimos).

Leccion metodologica de esta misma sesion que se aplica aqui a proposito:
probar un circuito aritmetico (sumador, QAOA) con la entrada trivial
|0...0> da un resultado degenerado (todo se mantiene "conocido" todo el
tiempo, porque un circuito reversible clasico sobre una entrada clasica
nunca genera superposicion real) -- el mismo error que ya cometimos una
vez con bond dimension en AQC-Tensor. Por eso aqui se aplica una capa de
Hadamards sobre los registros de entrada ANTES de correr el circuito, para
que la pregunta sea informativa (que tanto puede rastrearse un circuito
aritmetico usado dentro de un algoritmo mas grande, donde sus entradas SI
estan en superposicion -- el caso real de uso, p.ej. dentro de Shor).
"""

import numpy as np
import pyzx as zx


def gate_matrix(gate_type: str, phase=None, adjoint=False):
    """Matriz unitaria de 1 qubit para los tipos de compuerta relevantes."""
    if gate_type == "HAD":
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    if gate_type in ("XPhase", "NOT"):
        return np.array([[0, 1], [1, 0]])
    if gate_type == "ZPhase" or gate_type == "T" or gate_type == "S" or gate_type == "Z":
        defaults = {"S": 0.5, "Z": 1.0, "T": 0.25}
        p = float(phase) if phase is not None else defaults.get(gate_type, 0.5)
        if adjoint:
            p = -p
        return np.array([[1, 0], [0, np.exp(1j * np.pi * p)]])
    raise ValueError(f"tipo de compuerta no soportado: {gate_type}")


class UnionGroup:
    """Grupo de qubits entrelazados rastreado como vector de estado denso.

    BUG REAL ENCONTRADO Y CORREGIDO (esta sesion, tras el hallazgo de que
    el pipeline no escala mas alla de ~20 qubits): el constructor asignaba
    un vector de estado de tamano 2**len(qubits) SIN CONDICION, incluso
    para un grupo que se va a marcar `known=False` de inmediato (el caso
    "abandonar el rastreo" que es justamente el mecanismo que deberia
    mantener esto barato). Un circuito con suficiente entrelazamiento real
    (cualquier circuito no trivial de mas de un puñado de qubits) fusiona
    grupos "desconocidos" repetidamente via `_merge`, y cada fusion volvia
    a alocar un arreglo de `complex128` de tamano EXPONENCIAL en el numero
    de qubits acumulados -- para un grupo de 22 qubits eso son 2**22 * 16
    bytes = 64 MB de una sola vez, y crece a cientos de GB mucho antes de
    llegar a 30-32 qubits -- exactamente el rango de circuitos que colgo
    el pipeline en la prueba de generalizacion a gran escala. El estado
    denso NUNCA se lee ni se escribe para un grupo con `known=False`
    (`apply_1q`/`apply_2q`/`is_deterministic` retornan temprano antes de
    tocar `self.state`), asi que no hay ninguna razon para reservarlo.
    """
    def __init__(self, qubits, known=True):
        self.qubits = list(qubits)  # orden fijo -> posicion en el tensor
        self.known = known
        if known:
            dim = 2 ** len(self.qubits)
            self.state = np.zeros(dim, dtype=complex)
            self.state[0] = 1.0  # |0...0>
        else:
            # sin estado -- se evita la asignacion exponencial para grupos
            # ya descartados, que es la mayoria de los grupos grandes en
            # cualquier circuito con entrelazamiento real
            self.state = None

    def index_of(self, q):
        return self.qubits.index(q)

    def apply_1q(self, q, mat):
        if not self.known:
            return
        i = self.index_of(q)
        n = len(self.qubits)
        state = self.state.reshape([2] * n)
        state = np.tensordot(mat, state, axes=([1], [i]))
        state = np.moveaxis(state, 0, i)
        self.state = state.reshape(-1)

    def apply_2q(self, control, target, kind="cnot"):
        if not self.known:
            return
        ci, ti = self.index_of(control), self.index_of(target)
        n = len(self.qubits)
        state = self.state.reshape([2] * n)
        # construir el operador completo es caro para n grande, pero n_pcm
        # es pequeno por diseno (esa es la garantia de costo acotado)
        out = state.copy()
        idx = [slice(None)] * n
        if kind == "cnot":
            idx_c1 = idx.copy(); idx_c1[ci] = 1
            sl = tuple(idx_c1)
            sub = out[sl]
            out[sl] = np.take(sub, [1, 0], axis=ti if ti < ci else ti - 1)
        elif kind == "cz":
            idx_c1 = idx.copy(); idx_c1[ci] = 1; idx_c1[ti] = 1
            out[tuple(idx_c1)] *= -1
        self.state = out.reshape(-1)

    def is_deterministic(self):
        """True si el grupo esta en un unico estado computacional (prob=1)."""
        if not self.known:
            return False
        probs = np.abs(self.state) ** 2
        return np.isclose(probs.max(), 1.0, atol=1e-9)


class BoundedQCPTracker:
    def __init__(self, num_qubits, n_pcm=4):
        self.n_pcm = n_pcm
        self.groups = {q: UnionGroup([q]) for q in range(num_qubits)}

    def _merge(self, a, b):
        ga, gb = self.groups[a], self.groups[b]
        if ga is gb:
            return ga
        merged_qubits = ga.qubits + [q for q in gb.qubits if q not in ga.qubits]
        if not (ga.known and gb.known) or len(merged_qubits) > self.n_pcm:
            # CORREGIDO: known=False desde la construccion -- no se reserva
            # el vector de estado de tamano 2**len(merged_qubits) que de
            # todas formas nunca se iba a usar (ver docstring de UnionGroup)
            new_group = UnionGroup(merged_qubits, known=False)
        else:
            new_group = UnionGroup(merged_qubits, known=True)
            # producto tensorial de los dos estados conocidos, en el orden
            # merged_qubits (ga primero, luego los nuevos de gb)
            new_group.state = np.kron(ga.state, gb.state)
        for q in merged_qubits:
            self.groups[q] = new_group
        return new_group

    def _classical_control_value(self, q):
        """
        Si q esta en un grupo SINGLETON y su estado es exactamente |0> o |1>
        (control clasico determinista), devuelve 0 o 1. Si no (superposicion,
        entrelazado, o desconocido), devuelve None -- este es el caso central
        de "constant propagation": una compuerta de 2 qubits con un control
        clasico NO entrelaza, y no hace falta fusionar los grupos para
        seguir rastreandola (justo lo que permite a QCP, y a nuestra version
        acotada, no inflar grupos innecesariamente con CNOTs de control
        clasico -- que es exactamente el patron de un circuito reversible
        aritmetico operando sobre bits clasicos).
        """
        g = self.groups[q]
        if not g.known or len(g.qubits) != 1:
            return None
        probs = np.abs(g.state) ** 2
        if np.isclose(probs[0], 1.0, atol=1e-9):
            return 0
        if np.isclose(probs[1], 1.0, atol=1e-9):
            return 1
        return None

    def apply_gate(self, gate_type, qubits, phase=None, adjoint=False):
        if len(qubits) == 1:
            q = qubits[0]
            g = self.groups[q]
            if g.known:
                g.apply_1q(q, gate_matrix(gate_type, phase, adjoint))
        elif len(qubits) == 2:
            a, b = qubits
            kind = "cz" if gate_type in ("CZ",) else "cnot"

            # atajo de propagacion de constantes: control clasico determinista
            # -> no fusionar grupos, solo propagar el efecto clasico/de fase
            ctrl_val = self._classical_control_value(a)
            if ctrl_val is not None:
                if ctrl_val == 1:
                    gb = self.groups[b]
                    if gb.known:
                        mat = gate_matrix("XPhase") if kind == "cnot" else gate_matrix("Z")
                        gb.apply_1q(b, mat)
                return
            ctrl_val_b = self._classical_control_value(b) if kind == "cz" else None
            if kind == "cz" and ctrl_val_b is not None:
                if ctrl_val_b == 1:
                    ga = self.groups[a]
                    if ga.known:
                        ga.apply_1q(a, gate_matrix("Z"))
                return

            g = self._merge(a, b)
            if g.known:
                g.apply_2q(a, b, kind=kind)
        else:
            # gates de 3+ qubits (Toffoli ya decompuesto no deberian aparecer
            # aqui, pero por seguridad: marcar desconocido)
            for q in qubits:
                self.groups[q].known = False

    def known_fraction(self):
        seen = set()
        known_qubits = 0
        total = 0
        for q, g in self.groups.items():
            total += 1
            if id(g) in seen:
                continue
            seen.add(id(g))
        for q, g in self.groups.items():
            if g.known:
                known_qubits += 1
        return known_qubits / total

    def deterministic_fraction(self):
        seen_groups = {}
        for q, g in self.groups.items():
            seen_groups[id(g)] = g
        det_qubits = 0
        for q, g in self.groups.items():
            if g.known and g.is_deterministic():
                det_qubits += 1
        return det_qubits / len(self.groups)


def gate_qubits(gate):
    qs = []
    for attr in ("target", "control", "ctrl1", "ctrl2"):
        if hasattr(gate, attr):
            qs.append(getattr(gate, attr))
    return sorted(set(qs))


def run_tracker(qasm_path, n_pcm=4, superpose_inputs=True, circuit=None):
    """BUG REAL ENCONTRADO Y CORREGIDO (esta sesion): esta funcion siempre
    volvia a parsear el archivo QASM desde cero con
    `zx.Circuit.from_qasm_file`, incluso cuando el llamador
    (`technique_advisor.profile_circuit`) YA habia parseado el mismo
    archivo unas lineas antes. Para el circuito insignia de este proyecto
    (miles de compuertas), el parser de pyzx (basado en una gramatica
    `lark`, re-invocada por cada expresion de fase) es el costo dominante
    de todo el pipeline -- pagarlo DOS VECES por la misma llamada a
    `profile_circuit()` era puro desperdicio. Ahora se acepta un circuito
    ya parseado opcionalmente; si no se da uno, se preserva el
    comportamiento anterior (parsear desde `qasm_path`) para no romper
    a quien llame a `run_tracker()` de forma independiente."""
    if circuit is None:
        circuit = zx.Circuit.from_qasm_file(qasm_path).to_basic_gates()
    n = circuit.qubits
    tracker = BoundedQCPTracker(n, n_pcm=n_pcm)

    if superpose_inputs:
        # aplicar H seguido de S (preparar |+i>) a todos los qubits de
        # entrada ANTES del circuito real -- NO solo H, porque varios de
        # estos circuitos (QAOA) ya empiezan con su propia capa de H, y
        # una H extra se cancelaria exactamente (H*H=identidad), reproduciendo
        # el mismo caso degenerado de entrada clasica que ya nos dio un
        # resultado no informativo con bond dimension en AQC-Tensor. H+S
        # prepara un estado de un qubit que ninguna H posterior devuelve
        # trivialmente a |0>, así que la superposicion de entrada es
        # genuina sea cual sea la primera compuerta del circuito real.
        for q in range(n):
            tracker.apply_gate("HAD", [q])
            tracker.apply_gate("S", [q])

    for gate in circuit.gates:
        name = type(gate).__name__
        qs = gate_qubits(gate)
        phase = getattr(gate, "phase", None)
        adjoint = getattr(gate, "adjoint", False)
        tracker.apply_gate(name, qs, phase=phase, adjoint=adjoint)

    return {
        "num_qubits": n,
        "known_fraction": tracker.known_fraction(),
        "deterministic_fraction": tracker.deterministic_fraction(),
    }


if __name__ == "__main__":
    import sys

    circuits = {
        "QFT10": "qft10.qasm",
        "Sumador8": "adder8.qasm",
        "QAOA/farm": "qaoa_farm.qasm",
    }

    for n_pcm in [1, 2, 3, 4]:
        print(f"\n=== n_pcm = {n_pcm} ===")
        for name, path in circuits.items():
            r = run_tracker(path, n_pcm=n_pcm, superpose_inputs=True)
            print(f"  {name:12s}  n_qubits={r['num_qubits']:3d}  "
                  f"fraccion_conocida_al_final={r['known_fraction']*100:5.1f}%  "
                  f"fraccion_determinista={r['deterministic_fraction']*100:5.1f}%")
