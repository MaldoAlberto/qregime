"""
qasm_safe_export.py -- arregla un bug real encontrado en la prueba de
generalizacion a gran escala: `qiskit.qasm2.dumps()` escribe angulos muy
pequenos en NOTACION CIENTIFICA (p.ej. "9.587379924269612e-05"), y el
parser QASM de pyzx (gramatica `lark`, ver `pyzx/symbolic.py:parse_phase_list`)
no reconoce esa notacion -- falla con
`TypeError: Phase expression '...e-05' contains unbound variables`, como
si el numero fuera una variable simbolica no ligada. Verificado
directamente: el MISMO valor, escrito en notacion decimal plana
("0.00009587379924269612"), parsea sin problema. Esto tumbo 3 de 21
circuitos (QFT n=16,20,24) en la prueba de escala grande de este
proyecto.

Este modulo no toca pyzx -- reformatea el TEXTO QASM que Qiskit produce,
reemplazando cualquier literal en notacion cientifica por su expansion
decimal equivalente, antes de que ese texto llegue a pyzx.
"""
import re

_SCI_NOTATION = re.compile(r"(?<![\w.])(\d+\.?\d*)[eE]([+-]?\d+)")


def to_plain_decimal_qasm(qasm_str: str) -> str:
    """Reemplaza cada literal numerico en notacion cientifica del texto
    QASM por su valor equivalente en notacion decimal plana, sin tocar
    nada mas del texto (nombres de compuertas, registros, etc. no
    matchean el patron)."""

    def _replace(m):
        value = float(m.group(0))
        # 20 decimales alcanza y sobra para la precision de un float64 de
        # verdad (double tiene ~15-17 digitos significativos); se recorta
        # ceros de sobra para no ensuciar el archivo sin necesidad.
        s = f"{value:.20f}".rstrip("0")
        if s.endswith("."):
            s += "0"
        return s

    return _SCI_NOTATION.sub(_replace, qasm_str)


def safe_qasm2_dumps(qc) -> str:
    """Igual que `qiskit.qasm2.dumps(qc)`, pero garantiza que el texto
    resultante sea parseable por pyzx (sin notacion cientifica)."""
    from qiskit import qasm2
    return to_plain_decimal_qasm(qasm2.dumps(qc))
