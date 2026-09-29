"""qregime -- QEM, QEC or FTQC? A validated decision layer for quantum circuits.

Give it an OpenQASM 2 circuit and a hardware profile; it tells you which
circuit-reduction techniques are worth trying, what full fault tolerance
would cost, the best reliability reachable under a hard qubit budget, and
whether error mitigation, partial error correction or full fault tolerance
is the cheapest regime that meets your reliability target -- or, if none
does, how large the gap is.

See https://github.com/MaldoAlberto/qregime for the tutorial and the
accompanying paper.
"""
from .api import (
    HARDWARE_PROFILES,
    MAX_QUBITS_FOR_SIMULATION,
    HardwareProfile,
    analyze,
    best_within_budget,
    decide_regime,
    estimate_ftqc,
    profile,
    recommend,
)
from .qasm_export import safe_qasm2_dumps, to_plain_decimal_qasm

__version__ = "0.1.0"

__all__ = [
    "profile", "recommend", "estimate_ftqc", "best_within_budget",
    "decide_regime", "analyze", "HARDWARE_PROFILES", "HardwareProfile",
    "MAX_QUBITS_FOR_SIMULATION", "safe_qasm2_dumps", "to_plain_decimal_qasm",
    "__version__",
]
