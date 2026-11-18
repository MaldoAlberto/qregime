"""Fast OpenQASM loading for the analysis layers.

pyzx's QASM parser runs a full ``lark`` grammar on every gate parameter, twice
(once to split the parameter list, once per angle). On circuits with many
arbitrary rotations that is ~87% of ``qregime.profile``'s run time (see
research/comparison/PERF_STRESS.md). This module keeps pyzx's parser and its
exact semantics, and adds two shortcuts in front of it:

1. A parameter list with no nested parentheses is split on commas directly,
   instead of through the grammar.
2. Each angle string is converted once and cached. A plain decimal literal
   (``0.47``, ``-3``) is converted with the same two rounding steps pyzx
   applies, so the resulting ``Fraction`` is identical. Anything else
   (``pi/4``, ``2*pi/3``, expressions) goes through pyzx unchanged. That
   includes scientific notation (``1.5e-3``), which pyzx rejects; it is
   still rejected, with pyzx's own error.

Everything the shortcuts do not cover (OpenQASM 3, gate bodies with bound
parameters, nested parentheses) falls back to pyzx's own code path.
"""
from __future__ import annotations

import math
import os
import re
from fractions import Fraction

import pyzx as zx
from pyzx.circuit.qasmparser import QASMParser
from pyzx.utils import settings

_DECIMAL = re.compile(r"-?(?:\d+\.?\d*|\.\d+)")
_FLAT_CALL = re.compile(r"([^()\s]+)\(([^()]*)\)(.*)", re.S)


class _FastQASMParser(QASMParser):
    def __init__(self):
        super().__init__()
        self._phase_cache: dict[str, Fraction] = {}

    def extract_command_parts(self, c):
        if self.qasm_version != 2:
            return super().extract_command_parts(c)
        m = _FLAT_CALL.fullmatch(c)
        if m is None:
            return super().extract_command_parts(c)
        name, params, rest = m.groups()
        try:
            phases = [self.parse_phase_arg(v) for v in params.split(",") if v.strip()]
        except Exception:
            # Invalid parameters: let pyzx raise its own, usual error.
            return super().extract_command_parts(c)
        args = [s.strip() for s in rest.lstrip().split(",") if s.strip()]
        return name, phases, args

    def parse_phase_arg(self, val):
        if self._param_subst is not None:  # inside a parametrized gate body
            return super().parse_phase_arg(val)
        val = val.strip()
        cached = self._phase_cache.get(val)
        if cached is not None:
            return cached
        if _DECIMAL.fullmatch(val):
            # Same two steps as pyzx: its grammar first rounds the literal to
            # Fraction(float(x)).limit_denominator() (pyzx.symbolic, rule
            # `decimal`), then parse_phase_arg divides by pi and rounds again.
            const = Fraction(float(val)).limit_denominator()
            phase = Fraction(float(const) / math.pi).limit_denominator(
                settings.float_to_fraction_max_denominator)
        else:
            phase = super().parse_phase_arg(val)
        self._phase_cache[val] = phase
        return phase


def load_pyzx(qasm_path: str) -> zx.Circuit:
    """Drop-in replacement for ``zx.Circuit.from_qasm_file``."""
    with open(qasm_path) as f:
        circuit = _FastQASMParser().parse(f.read())
    circuit.name = os.path.basename(qasm_path)
    return circuit
