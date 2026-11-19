"""A small, dependency-free OpenQASM 2 reader.

qregime only needs a flat list of gates, so it reads OpenQASM 2 itself
instead of depending on a quantum SDK. Any framework that exports
OpenQASM 2 (Qiskit, Cirq, PennyLane, Braket, tket, ...) can feed qregime.

What it reads:
  - ``qreg``/``creg`` declarations (several registers are flattened in
    declaration order, the convention used by most SDKs);
  - the standard library ``qelib1.inc`` (bundled), other ``include`` files
    relative to the circuit, and ``gate`` definitions;
  - gate calls with parameter expressions (``pi``, ``+ - * / ^``,
    ``sin cos tan exp ln sqrt asin acos atan``) and register broadcasting;
  - ``measure``, ``reset``, ``barrier`` and ``if`` statements, which are kept
    as non-unitary instructions so callers can decide what to do with them.

``Circuit.instructions`` keeps the gates as written (top level, not
expanded). ``Circuit.basis_ops()`` expands every gate into the primitive
``U(theta, phi, lambda)`` and ``CX`` gates, which is what the simulators use.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NON_UNITARY = ("measure", "reset", "barrier", "if")


class QASMError(ValueError):
    """The file is not valid OpenQASM 2, or uses something qregime cannot read."""


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"""
    (?P<ws>\s+|//[^\n]*)
  | (?P<string>"[^"]*")
  | (?P<real>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>->|==|[{}()\[\];,+\-*/^])
""", re.VERBOSE)


@dataclass
class _Tok:
    kind: str
    text: str
    start: int
    end: int


def _tokenize(text: str, source: str) -> list:
    tokens, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            line = text.count("\n", 0, pos) + 1
            raise QASMError(f"{source}:{line}: unexpected character {text[pos]!r}")
        if m.lastgroup != "ws":
            tokens.append(_Tok(m.lastgroup, m.group(), m.start(), m.end()))
        pos = m.end()
    return tokens


# --------------------------------------------------------------------------
# parameter expressions
# --------------------------------------------------------------------------

_FUNCS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp,
          "ln": math.log, "sqrt": math.sqrt, "asin": math.asin, "acos": math.acos,
          "atan": math.atan}


def _evaluate(expr, env: dict) -> float:
    kind = expr[0]
    if kind == "num":
        return expr[1]
    if kind == "var":
        if expr[1] not in env:
            raise QASMError(f"unknown parameter {expr[1]!r}")
        return env[expr[1]]
    if kind == "neg":
        return -_evaluate(expr[1], env)
    if kind == "call":
        return _FUNCS[expr[1]](_evaluate(expr[2], env))
    a, b = _evaluate(expr[1], env), _evaluate(expr[2], env)
    if kind == "+":
        return a + b
    if kind == "-":
        return a - b
    if kind == "*":
        return a * b
    if kind == "/":
        return a / b
    return a ** b  # "^"


# --------------------------------------------------------------------------
# IR
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Instruction:
    """One top-level operation. ``qubits`` are flat indices."""
    name: str
    qubits: tuple
    params: tuple = ()
    statement: int = -1     # index into Circuit.statements (source span)


@dataclass
class _GateDef:
    params: list
    qargs: list
    body: list              # [(name, [param exprs], [qarg names])]
    opaque: bool = False


@dataclass
class Circuit:
    num_qubits: int
    instructions: list
    gate_defs: dict
    statements: list = field(default_factory=list)   # (start, end, kind) in source text
    text: str = ""
    name: str = ""
    num_clbits: int = 0

    def gates(self):
        """Top-level unitary gates (no measure/reset/barrier/if)."""
        return [i for i in self.instructions if i.name not in NON_UNITARY]

    def count_ops(self) -> dict:
        counts: dict = {}
        for inst in self.instructions:
            counts[inst.name] = counts.get(inst.name, 0) + 1
        return counts

    # -- expansion to U and CX ---------------------------------------------

    def expand(self, inst: Instruction) -> list:
        """``inst`` as a list of ("U", (q,), (theta, phi, lam)) and ("CX", (c, t), ())."""
        out: list = []
        self._expand(inst.name, list(inst.params), list(inst.qubits), out, depth=0)
        return out

    def _expand(self, name, params, qubits, out, depth):
        if depth > 64:
            raise QASMError(f"gate {name!r}: definitions nest too deeply (recursive?)")
        if name == "U":
            out.append(("U", (qubits[0],), tuple(params)))
            return
        if name == "CX":
            out.append(("CX", (qubits[0], qubits[1]), ()))
            return
        gate = self.gate_defs.get(name)
        if gate is None:
            raise QASMError(f"gate {name!r} is not defined")
        if gate.opaque:
            raise QASMError(f"gate {name!r} is opaque: no definition to simulate")
        env = dict(zip(gate.params, params))
        env["pi"] = math.pi
        binding = dict(zip(gate.qargs, qubits))
        for sub_name, sub_params, sub_qargs in gate.body:
            if sub_name == "barrier":
                continue
            self._expand(sub_name, [_evaluate(p, env) for p in sub_params],
                         [binding[a] for a in sub_qargs], out, depth + 1)

    def basis_ops(self, gates_only: bool = True) -> list:
        """Every gate expanded into U/CX, in order, each tagged with the index
        of the top-level instruction it came from: (kind, qubits, params, index)."""
        ops = []
        for k, inst in enumerate(self.instructions):
            if inst.name in NON_UNITARY:
                if gates_only:
                    continue
                ops.append((inst.name, inst.qubits, inst.params, k))
                continue
            for kind, qubits, params in self.expand(inst):
                ops.append((kind, qubits, params, k))
        return ops


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

class _Parser:
    def __init__(self, text: str, source: str, base_dir: Optional[str]):
        self.text, self.source, self.base_dir = text, source, base_dir
        self.toks = _tokenize(text, source)
        self.i = 0
        self.qregs: dict = {}
        self.cregs: dict = {}
        self.n_qubits = 0
        self.n_clbits = 0
        self.gate_defs: dict = {}
        self.instructions: list = []
        self.statements: list = []

    # token helpers
    def _peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None

    def _error(self, msg, tok=None):
        tok = tok or self._peek() or (self.toks[-1] if self.toks else None)
        line = self.text.count("\n", 0, tok.start) + 1 if tok else 1
        raise QASMError(f"{self.source}:{line}: {msg}")

    def _next(self):
        tok = self._peek()
        if tok is None:
            self._error("unexpected end of file")
        self.i += 1
        return tok

    def _expect(self, text):
        tok = self._next()
        if tok.text != text:
            self._error(f"expected {text!r}, found {tok.text!r}", tok)
        return tok

    def _accept(self, text):
        tok = self._peek()
        if tok is not None and tok.text == text:
            self.i += 1
            return True
        return False

    def _ident(self):
        tok = self._next()
        if tok.kind != "id":
            self._error(f"expected a name, found {tok.text!r}", tok)
        return tok.text

    def _int(self):
        tok = self._next()
        if tok.kind != "real" or not tok.text.isdigit():
            self._error(f"expected an integer, found {tok.text!r}", tok)
        return int(tok.text)

    # expressions: sum := term (('+'|'-') term)*, term := unary (('*'|'/') unary)*,
    # unary := '-' unary | power, power := atom ('^' unary)?
    def _expr(self):
        node = self._term()
        while self._peek() is not None and self._peek().text in ("+", "-"):
            op = self._next().text
            node = (op, node, self._term())
        return node

    def _term(self):
        node = self._unary()
        while self._peek() is not None and self._peek().text in ("*", "/"):
            op = self._next().text
            node = (op, node, self._unary())
        return node

    def _unary(self):
        if self._accept("-"):
            return ("neg", self._unary())
        if self._accept("+"):
            return self._unary()
        node = self._atom()
        if self._accept("^"):
            node = ("^", node, self._unary())
        return node

    def _atom(self):
        tok = self._next()
        if tok.kind == "real":
            return ("num", float(tok.text))
        if tok.text == "(":
            node = self._expr()
            self._expect(")")
            return node
        if tok.kind == "id":
            if tok.text == "pi":
                return ("num", math.pi)
            if tok.text in _FUNCS:
                self._expect("(")
                arg = self._expr()
                self._expect(")")
                return ("call", tok.text, arg)
            return ("var", tok.text)
        self._error(f"unexpected {tok.text!r} in expression", tok)

    def _expr_list(self, closing=")"):
        items = []
        if self._accept(closing):
            return items
        while True:
            items.append(self._expr())
            if self._accept(closing):
                return items
            self._expect(",")

    # registers and arguments
    def _arg(self, regs, kind):
        name = self._ident()
        if name not in regs:
            self._error(f"unknown {kind} register {name!r}")
        offset, size = regs[name]
        if self._accept("["):
            idx = self._int()
            self._expect("]")
            if idx >= size:
                self._error(f"index {idx} out of range for {name}[{size}]")
            return [offset + idx]
        return [offset + k for k in range(size)]

    def _broadcast(self, args):
        sizes = {len(a) for a in args if len(a) > 1}
        if len(sizes) > 1:
            self._error("registers of different sizes in one statement")
        n = sizes.pop() if sizes else 1
        return [tuple(a[k] if len(a) > 1 else a[0] for a in args) for k in range(n)]

    # statements
    def parse(self, top_level=True):
        if top_level and self._peek() is not None and self._peek().text == "OPENQASM":
            self._next()
            version = self._next().text
            if not version.startswith("2"):
                self._error(f"only OpenQASM 2 is supported, found version {version}")
            self._expect(";")
        while self._peek() is not None:
            start = self._peek().start
            kind = self._statement()
            self.statements.append((start, self.toks[self.i - 1].end, kind))

    def _statement(self):
        tok = self._peek()
        word = tok.text
        if word == "include":
            self._next()
            fname = self._next().text.strip('"')
            self._expect(";")
            self._include(fname)
            return "include"
        if word in ("qreg", "creg"):
            self._next()
            name = self._ident()
            self._expect("[")
            size = self._int()
            self._expect("]")
            self._expect(";")
            if word == "qreg":
                self.qregs[name] = (self.n_qubits, size)
                self.n_qubits += size
            else:
                self.cregs[name] = (self.n_clbits, size)
                self.n_clbits += size
            return word
        if word in ("gate", "opaque"):
            self._gate_definition(opaque=word == "opaque")
            return word
        if word == "measure":
            self._next()
            q = self._arg(self.qregs, "quantum")
            self._expect("->")
            c = self._arg(self.cregs, "classical")
            self._expect(";")
            if len(q) != len(c):
                self._error("measure: quantum and classical registers differ in size")
            sid = len(self.statements)
            self.instructions += [Instruction("measure", (qa,), (), sid) for qa in q]
            return "measure"
        if word == "reset":
            self._next()
            q = self._arg(self.qregs, "quantum")
            self._expect(";")
            sid = len(self.statements)
            self.instructions += [Instruction("reset", (qa,), (), sid) for qa in q]
            return "reset"
        if word == "barrier":
            self._next()
            qubits = []
            while True:
                qubits += self._arg(self.qregs, "quantum")
                if self._accept(";"):
                    break
                self._expect(",")
            self.instructions.append(Instruction("barrier", tuple(qubits), (), len(self.statements)))
            return "barrier"
        if word == "if":
            self._next()
            self._expect("(")
            self._ident()
            self._expect("==")
            self._int()
            self._expect(")")
            first = len(self.instructions)
            self._statement()
            touched = sorted({q for inst in self.instructions[first:] for q in inst.qubits})
            del self.instructions[first:]
            self.instructions.append(Instruction("if", tuple(touched), (), len(self.statements)))
            return "if"
        self._gate_call()
        return "call"

    def _include(self, fname):
        if fname == "qelib1.inc":
            path = os.path.join(_DATA, "qelib1.inc")
        else:
            path = os.path.join(self.base_dir or ".", fname)
            if not os.path.exists(path):
                self._error(f"include file {fname!r} not found")
        for name, gate in _library(path).items():
            self.gate_defs.setdefault(name, gate)

    def _gate_definition(self, opaque):
        self._next()
        name = self._ident()
        params = []
        if self._accept("("):
            if not self._accept(")"):
                while True:
                    params.append(self._ident())
                    if self._accept(")"):
                        break
                    self._expect(",")
        qargs = [self._ident()]
        while self._accept(","):
            qargs.append(self._ident())
        if opaque:
            self._expect(";")
            self.gate_defs[name] = _GateDef(params, qargs, [], opaque=True)
            return
        self._expect("{")
        body = []
        while not self._accept("}"):
            sub = self._ident()
            sub_params = self._expr_list() if self._accept("(") else []
            sub_qargs = [self._ident()]
            while self._accept(","):
                sub_qargs.append(self._ident())
            self._expect(";")
            for a in sub_qargs:
                if a not in qargs:
                    self._error(f"gate {name!r}: unknown qubit argument {a!r}")
            body.append((sub, sub_params, sub_qargs))
        self.gate_defs[name] = _GateDef(params, qargs, body)

    def _gate_call(self):
        tok = self._peek()
        name = self._ident()
        exprs = self._expr_list() if self._accept("(") else []
        args = [self._arg(self.qregs, "quantum")]
        while self._accept(","):
            args.append(self._arg(self.qregs, "quantum"))
        self._expect(";")
        if name not in ("U", "CX") and name not in self.gate_defs:
            self._error(f"gate {name!r} is not defined (missing include \"qelib1.inc\"?)", tok)
        expected = (3, 1) if name == "U" else (0, 2) if name == "CX" else (
            len(self.gate_defs[name].params), len(self.gate_defs[name].qargs))
        if (len(exprs), len(args)) != expected:
            self._error(f"gate {name!r} takes {expected[0]} parameters and {expected[1]} "
                        f"qubits, got {len(exprs)} and {len(args)}", tok)
        params = tuple(_evaluate(e, {"pi": math.pi}) for e in exprs)
        sid = len(self.statements)
        for qubits in self._broadcast(args):
            if len(set(qubits)) != len(qubits):
                self._error(f"gate {name!r} applied to the same qubit twice", tok)
            # SDK convention: U and CX are written as u and cx
            self.instructions.append(Instruction({"U": "u", "CX": "cx"}.get(name, name),
                                                 qubits, params, sid))


@lru_cache(maxsize=None)
def _library(path: str) -> dict:
    with open(path) as f:
        text = f.read()
    p = _Parser(text, os.path.basename(path), os.path.dirname(path))
    p.parse(top_level=False)
    return p.gate_defs


def parse(text: str, source: str = "<string>", base_dir: Optional[str] = None) -> Circuit:
    """Parse OpenQASM 2 source text.

    The standard-library gates are available even without ``include
    "qelib1.inc"`` (lenient, like Qiskit's legacy reader for its builtins);
    a definition in the file overrides them.
    """
    p = _Parser(text, source, base_dir)
    p.gate_defs = dict(_library(os.path.join(_DATA, "qelib1.inc")))
    p.parse()
    return Circuit(num_qubits=p.n_qubits, instructions=p.instructions, gate_defs=p.gate_defs,
                   statements=p.statements, text=text, name=os.path.splitext(source)[0],
                   num_clbits=p.n_clbits)


def load(path: str) -> Circuit:
    """Parse an OpenQASM 2 file."""
    with open(path) as f:
        text = f.read()
    return parse(text, os.path.basename(path), os.path.dirname(os.path.abspath(path)))


def strip_final_measurements(circ: Circuit) -> Optional[str]:
    """OpenQASM text of ``circ`` without barriers, final measurements and
    classical registers (see :func:`to_qasm`), or None if there is nothing
    to strip.

    A measurement is final when no later operation touches its qubit.
    Mid-circuit measurements, resets and classically controlled gates
    change what the circuit computes, so they raise ``QASMError``.
    """
    if not any(inst.name in NON_UNITARY for inst in circ.instructions):
        return None
    later: set = set()
    blocking: set = set()
    for inst in reversed(circ.instructions):
        if inst.name == "barrier":
            continue
        if inst.name == "measure" and inst.qubits[0] not in later:
            continue    # final measurement
        if inst.name in ("measure", "reset", "if"):
            blocking.add(inst.name)
        later.update(inst.qubits)
    if blocking:
        raise QASMError(
            f"mid-circuit {'/'.join(sorted(blocking))} found. Dynamic circuits are not "
            "supported yet; only final measurements are accepted.")
    return to_qasm(circ, gates_only=True)


def _number(x: float) -> str:
    """Plain decimal (no scientific notation), exact enough to round-trip."""
    text = repr(float(x))
    if "e" in text or "E" in text:
        text = f"{x:.17f}".rstrip("0")
        if text.endswith("."):
            text += "0"
    return text


def to_qasm(circ: Circuit, gates_only: bool = False) -> str:
    """OpenQASM 2 text for ``circ``: the file's own include and gate
    definitions verbatim, one flat register ``q``, and every gate call with
    its parameters evaluated to plain decimals. With ``gates_only``,
    measurements, barriers, resets and classical registers are left out."""
    lines = ["OPENQASM 2.0;"]
    for start, end, kind in circ.statements:
        if kind in ("include", "gate", "opaque"):
            lines.append(circ.text[start:end])
    lines.append(f"qreg q[{circ.num_qubits}];")
    if not gates_only and circ.num_clbits:
        lines.append(f"creg c[{circ.num_clbits}];")
    for inst in circ.instructions:
        if inst.name in NON_UNITARY:
            if gates_only:
                continue
            raise QASMError("to_qasm: only gates_only output is supported for non-unitary operations")
        params = f"({','.join(_number(x) for x in inst.params)})" if inst.params else ""
        lines.append(f"{inst.name}{params} {','.join(f'q[{q}]' for q in inst.qubits)};")
    return "\n".join(lines) + "\n"
