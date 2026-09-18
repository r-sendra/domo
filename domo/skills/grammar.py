"""
The DOMO skill-composition grammar.

A composition program is a string over the skill library:

    program  := seq
    seq      := fallback ('>>' fallback)*          # S1 then S2 (on success)
    fallback := layer ('|' layer)*                 # S1, or S2 if S1 fails
    layer    := post ('@' post)*                   # S1 executed ON TOP OF S2
                                                   #   (right-assoc; rightmost
                                                   #    is the motor skill)
    post     := atom modifier*
    modifier := '.for(' NUMBER ')'                 # succeed after T seconds
              | '.until(' cond ')'                 # succeed when cond holds
              | '.repeat(' INT ')'                 # loop the child N times
    atom     := NAME [ '(' kv (',' kv)* ')' ]      # skill with parameters
              | '(' seq ')'
    cond     := NAME [ '(' NUMBER (',' NUMBER)* ')' ]
    kv       := NAME '=' NUMBER

Precedence (tightest first): modifiers, '@', '|', '>>'.

Examples:
    walk(vx=0.5).for(4)
    (avoid @ walk(vx=0.6)).until(moved(3.0)) >> stand.for(2)
    climb | (avoid @ walk(vx=0.3)).for(10) | stand

The parser produces a plain AST (dataclasses below); typing and
instantiation against a SkillLibrary happen in library.compile().
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "CondRef",
    "Fallback",
    "GrammarError",
    "Layer",
    "Modified",
    "Sequence",
    "SkillRef",
    "parse",
]


class GrammarError(ValueError):
    pass


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass
class CondRef:
    name: str
    args: tuple[float, ...] = ()

    def to_text(self):
        return self.name + (f"({', '.join(_num(a) for a in self.args)})"
                            if self.args else "")


@dataclass
class SkillRef:
    name: str
    params: dict = field(default_factory=dict)

    def to_text(self):
        return self.name + (
            "(" + ", ".join(f"{k}={_num(v)}" for k, v in self.params.items()) + ")"
            if self.params else "")


@dataclass
class Layer:
    top: object                 # CommandSkill-typed node
    base: object                # motor-typed node

    def to_text(self):
        return f"{self.top.to_text()} @ {self.base.to_text()}"


@dataclass
class Sequence:
    children: list[object]

    def to_text(self):
        return " >> ".join(_paren(c, (Sequence,)) for c in self.children)


@dataclass
class Fallback:
    children: list[object]

    def to_text(self):
        return " | ".join(_paren(c, (Sequence, Fallback)) for c in self.children)


@dataclass
class Modified:
    child: object
    for_s: float | None = None
    until: CondRef | None = None
    repeat: int | None = None

    def to_text(self):
        t = _paren(self.child, (Sequence, Fallback, Layer))
        if self.for_s is not None:
            t += f".for({_num(self.for_s)})"
        if self.until is not None:
            t += f".until({self.until.to_text()})"
        if self.repeat is not None:
            t += f".repeat({self.repeat})"
        return t


def _num(x):
    return str(int(x)) if float(x) == int(x) else str(x)


def _paren(node, wrap_types):
    text = node.to_text()
    return f"({text})" if isinstance(node, wrap_types) else text


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    (?P<SEQ>>>)
  | (?P<AT>@)
  | (?P<OR>\|)
  | (?P<DOT>\.)
  | (?P<LP>\()
  | (?P<RP>\))
  | (?P<COMMA>,)
  | (?P<EQ>=)
  | (?P<NUM>-?\d+\.?\d*)
  | (?P<NAME>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<WS>\s+)
""", re.VERBOSE)


def _tokenize(text: str):
    tokens, pos = [], 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise GrammarError(f"unexpected character {text[pos]!r} at {pos} "
                               f"in: {text}")
        pos = m.end()
        kind = m.lastgroup
        if kind != "WS":
            tokens.append((kind, m.group()))
    tokens.append(("EOF", ""))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens, text):
        self.toks = tokens
        self.i = 0
        self.text = text

    def peek(self):
        return self.toks[self.i][0]

    def next(self, expect=None):
        kind, val = self.toks[self.i]
        if expect and kind != expect:
            raise GrammarError(
                f"expected {expect}, got {kind} '{val}' in: {self.text}")
        self.i += 1
        return val

    # program := seq EOF
    def parse(self):
        node = self.seq()
        if self.peek() != "EOF":
            raise GrammarError(f"trailing input from token "
                               f"'{self.toks[self.i][1]}' in: {self.text}")
        return node

    def seq(self):
        children = [self.fallback()]
        while self.peek() == "SEQ":
            self.next()
            children.append(self.fallback())
        return children[0] if len(children) == 1 else Sequence(children)

    def fallback(self):
        children = [self.layer()]
        while self.peek() == "OR":
            self.next()
            children.append(self.layer())
        return children[0] if len(children) == 1 else Fallback(children)

    def layer(self):
        parts = [self.post()]
        while self.peek() == "AT":
            self.next()
            parts.append(self.post())
        node = parts[-1]                      # right-assoc: base is rightmost
        for top in reversed(parts[:-1]):
            node = Layer(top=top, base=node)
        return node

    def post(self):
        node = self.atom()
        while self.peek() == "DOT":
            self.next()
            mod = self.next("NAME")
            self.next("LP")
            if mod == "for":
                node = self._merge_mod(node, for_s=float(self.next("NUM")))
            elif mod == "until":
                node = self._merge_mod(node, until=self.cond())
            elif mod == "repeat":
                node = self._merge_mod(node, repeat=int(float(self.next("NUM"))))
            else:
                raise GrammarError(f"unknown modifier '.{mod}(' "
                                   f"(known: for, until, repeat)")
            self.next("RP")
        return node

    def _merge_mod(self, node, **kw):
        # Each modifier wraps a fresh node: 'x.for(2).repeat(3)' is
        # Modified(repeat=3, child=Modified(for=2, child=x)) — the repeat
        # loops the timed inner node, which is the intuitive reading.
        return Modified(child=node, **kw)

    def atom(self):
        if self.peek() == "LP":
            self.next()
            node = self.seq()
            self.next("RP")
            return node
        name = self.next("NAME")
        params = {}
        if self.peek() == "LP":
            self.next()
            while True:
                key = self.next("NAME")
                self.next("EQ")
                params[key] = float(self.next("NUM"))
                if self.peek() == "COMMA":
                    self.next()
                    continue
                break
            self.next("RP")
        return SkillRef(name, params)

    def cond(self):
        name = self.next("NAME")
        args = []
        if self.peek() == "LP":
            self.next()
            while True:
                args.append(float(self.next("NUM")))
                if self.peek() == "COMMA":
                    self.next()
                    continue
                break
            self.next("RP")
        return CondRef(name, tuple(args))


def parse(text: str):
    """Parse a composition program into an AST. Raises GrammarError."""
    return _Parser(_tokenize(text), text).parse()


def parse_condition(text: str) -> CondRef:
    """Parse a lone condition expression, e.g. 'tipped(0.9)'."""
    parser = _Parser(_tokenize(text), text)
    ref = parser.cond()
    if parser.peek() != "EOF":
        raise GrammarError(f"trailing input in condition: {text}")
    return ref
