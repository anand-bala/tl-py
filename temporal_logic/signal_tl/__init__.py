"""The STL package

This package defines STL grammar and semantics.
"""

from typing import Tuple, Union
from collections import deque

import sympy

from .grammar.atoms import Atom, TLTrue, TLFalse, Predicate, true, false
from .grammar.basic_ops import Or, And, Not, Implies
from .grammar.expression import Expression, as_Expression
from .grammar.temporal_ops import Until, Always, Eventually

# from . import semantics as monitors



U = Until

G = Always
Globally = Always
Alw = Always

F = Eventually
Finally = Eventually
Ev = Eventually


def signals(sig):
    return sympy.symbols(sig)


def preorder_iterator(expr: Expression):
    if expr is None:
        return None

    stack = deque()
    stack.append(expr)
    while len(stack) > 0:
        node = stack.pop()  # type: Expression
        yield node
        stack.extend(list(reversed(node.args)))


def get_atoms(expr: Expression) -> Union[Tuple[Atom, ...], None]:
    if expr is None:
        return None
    atoms = deque()
    for e in preorder_iterator(expr):
        if e.is_Atom:
            atoms.append(e)
    return tuple(atoms)


__all__ = [
    "Atom",
    "Predicate",
    "TLTrue",
    "true",
    "TLFalse",
    "false",
    "Not",
    "Or",
    "And",
    "Implies",
    "Until",
    "U",
    "Always",
    "Alw",
    "G",
    "Eventually",
    "Ev",
    "F",
    "preorder_iterator",
]
