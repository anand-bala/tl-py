"""The STL package

This package defines STL grammar and semantics.
"""

import sympy

from .grammar.expression import Expression
from .grammar.atoms import Atom, TLFalse, TLTrue, true, false, Predicate
from .grammar.basic_ops import And, Or, Not, Implies
from .grammar.temporal_ops import Eventually, Always, Until

from . import semantics as monitors

from collections import deque

from typing import Tuple, Union

U = Until

G = Always
Globally = Always
Alw = Always

F = Eventually
Finally = Eventually
Ev = Eventually


def signals(sig):
    if isinstance(sig, (tuple, list)):
        sig = ' '.join(sig)
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


__all__ = ['Atom', 'Predicate',
           'TLTrue', 'true', 'TLFalse', 'false',
           'Not', 'Or', 'And', 'Implies',
           'Until', 'U',
           'Always', 'Alw', 'G',
           'Eventually', 'Ev', 'F',
           'preorder_iterator']
