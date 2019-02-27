"""The STL package

This package defines STL grammar and semantics.
"""

import sympy

from .grammar.expression import Expression, as_Expression
from .grammar.atoms import Atom, TLFalse, TLTrue, true, false, Predicate
from .grammar.basic_ops import And, Or, Not, Implies
from .grammar.temporal_ops import Eventually, Always, Until

# from . import semantics as monitors

from collections import deque

from typing import Tuple, Union

U = Until

G = Always
Globally = Always
Alw = Always

F = Eventually
Finally = Eventually
Ev = Eventually

class Signal(sympy.Symbol):
    def __ge__(self, other):
        return stl.Predicate(self >= other)

    def __gt__(self, other):
        return stl.Predicate(self > other)

    def __le__(self, other):
        return stl.Predicate(self <= other)

    def __lt__(self, other):
        return stl.Predicate(self < other)


class Parameter(sympy.Symbol):
    pass

def signals(sig):
    return sympy.symbols(sig, cls=Signal)


def preorder_iterator(expr: Expression):
    if isinstance(expr, Expression):
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
