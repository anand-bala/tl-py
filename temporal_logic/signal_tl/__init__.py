"""The STL package

This package defines STL grammar and semantics.
"""

from collections import deque
from typing import Tuple, Union

import sympy
from .core.atoms import Atom, TLFalse, TLTrue, true, false, Predicate
from .core.base import Expression, Signal, Parameter
from .core.basic_ops import And, Or, Not, Implies
from .core.temporal_ops import Eventually, Always, Until, Releases

# from . import semantics as monitors

U = Until
R = Releases

G = Always
Globally = Always
Alw = Always

F = Eventually
Finally = Eventually
Ev = Eventually


def signals(names):
    """
    Declare a bunch of signals. see: :func:`sympy.symbols` for more info on how to use
    :param names: List of signal names
    :return: tuple of signal variables
    :rtype: Tuple[Signal]
    """
    return sympy.symbols(names, cls=Signal)


def params(names):
    """
    Declare a bunch of parameters in the STL formula
    :param names: List of parameter names
    :return: Tuple[Parameter]
    """
    return sympy.symbols(names, cls=Parameter)


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
