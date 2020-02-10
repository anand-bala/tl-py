"""The STL package

This package defines STL grammar and semantics.
"""

from collections import deque
from typing import Tuple, Union, Iterator, Iterable, Deque

import sympy

import temporal_logic.signal_tl.core.atoms as atoms
import temporal_logic.signal_tl.core.base as base
import temporal_logic.signal_tl.core.basic_ops as basic_ops
import temporal_logic.signal_tl.core.temporal_ops as temporal_ops

# from . import semantics as monitors

U = temporal_ops.Until
R = temporal_ops.Releases

G = temporal_ops.Always
Globally = temporal_ops.Always
Alw = temporal_ops.Always

F = temporal_ops.Eventually
Finally = temporal_ops.Eventually
Ev = temporal_ops.Eventually


class Signal:
    _symbol: sympy.Symbol

    def __init__(self, sym: sympy.Symbol):
        self._symbol = sym

    def __ge__(self, other: sympy.Expr) -> atoms.Predicate:
        return atoms.Predicate(self._symbol >= other)

    def __gt__(self, other: sympy.Expr) -> atoms.Predicate:
        return atoms.Predicate(self._symbol > other)

    def __le__(self, other: sympy.Expr) -> atoms.Predicate:
        return atoms.Predicate(self._symbol <= other)

    def __lt__(self, other: sympy.Expr) -> atoms.Predicate:
        return atoms.Predicate(self._symbol < other)


def signals(names: Iterable[str]) -> Tuple[Signal, ...]:
    """
    Declare a bunch of signals. see: :func:`sympy.symbols` for more info on how to use
    :param names: List of signal names
    :return: tuple of signal variables
    :rtype: Tuple[Signal]
    """
    symbols = sympy.symbols(names)
    return tuple(map(Signal, symbols))


def params(names):
    """
    Declare a bunch of parameters in the STL formula
    """
    return NotImplementedError


def preorder_iterator(expr: base.Expression) -> Iterator[base.Expression]:
    if expr is None:
        return None

    stack: Deque[base.Expression] = deque()
    stack.append(expr)
    while len(stack) > 0:
        node = stack.pop()  # type: base.Expression
        yield node
        stack.extend(list(reversed(node.args)))


def postorder_iterator(expr: base.Expression) -> Iterator[base.Expression]:
    # TODO: Single stack implementation?
    if expr is None:
        return None
    stack1: Deque[base.Expression] = deque()
    stack2: Deque[base.Expression] = deque()

    root = expr
    stack1.append(root)
    while len(stack1) > 0:
        node = stack1.pop()
        stack1.extend(list(node.args))
        stack2.append(node)

    yield stack2.pop()


def get_atoms(expr: base.Expression) -> Union[Tuple[atoms.Atom, ...], None]:
    if expr is None:
        return None
    ats: Deque[atoms.Atom] = deque()
    for e in preorder_iterator(expr):
        if isinstance(e, atoms.Atom):
            ats.append(e)
    return tuple(ats)


def is_nnf(phi: base.Expression) -> bool:
    if phi is None:
        return True
    check = True
    for expr in preorder_iterator(phi):
        if isinstance(expr, basic_ops.Not):
            check = check and isinstance(expr.args[0], atoms.Atom)
    return check


__all__ = [
    'Atom', 'Predicate', 'TLTrue', 'true', 'TLFalse', 'false', 'Not', 'Or',
    'And', 'Implies', 'Until', 'U', 'Always', 'Alw', 'G', 'Eventually', 'Ev',
    'F', 'preorder_iterator'
]