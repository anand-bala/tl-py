from sympy import latex

from temporal_logic.signal_tl.core.base import Expression
from temporal_logic.signal_tl.core.temporal_ops import *


class LogicOp(Expression):
    is_LogicOp = True
    _symbol = ''

    def _latex(self):
        symbol = self._symbol
        if self.nargs is None:
            tex_args = tuple(r'\left( {} \right)'.format(latex(arg))
                             for arg in self.args)
            return ' {} '.format(symbol).join(tex_args)
        if self.nargs == 1:
            return r'{} \left( {} \right)'.format(symbol, latex(self.args[0]))
        if self.nargs == 2:
            a, b = self.args
            return r'\left( {} \right) {} \left( {} \right)'.format(latex(a), symbol, latex(b))


class Not(LogicOp):
    """The Not operator.

    The Not operator contains a single operand, which can be an STL expression.
    """

    nargs = 1
    _symbol = r'\neg'

    def to_nnf(self):
        phi = self.args[0]  # type: Expression
        func = phi.func
        if phi.is_Singleton:
            return Expression.convert(not phi)
        if phi.is_Predicate:
            return self
        if isinstance(phi, Not):
            return phi.args[0]
        if isinstance(phi, And):
            return Or(*tuple(map(Not, phi.args)))
        if isinstance(phi, Or):
            return And(*tuple(map(Not, phi.args)))
        if isinstance(phi, Implies):
            a, b = phi.args
            return And(a, ~b)
        if isinstance(phi, Eventually):
            I = phi.interval
            a = phi.args[0]  # type: Expression
            return Always(~a, I)
        if isinstance(phi, Always):
            I = phi.interval
            a = phi.args[0]  # type: Expression
            return Eventually(~a, I)
        if isinstance(phi, Until):
            I = phi.interval
            a, b = phi.args  # type: Expression
            return Releases(~a, ~b, I)
        if isinstance(phi, Releases):
            I = phi.interval
            a, b = phi.args  # type: Expression
            return Until(~a, ~b, I)
        raise ValueError('Illegal operator: {}'.format(func))


class Or(LogicOp):
    """The Or operator

    This is a n-ary operator where n >= 2.
    """
    nargs = None
    _symbol = r'\lor'

    @classmethod
    def _filter_args(cls, *args) -> tuple:
        args = super()._filter_args(*args)
        new_args = []
        for arg in args:
            if isinstance(arg, Or):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        return tuple(new_args)


class And(LogicOp):
    """The And operator

    This is a n-ary operator where n >= 2.
    """
    nargs = None
    _symbol = r'\land'

    @classmethod
    def _filter_args(cls, *args) -> tuple:
        args = super()._filter_args(*args)
        new_args = []
        for arg in args:
            if isinstance(arg, And):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        return tuple(new_args)


class Implies(LogicOp):
    nargs = 2
    _symbol = r'\implies'


__all__ = [
    'And', 'Or', 'Not', 'Implies',
]
