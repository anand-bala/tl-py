from abc import abstractmethod

import numpy as np
import pandas as pd
import sympy
from sympy import sympify
from sympy.core.relational import Ge, Gt, Le, Lt

from temporal_logic.signal_tl.core.base import Expression, Signal


class Atom(Expression):
    is_Atom = True
    nargs = 0

    @abstractmethod
    def eval(self, *args):
        pass


class TLTrue(Atom):
    """Atomic True"""
    is_Singleton = True

    def eval(self):
        return True

    def __nonzero__(self):
        return True

    __bool__ = __nonzero__

    def __hash__(self):
        return hash(True)

    def _latex(self):
        return r' \top '


class TLFalse(Atom):
    """Atomic False"""
    is_Singleton = True

    def eval(self):
        return False

    def __nonzero__(self):
        return False

    __bool__ = __nonzero__

    def __hash__(self):
        return hash(False)

    def _latex(self):
        return r' \bot '


true = TLTrue()
false = TLFalse()


class Predicate(Atom):
    r"""A signal predicate in the form of :math:`f(x_i) \geq 0`

    Define a predicate on a signal in the form of :math:`f(x_i) \geq 0`.
    Here,
        :math:`i \in \mathbb{N}`
        :math:`x_i` are the parameters of the signal.

    """
    _predicate = None
    is_Predicate = True
    _signals = set()
    _f = lambda *args: np.empty(0)

    def __new__(cls, *args, **kwargs):
        if len(args) != 1:
            raise ValueError('Must provide the predicate as an argument')
        predicate = cls._get_predicate(args[0])
        obj = super(Predicate, cls).__new__(cls, *args, **kwargs)
        obj._predicate = predicate
        obj._signals = set(map(str, predicate.atoms(Signal)))
        obj._f = sympy.lambdify(obj._signals, obj._predicate.gts, 'numpy')
        return obj

    @classmethod
    def _get_predicate(cls, args):
        """Return the predicate in the form f(x) >= 0"""
        pred_default = sympify(args)
        if isinstance(pred_default, (Ge, Gt, Le, Lt)):

            new_lhs = pred_default.gts - pred_default.lts

            if isinstance(pred_default, (Ge, Gt)):
                return pred_default.func(new_lhs, 0)
            if isinstance(pred_default, Lt):
                return Ge(new_lhs, 0)
            if isinstance(pred_default, Le):
                return Gt(new_lhs, 0)

        raise TypeError('The given predicate is not an inequality')

    @property
    def expr(self):
        return self._predicate.gts

    @property
    def predicate(self):
        return self._predicate

    @property
    def signals(self):
        return self._signals  # type: set

    def f(self, trace):
        """
        Evaluate the RHS of predicate

        Assumption: 
            The name of the symbols in this predicate are the same as the name of the columns in the DataFrame.
            If the trace is a series, the number of signals used in the predicate must be equal to 1
        """
        if isinstance(trace, pd.DataFrame):
            assert self.signals.issubset(
                trace.columns), 'The signals used in the predicates are not a subset of the column names of the trace'
            signals = tuple(trace[i].values for i in self.signals)
            return self._f(*signals)

        elif isinstance(trace, pd.Series):
            assert len(
                self.signals) == 1, 'Predicate uses more than 1 symbol, got 1-D trace'
            signal = trace.values
            return self._f(signal)
        else:
            raise ValueError(
                'Expected pandas DataFrame or Series, got {}'.format(trace.__qualname__))

    def eval(self, trace):
        """
        Evaluate the predicate.

        :returns: Boolean signal
        """
        if isinstance(self.predicate, sympy.Ge):
            return self.f(trace) >= 0
        return self.f(trace) > 0

    def _latex(self):
        return sympy.latex(self.predicate)


__all__ = [
    'TLFalse', 'TLTrue', 'true', 'false',
    'Predicate'
]
