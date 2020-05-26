from abc import abstractmethod

import sympy
from sympy import sympify
from sympy.core.relational import Ge, Gt, Le, Lt

from .expression import Expression


class Atom(Expression):
    is_Atom = True
    nargs = 0

    @abstractmethod
    def eval(self, *args):
        pass

    @classmethod
    def _filter_args(cls, *args):
        return ()


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

    def tex_print(self):
        return r" \top "


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

    def tex_print(self):
        return r" \bot "


true = TLTrue()
false = TLFalse()


class Predicate(Atom):
    """A signal predicate in the form of :math:`f(x_i) \geq 0`

    Define a predicate on a signal in the form of :math:`f(x_i) \geq 0`.
    Here,
        :math:`i \in \mathbb{N}`
        :math:`x_i` are the parameters of the signal.

    """

    _predicate = None
    is_Predicate = True

    def __new__(cls, *args, **kwargs):
        if len(args) != 1:
            raise ValueError("Must provide the predicate as an argument")
        predicate = cls._get_predicate(args[0])
        obj = super(Predicate, cls).__new__(cls, *args, **kwargs)
        obj._predicate = predicate
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

        raise TypeError("The given predicate is not an inequality")

    @property
    def expr(self):
        return self._predicate.gts

    @property
    def predicate(self):
        return self._predicate

    def eval(self, signals, *x):
        assert len(signals) == len(x)
        fn = sympy.lambdify(signals, self.expr)
        return fn(*x)

    def tex_print(self):
        return sympy.latex(self.predicate)


__all__ = ["TLFalse", "TLTrue", true, false, "Predicate"]
