from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import map
from future import standard_library
standard_library.install_aliases()
import math
from abc import ABC, abstractmethod
from collections import deque

import sympy

import temporal_logic.signal_tl as stl


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


class Expression(ABC):
    """Expression base class

    This class just holds a bunch of methods that determine what kind of expression it is (defined below)
    """

    is_Atom = False
    is_Singleton = False
    is_Predicate = False
    is_LogicOp = False
    is_TemporalOp = False

    nargs = 0

    _params = []

    def __new__(cls, *args, **kwargs):

        _args = cls._filter_args(*args)
        if cls.nargs is None or math.isinf(cls.nargs):
            pass
        elif cls.nargs != len(_args):
            raise ValueError('Incompatible number of args')

        obj = object.__new__(cls)
        obj._args = obj._filter_args(*args)
        for a in obj._args:
            a._parent = obj
        obj._mhash = None
        obj._depth = obj._calc_depth()
        obj._size = obj._calc_size()
        obj._parent = None
        return obj

    @classmethod
    def convert(cls, other):
        from . import atoms
        from . import basic_ops

        sym_2_expression = {
            'And': basic_ops.And,
            'Or': basic_ops.Or,
            'Not': basic_ops.Not,
            'Implies': basic_ops.Implies,
        }

        if isinstance(other, atoms.Expression):
            return other
        if isinstance(other, (bool, sympy.boolalg.BooleanAtom)):
            return atoms.true if other else atoms.false
        if isinstance(other, (sympy.And, sympy.Or, sympy.Not, sympy.Implies)):
            return sym_2_expression[type(other).__name__](*other.args)
        if isinstance(other, (sympy.Ge, sympy.Gt, sympy.Le, sympy.Lt)):
            return atoms.Predicate(other)
        raise TypeError('Incompatible argument type: %s',
                        other.__module__ + "." + other.__class__.__qualname__)

    @classmethod
    def _filter_args(cls, *args):
        # type: *Union[Expression, sympy.Expr] -> Tuple[Expression]
        return tuple(map(cls.convert, args))

    @property
    def func(self):
        return self.__class__

    @property
    def args(self):
        return self._args

    def __and__(self, other):
        """Logical And"""
        from . import basic_ops
        return basic_ops.And(self, other)

    __rand__ = __and__

    def __or__(self, other):
        """Logical Or"""
        from . import basic_ops
        return basic_ops.Or(self, other)

    __ror__ = __or__

    def __neg__(self):
        """Logical Negation"""
        from . import basic_ops
        return basic_ops.Not(self)

    def __invert__(self):
        """Logical Negation"""
        from . import basic_ops
        return basic_ops.Not(self)

    def __rshift__(self, other):
        """Logical Implication"""
        from . import basic_ops
        return basic_ops.Or(basic_ops.Not(self), other)

    def __lshift__(self, other):
        """Logical Reverse Implication"""
        from . import basic_ops
        return basic_ops.Or(basic_ops.Not(other), self)

    __rrshift__ = __lshift__
    __rlshift__ = __rshift__

    def __hash__(self):
        h = self._mhash
        if h is None:
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h

    def _hashable_content(self):
        return self.args

    def _noop(self, other=None):
        raise TypeError('BooleanAtom not allowed in this context.')

    __add__ = _noop
    __radd__ = _noop
    __sub__ = _noop
    __rsub__ = _noop
    __mul__ = _noop
    __rmul__ = _noop
    __pow__ = _noop
    __rpow__ = _noop
    __rdiv__ = _noop
    __truediv__ = _noop
    __div__ = _noop
    __rtruediv__ = _noop
    __mod__ = _noop
    __rmod__ = _noop
    _eval_power = _noop

    @property
    def depth(self):
        return self._depth

    def _calc_depth(self):
        if self.is_Atom:
            return 0
        return 1 + max(map(lambda arg: arg.depth, self.args))

    @property
    def size(self):
        return self._size

    def _calc_size(self):
        if self.is_Atom:
            return 1
        return 1 + sum(map(lambda arg: arg.depth, self.args))

    @property
    def parent(self):
        return self._parent

    @abstractmethod
    def _latex(self):
        pass

    def to_nnf(self):
        return self

    def to_cnf(self):
        return self
