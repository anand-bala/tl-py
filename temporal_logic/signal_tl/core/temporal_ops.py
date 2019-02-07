import math

from sympy import latex
import sympy

from temporal_logic.signal_tl.core.base import Expression, Parameter


class Interval:
    def __init__(self, a, b, lopen=False, ropen=False):
        self.a = a
        self.b = b
        self._lopen = lopen if not math.isinf(-a) else True
        self._ropen = ropen if not math.isinf(b) else True
        self.is_parametric = isinstance(a, Parameter) or isinstance(b, Parameter)

    @property
    def interval(self):
        return self.a, self.b

    @property
    def open_on(self):
        return self._lopen, self._ropen

    @property
    def lopen(self):
        return self._lopen

    @property
    def ropen(self):
        return self._ropen

    @property
    def left_unbounded(self):
        if self.is_parametric:
            return False
        return math.isinf(-self.a)

    @property
    def right_unbounded(self):
        if self.is_parametric:
            return False
        return math.isinf(self.b)

    @property
    def unbounded(self):
        if self.is_parametric:
            return False
        return self.left_unbounded or self.right_unbounded

    @property
    def left(self):
        return self.a

    @property
    def right(self):
        return self.b

    def latex(self):
        i = sympy.Interval(self.a, self.b, self.lopen, self.ropen)
        return latex(i)


class TemporalOp(Expression):
    is_TemporalOp = True

    _interval = Interval(0, math.inf)
    _symbol = ''

    def __new__(cls, *args, **kwargs):
        obj = super(TemporalOp, cls).__new__(cls, *args, **kwargs)
        obj._interval = cls._get_interval(*args, **kwargs)
        return obj

    @property
    def interval(self):
        return self._interval

    @classmethod
    def _filter_args(cls, *args):
        if isinstance(args[-1], (tuple, list, Interval)):
            args = args[:-1]
        return super(TemporalOp, cls)._filter_args(args)

    @classmethod
    def _get_interval(cls, *args, **kwargs):
        interval = kwargs.get('interval', args[-1])
        if isinstance(interval, str):
            # TODO: string interval argument
            raise NotImplementedError('string interval argument not supported')
        if isinstance(interval, (tuple, list)):
            assert len(interval) == 2, "Expected interval with left and right bounds"
            a, b = interval
            return Interval(a, b)
        if isinstance(interval, Interval):
            return interval
        return Interval(0, math.inf)

    def _latex(self):
        interval = self._interval
        symbol = self._symbol
        if not interval.unbounded:
            symbol = r' {}_{ {} }'.format(symbol, interval.latex())
        if self.nargs == 1:
            return r'{} \left( {} \right)'.format(symbol, latex(self.args[0]))
        if self.nargs == 2:
            a, b = self.args
            return r'\left( {} \right) {} \left( {} \right)'.format(latex(a), symbol, latex(b))
        raise ValueError('There is no temporal op with nargs > 2')


class Until(TemporalOp):
    """The Until operator"""
    nargs = 2
    _symbol = r'\mathbf{U}'


class Releases(TemporalOp):
    """The Releases operator"""
    nargs = 2
    _symbol = r'\mathbf{R}'


class Always(TemporalOp):
    """The Always/Globally operator"""
    nargs = 1
    _symbol = r'\mathbf{G}'


class Eventually(TemporalOp):
    """The Eventually/Finally operator"""
    nargs = 1
    _symbol = r'\mathbf{F}'


__all__ = [
    'Until', 'Eventually', 'Always', 'Releases'
]
