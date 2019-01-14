import math
from .expression import Expression, as_Expression


class TemporalOp(Expression):
    is_TemporalOp = True

    _interval = (0, math.inf)
    _lopen = False
    _ropen = False

    def __new__(cls, *args, **kwargs):
        obj = super(TemporalOp, cls).__new__(cls, *args, **kwargs)
        obj._lopen, obj._ropen = kwargs.get('open_on', (False, True))
        obj._interval = cls._get_interval(*args)
        return obj

    @property
    def interval(self):
        return self._interval

    @property
    def interval_opening(self):
        return self._lopen, self._ropen

    @classmethod
    def _filter_args(cls, *args):
        if isinstance(args[-1], (tuple, list)):
            return tuple(map(as_Expression, args[:-1]))
        return tuple(map(as_Expression, args))

    @classmethod
    def _get_interval(cls, *args):
        if isinstance(args[-1], (tuple, list)):
            return args[-1]
        return 0, math.inf


class Until(TemporalOp):
    """The Until operator"""
    nargs = 2

    def tex_print(self):
        args_tex = tuple(arg.tex_print for arg in self.args)
        a, b = self.interval
        if math.isinf(b):
            return r' U '.join(args_tex)


class Always(TemporalOp):
    """The Always/Globally operator"""
    nargs = 1

    def tex_print(self):
        arg_tex = self.args[0].tex_print()
        a, b = self.interval
        if math.isinf(b):
            return r'$G({})$'.format(arg_tex)

        return r' G_{{ [{a}, {b}] }} ( {phi} ) '.format(a=a, b=b, phi=arg_tex)


class Eventually(TemporalOp):
    """The Eventually/Finally operator"""
    nargs = 1

    def tex_print(self):
        arg_tex = self.args[0].tex_print()
        a, b = self.interval
        if math.isinf(b):
            return r'$F({})$'.format(arg_tex)

        return r' F_{{ [{a}, {b}] }} ( {phi} ) '.format(a=a, b=b, phi=arg_tex)
