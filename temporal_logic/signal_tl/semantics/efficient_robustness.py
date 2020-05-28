import numpy as np
import sympy
from numba import jit, njit
from scipy.ndimage import shift
from scipy.interpolate import interp1d
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d

from temporal_logic import signal_tl
from temporal_logic.signal_tl import as_Expression

from .base import BaseMonitor

BOTTOM = -np.inf
TOP = np.inf


def TOP_FN(_):
    return np.inf


def BOTTOM_FN(_):
    return -np.inf


def _get_atom_fn(inputs, expr):
    if isinstance(expr, signal_tl.TLTrue):
        return TOP_FN
    if isinstance(expr, signal_tl.TLFalse):
        return BOTTOM_FN
    if isinstance(expr, signal_tl.Predicate):
        return sympy.lambdify(inputs, expr.expr)
    raise TypeError(
        "Invalid input type: must be of type signal_tl.Atom, got {}".format(type(expr))
    )


class EfficientRobustnessMonitor(BaseMonitor):
    @property
    def horizon(self):
        return np.inf

    def __init__(self, spec, signals):
        """
        Initialize robustness computer for spec
        :param spec: STL specification to monitor
        :param signals: List of signal parameters. This must match the symbols used in the predicates
        """
        self._spec = as_Expression(spec)
        self._signals = signals
        self._atoms = frozenset(signal_tl.get_atoms(spec))
        self._reset()

    def _reset(self):
        self.atom_functions = dict(zip(self._atoms, [BOTTOM_FN] * len(self._atoms)))
        self.atom_signals = dict(zip(self._atoms, [None] * len(self._atoms)))

    @property
    def spec(self):
        return self._spec

    @property
    def signals(self):
        return self._signals

    @property
    def atoms(self):
        return self._atoms

    def __call__(self, w, t=None, dt=1.0):
        """
        Compute the robustness of the given trace containing the specified signals

        :param w: The trace of the systems containing the signals specified above with dimensions (len(t), len(signals))
        :param t: The timestamps corresponding to the sample points in the trace
        :param dt: the minimum time difference to resample the trace and the dt. min(actual_dt, dt) will be used
        :return: Robustness signal of `w` corresponding to the timestamps `t`
        """
        self._reset()
        if w.ndim == 1:
            w = np.reshape(w, (len(w), 1))
        if len(self.signals) != w.shape[1]:
            raise ValueError(
                "Expected shape of w to be (n_samples, {}, ...), got {}".format(
                    len(self.signals), w.shape
                )
            )
        if t is None:
            t = np.arange(w.shape[0])

        orig_t = t
        ediff = np.ediff1d(t)
        if len(ediff) == 0:
            ediff = [1]
        min_dt = min(dt, np.amin(ediff))
        n_samples = int(np.ceil((t[-1] - t[0]) / min_dt) + 1)
        t, dt = np.linspace(t[0], t[-1], num=n_samples, retstep=True)
        max_t, min_t = t[-1], t[0]
        t = np.concatenate((t, t[-1:]))
        w = np.concatenate((w, w[-1:]))
        trace = interp1d(t, w, axis=0)
        w = trace(t)

        self.atom_functions = dict(
            zip(self._atoms, [_get_atom_fn(self.signals, atom) for atom in self._atoms])
        )

        y_signal = self.robustness_signal(self.spec, w)
        y = interp1d(t, y_signal, axis=0)
        return y(orig_t)

    def robustness_signal(self, phi, w):
        """
        Given a spec an
        :param phi:
        :param w:
        :return:
        """
        if isinstance(phi, signal_tl.Atom):
            fn = self.atom_functions[phi]
            return np.array([fn(*s) for s in w])

        if isinstance(phi, signal_tl.Not):
            return compute_not(self.robustness_signal(phi.args[0], w))

        if isinstance(phi, (signal_tl.And, signal_tl.Or)):
            y_signals = np.transpose(
                np.array([self.robustness_signal(arg, w) for arg in phi.args])
            )
            if isinstance(phi, signal_tl.And):
                return compute_and(y_signals)
            if isinstance(phi, signal_tl.Or):
                return compute_or(y_signals)

        if isinstance(phi, (signal_tl.Eventually, signal_tl.Always)):
            y = self.robustness_signal(phi.args[0], w)
            if isinstance(phi, signal_tl.Eventually):
                return compute_ev(y, phi.interval)
            if isinstance(phi, signal_tl.Always):
                return compute_alw(y, phi.interval)

        if isinstance(phi, signal_tl.Until):
            y1 = self.robustness_signal(phi.args[0], w)
            y2 = self.robustness_signal(phi.args[1], w)
            return compute_until(y1, y2, phi.interval)

        return np.full(len(w), fill_value=-np.inf)


@njit(parallel=True)
def compute_not(y):
    return -1 * y


@njit(parallel=True)
def compute_or(y_signals):
    return np.amax(y_signals, axis=1)


@njit(parallel=True)
def compute_or_binary(x, y):
    return np.maximum(x, y)


@njit(parallel=True)
def compute_and(y_signals):
    return np.amin(y_signals, axis=1)


@njit(parallel=True)
def compute_and_binary(x, y):
    return np.minimum(x, y)


@njit(parallel=True)
def compute_ev(y, interval):
    a, b = interval
    if a > 0:
        y = shift(y, -a, mode="nearest")
    if b - a <= 0:
        return y
    elif b - a >= len(y):
        return _compute_eventually(y)
    else:
        return _compute_bounded_eventually(y, b - a)


@njit(parallel=True)
def _compute_eventually(y):
    z = np.full_like(y, BOTTOM)

    y = np.append(y, y[-1])
    dy = np.gradient(y)

    z_max = BOTTOM
    for i in reversed(range(len(y) - 1)):
        if dy[i] >= 0:
            z[i] = max(y[i + 1], z_max)
        elif y[i + 1] >= z_max:
            z[i] = y[i]
        elif z_max >= y[i]:
            z[i] = z_max
        else:
            z[i] = y[i]
            # TODO(anand): There is an intermediate value. But I don't see how this is needed for the discrete case
        z_max = z[i]
    return z


@njit(parallel=True)
def _compute_bounded_eventually(x, a):
    z1 = maximum_filter1d(x, a, mode="nearest")
    z2 = shift(x, -a, cval=BOTTOM)
    z3 = compute_or_binary(z2, z1)
    z = compute_or_binary(x, z3)
    return z


@njit(parallel=True)
def compute_alw(y, interval):
    return -1 * compute_ev(-1 * y, interval)


@njit(parallel=True)
def _compute_bounded_globally(x, a):
    z1 = minimum_filter1d(x, a, mode="nearest")
    z2 = shift(x, -a, cval=TOP)
    z3 = compute_and_binary(z2, z1)
    z = compute_and_binary(x, z3)
    return z


@njit()
def compute_until(x, y, interval):
    a, b = interval
    if np.isinf(b):
        if a == 0:
            return _compute_unbounded_until(x, y)
        else:
            yalw1 = _compute_bounded_globally(x, a)
            ytmp = shift(_compute_unbounded_until(x, y), -a, mode="nearest")
            return compute_and_binary(yalw1, ytmp)
    else:
        z2 = _compute_bounded_eventually(y, b - a)
        z3 = _compute_unbounded_until(x, y)
        z4 = compute_and_binary(z2, z3)
        if a > 0:
            z1 = _compute_bounded_globally(x, a)
            z4 = shift(z4, -a, mode="nearest")
            return compute_and_binary(z1, z4)
        else:
            return compute_and_binary(x, z4)


@njit()
def _compute_unbounded_until(x, y):
    z = np.full_like(x, BOTTOM)

    x = np.append(x, x[-1])
    y = np.append(y, y[-1])
    dx = np.gradient(x)

    z0 = np.array([BOTTOM] * 2)

    for i in reversed(range(len(x) - 1)):
        seg = [i, i + 1]
        con = [i, i]
        if dx[i] <= 0:
            z1 = _compute_eventually(y[seg])
            z2 = compute_and_binary(z1, x[seg])
            z3 = compute_and_binary(x[con + 1], z0)
            z[i] = compute_or_binary(z2, z3)[0]
        else:
            z1 = compute_and_binary(y[seg], x[seg])
            z2 = _compute_eventually(z1)
            z3 = compute_and_binary(x[seg], z0)
            z[i] = compute_or_binary(z2, z3)[0]
        z0 = z[con]
    return z
