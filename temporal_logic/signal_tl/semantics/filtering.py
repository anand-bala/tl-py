"""Implement robustness metric defined in [1].

In  this, basic signal filtering is used to define the robust satisfaction degree of a given trace.


[1] A. Rodionova, E. Bartocci, D. Nickovic, and R. Grosu, “Temporal Logic as Filtering,”
    Proceedings of the 19th International Conference on Hybrid Systems: Computation and Control - HSCC ’16, pp. 11–20, 2016.
"""
import numpy as np
import scipy
import scipy.signal
from scipy.interpolate import interp1d
from scipy.ndimage import shift
from scipy.ndimage.filters import minimum_filter1d

from temporal_logic import signal_tl
from temporal_logic.signal_tl import as_Expression


from .base import BaseMonitor

BOTTOM = 0.0
TOP = 1.0


def get_spec_interval(spec: signal_tl.Expression):
    """

    :param spec:
    :return:
    """
    root = spec
    if isinstance(root, signal_tl.grammar.TemporalOp):
        a, b = root.interval
        interval_width = abs(b - a) + 1
        return interval_width
    else:
        return 0


def _compute_or(y_signals):
    return np.amax(y_signals, axis=1)


def _compute_or_binary(x, y):
    return np.maximum(x, y)


def _compute_and(y_signals):
    return np.amin(y_signals, axis=1)


def _compute_and_binary(x, y):
    return np.minimum(x, y)


def _compute_ev(y, window):
    return scipy.convolve(y, window, mode='same')


def _compute_alw(y, interval):
    a, b = interval
    if a == b:
        return y
    y = shift(y, -a, mode='nearest')
    b = min(b, len(y))

    # compute offset to left end of window from center
    width = int(abs(b - a)) + 1
    center = width // 2

    return minimum_filter1d(y, b - a + 1, mode='nearest', origin=-center)


class FilteringMonitor(BaseMonitor):

    @property
    def horizon(self):
        print("WARNING: There is no well defined horizon for filtering semantics.")
        return np.inf

    def __init__(self, spec, signals, window='boxcar'):
        """
        Initialize the monitor for the given spec and window (used for the convolution semantics of Temporal Ops)

        :param spec: STL Specification to monitor
        :type spec: Expression
        :param signals: List of signal parameters. This must match the symbols used in the predicates
        :param window: Convolution window to use. Provide the first argument for function in
                       https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html#scipy.signal.windows.get_window
        """
        self._spec = as_Expression(spec)
        self._signals = signals
        self._window_fn = window

        self._atoms = frozenset(signal_tl.get_atoms(spec))
        self._reset()

    def _reset(self):
        self.atom_functions = dict(
            zip(self._atoms, [BOTTOM] * len(self._atoms)))
        self.atom_signals = dict(zip(self._atoms, [None] * len(self._atoms)))
        self.node_robustness_signals = dict()

    @property
    def spec(self):
        return self._spec

    @property
    def signals(self):
        return self._signals

    @property
    def atoms(self):
        return self._atoms

    @property
    def window_fn(self):
        return self._window_fn

    @window_fn.setter
    def window_fn(self, window):
        self._window_fn = window

    def window(self, interval_size):
        return scipy.signal.windows.get_window(self.window_fn, interval_size)

    def __call__(self, w, t=None, dt=np.inf):
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
                'Expected shape of w to be (n_samples, n_states, ...), got {}'.format(w.shape))
        if not t:
            t = np.arange(w.shape[0])

        orig_t = t
        min_dt = min(dt, np.amin(np.ediff1d(t)))
        n_samples = np.ceil((t[-1] - t[0]) / min_dt) + 1
        t, dt = np.linspace(t[0], t[-1], num=n_samples, retstep=True)
        max_t, min_t = t[-1], t[0]

        trace = interp1d(t, w, axis=0)
        w = trace(t)

        y_signal = self._robustness_signal(self.spec, w)
        y = interp1d(t, y_signal, axis=0)
        return y(orig_t)

    def _robustness_signal(self, phi, w):
        """
        This function returns the robustness signal of the given trace against phi. This is meant to be called
        recursively over the expression tree of the STL formula.

        :param phi: The STL property to compute robustness against
        :param w: The trace to monitor
        :return: Robustness signal of phi on w
        """

        if phi in self.node_robustness_signals:
            return self.node_robustness_signals[phi]
        if isinstance(phi, signal_tl.Atom):
            y_raw = np.array([phi.eval(self.signals, *s) for s in w])
            y = (y_raw > 0).astype(np.uint8)
            self.node_robustness_signals[phi] = y
            return y
        if isinstance(phi, signal_tl.Not):
            return 1 - self._robustness_signal(phi.args[0], w)
        if isinstance(phi, (signal_tl.And, signal_tl.Or)):
            y_signals = np.transpose(
                np.array([self._robustness_signal(arg, w) for arg in phi.args]))
            if isinstance(phi, signal_tl.And):
                z = _compute_and(y_signals)
                self.node_robustness_signals[phi] = z
                return z
            if isinstance(phi, signal_tl.Or):
                z = _compute_or(y_signals)
                self.node_robustness_signals[phi] = z
                return z
        if isinstance(phi, (signal_tl.Eventually, signal_tl.Always)):
            y = self._robustness_signal(phi.args[0], w)
            interval_width = get_spec_interval(phi)
            window_width = min(len(y), interval_width)
            window = self.window(window_width)
            if isinstance(phi, signal_tl.Eventually):
                z = _compute_ev(y, window)
                return z
            if isinstance(phi, signal_tl.Always):
                z = _compute_alw(y, phi.interval)
                return z
