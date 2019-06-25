"""Implement robustness metric defined in [1].

In  this, basic signal filtering is used to define the robust satisfaction degree of a given trace.


[1] A. Rodionova, E. Bartocci, D. Nickovic, and R. Grosu, “Temporal Logic as Filtering,”
    Proceedings of the 19th International Conference on Hybrid Systems: Computation and Control - HSCC ’16, pp. 11–20, 2016.
"""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import int
from future import standard_library
standard_library.install_aliases()

from collections import deque, namedtuple
from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import convolve, get_window

import temporal_logic.signal_tl as stl

import sympy

BOTTOM = 0
TOP = 1


def lti_filter_monitor(phi, w, t=None, dt=-1.0, window='boxcar'):
    # type: stl.Expression, pd.DataFrame, Optional[Union[List, Tuple, np.ndarray]], Optional[float] -> pd.Series
    """
    Compute the robustness of `w` against STL property `phi` wrt filtering semantics of robustness.

    :param phi: STL property defined on the trace
    :type phi: stl.Expression
    :param w: A signal trace of a system. It must be convertible to a Pandas DataFrame where the names of the columns correspond to the signals defined in the STL Expression.
    :type w: pd.DataFrame
    :param t: List of time points to get the robustness from (default: all points in `w`)
    :type t: Optional[Union[List, Tuple, np.ndarray]]
    :param dt: Time delta to resample the trace at, assuming linear interpolation (default: resample with minimum dt in trace)
    :type dt: float
    :param window: First argument to `scipy.signal.get_window` function for filtering
    :type window: str or tuple or float
    :return: The robustness signal for the given trace
    :rtype: pd.Series
    """
    trace = pd.DataFrame(w)

    signals = set()
    for atom in stl.get_atoms(phi):
        if isinstance(atom, stl.Predicate):
            signals.update(atom.signals)

    assert signals.issubset(
        w.columns), 'Signals: {} not subset of Columns:{}'.format(signals, w.columns)

    if dt <= 0:
        dt = min(w.index.to_series().diff().min(), -
                 w.index.to_series().diff(-1).max())
    s, t = w.idxmin, w.idxmax

    w = trace \
        .combine_first(pd.Series(index=np.arange(s, t + dt, dt))) \
        .interpolate('values', limit_direction='both') \
        .reindex(np.arange(s, t + dt, dt))

    robust_signals = pd.DataFrame(index=w.index)

    for expr in stl.postorder_iterator(phi):
        col = sympy.latex(expr)
        if col in robust_signals.columns:
            continue
        robust_signals[col] = _robustness_signal(expr, w, dt, window)
    z = robust_signals[sympy.latex(phi)]
    if t is not None and isinstance(t, (list, tuple, np.ndarray)):
        z = z.reindex(z.index.union(t)).interpolate(
            'values', limit_direction='both')
        return z[t]
    return z \
        .reindex_like(trace) \
        .interpolate('values', limit_direction='both')


def _robustness_signal(phi, w, dt,  window):
    z = None

    if isinstance(phi, stl.Atom):
        if isinstance(phi, stl.TLTrue):
            z = pd.Series(TOP, index=w.index)
        elif isinstance(phi, stl.TLFalse):
            z = pd.Series(BOTTOM, index=w.index)
        elif isinstance(phi, stl.Predicate):
            z = phi.eval(w) + 0.0

    # Unary ops
    elif isinstance(phi, (stl.Not, stl.Eventually, stl.Always)):
        y = lti_filter_monitor(phi.args[0], w)
        if isinstance(phi, stl.Not):
            z = compute_not(y)
        elif isinstance(phi, stl.Eventually):
            z = compute_eventually(y, phi.interval, dt, window)
        elif isinstance(phi, stl.Always):
            z = compute_globally(y, phi.interval, dt)

    # Binary ops
    elif isinstance(phi, stl.Until):
        y1 = lti_filter_monitor(phi.args[0], w)
        y2 = lti_filter_monitor(phi.args[1], w)
        z = compute_until(y1, y2, phi.interval)

    # N-ary ops
    elif isinstance(phi, (stl.Or, stl.And)):
        y_signals = pd.concat(
            [lti_filter_monitor(op, w) for op in phi.args],
            axis=1,
        )  # Create a dataframe of signals
        if isinstance(phi, stl.Or):
            z = compute_or(y_signals)
        elif isinstance(phi, stl.And):
            z = compute_and(y_signals)
    else:
        raise ValueError(
            'phi ({}) is not a supported STL operation'.format(phi.func))
    z.sort_index(inplace=True)
    return z


def discrete_to_continuous(interval, dt):
    a, b = interval
    return stl.Interval(
        a * dt, b * dt,
    )


def continuous_to_discrete(interval, dt):
    a, b = interval
    return stl.Interval(
        a // dt, b // dt,
    )


def compute_not(y):
    return 1.0 - y


def compute_or(y_signals):
    return y_signals.max(axis=1)


def compute_or_binary(x, y):
    return pd.concat([x, y], axis=1).interpolate('values', limit_direction='both').max(axis=1)


def compute_and(y_signals):
    return y_signals.min(axis=1)


def compute_and_binary(x, y):
    return pd.concat([x, y], axis=1).interpolate('values', limit_direction='both').max(axis=1)


def compute_eventually(y, interval, dt, window):
    a, b = interval
    y = y.shift(-int(a // dt)).interpolate(limit_direction='both')

    window_samples = min(int(abs(b - a) // dt) + 1, len(y))
    if len(window_samples) == 1:
        return y
    win = get_window(window, window_samples)
    # Normalize the window (sum == 1.0)
    win = win / win.sum()
    z = pd.Series(
        convolve(y.values, win, mode='same'),
        index=y.index
    )
    return z


def compute_globally(y, interval, dt):
    a, b = interval
    y = y.shift(-int(a // dt)).interpolate(limit_direction='both')

    win_samples = min(int(abs(b - a) // dt) + 1, len(y))
    if len(win_samples) == 1:
        return y

    if not interval.right_unbounded:
        return y.iloc[::-1] \
            .rolling(win_samples, min_periods=1).min().iloc[::-1]
    else:
        return y.iloc[::-1].expanding(1).min().iloc[::-1]


def compute_until(x, y, interval, dt):
    raise NotImplementedError('Robustness of the until operator has not been implemented')
