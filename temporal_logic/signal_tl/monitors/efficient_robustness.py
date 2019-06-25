"""
Efficient robust monitoring for STL.

It's mostly a verbatim port of the `robusthom` code from the Breach source. 

[1] A. Donzé, T. Ferrère, and O. Maler, “Efficient Robust Monitoring for STL,”
    in Computer Aided Verification, 2013, pp. 264–279.
"""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from collections import deque, namedtuple
from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd

import temporal_logic.signal_tl as stl

# TODO: Can use Numba to jit compile most of these


BOTTOM = -np.inf
TOP = np.inf

Point = namedtuple('Point', ('value', 'time'))
Sample = namedtuple('Sample', ('value', 'time', 'derivative'))


def efficient_robustness(phi, w, t=None):
    # type: stl.Expression, pd.DataFrame, Optional[Union[List, Tuple, np.ndarray]] -> pd.Series
    """
    Compute the robustness of the given trace `w` against a STL property `phi`.

    :param phi: STL property defined on the trace
    :type phi: stl.Expression
    :param w: A signal trace of a system. It must be convertible to a Pandas DataFrame where the names of the columns correspond to the signals defined in the STL Expression.
    :type w: pd.DataFrame
    :param t: List of time points to get the robustness from (default: all points in `w`)
    :type t: Optional[Union[List, Tuple, np.ndarray]]
    :return: The robustness signal for the given trace
    :rtype: pd.Series
    """
    w = pd.DataFrame(w)

    signals = set()
    for atom in stl.get_atoms(phi):
        if isinstance(atom, stl.Predicate):
            signals.update(atom.signals)

    assert signals.issubset(
        w.columns), 'Signals: {} not subset of Columns:{}'.format(signals, w.columns)

    z = pd.Series()

    if isinstance(phi, stl.Atom):
        if isinstance(phi, stl.TLTrue):
            z = pd.Series(TOP, index=w.index)
        elif isinstance(phi, stl.TLFalse):
            z = pd.Series(BOTTOM, index=w.index)
        elif isinstance(phi, stl.Predicate):
            z = phi.f(w)

    # Unary ops
    elif isinstance(phi, (stl.Not, stl.Eventually, stl.Always)):
        y = efficient_robustness(phi.args[0], w)
        if isinstance(phi, stl.Not):
            z = compute_not(y)
        elif isinstance(phi, stl.Eventually):
            z = compute_eventually(y, phi.interval)
        elif isinstance(phi, stl.Always):
            z = compute_globally(y, phi.interval)

    # Binary ops
    elif isinstance(phi, stl.Until):
        y1 = efficient_robustness(phi.args[0], w)
        y2 = efficient_robustness(phi.args[1], w)
        z = compute_until(y1, y2, phi.interval)

    # N-ary ops
    elif isinstance(phi, (stl.Or, stl.And)):
        y_signals = pd.concat(
            [efficient_robustness(op, w) for op in phi.args],
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
    if t is not None and isinstance(t, (list, tuple, np.ndarray)):
        z = z.reindex(z.index.union(t)).interpolate(
            'values', limit_direction='both')
        return z[t]
    return z


def compute_not(y):
    return -y


def compute_or(y_signals):
    return y_signals.max(axis=1)


def compute_or_binary(x, y):
    return pd.concat([x, y], axis=1).interpolate('values', limit_area='inside').max(axis=1)


def compute_and(y_signals):
    return y_signals.min(axis=1)


def compute_and_binary(x, y):
    return pd.concat([x, y], axis=1).min(axis=1)


def compute_eventually(signal, interval = stl.Interval(0, np.inf)):
    a, b = interval

    if a > 0:
        signal.index -= a

    if b - a == 0:
        return signal
    elif b - a >= signal.index[-1] - signal.index[0]:
        return _unbounded_eventually(signal)
    else:
        return _bounded_eventually(signal, b - a)


def _bounded_eventually(y, window):
    z2 = y.copy()
    z2.index -= window
    z3 = compute_or_binary(
        z2,
        plateau_maxmin(y, window, 'max')
    )
    return compute_or_binary(y, z3).reindex(y.index)


def _unbounded_eventually(y):
    return y.iloc[::-1].expanding(1).max().iloc[::-1]


def compute_globally(y, interval = stl.Interval(0, np.inf)):
    return -compute_eventually(-y, interval)


def compute_until(signal1, signal2, interval):
    a, b = interval
    if np.isinf(b):
        if a == 0:
            return _unbounded_until(signal1, signal2)
        else:
            yalw1 = _bounded_globally(signal1, a)
            yuntmp = _unbounded_until(signal1, signal2)
            yuntmp.index -= a
            return compute_and_binary(yalw1, yuntmp)
    else:
        return _timed_until(signal1, signal2, interval)


def plateau_maxmin(x, a, fn='max'):
    """
    :param x: 1-D array with data points
    :param a: size of interval (can be float timestamp diff)
    :param fn: 'max' or 'min'
    """
    M = deque()

    # Get local maximums/minimums
    if fn == 'max':
        y = x.rolling(2).max()
    elif fn == 'min':
        y = x.rolling(2).min()
    else:
        raise ValueError("fn=('min'|'max'), got {}".format(fn))
    y[0] = x[0]

    # Read values in [0, a)
    threshold = y.idxmin() + a
    i = 0
    for t, val in y.iteritems():
        if t < threshold:
            break
        while len(M) != 0 and val <= M[-1]:
            M.pop()
        M.append(Point(val, t))
        i += 1

    new_candidate = (y.index[i] == threshold)
    end_candidate = False
    t = x.idxmin()
    cont = True

    z = x.copy()

    while cont:
        # Update candidate list
        if end_candidate:
            M.popleft()
        if new_candidate:
            while len(M) != 0 and y.iloc[i] <= M[-1].value:
                M.pop()
            if len(M) != 0:
                new_candidate = False
            M.append(Point(y.iloc[i], y.index[i]))
            i += 1

        # Output new minimum
        if len(M) == 0:
            z[t] = TOP
        else:
            z[t] = M[0].value

        # Next Event Detection
        if len(M) != 0:
            if i < y.size:
                if y.index[i] - a == M[0].time:
                    t = M[0].time
                    new_candidate, end_candidate = True, True
                elif y.index[i] - a < M[0].time:
                    t = y.index[i] - a
                    new_candidate, end_candidate = True, False
                else:
                    t = M[0].time
                    new_candidate, end_candidate = False, True
            else:
                t = M[0].time
                new_candidate, end_candidate = False, True
        else:
            if i < y.size:
                t = y.index[i] - a
                new_candidate, end_candidate = True, False
            else:
                new_candidate, end_candidate = False, False
        cont = new_candidate or end_candidate

    return z.reindex(x.index)


def _unbounded_until(signal1, signal2):
    z_max = BOTTOM
    z = pd.Series()

    s = max(signal1.idxmin, signal2.idxmin)
    t = min(signal1.idxmax, signal2.idxmax)

    x = signal1[s:t]
    y = signal2[s:t]

    for tau in reversed(x.index):
        _segment_until(x, y, tau, t, z_max, z)
        z_max = z.min()
        t = tau
    return z


def _timed_until(x, y, interval):
    a, b = interval

    z2 = _bounded_eventually(y, b-a)
    z3 = _unbounded_until(x, y)
    z4 = compute_and_binary(z2, z3)
    if a > 0:
        z1 = _bounded_globally(x, a)
        z4 -= a
        return compute_and_binary(
            z1, z4
        )
    else:
        return compute_and_binary(
            x, z4
        )


def _segment_until(signal1, signal2, s, t, z_max, out = None):

    z = pd.Series()
    x = signal1.reindex(signal1.index.union([s, t])).interpolate(
        'values', limit_direction='both')
    dx = -signal1.diff(-1).fillna(0)

    y = signal2.reindex(signal2.index.union([s, t])).interpolate(
        'values', limit_direction='both')
    dy = -signal2.diff(-1).fillna(0)

    if dx[s] <= 0:
        z1 = _compute_segment_and(x, y, s, t)
        z2 = _compute_partial_eventually(z1, s, t)

        i = pd.Series()
        i[s] = min(z_max, x[t])
        z = _compute_segment_or(i, z2, s, t)
    else:
        z1 = _compute_partial_eventually(y, s, t)
        z2 = _compute_segment_and(x, z1, s, t)
        z3 = pd.Series()
        z3[s] = z_max
        z1 = _compute_segment_and(x, z3, s, t)
        z = _compute_segment_or(z1, z2, s, t)

    z.sort_index(inplace=True)
    if out is not None and isinstance(out, pd.Series):
        out.update(z)
    return z


def _bounded_globally(x, window):
    z2 = x.copy()
    z2.index -= window
    z3 = compute_and_binary(
        z2,
        plateau_maxmin(x, window, 'min')
    )
    return compute_and_binary(x, z3)


def _partial_or(x, y, s, t, out = None):
    z = pd.Series()
    x = x.reindex(x.index.union([s, t])).interpolate(
        'values', limit_direction='both')
    y = y.reindex(y.index.union([s, t])).interpolate(
        'values', limit_direction='both')
    for tau in reversed(x.index):
        _compute_segment_or(x, y, tau, t, z)
        t = tau
    z.sort_index(inplace=True)
    if out is not None and isinstance(out, pd.Series):
        out.update(z)
    return z


def _partial_and(x, y, s, t, out = None):
    z = pd.Series()
    x = x.reindex(x.index.union([s, t])).interpolate(
        'values', limit_direction='both')
    y = y.reindex(y.index.union([s, t])).interpolate(
        'values', limit_direction='both')
    for tau in reversed(x.index):
        _compute_segment_and(x, y, tau, t, z)
        t = tau
    z.sort_index(inplace=True)
    if out is not None and isinstance(out, pd.Series):
        out.update(z)
    return z


def _compute_partial_eventually(signal, s, t, out = None):
    z = pd.Series()
    continued = False
    z_max = BOTTOM
    y = signal.reindex(signal.index.union([s, t])).interpolate(
        'values', limit_direction='both')

    # Compute dy = y_{i+1} - y_i. Pandas does the opposite calculation...
    dy = -y.diff(-1).fillna(0)[s:t]
    y = y[s:t]

    for idx, (t_i, v) in reversed(tuple(enumerate(y.iteritems()))):
        if dy[t_i] >= 0:
            if z_max < y[t]:
                if continued:
                    z[t] = z_max
                z_max = y[t]
            continued = True
        elif y[t] >= z_max:
            if continued:
                z[t] = z_max
                continued = False
            z_max = v
            z[t_i] = v
        elif z_max >= v:
            continued = True
        else:
            cross_time = t_i + (z_max - v) / dy[t_i]
            z[cross_time] = z_max
            z[t_i] = v
            z_max = v
            continued = False

        t = t_i

    z.sort_index(inplace=True)
    if out is not None and isinstance(out, pd.Series):
        out.update(z)
    return z


def _compute_segment_and(signal1, signal2, begin, end, out = None):
    z = pd.Series()

    continued = False

    s, t = begin, end

    x = signal1.reindex(signal1.index.union([s, t])).interpolate(
        'values', limit_direction='both')
    x = x[s:t]
    dx = -x.diff(-1).fillna(0)[s:t]

    y = signal2.reindex(signal2.index.union([s, t])).interpolate(
        'values', limit_direction='both')
    y = y[s:t]
    dy = -y.diff(-1).fillna(0)[s:t]

    for tau in reversed(y.index):
        if x[t] < y[t]:
            if x[tau] > y[tau]:
                tau_star = time_intersect(
                    Sample(x[s], s, dx[s]),
                    Sample(t[tau], tau, dy[tau])
                )
                z[tau_star] = x[tau_star]
                z[tau] = y[tau]
                continued = False
            else:
                continued = True
        elif x[t] == y[t]:
            if x[tau] > y[tau]:
                if continued:
                    z[t] = x[t]
                    continued = False
                z[tau] = y[tau]
            else:
                continued = True
        else:
            # TODO: This block is suspect
            if x[tau] < y[tau]:
                if continued:
                    z[t] = x[t]
                tau_star = time_intersect(
                    Sample(x[s], s, dx[s]),
                    Sample(y[t], t, dy[t])
                )
                z[tau_star] = y[tau_star]
                continued = True
            else:
                if continued:
                    z[t] = x[t]
                    continued = False
                z[tau] = y[tau]
        t = tau
    z.sort_index(inplace=True)
    if out is not None and isinstance(out, pd.Series):
        out.update(z)
    return z


def _compute_segment_or(signal1, signal2, begin, end, out = None):
    z = pd.Series()

    continued = False

    s, t = begin, end

    x = signal1.reindex(signal1.index.union([s, t])).interpolate(
        'values', limit_direction='both')
    x = x[s:t]
    dx = -x.diff(-1).fillna(0)[s:t]

    y = signal2.reindex(signal2.index.union([s, t])).interpolate(
        'values', limit_direction='both')
    y = y[s:t]
    dy = -y.diff(-1).fillna(0)[s:t]

    for tau in reversed(y.index):
        if x[t] > y[t]:
            if x[tau] < y[tau]:
                tau_star = time_intersect(
                    Sample(x[s], s, dx[s]),
                    Sample(t[tau], tau, dy[tau])
                )
                z[tau_star] = x[tau_star]
                z[tau] = y[tau]
                continued = False
            else:
                continued = True
        elif x[t] == y[t]:
            if x[tau] < y[tau]:
                if continued:
                    z[t] = x[t]
                    continued = False
                z[tau] = y[tau]
            else:
                continued = True
        else:
            # TODO: This block is suspect
            if x[tau] > y[tau]:
                if continued:
                    z[t] = x[t]
                tau_star = time_intersect(
                    Sample(x[s], s, dx[s]),
                    Sample(y[t], t, dy[t])
                )
                z[tau_star] = y[tau_star]
                continued = True
            else:
                if continued:
                    z[t] = x[t]
                    continued = False
                z[tau] = y[tau]
        t = tau
    z.sort_index(inplace=True)

    if out is not None and isinstance(out, pd.Series):
        out.update(z)
    return z


def time_intersect(x, y):
    """
    Intersection of two lines given a point slope form
    """
    return (x.value - y.value + (y.derivative * y.time) - (x.derivative * x.time)) / (y.derivative - x.derivative)
