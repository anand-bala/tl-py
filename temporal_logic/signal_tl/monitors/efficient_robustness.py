"""
Efficient robust monitoring for STL.

[1] A. Donzé, T. Ferrère, and O. Maler, “Efficient Robust Monitoring for STL,”
    in Computer Aided Verification, 2013, pp. 264–279.
"""
from collections import deque, namedtuple

import numpy as np
import pandas as pd

import temporal_logic.signal_tl as stl

# TODO: Can use Numba to jit compile most of these


BOTTOM = -np.inf
TOP = np.inf


def compute_not(y: pd.Series) -> pd.Series:
    return -1 * y


def compute_or(y_signals: pd.DataFrame) -> pd.Series:
    return y_signals.max(axis=1)


def compute_or_binary(x: pd.Series, y: pd.Series):
    return pd.concat([x, y], axis=1).interpolate(limit_area='inside').max(axis=1)


def compute_and(y_signals: pd.DataFrame) -> pd.Series:
    return y_signals.min(axis=1)


def compute_and_binary(x: pd.Series, y: pd.Series) -> pd.Series:
    return pd.concat([x, y], axis=1).min(axis=1)


def compute_eventually(signal: pd.Series, interval: stl.Interval) -> pd.Series:
    a, b = interval

    if a > 0:
        signal.index -= a

    def bounded(y: pd.Series, window) -> pd.Series:
        z2 = y.copy()
        z2.index -= window
        z3 = compute_or_binary(
            z2,
            plateau_maxmin(y, window, 'max')
        )
        return compute_or_binary(y, z3).reindex(y.index)

    def unbounded(y: pd.Series) -> pd.Series:
        return y.iloc[::-1].expanding(1).max().iloc[::-1]

    if b - a == 0:
        return signal
    elif b - a >= signal.index[-1] - signal.index[0]:
        return unbounded(signal)
    else:
        return bounded(signal, b-a)


def compute_globally(y: pd.Series, interval: stl.Interval) -> pd.Series:
    return -1 * compute_eventually(-1 * y, interval)


def compute_until(signal: pd.Series, interval: stl.Interval) -> pd.Series:

    def _bounded_globally(x: pd.Series, window):
        pass


def efficient_robustness(phi: stl.Expression, w: pd.DataFrame):
    pass


def plateau_maxmin(x: pd.Series, a: int or float, fn='max') -> np.ndarray:
    """
    :param x: 1-D array with data points
    :param a: size of interval (can be float timestamp diff)
    """
    Point = namedtuple('Point', ('value', 'time'))
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
