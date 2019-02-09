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

Point = namedtuple('Point', ('value', 'time'))
Sample = namedtuple('Sample', ('value', 'time', 'derivative'))


def efficient_robustness(phi: stl.Expression, w: pd.DataFrame) -> pd.Series:
    """
    Compute the robustness of the given trace `w` against a STL property `phi`.

    :param phi: STL property defined on the trace
    :type phi: stl.Expression
    :param w: A signal trace of a system. It must be convertible to a Pandas DataFrame where the names of the columns correspond to the signals defined in the STL Expression.
    :type w: pd.DataFrame
    :return: The robustness signal for the given trace
    :rtype: pd.Series
    """
    w = pd.DataFrame(w)

    signals = set()
    for atom in stl.get_atoms(phi):
        if isinstance(atom, stl.Predicate):
            signals.update(atom.signals)

    assert signals.issubset(w.columns), 'Signals: {} not subset of Columns:{}'.format(signals, w.columns)

    if isinstance(phi, stl.Atom):
        if isinstance(phi, stl.TLTrue):
            return pd.Series(TOP, index=w.index)
        if isinstance(phi, stl.TLFalse):
            return pd.Series(BOTTOM, index=w.index)
        if isinstance(phi, stl.Predicate):
            return phi.f(w)

    # Unary ops
    elif isinstance(phi, (stl.Not, stl.Eventually, stl.Always)):
        y = efficient_robustness(phi.args[0], w)
        if isinstance(phi, stl.Not):
            return compute_not(y)
        if isinstance(phi, stl.Eventually):
            return compute_eventually(y, phi.interval)
        if isinstance(phi, stl.Always):
            return compute_globally(y, phi.interval)

    # Binary ops
    elif isinstance(phi, stl.Until):
        y1 = efficient_robustness(phi.args[0], w)
        y2 = efficient_robustness(phi.args[1], w)
        return compute_until(y1, y2, phi.interval)

    # N-ary ops
    elif isinstance(phi, (stl.Or, stl.And)):
        y_signals = pd.concat(
            [efficient_robustness(op, w) for op in phi.args],
            axis=1,
        )  # Create a dataframe of signals
        if isinstance(phi, stl.Or):
            return compute_or(y_signals)
        if isinstance(phi, stl.And):
            return compute_and(y_signals)
    else:
        raise ValueError('phi ({}) is not a supported STL operation'.format(phi.func))


def compute_not(y: pd.Series) -> pd.Series:
    return -y


def compute_or(y_signals: pd.DataFrame) -> pd.Series:
    return y_signals.max(axis=1)


def compute_or_binary(x: pd.Series, y: pd.Series):
    return pd.concat([x, y], axis=1).interpolate(limit_area='inside').max(axis=1)


def compute_and(y_signals: pd.DataFrame) -> pd.Series:
    return y_signals.min(axis=1)


def compute_and_binary(x: pd.Series, y: pd.Series) -> pd.Series:
    return pd.concat([x, y], axis=1).min(axis=1)


def compute_eventually(signal: pd.Series, interval: stl.Interval = stl.Interval(0, np.inf)) -> pd.Series:
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
        return bounded(signal, b - a)


def compute_globally(y: pd.Series, interval: stl.Interval = stl.Interval(0, np.inf)) -> pd.Series:
    return -compute_eventually(-y, interval)


def compute_until(signal1: pd.Series, signal2: pd.Series, interval: stl.Interval) -> pd.Series:


    def _unbounded(x: pd.Series, y: pd.Series):
        z_max = BOTTOM
        z = pd.Series(index=x.index.intersection(y))

        x[x.idxmax() + 1] = x.iloc[-1]
        y[y.idxmax() + 1] = x.iloc[-1]
        tx = x.index
        ty = y.index
        t =
        # i = x iterator
        # j = y iterator
        # s = z begin, t = z end

        for i in reversed(range(len(x) - 1)):
            seg =
            if x[i] >= x[i + 1]:  # dx <= 0
                z1 = compute_eventually(y[ty[i]: ty[i + 1] + 0.01])
                z2 = compute_and_binary(z1, x[tx[i]: tx[i + 1] + 0.01])
                z3 = compute_and_binary(z0, const_x)

    def _timed_until(x: pd.Series, y: pd.Series, interval: stl.Interval) -> pd.Series:
        z2 = compute_eventually(y, interval)
        z3 = _unbounded(x, y)
        z4 = compute_and_binary(z2, z3)

        a, b = interval
        if a > 0:
            z1 = _bounded_globally(x, a)
            z4 -= a
            return compute_and_binary(
                z2, z3
            )
        else:
            return compute_and_binary(
                x, z4
            )

    def _segment_until(y: pd.Series, x: Sample, t: float, z_max):
        """
        For the current sample of x and y[]
        """
        pass

    def _segment_or(y: pd.Series, sample: Sample, t: float, out: pd.Series):
        pass

    def _segment_and(y: pd.Series, sample: Sample, t: float, out: pd.Series):
        pass

    def _bounded_globally(x: pd.Series, window: int or float):
        z2 = x.copy()
        z2.index -= window
        z3 = compute_and_binary(
            z2,
            plateau_maxmin(x, window, 'min')
        )
        return compute_and_binary(x, z3)

    a, b = interval
    if np.isinf(b):
        if a == 0:
            return _unbounded(signal1, signal2)
        else:
            yalw1 = _bounded_globally(signal1, a)
            yuntmp = _unbounded(signal1, signal2)
            yuntmp.index -= a
            return compute_and_binary(yalw1, yuntmp)
    else:
        return _timed_until(signal1, signal2, interval)



def plateau_maxmin(x: pd.Series, a: int or float, fn='max') -> pd.Series:
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


def _compute_partial_eventually(signal: pd.Series, s, t, z: pd.Series = None) -> pd.Series:
    if z is None:
        z = pd.Series()
    continued = False
    z_max = BOTTOM
    y = signal.reindex(signal.index.union([s, t])).interpolate('values', limit_direction='both')

    dy = -y.diff(-1).fillna(0)[s:t]  # Compute dy = y_{i+1} - y_i. Pandas does the opposite calculation...
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
    return z

def _compute_segment_and(signal1: pd.Series, signal2: pd.Series, begin, end, z: pd.Series = None) -> pd.Series:
    if z is None:
        z = pd.Series()
    continued = False

    y = signal1.reindex(signal1.index.union([begin, end])).interpolate('values', limit_direction='both')
    y = y[begin:end]

    x = signal2.reindex(signal2.index.union([begin, end])).interpolate('values', limit_direction='both')
    x = x[begin:end]

    i = Sample(signal2[begin], begin, 0)

    for s, v_y in reversed(tuple(y.iteritems())):
        if x[t] < y[t]:
            if x[t_y] > v_y:


