"""Robust Online Monitoring

Online monitoring of robustness as defined in [1].

[1] J. V. Deshmukh, A. Donzé, S. Ghosh, X. Jin, G. Juniwal, and S. A. Seshia,
    “Robust online monitoring of signal temporal logic,”
    Form Methods Syst Des, vol. 51, no. 1, pp. 5–30, Aug. 2017.

"""

import sympy
import numpy as np

import temporal_logic.signal_tl as stl

BOTTOM = -np.inf
TOP = np.inf


def TOP_FN(_): return np.inf


def BOTTOM_FN(_): return -np.inf


def _get_atom_fn(inputs, expr):
    if isinstance(expr, stl.TLTrue):
        return TOP_FN
    if isinstance(expr, stl.TLFalse):
        return BOTTOM_FN
    if isinstance(expr, stl.Predicate):
        return sympy.lambdify(inputs, expr.expr)
    raise TypeError(
        'Invalid input type: must be of type signal_tl.Atom, got {}'.format(type(expr)))


def minkowski_sum(i1, i2):
    a1, b1 = i1
    a2, b2 = i2
    return a1 + a2, b1 + b2


class OnlineRobustness:

    def __init__(self, signals, spec):
        self._signals = signals
        self._spec = spec

        self._reset()

    def _reset(self):
        self._atoms = frozenset(stl.get_atoms(self.spec))

        self.atom_functions = dict(
            zip(
                self._atoms,
                [_get_atom_fn(self._signals, atom) for atom in self._atoms]
            )
        )

        self.horizons = self._get_horizons(self.spec)
        self.worklist = self._init_worklist(self.spec)

    @classmethod
    def _get_horizons(cls, spec):
        """
        Get the horizons of each node in the expression tree, as defined in [1].

        :param spec: STL specification to get the horizon of
        :type spec: stl.Expression
        :return:
        """
        horizons = dict()
        for node in stl.preorder_iterator(spec):
            if node.parent is None:
                horizons[node] = (0, 0)
            elif isinstance(node.parent, temporal_rl.temporal_logic.signal_tl.grammar.TemporalOp):
                parent_horizon = horizons[node.parent]
                horizons[node] = minkowski_sum(node.interval, parent_horizon)
            else:
                horizons[node] = horizons[node.parent]

        return horizons

    @classmethod
    def _init_worklist(cls, spec):
        worklist = dict()
        for node in stl.preorder_iterator(spec):
            # TODO(anand): May need to optimize
            # TODO(anand): Check the initial capacity of the array
            worklist[node] = np.zeros((2, 20))

        return worklist

    def update_worklist(self, phi, t, x):
        if isinstance(phi, stl.Atom):
            hor = self.horizons[phi]
            if hor[0] <= t <= hor[1]:
                f = self.atom_functions[phi](*x)
                self._add_to_worklist(phi, t, [f, f])
            return
        if isinstance(phi, stl.Not):
            self.update_worklist(phi.args[0], t, x)
            self.worklist[phi] = -1 * self.worklist[phi.args[0]]
            return
        if isinstance(phi, stl.And):
            for arg in phi.args:
                self.update_worklist(arg, t, x)
            self.worklist[phi] = np.maximum.reduce(
                [self.worklist[arg] for arg in phi.args])
            return
        if isinstance(phi, stl.Always):
            self.update_worklist(phi.args[0], t, x)

    def _add_to_worklist(self, phi, t, interval):
        if len(self.worklist[phi]) <= t:
            np.append(self.worklist[phi], np.zeros((2, 20)), axis=1)
        self.worklist[phi][:, t] = np.array(interval).reshape((2, 1))

    @property
    def spec(self):
        return self._spec

    @property
    def atoms(self):
        return self._atoms

    @property
    def signals(self):
        return self._signals
