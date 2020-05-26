import numpy as np

from .base import BaseMonitor
from .filtering import FilteringMonitor
from .online_monitoring import OnlineRobustness
from .efficient_robustness import EfficientRobustnessMonitor


def check_sat(phi, signals, trace, t=None, dt=np.inf):
    """A wrapper that checks sat of a trace given a phi

    Internally uses the EfficientRobustnessMonitor
    """

    monitor = EfficientRobustnessMonitor(phi, signals)
    return monitor(trace, t, dt)[0] >= 0
