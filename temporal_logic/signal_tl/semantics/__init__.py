import numpy as np
from .base import BaseMonitor
from .efficient_robustness import EfficientRobustnessMonitor
from .online_monitoring import OnlineRobustness
from .filtering import FilteringMonitor


def check_sat(phi, signals, trace, t=None, dt=np.inf):
    """A wrapper that checks sat of a trace given a phi

    Internally uses the EfficientRobustnessMonitor
    """

    monitor = EfficientRobustnessMonitor(phi, signals)
    return monitor(trace, t, dt)
