from abc import ABC, abstractmethod
from typing import Dict, Sequence, Union, Optional

from temporal_logic.types import SignalType, TraceType, NumericType


class BaseMonitor(ABC):
    @abstractmethod
    def __call__(self, w: TraceType, t: Optional[SignalType] = None, dt: Optional[NumericType] = None) -> SignalType:
        pass

    @property
    @abstractmethod
    def horizon(self) -> NumericType:
        pass
