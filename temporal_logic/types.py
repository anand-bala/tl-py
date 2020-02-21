from typing import Dict, Sequence, Union, Optional, TypeVar


NumericType = TypeVar('NumericType', int, float)
SignalType = TypeVar('SignalType', Sequence[NumericType])
TraceType = TypeVar('TraceType', Dict[str, SignalType])


__all__ = [
    "NumericType",
    "SignalType",
    "TraceType",
]
