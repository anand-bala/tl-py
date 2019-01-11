from abc import ABC, abstractmethod


class BaseMonitor(ABC):

    @abstractmethod
    def __call__(self, w, t, dt):
        pass

    @property
    @abstractmethod
    def horizon(self):
        pass
