from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from abc import ABC, abstractmethod


class BaseMonitor(ABC):

    @abstractmethod
    def __call__(self, w, t, dt):
        pass

    @property
    @abstractmethod
    def horizon(self):
        pass
