from enum import Enum
from collections import namedtuple


Bbox = namedtuple("Bbox", ["xmin", "ymin", "xmax", "ymax"])


class Inconsistency(Enum):
    BBOX = 0
    TEXT = 1
    COLOR = 2
