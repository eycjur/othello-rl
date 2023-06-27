from enum import Enum, IntEnum, auto

from src.utils.dataclass import Place


class Stone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


class Direction(Enum):
    UP = Place(y=-1, x=0)
    DOWN = Place(y=1, x=0)
    LEFT = Place(y=0, x=-1)
    RIGHT = Place(y=0, x=1)
    UP_LEFT = Place(y=-1, x=-1)
    UP_RIGHT = Place(y=-1, x=1)
    DOWN_LEFT = Place(y=1, x=-1)
    DOWN_RIGHT = Place(y=1, x=1)


class Result(Enum):
    WIN = auto()
    LOSE = auto()
    DRAW = auto()
