from dataclasses import dataclass
from typing import Any


@dataclass
class Place:
    y: int
    x: int

    def __add__(self, other: "Place") -> "Place":
        return Place(y=self.y + other.y, x=self.x + other.x)

    def __mul__(self, other: int) -> "Place":
        return Place(y=self.y * other, x=self.x * other)

    def __rmul__(self, other: int) -> "Place":
        return Place(y=self.y * other, x=self.x * other)

    def __hash__(self, *args: Any, **kwargs: Any) -> int:
        return hash((self.y, self.x))


class Pass:
    pass
