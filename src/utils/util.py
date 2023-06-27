from typing import Optional

from src.utils.enums import Result, Stone


def change_stone(stone: Stone) -> Stone:
    if stone == Stone.BLACK:
        return Stone.WHITE
    elif stone == Stone.WHITE:
        return Stone.BLACK
    else:
        raise ValueError("Stone must be BLACK or WHITE")


class Turn:
    def __init__(self, stone: Optional[Stone] = None) -> None:
        if stone is None:
            self._stone = Stone.BLACK
        else:
            self._stone = stone

    def change_turn(self) -> "Turn":
        return Turn(change_stone(self._stone))

    @property
    def stone(self) -> Stone:
        return self._stone


def stone2mark(value: int) -> str:
    if value == Stone.EMPTY:
        return " "
    elif value == Stone.BLACK:
        return "●"
    elif value == Stone.WHITE:
        return "○"
    else:
        return str(value)


def result2reward(result: Result) -> int:
    if result == Result.WIN:
        return 1
    elif result == Result.LOSE:
        return -1
    else:
        return 0
