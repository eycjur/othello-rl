import copy
from typing import Generator

import numpy as np

from src.utils.dataclass import Pass, Place
from src.utils.enums import Direction, Stone
from src.utils.errors import GoOutOfBoardError
from src.utils.util import change_stone, stone2mark


class Board:
    def __init__(self, size: int) -> None:
        if size % 2 != 0:
            raise ValueError("Board size must be even")

        self.size = size
        self._array = np.full((self.size, self.size), Stone.EMPTY, "int8")
        self.reset()

    def reset(self) -> None:
        """盤面を初期化する"""
        self._array.fill(Stone.EMPTY)
        centor = self.size // 2
        self._array[centor - 1][centor - 1] = Stone.BLACK
        self._array[centor][centor - 1] = Stone.WHITE
        self._array[centor - 1][centor] = Stone.WHITE
        self._array[centor][centor] = Stone.BLACK

    def _check_out_of_board(self, place: Place) -> None:
        """盤面の外に出ていないかチェックする

        Args:
            place (Place): 盤面の座標
        """
        if not (0 <= place.x < self.size and 0 <= place.y < self.size):
            raise GoOutOfBoardError("Go out of board")

    def __getitem__(self, key: Place) -> Stone:
        """盤面の座標を指定して石を取得する

        Args:
            key (Place): 盤面の座標

        Returns:
            Stone: 石
        """
        self._check_out_of_board(key)
        return Stone.stone_from_int(self._array[key.y, key.x])

    def __setitem__(self, key: Place, stone: Stone) -> None:
        """盤面の座標を指定して石を置く

        Args:
            key (Place): 盤面の座標
            stone (Stone): 石
        """
        self._check_out_of_board(key)
        self._array[key.y, key.x] = stone

    def __str__(self) -> str:
        """盤面を文字列に変換する

        Returns:
            str: 盤面の文字列
        """
        s = "  " + " ".join([str(i) for i in range(self.size)])
        for i, row in enumerate(self._array):
            s += f"\n{i} " + " ".join([stone2mark(stone) for stone in row])
        return s

    def __iter__(self) -> Generator[tuple[Place, Stone], None, None]:
        """盤面の全ての座標と石を取得する

        Yields:
            Generator[tuple[Place, Stone], None, None]: 盤面の座標と石
        """
        for i, row in enumerate(self._array):
            for j, stone in enumerate(row):
                yield Place(y=i, x=j), stone

    def __copy__(self) -> "Board":
        """盤面をコピーする

        Returns:
            Board: コピーした盤面
        """
        board = Board(self.size)
        board._array = self._array.copy()
        return board

    def get_array(self) -> np.ndarray:
        """盤面の状態を取得する

        Returns:
            np.ndarray: 盤面の状態
        """
        return self._array.astype(np.float64)

    def count_stone(self, stone: Stone) -> int:
        """石の数を数える

        Args:
            stone (Stone): 石

        Returns:
            int: 石の数
        """
        return np.count_nonzero(self._array == stone)


class Othello:
    def __init__(self, size: int = 6) -> None:
        self._board = Board(size)

    def reset(self) -> None:
        """ゲームを初期化する"""
        self._board.reset()

    def _return(
        self, place: Place, stone: Stone, direction: Direction, check_only: bool = True
    ) -> bool:
        """指定した方向の石を返す

        Args:
            place (Place): 石を置く座標
            stone (Stone): 石
            direction (Direction): 方向
            check_only (bool, optional): 石を返すかどうか. Defaults to True.

        Returns:
            bool: 石を返せるかどうか
        """
        # 該当マスが空いていない
        if self._board[place] != Stone.EMPTY:
            return False

        for i in range(1, self._board.size):
            try:
                next_place = place + i * direction.value
                stone_i = self._board[next_place]
                # 自分の石がある場合
                if stone_i == stone:
                    # 隣のマス以外なら置ける、隣のマスなら置けない
                    return i != 1
                # 相手の石がある場合は次のマス
                elif stone_i == change_stone(stone):
                    if not check_only:
                        self._board[next_place] = stone
                    continue
                # 空いている場合は置けない
                else:
                    return False

            # 盤の外に出る場合は置けない
            except GoOutOfBoardError:
                return False

        return False

    def _check_put(self, place: Place | Pass, stone: Stone) -> bool:
        """石を置けるかどうかチェックする

        Args:
            place (Place | Pass): 石を置く座標
            stone (Stone): 石

        Returns:
            bool: 石を置けるかどうか
        """
        if isinstance(place, Pass):
            return True

        # 該当マスが空いていない
        if self._board[place] != Stone.EMPTY:
            return False

        for direction in Direction:
            if self._return(place, stone, direction, check_only=True):
                # 1つでも置ける方向があれば置ける
                return True

        return False

    def search_place_candidate(self, stone: Stone) -> list[Place]:
        """石を置ける場所を探す

        Args:
            stone (Stone): 石

        Returns:
            list[Place]: 石を置ける場所
        """
        return [place for place, _ in self._board if self._check_put(place, stone)]

    def put(self, place: Place | Pass, stone: Stone) -> None:
        """石を置く

        Args:
            place (Place | Pass): 石を置く座標
            stone (Stone): 石

        Raises:
            ValueError: 石を置けない場所に置こうとした場合
        """
        if isinstance(place, Pass):
            return

        if not self._check_put(place, stone):
            raise ValueError("Can't set stone")

        for direction in Direction:
            if self._return(place, stone, direction, check_only=True):
                self._return(place, stone, direction, check_only=False)
        self._board[place] = stone

    def __str__(self) -> str:
        """盤面を文字列に変換する

        Returns:
            str: 盤面の文字列
        """
        return str(self._board)

    def count_stone(self, stone: Stone) -> int:
        """石の数を数える

        Args:
            stone (Stone): 石

        Returns:
            int: 石の数
        """
        return self._board.count_stone(stone)

    def get_board(self) -> Board:
        """盤面を取得する

        Returns:
            Board: 盤面
        """
        return copy.copy(self._board)
