from typing import Optional

import gym
import numpy as np
from gym import spaces

from src.agent import AgentInterface, RandomAgent
from src.othello import Board, Othello
from src.utils.dataclass import Pass, Place
from src.utils.enums import Result, Stone
from src.utils.util import Turn, change_stone, result2reward


class OthelloEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, size: int = 6) -> None:
        super().__init__()

        self.size = size
        self._othello = Othello(self.size)
        self.turn = Turn(Stone.BLACK)

        self.action_space = spaces.Discrete(self.size * self.size + 1)
        self.observation_space = spaces.Box(
            low=min(Stone).value,
            high=max(Stone).value,
            shape=(self.size, self.size),
            dtype=np.uint8,
        )

    def reset(  # type: ignore[override]
        self, mystone: Stone, opponent_agent: Optional[AgentInterface] = None
    ) -> Board:
        """ゲームを初期化する

        Args:
            mystone (Stone): 自分の石
            opponent_agent (Optional[AgentInterface], optional):
                対戦相手のエージェント. Defaults to None.

        Returns:
            Board: 盤面
        """
        if opponent_agent is None:
            opponent_agent = RandomAgent()

        self._othello.reset()
        self.mystone = mystone
        self.turn = Turn(Stone.BLACK)
        self.opponent_agent = opponent_agent
        return self._othello.get_board()

    def step(self, action: Place | Pass) -> tuple[Board, float, bool, dict]:  # type: ignore[override]
        """ゲームを進める

        Args:
            action (Place | Pass): 行動

        Returns:
            tuple[Board, float, bool, dict]: 盤面, 報酬, ゲーム終了フラグ, その他の情報
        """
        if self.mystone == self.turn.stone:
            self._othello.put(action, self.turn.stone)
            self.turn = self.turn.change_turn()

        action_oppoent = self.opponent_agent.get_action(
            self._othello.get_board(),
            self._othello.search_place_candidate(self.turn.stone),
            self.turn,
            change_stone(self.mystone),
        )
        self._othello.put(action_oppoent, self.turn.stone)
        self.turn = self.turn.change_turn()

        if isinstance(action, Pass) and isinstance(action_oppoent, Pass):
            result = self.calc_result()
            return self._othello.get_board(), result2reward(result), True, {}

        return self._othello.get_board(), 0, False, {}

    def render(self, mode: str = "human", close: bool = False) -> Optional[Board]:  # type: ignore[override]
        """ゲームの状態を描画する

        Args:
            mode (str, optional): 描画モード. Defaults to "human".
            close (bool, optional): 描画を閉じるかどうか. Defaults to False.

        Returns:
            Board: 盤面
        """
        if close:
            return None
        return self._othello.get_board()

    # 以下の関数は、自分で追加したもの
    def search_action_candidate(self, turn: Turn) -> list[Place]:
        """置ける場所を探索する

        Args:
            turn (Turn): 石

        Returns:
            list[Place]: 置ける場所のリスト
        """
        return self._othello.search_place_candidate(turn.stone)

    def calc_result(self) -> Result:
        """結果を計算する

        Returns:
            Result: 結果
        """
        my_score = self._othello.count_stone(self.mystone)
        oppoent_score = self._othello.count_stone(change_stone(self.mystone))
        if my_score > oppoent_score:
            return Result.WIN
        elif my_score < oppoent_score:
            return Result.LOSE
        else:
            return Result.DRAW
