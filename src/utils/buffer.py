import random
from collections import deque
from typing import Deque


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[tuple] = deque(maxlen=capacity)
        self.sample_count = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, experience: tuple) -> None:
        """経験を追加する

        Args:
            experience (tuple):
                (state, action, reward, next_state, next_action_candidates)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[tuple]:
        """経験をランダムに取得する

        Args:
            batch_size (int): バッチサイズ

        Returns:
            list[tuple]: 経験のリスト
        """
        self.sample_count += 1
        sample_size = min(len(self.buffer), batch_size)
        return random.sample(self.buffer, sample_size)
