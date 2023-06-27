import copy
import random
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

import dill
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dropout  # noqa
from keras.layers import Activation, Add, Conv2D, Dense, Flatten, Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

from src.othello import Board
from src.utils.buffer import ReplayBuffer
from src.utils.dataclass import Pass, Place
from src.utils.enums import Stone
from src.utils.util import Turn


class AgentInterface(metaclass=ABCMeta):
    def test(self) -> None:
        pass

    @abstractmethod
    def get_action(
        self, state: Board, choice: List[Place], turn: Turn, mystone: Stone
    ) -> Place | Pass:
        pass

    def save(self, file_name: str) -> None:
        with open(file_name, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, file_name: str) -> "AgentInterface":
        with open(file_name, "rb") as f:
            instance = dill.load(f)
        return instance

    def copy(self) -> "AgentInterface":
        return copy.deepcopy(self)


class TrainAgentInterface(AgentInterface):
    def __init__(self) -> None:
        self.model: Model
        self.memory: ReplayBuffer

    @abstractmethod
    def remember(
        self,
        state: Board,
        action: Place | Pass,
        reward: float,
        next_state: Board,
        next_action_candidates: List[Place],
        turn: Turn,
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def sync_model(self, opponent_agent: AgentInterface) -> None:
        pass

    @abstractmethod
    def replay(self, batch_size: int, opponent_agent: AgentInterface) -> None:
        pass

    @abstractmethod
    def plot_history(self) -> None:
        pass

    @classmethod
    def load(cls, file_name: str) -> "TrainAgentInterface":
        with open(file_name, "rb") as f:
            instance = dill.load(f)
        return instance


class RandomAgent(AgentInterface):
    def get_action(
        self, state: Board, choice: List[Place], turn: Turn, mystone: Stone
    ) -> Place | Pass:
        if len(choice) == 0:
            return Pass()
        return random.choice(choice)


class HumanAgent(AgentInterface):
    def get_action(
        self, state: Board, choice: List[Place], turn: Turn, mystone: Stone
    ) -> Place | Pass:
        if len(choice) == 0:
            return Pass()

        while True:
            print("")
            print(state)
            print("choice: ", choice)
            try:
                y, x = list(int(x) for x in input("yx: "))
                place = Place(y=y, x=x)
                if place in choice:
                    return place
            except Exception:
                pass

            print("Invalid choice. Please try again.")


class DQNAgent(TrainAgentInterface):
    def __init__(
        self,
        shape: tuple[int, int],
        replay_buffer: ReplayBuffer = ReplayBuffer(1000),
        gamma: float = 0.98,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.1,
        num_block: int = 2,
        filter: int = 32,
        activation: str = "relu",
        dropout: float = 0,
        loss_func: Any = "mean_squared_error",  # keras.losses.Huber(),
        optimizer: Any = Adam(lr=0.01),
    ) -> None:
        """Q関数を用いた方策

        Args:
            gamma (float): 割引率
            epsilon (float): ε-greedyのε(1-εで最良の行動を取り、εでランダムに行動を取る)
        """
        self._shape = shape
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self.memory = replay_buffer
        self._num_block = num_block
        self._filter = filter
        self._activation = activation
        self._dropout = dropout
        self._loss_func = loss_func
        self._optimizer = optimizer

        self.model = self._build_model()
        self._target_model = self._build_model()
        self._history: dict[str, list[float]] = {"loss": []}

        super().__init__()

    def test(self) -> None:
        self._epsilon = 0

    def remember(
        self,
        state: Board,
        action: Place | Pass,
        reward: float,
        next_state: Board,
        next_action_candidates: List[Place],
        turn: Turn,
        done: bool,
    ) -> None:
        """Q学習

        Info:
            double dqnでのtarget
                target = reward + gamma * Q(s', argmax_a' Q(s', a'))

        XXX:
            高速化のため計算済みの値を保存しておくが、改良する必要がある
        """
        if isinstance(action, Pass):
            return

        if next_action_candidates:
            X = self._create_X(next_state, turn)
            place_map_value = self._predict_action_candidates(
                X, next_action_candidates, True
            )
            argmax_a = max(place_map_value, key=place_map_value.get)  # type: ignore
            q = self._predict_action_candidates(X, next_action_candidates)[argmax_a]
        else:
            q = 0

        target = reward
        if not done:
            target += self._gamma * q
        X = self._create_X(state, turn)
        y = self._create_y(X, action, target)
        self.memory.add((X, y))

    def replay(self, batch_size: int) -> None:
        experiences = self.memory.sample(batch_size)
        # データ拡張
        experiences = [self._data_augmentation(*Xy) for Xy in experiences]
        X = np.concatenate([X for X, _ in experiences])
        y = np.concatenate([y for _, y in experiences])
        history = self.model.fit(X, y, epochs=10, verbose=0)
        self._history["loss"].extend(history.history["loss"])

        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def sync_model(self, opponent_agent: AgentInterface) -> None:
        self._target_model.set_weights(copy.deepcopy(self.model.get_weights()))
        opponent_agent.model.set_weights(copy.deepcopy(self.model.get_weights()))

    def get_action(
        self, state: Board, choice: List[Place], turn: Turn, mystone: Stone
    ) -> Place | Pass:
        """行動選択

        Args:
            state (Board): 状態
            choice (List[Place]): 選択肢
            turn (Turn): 手番

        Returns:
            Place | Pass: 行動
        """
        if turn.stone != mystone:  # 白番初手はパス
            return Pass()

        if len(choice) == 0:
            return Pass()

        # ランダムな行動を取る
        if random.random() < self._epsilon:
            return random.choice(choice)

        # 最良の行動を取る
        X = self._create_X(state, turn)
        place_map_value = self._predict_action_candidates(X, choice)
        return max(place_map_value, key=place_map_value.get)  # type: ignore

    def plot_history(self, path: Path = Path("images")) -> None:
        path.mkdir(exist_ok=True, parents=True)
        for key, value in self._history.items():
            plt.figssize = (100, 100)
            plt.plot(value, label=key)
            plt.title(key)
            plt.yscale("log")
            plt.savefig(path / f"{key}.png")
            plt.close()

    def _build_model(self) -> Sequential:
        """モデルの構築

        Returns:
            Sequential: モデル
        """

        def Block(x: tf.Tensor, num_conv: int = 2, kernel_size: int = 3) -> tf.Tensor:
            for _ in range(num_conv):
                x = Conv2D(
                    self._filter,
                    kernel_size=(kernel_size, kernel_size),
                    padding="same",
                )(x)
                # x = BatchNormalization()(x)
                x = Activation(self._activation)(x)
            x = Dropout(self._dropout)(x)
            return x

        inputs = Input(shape=(*self._shape, len(Stone)))
        x = Block(inputs, num_conv=2, kernel_size=3)
        for i in range(self._num_block - 1):
            x = Block(x, num_conv=2, kernel_size=3)
        v = Flatten()(x)
        v = Dense(256, activation=self._activation)(v)
        v = Dropout(self._dropout)(v)
        v = Dense(1, activation="linear")(v)
        adv = Block(x, num_conv=2, kernel_size=3)
        adv = Conv2D(1, kernel_size=(3, 3), padding="same")(adv)
        outputs = Add()([v, adv - K.mean(adv, axis=(1, 2), keepdims=True)])
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self._loss_func, optimizer=self._optimizer)
        model.summary()
        return model

    def _predict_action_candidates(
        self, X: np.ndarray, choice: List[Place], is_target_model: bool = False
    ) -> dict[Place, float]:
        """可能な行動における、行動価値の予測

        Args:
            X (np.ndarray): 入力データ
            choice (List[Place]): 選択肢

        Returns:
            dict[Place, float]: 行動価値
        """
        model = self._target_model if is_target_model else self.model
        np_reward = model.predict(X, verbose=0)[0]
        return {place: np_reward[place.y, place.x] for place in choice}

    def _create_X(self, state: Board, turn: Turn) -> np.ndarray:
        """入力データの作成

        Args:
            state (Board): 状態

        Returns:
            np.ndarray: 入力データ
        """
        # 白番と黒番で同じモデルを使うので、白番の場合は盤面を反転させる
        encoded_matrix = to_categorical(state.get_array() * turn.stone.value - min(Stone).value, num_classes=len(Stone))
        return np.expand_dims(encoded_matrix, 0)

    def _create_y(self, X: np.ndarray, action: Place, target: float) -> np.ndarray:
        """教師データの作成

        Args:
            state (Board): 状態
            action (Place): 行動
            target (float): 教師信号

        Returns:
            np.ndarray: 教師データ
        """
        # 行動した以外の価値はそのままに、行動した価値のみ教師信号に置き換える
        target_ = self.model.predict(X, verbose=0)
        target_[0][action.y, action.x] = target
        return target_

    def _data_augmentation(self, X: np.ndarray, y: np.ndarray, k: Optional[int] = None) -> None:
        """データ拡張

        Args:
            X (np.ndarray): 入力データ
            y (np.ndarray): 教師データ
            k (Optional[int], optional): 回転回数. Defaults to None.
        """
        if k is None:
            k = random.randrange(0, 3)
        X = np.rot90(X, k, axes=(1, 2)).copy()
        y = np.rot90(y, k, axes=(1, 2)).copy()
        return X, y
