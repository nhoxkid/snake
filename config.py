# config.py - All configuration in one place
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @property
    def opposite(self) -> Direction:
        return _OPPOSITES[self]

    @property
    def delta(self) -> tuple[int, int]:
        return _DELTAS[self]

    @property
    def left_turn(self) -> Direction:
        return _LEFT[self]

    @property
    def right_turn(self) -> Direction:
        return _RIGHT[self]


_OPPOSITES = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT,
}
_LEFT = {
    Direction.UP: Direction.LEFT, Direction.LEFT: Direction.DOWN,
    Direction.DOWN: Direction.RIGHT, Direction.RIGHT: Direction.UP,
}
_RIGHT = {
    Direction.UP: Direction.RIGHT, Direction.RIGHT: Direction.DOWN,
    Direction.DOWN: Direction.LEFT, Direction.LEFT: Direction.UP,
}


@dataclass(frozen=True)
class GameConfig:
    width: int = 600
    height: int = 400
    cell: int = 20
    fps: int = 15

    @property
    def grid_w(self) -> int:
        return self.width // self.cell

    @property
    def grid_h(self) -> int:
        return self.height // self.cell


@dataclass(frozen=True)
class TrainConfig:
    state_size: int = 14       # full state vector length
    seq_len: int = 8           # LSTM lookback frames
    hidden: int = 64           # LSTM + dense units
    n_actions: int = 4
    n_envs: int = 8            # parallel environments for batch inference
    batch: int = 64
    memory: int = 50_000
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.998
    lr: float = 0.001
    target_sync: int = 10     # episodes between target net sync
    episodes: int = 500
    max_steps: int = 1000
    starvation: int = 250     # steps without food before death


# Singletons — import these everywhere
GAME = GameConfig()
TRAIN = TrainConfig()

# Precompute deltas using the singleton cell size
_DELTAS = {
    Direction.UP:    (0, -GAME.cell),
    Direction.DOWN:  (0,  GAME.cell),
    Direction.LEFT:  (-GAME.cell, 0),
    Direction.RIGHT: ( GAME.cell, 0),
}
