# environment.py - Gym-style SnakeEnv + thread-safe game snapshot
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from config import Direction, GAME, TRAIN
from snake import Snake, Food


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    score: int
    ate_food: bool


@dataclass
class GameSnapshot:
    """Thread-safe copy of game state for rendering from another thread."""
    body: list[tuple[int, int]]
    food_pos: tuple[int, int]
    score: int
    alive: bool
    episode: int = 0
    step: int = 0
    epsilon: float = 0.0
    reward: float = 0.0


class SnakeEnv:
    """Gym-style environment.  reset() -> state,  step(action) -> StepResult."""

    def __init__(self) -> None:
        self.snake = Snake()
        self.food = Food()
        self._prev_dist: float = 0.0

    def reset(self) -> np.ndarray:
        self.snake.reset()
        self.food.respawn(self.snake._body_set)
        self._prev_dist = self._manhattan()
        return self.snake.observe(self.food.position)

    def step(self, action: int) -> StepResult:
        direction = Direction(action)
        self.snake.set_direction(direction)
        self.snake.move()

        ate = self.snake.head == self.food.position
        if ate:
            self.snake.grow()
            self.food.respawn(self.snake._body_set)
        else:
            self.snake.trim_tail()

        died = self.snake.check_collision()
        starved = self.snake.steps_since_food > TRAIN.starvation
        if starved:
            died = True
            self.snake.alive = False

        reward = self._reward(ate, died)
        state = self.snake.observe(self.food.position)
        return StepResult(state, reward, died, self.snake.score, ate)

    def snapshot(self, episode=0, step=0, epsilon=0.0, reward=0.0) -> GameSnapshot:
        """Create a thread-safe copy of current game state."""
        return GameSnapshot(
            body=list(self.snake.body),
            food_pos=self.food.position,
            score=self.snake.score,
            alive=self.snake.alive,
            episode=episode, step=step,
            epsilon=epsilon, reward=reward,
        )

    def _reward(self, ate: bool, died: bool) -> float:
        if died:
            return -10.0
        if ate:
            self._prev_dist = self._manhattan()
            return 10.0
        dist = self._manhattan()
        r = 0.1 if dist < self._prev_dist else -0.15
        self._prev_dist = dist
        return r

    def _manhattan(self) -> float:
        hx, hy = self.snake.head
        fx, fy = self.food.position
        return float(abs(hx - fx) + abs(hy - fy))
