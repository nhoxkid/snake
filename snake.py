# snake.py - Game entities: Snake and Food
from __future__ import annotations
import random
import numpy as np
from config import Direction, GAME


class Snake:
    __slots__ = ('body', '_body_set', 'direction', 'alive', 'score',
                 'steps', 'steps_since_food')

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        start = (GAME.width // 2, GAME.height // 2)
        self.body: list[tuple[int, int]] = [start]
        self._body_set: set[tuple[int, int]] = {start}
        self.direction = Direction.RIGHT
        self.alive = True
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]

    def set_direction(self, d: Direction) -> None:
        if self.direction.opposite != d:
            self.direction = d

    def move(self) -> None:
        dx, dy = self.direction.delta
        new_head = (self.head[0] + dx, self.head[1] + dy)
        self.body.insert(0, new_head)
        self._body_set.add(new_head)
        self.steps += 1
        self.steps_since_food += 1

    def grow(self) -> None:
        self.score += 1
        self.steps_since_food = 0

    def trim_tail(self) -> None:
        tail = self.body.pop()
        # Only remove from set if tail position isn't duplicated
        if tail not in self.body:
            self._body_set.discard(tail)

    def check_collision(self) -> bool:
        hx, hy = self.head
        if hx < 0 or hx >= GAME.width or hy < 0 or hy >= GAME.height:
            self.alive = False
        elif len(self.body) > 1 and self.head in set(self.body[1:]):
            self.alive = False
        return not self.alive

    # --- State observation ---

    def danger_in(self, d: Direction) -> bool:
        dx, dy = d.delta
        nx, ny = self.head[0] + dx, self.head[1] + dy
        if nx < 0 or nx >= GAME.width or ny < 0 or ny >= GAME.height:
            return True
        return (nx, ny) in self._body_set and (nx, ny) != self.body[-1]

    def observe(self, food_pos: tuple[int, int]) -> np.ndarray:
        """Build a 14-element float32 state vector."""
        hx, hy = self.head
        fx, fy = food_pos
        w, h = GAME.width, GAME.height

        state = np.empty(14, dtype=np.float32)
        # Direction one-hot [0..3]
        state[0:4] = 0.0
        state[int(self.direction)] = 1.0
        # Relative food position [4..5]
        state[4] = (fx - hx) / w
        state[5] = (fy - hy) / h
        # Wall distances [6..9]
        state[6] = hy / h
        state[7] = (h - hy - GAME.cell) / h
        state[8] = hx / w
        state[9] = (w - hx - GAME.cell) / w
        # Danger straight / left / right [10..12]
        state[10] = float(self.danger_in(self.direction))
        state[11] = float(self.danger_in(self.direction.left_turn))
        state[12] = float(self.danger_in(self.direction.right_turn))
        # Body length normalized [13]
        state[13] = len(self.body) / (GAME.grid_w * GAME.grid_h)
        return state


class Food:
    __slots__ = ('position',)

    def __init__(self, excluded: set[tuple[int, int]] | None = None) -> None:
        self.position = self._random(excluded or set())

    def respawn(self, excluded: set[tuple[int, int]] | None = None) -> tuple[int, int]:
        self.position = self._random(excluded or set())
        return self.position

    @staticmethod
    def _random(excluded: set[tuple[int, int]]) -> tuple[int, int]:
        while True:
            pos = (
                random.randint(0, GAME.grid_w - 1) * GAME.cell,
                random.randint(0, GAME.grid_h - 1) * GAME.cell,
            )
            if pos not in excluded:
                return pos
