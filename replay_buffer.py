# replay_buffer.py - Pre-allocated numpy ring buffer for speed
from __future__ import annotations
import numpy as np
from config import TRAIN


class ReplayBuffer:
    """Fixed-capacity ring buffer backed by pre-allocated numpy arrays.

    ~10x faster sampling than deque-of-tuples because no per-sample
    python object creation or np.array() conversion on each call.
    """

    def __init__(self, capacity: int = TRAIN.memory) -> None:
        seq = TRAIN.seq_len
        ss = TRAIN.state_size
        self._cap = capacity
        self._pos = 0
        self._size = 0
        self.states = np.zeros((capacity, seq, ss), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, seq, ss), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, s: np.ndarray, a: int, r: float,
             ns: np.ndarray, done: float) -> None:
        i = self._pos
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = ns
        self.dones[i] = done
        self._pos = (i + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, n: int) -> tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self._size, size=n)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self._size
