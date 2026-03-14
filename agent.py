# agent.py - DQRN Agent: batch inference, 1-pass diagnostics, multi-env support
from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import TRAIN
from replay_buffer import ReplayBuffer


@dataclass
class NetworkSnapshot:
    """All values from a SINGLE forward pass + gradient — nothing redundant."""
    input_vals: np.ndarray
    input_saliency: np.ndarray
    lstm_hidden: np.ndarray
    lstm_cell: np.ndarray
    dense_act: np.ndarray
    q_values: np.ndarray
    chosen_action: int
    was_random: bool
    prev_q: np.ndarray | None
    dense1_kernel: np.ndarray
    output_kernel: np.ndarray
    loss: float
    td_error: float


class DQRNAgent:
    """Double-DQN with LSTM. Supports N parallel envs with batch inference."""

    def __init__(self, n_envs: int = 1) -> None:
        self.n_actions = TRAIN.n_actions
        self.seq_len = TRAIN.seq_len
        self.state_size = TRAIN.state_size
        self.n_envs = n_envs

        self.epsilon = TRAIN.eps_start

        self.policy, self._diag = self._build_with_diag()
        self.target = self._build_simple()
        self.sync_target()

        self.memory = ReplayBuffer()

        # Per-env state histories
        self._zero = np.zeros(self.state_size, dtype=np.float32)
        self._histories: list[deque] = []
        for _ in range(n_envs):
            h = deque(maxlen=self.seq_len)
            for _ in range(self.seq_len):
                h.append(self._zero.copy())
            self._histories.append(h)

        # Training state
        self._opt = Adam(learning_rate=TRAIN.lr)
        self._loss_fn = tf.keras.losses.Huber()
        self._compiled_train = tf.function(self._train_on_batch)
        self._last_loss: float = 0.0
        self._last_td: float = 0.0
        self._prev_q: np.ndarray | None = None

        # Cache weight references (avoid get_layer lookup every frame)
        self._dense1_layer = self.policy.get_layer('dense1')
        self._output_layer = self.policy.get_layer('q_output')

    def _build_with_diag(self) -> tuple[Model, Model]:
        inp = Input(shape=(self.seq_len, self.state_size), name='seq_input')
        lstm = LSTM(TRAIN.hidden, return_sequences=False,
                    return_state=True, name='lstm')
        lstm_out, lstm_h, lstm_c = lstm(inp)
        dense1 = Dense(TRAIN.hidden, activation='relu', name='dense1')
        d1_out = dense1(lstm_out)
        dense2 = Dense(self.n_actions, activation='linear', name='q_output')
        q_out = dense2(d1_out)

        policy = Model(inputs=inp, outputs=q_out, name='policy')
        diag = Model(inputs=inp,
                     outputs=[lstm_h, lstm_c, d1_out, q_out],
                     name='diag')
        return policy, diag

    def _build_simple(self) -> Model:
        inp = Input(shape=(self.seq_len, self.state_size))
        x = LSTM(TRAIN.hidden, return_sequences=False)(inp)
        x = Dense(TRAIN.hidden, activation='relu')(x)
        out = Dense(self.n_actions, activation='linear')(x)
        return Model(inputs=inp, outputs=out)

    def sync_target(self) -> None:
        self.target.set_weights(self.policy.get_weights())

    # --- Per-env state management ---

    def clear_history(self, env_idx: int = 0) -> None:
        h = self._histories[env_idx]
        h.clear()
        for _ in range(self.seq_len):
            h.append(self._zero.copy())
        if env_idx == 0:
            self._prev_q = None

    def clear_all_histories(self) -> None:
        for i in range(self.n_envs):
            self.clear_history(i)

    def push_state(self, state: np.ndarray, env_idx: int = 0) -> None:
        self._histories[env_idx].append(state)

    def state_seq(self, env_idx: int = 0) -> np.ndarray:
        return np.array(self._histories[env_idx], dtype=np.float32)

    def all_state_seqs(self) -> np.ndarray:
        """(n_envs, seq_len, state_size) batch for all envs."""
        return np.array([list(h) for h in self._histories], dtype=np.float32)

    # --- Batch inference (the main speedup) ---

    def act_batch(self, seqs: np.ndarray) -> list[int]:
        """Single forward pass for ALL envs. Returns list of actions."""
        q_all = self.policy(seqs, training=False).numpy()  # (n_envs, n_actions)
        actions = []
        for i in range(len(seqs)):
            if random.random() < self.epsilon:
                actions.append(random.randint(0, self.n_actions - 1))
            else:
                actions.append(int(np.argmax(q_all[i])))
        return actions

    # --- Single-pass diagnostics (only when rendering) ---

    def act_and_explain(self, seq: np.ndarray, action: int | None = None) -> NetworkSnapshot:
        """ONE forward pass: diag model + gradient saliency. No redundant calls.

        If action is None, selects one via epsilon-greedy.
        If action is provided (from act_batch), reuses it.
        """
        seq_t = tf.constant(seq[np.newaxis], dtype=tf.float32)

        # Single forward through diag model WITH GradientTape for saliency
        with tf.GradientTape() as tape:
            tape.watch(seq_t)
            lstm_h, lstm_c, d1_out, q_out = self._diag(seq_t, training=False)

            q_np = q_out[0].numpy()

            if action is None:
                was_random = random.random() < self.epsilon
                greedy = int(np.argmax(q_np))
                action = random.randint(0, self.n_actions - 1) if was_random else greedy
            else:
                was_random = False  # action came from act_batch

            q_chosen = q_out[0, action]

        # Gradient of chosen Q w.r.t. input (saliency)
        grads = tape.gradient(q_chosen, seq_t)
        saliency = np.abs(grads[0, -1].numpy())

        snap = NetworkSnapshot(
            input_vals=seq[-1],
            input_saliency=saliency,
            lstm_hidden=lstm_h[0].numpy(),
            lstm_cell=lstm_c[0].numpy(),
            dense_act=d1_out[0].numpy(),
            q_values=q_np,
            chosen_action=action,
            was_random=was_random,
            prev_q=self._prev_q.copy() if self._prev_q is not None else None,
            dense1_kernel=self._dense1_layer.get_weights()[0],
            output_kernel=self._output_layer.get_weights()[0],
            loss=self._last_loss,
            td_error=self._last_td,
        )
        self._prev_q = q_np.copy()
        return snap

    def decay_epsilon(self) -> None:
        self.epsilon = max(TRAIN.eps_end, self.epsilon * TRAIN.eps_decay)

    # --- Training ---

    def remember(self, s, a, r, ns, done):
        self.memory.push(s, a, r, ns, done)

    def train_step(self) -> float:
        if len(self.memory) < TRAIN.batch:
            return 0.0
        states, actions, rewards, next_states, dones = self.memory.sample(TRAIN.batch)
        loss, td = self._compiled_train(
            tf.constant(states), tf.constant(actions),
            tf.constant(rewards), tf.constant(next_states),
            tf.constant(dones),
        )
        self._last_loss = float(loss)
        self._last_td = float(td)
        return self._last_loss

    def _train_on_batch(self, states, actions, rewards, next_states, dones):
        next_q_policy = self.policy(next_states, training=False)
        next_q_target = self.target(next_states, training=False)
        best_a = tf.argmax(next_q_policy, axis=1)
        best_q = tf.gather_nd(
            next_q_target,
            tf.stack([tf.range(TRAIN.batch), tf.cast(best_a, tf.int32)], axis=1))
        targets = rewards + TRAIN.gamma * best_q * (1.0 - dones)

        with tf.GradientTape() as tape:
            q_vals = self.policy(states, training=True)
            indices = tf.stack([tf.range(TRAIN.batch), actions], axis=1)
            predicted = tf.gather_nd(q_vals, indices)
            loss = self._loss_fn(targets, predicted)

        grads = tape.gradient(loss, self.policy.trainable_variables)
        self._opt.apply_gradients(zip(grads, self.policy.trainable_variables))
        mean_td = tf.reduce_mean(tf.abs(targets - predicted))
        return loss, mean_td

    # --- Persistence ---

    def save(self, path: str | Path) -> None:
        self.policy.save_weights(str(path))

    def load(self, path: str | Path) -> None:
        self.policy.load_weights(str(path))
        self.sync_target()
