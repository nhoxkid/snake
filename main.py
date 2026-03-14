# main.py - Threaded trainer: training in background, rendering in main thread
#
# Architecture:
#   Main thread  — pygame event loop + rendering at FPS (must be main thread)
#   Train thread — runs N envs with batch inference, fills replay, trains
#   Communication — shared (GameSnapshot, NetworkSnapshot) behind a lock
#
# Why threading works here:
#   TensorFlow ops release the GIL. Pygame rendering is mostly C calls that
#   also release the GIL. So training and rendering genuinely overlap.
from __future__ import annotations
import sys
import threading
import numpy as np
import pygame

from config import GAME, TRAIN
from environment import SnakeEnv, GameSnapshot
from agent import DQRNAgent, NetworkSnapshot
from renderer import GameRenderer, NetworkVisualizer

VIS_W = 520


class Trainer:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((GAME.width + VIS_W, GAME.height))
        pygame.display.set_caption("DQRN Snake")
        self.clock = pygame.time.Clock()

        self.game_surf = self.screen.subsurface((0, 0, GAME.width, GAME.height))
        self.net_surf = self.screen.subsurface((GAME.width, 0, VIS_W, GAME.height))
        self.game_view = GameRenderer(self.game_surf)
        self.net_view = NetworkVisualizer(self.net_surf)

        # Shared state between threads
        self._lock = threading.Lock()
        self._latest_game: GameSnapshot | None = None
        self._latest_net: NetworkSnapshot | None = None
        self._quit = threading.Event()
        self._status: str = "Initializing..."
        self._demo_mode = False

    # ── Main thread: rendering ──────────────────────────────────

    def run(self) -> None:
        train_thread = threading.Thread(target=self._train_loop, daemon=True)
        train_thread.start()

        while not self._quit.is_set():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._quit.set()

            with self._lock:
                gs = self._latest_game
                ns = self._latest_net

            if gs is not None:
                self.game_view.draw(gs)
            if ns is not None:
                self.net_view.draw(ns)

            pygame.display.flip()
            fps = GAME.fps if self._demo_mode else GAME.fps * 3
            self.clock.tick(fps)

        train_thread.join(timeout=2)
        pygame.quit()

    # ── Training thread ─────────────────────────────────────────

    def _train_loop(self) -> None:
        n = TRAIN.n_envs
        envs = [SnakeEnv() for _ in range(n)]
        agent = DQRNAgent(n_envs=n)

        # Reset all envs
        for i, env in enumerate(envs):
            state = env.reset()
            agent.push_state(state, env_idx=i)

        best_score = 0
        scores: list[int] = []
        ep_count = 0
        step_count = 0
        # Track per-env episode step counts
        env_steps = [0] * n

        while not self._quit.is_set() and ep_count < TRAIN.episodes:
            # 1. Batch inference — ONE forward pass for all N envs
            seqs = agent.all_state_seqs()
            actions = agent.act_batch(seqs)

            # 2. Step all envs, collect transitions
            for i in range(n):
                seq_before = seqs[i]
                result = envs[i].step(actions[i])
                agent.push_state(result.state, env_idx=i)
                next_seq = agent.state_seq(env_idx=i)
                env_steps[i] += 1

                agent.remember(seq_before, actions[i], result.reward,
                               next_seq, float(result.done))

                if result.done:
                    ep_count += 1
                    scores.append(envs[i].snake.score)
                    best_score = max(best_score, envs[i].snake.score)
                    env_steps[i] = 0

                    if ep_count % TRAIN.target_sync == 0:
                        agent.sync_target()
                    agent.decay_epsilon()

                    if ep_count % 20 == 0:
                        avg = np.mean(scores[-50:]) if scores else 0
                        print(f"Ep {ep_count:4d} | Score {scores[-1]:3d} | "
                              f"Best {best_score:3d} | Avg50 {avg:5.1f} | "
                              f"\u03b5 {agent.epsilon:.3f} | Buf {len(agent.memory):,}")

                    new_state = envs[i].reset()
                    agent.clear_history(env_idx=i)
                    agent.push_state(new_state, env_idx=i)

                # Publish display env (env 0) snapshot for rendering
                if i == 0:
                    gs = envs[0].snapshot(
                        episode=ep_count,
                        step=env_steps[0],
                        epsilon=agent.epsilon,
                        reward=result.reward,
                    )
                    # Only compute expensive explain() ~30 times/sec
                    # (check if lock is free = renderer consumed previous frame)
                    if not self._lock.locked():
                        snap = agent.act_and_explain(seqs[0], actions[0])
                        with self._lock:
                            self._latest_game = gs
                            self._latest_net = snap

            # 3. Train on replay buffer
            agent.train_step()
            step_count += 1

        # Training done — run demo
        print(f"{'=' * 60}\nDone. Best: {best_score}. Demo...")
        agent.save("best_weights.weights.h5")
        self._demo_mode = True
        self._run_demo(agent, envs[0])

    def _run_demo(self, agent: DQRNAgent, env: SnakeEnv) -> None:
        agent.epsilon = 0.0
        state = env.reset()
        agent.clear_history(env_idx=0)
        agent.push_state(state, env_idx=0)

        while not self._quit.is_set() and env.snake.alive:
            seq = agent.state_seq(env_idx=0)
            snap = agent.act_and_explain(seq)
            result = env.step(snap.chosen_action)
            agent.push_state(result.state, env_idx=0)

            gs = env.snapshot(step=env.snake.steps, reward=result.reward)
            with self._lock:
                self._latest_game = gs
                self._latest_net = snap

            if result.done:
                break
            # Pace demo to render speed
            import time
            time.sleep(1.0 / GAME.fps)

        print(f"Demo score: {env.snake.score}")
        # Keep window open until user closes
        while not self._quit.is_set():
            import time
            time.sleep(0.1)


if __name__ == '__main__':
    try:
        Trainer().run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        pygame.quit()
        sys.exit()
