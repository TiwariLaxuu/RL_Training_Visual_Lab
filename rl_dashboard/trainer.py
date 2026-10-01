"""VisualTrainer — a thin wrapper around Stable-Baselines3 that trains in a
background thread and streams everything interesting about the process
(rewards, losses, gradient norms, raw weights, rendered frames) onto a
queue, without changing how SB3 itself trains.
"""

from __future__ import annotations

import queue
import threading

from gymnasium import spaces
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

from rl_dashboard.callbacks import LiveMetricsCallback
from rl_dashboard.env_docs import classify_episode
from rl_dashboard.envs import make_env

ALGOS = {"PPO": PPO, "A2C": A2C, "DQN": DQN}


def compatible_algos(action_space) -> list[str]:
    """Which of our algorithms can act in this action space."""
    if isinstance(action_space, spaces.Discrete):
        return ["PPO", "A2C", "DQN"]
    if isinstance(action_space, (spaces.Box, spaces.MultiDiscrete, spaces.MultiBinary)):
        return ["PPO", "A2C"]
    return []


def attach_grad_norm_hook(model, on_grad_norm):
    """Wrap the policy optimizer's .step() to report the gradient norm of
    every trainable parameter right before each update is applied.

    This is the actual `torch.optim.Optimizer.step()` SB3 calls internally
    — we're not simulating anything, just observing the real update.
    """
    optimizer = model.policy.optimizer
    original_step = optimizer.step

    def step_with_grad_norm(*args, **kwargs):
        total_sq = 0.0
        for p in model.policy.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.data.norm(2).item()) ** 2
        on_grad_norm(total_sq**0.5)
        return original_step(*args, **kwargs)

    optimizer.step = step_with_grad_norm


class VisualTrainer:
    """Owns one training run: builds env + model, runs .learn() in a
    background thread, and can be asked to stop early.
    """

    def __init__(
        self,
        env_id: str,
        algo: str = "PPO",
        total_timesteps: int = 20_000,
        learning_rate: float = 3e-4,
        seed: int | None = None,
    ):
        if algo not in ALGOS:
            raise ValueError(f"Unknown algorithm '{algo}'. Choose from {list(ALGOS)}")

        self.env_id = env_id
        self.algo = algo
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.seed = seed

        self.event_queue: "queue.Queue[dict]" = queue.Queue(maxsize=2000)
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._render_env = None

        # Set once the model exists (even mid-training), so the Report tab
        # can inspect/test it. `trained_at_step` records how far training
        # had gotten the moment it stopped, for an honest "how trained is
        # this?" caveat on the Report page.
        self.model = None
        self.trained_at_step = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_test_episodes(self, n_episodes: int = 10, deterministic: bool = True) -> list[dict]:
        """Roll out the current model greedily (no exploration) and grade
        each episode success/fail using env_docs' per-environment rules.
        """
        if self.model is None:
            raise RuntimeError("No trained model yet — start and finish a training run first.")

        env = make_env(self.env_id, render=False)
        max_steps = getattr(env.spec, "max_episode_steps", None)

        results = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
                length += 1

            success, reason = classify_episode(self.env_id, total_reward, length, terminated, truncated, max_steps)
            results.append(
                {
                    "episode": ep + 1,
                    "reward": round(total_reward, 3),
                    "length": length,
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": success,
                    "reason": reason,
                }
            )
        env.close()
        return results

    def _run(self):
        try:
            raw_env = make_env(self.env_id, render=True)
            self._render_env = raw_env
            train_env = Monitor(raw_env)

            algo_cls = ALGOS[self.algo]
            model_kwargs = {"learning_rate": self.learning_rate, "verbose": 0}
            if self.seed is not None:
                model_kwargs["seed"] = self.seed

            model = algo_cls("MlpPolicy", train_env, **model_kwargs)
            self.model = model

            self.event_queue.put_nowait(
                {
                    "type": "status",
                    "state": "starting",
                    "step": 0,
                    "env_id": self.env_id,
                    "algo": self.algo,
                    "total_timesteps": self.total_timesteps,
                    "n_parameters": sum(p.numel() for p in model.policy.parameters()),
                }
            )

            attach_grad_norm_hook(
                model,
                lambda norm: self.event_queue.put_nowait(
                    {"type": "grad_norm", "step": int(model.num_timesteps), "value": norm}
                ),
            )

            callback = LiveMetricsCallback(
                event_queue=self.event_queue,
                stop_event=self.stop_event,
                render_env=raw_env,
                frame_every_steps=max(1, self.total_timesteps // 300),
                weight_every_steps=max(1, self.total_timesteps // 20),
            )

            model.learn(total_timesteps=self.total_timesteps, callback=callback)

        except Exception as exc:  # surface to the UI instead of dying silently
            self.event_queue.put_nowait({"type": "status", "state": "error", "message": str(exc)})
        finally:
            if self.model is not None:
                self.trained_at_step = int(self.model.num_timesteps)
            if self._render_env is not None:
                try:
                    self._render_env.close()
                except Exception:
                    pass
