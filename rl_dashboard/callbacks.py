"""SB3 callback that turns a normal training run into a stream of events.

Every event is a small JSON-serializable dict pushed onto a thread-safe
queue. The FastAPI websocket layer just drains that queue and forwards it
to the browser — this module doesn't know anything about HTTP.
"""

from __future__ import annotations

import base64
import io
import queue
import threading
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


def encode_frame(frame: np.ndarray, max_width: int = 360) -> str | None:
    """Downscale + JPEG-encode a rendered rgb_array frame as a data URI."""
    if not _HAS_PIL or frame is None:
        return None
    img = Image.fromarray(frame)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, max(1, int(img.height * ratio))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_weight_snapshot(model, max_dim: int = 24) -> dict[str, list]:
    """Grab a small readable slice of the policy network's weights.

    Cropped rather than resized so the numbers on screen are the model's
    real weights, not an interpolated approximation of them.
    """
    layers: dict[str, list] = {}
    policy = model.policy

    named_linear = []
    for name, module in policy.named_modules():
        if module.__class__.__name__ == "Linear":
            named_linear.append((name, module))

    # first Linear layer is the most legible one: raw-observation weights.
    # For the second, prefer the layer that actually produces the agent's
    # decision (action logits / Q-values) over an internal critic head.
    picks = []
    if named_linear:
        picks.append(named_linear[0])
        by_name = dict(named_linear)
        head = by_name.get("action_net") or by_name.get("q_net.q_net.2") or by_name.get("q_net.q_net.4")
        if head is not None:
            picks.append(("action_net" if "action_net" in by_name else "q_head", head))
        elif len(named_linear) > 1:
            picks.append(named_linear[-1])

    for name, module in picks:
        w = module.weight.detach().cpu().numpy()
        cropped = w[:max_dim, :max_dim]
        layers[name] = np.round(cropped, 3).tolist()

    return layers


class LiveMetricsCallback(BaseCallback):
    """Streams episode rewards, SB3 loss diagnostics, and rendered frames."""

    def __init__(
        self,
        event_queue: "queue.Queue",
        stop_event: threading.Event,
        render_env=None,
        frame_every_steps: int = 200,
        weight_every_steps: int = 2000,
        min_frame_interval_s: float = 0.08,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.render_env = render_env
        self.frame_every_steps = max(1, frame_every_steps)
        self.weight_every_steps = max(1, weight_every_steps)
        self.min_frame_interval_s = min_frame_interval_s
        self._last_logged_metrics: dict = {}
        self._last_frame_time = 0.0

    def _emit(self, event: dict):
        event.setdefault("step", int(self.num_timesteps))
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass

    def _on_training_start(self) -> None:
        self._emit({"type": "status", "state": "training"})

    def _on_step(self) -> bool:
        if self.stop_event.is_set():
            self._emit({"type": "status", "state": "stopped"})
            return False

        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._emit(
                    {
                        "type": "episode",
                        "reward": float(ep["r"]),
                        "length": int(ep["l"]),
                    }
                )

        # SB3's logger only gets new values once per rollout/update, so we
        # dedupe by comparing against the last snapshot we sent.
        current = dict(getattr(self.model.logger, "name_to_value", {}))
        changed = {
            k: float(v)
            for k, v in current.items()
            if k.startswith("train/") and self._last_logged_metrics.get(k) != v
        }
        if changed:
            self._last_logged_metrics.update(changed)
            self._emit({"type": "train_metrics", "metrics": changed})

        if self.render_env is not None and self.num_timesteps % self.frame_every_steps == 0:
            now = time.time()
            if now - self._last_frame_time >= self.min_frame_interval_s:
                self._last_frame_time = now
                try:
                    frame = self.render_env.render()
                    data_uri = encode_frame(frame)
                    if data_uri:
                        self._emit({"type": "frame", "image": data_uri})
                except Exception:
                    pass

        if self.num_timesteps % self.weight_every_steps == 0:
            try:
                layers = extract_weight_snapshot(self.model)
                if layers:
                    self._emit({"type": "weights", "layers": layers})
            except Exception:
                pass

        return True

    def _on_training_end(self) -> None:
        self._emit({"type": "status", "state": "finished"})
