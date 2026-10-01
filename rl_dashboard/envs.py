"""Discover and safely construct Gymnasium environments for the dashboard.

The full Gymnasium registry includes environments that need optional
system packages (Box2D, MuJoCo, Atari ROMs via shimmy). We list everything
so the web portal can show it, but tag each entry with the dependency
family it needs so the UI can warn before the user clicks "train" on
something that isn't installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym

# Order matters: first matching prefix wins.
_FAMILY_PREFIXES = [
    ("gymnasium.envs.classic_control", "classic_control", None),
    ("gymnasium.envs.phys2d", "classic_control", None),
    ("gymnasium.envs.tabular", "toy_text", None),
    ("gymnasium.envs.toy_text", "toy_text", None),
    ("gymnasium.envs.box2d", "box2d", "box2d-py (pip install gymnasium[box2d])"),
    ("gymnasium.envs.mujoco", "mujoco", "mujoco (pip install gymnasium[mujoco])"),
]

# A hand-picked shortlist that trains fast and renders well — used to sort
# the dropdown so good defaults show up first instead of being buried.
FEATURED = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "LunarLander-v3",
    "LunarLanderContinuous-v3",
    "BipedalWalker-v3",
    "FrozenLake-v1",
    "CliffWalking-v0",
    "Taxi-v3",
]


@dataclass
class EnvInfo:
    id: str
    family: str
    requires: str | None
    featured: bool


def list_envs() -> list[EnvInfo]:
    """Return every registered Gymnasium env, tagged by dependency family."""
    infos: list[EnvInfo] = []
    for env_id, spec in gym.envs.registry.items():
        entry_point = spec.entry_point
        mod = entry_point.split(":")[0] if isinstance(entry_point, str) else ""

        family, requires = "other", None
        for prefix, fam, req in _FAMILY_PREFIXES:
            if mod.startswith(prefix):
                family, requires = fam, req
                break
        else:
            if not isinstance(entry_point, str):
                # unresolvable entry points (mujoco-py / atari via shimmy
                # when the extra isn't installed) raise a helper function
                # instead of pointing at a real module.
                family, requires = "unavailable", "missing optional dependency"

        infos.append(EnvInfo(env_id, family, requires, env_id in FEATURED))

    infos.sort(key=lambda e: (not e.featured, e.family, e.id))
    return infos


def make_env(env_id: str, render: bool = True):
    """Create an env, optionally with rgb_array rendering enabled.

    Raises whatever gym.make raises (e.g. missing dependency) — the caller
    is expected to turn that into a clean error message for the frontend.
    """
    kwargs = {}
    if render:
        kwargs["render_mode"] = "rgb_array"
    return gym.make(env_id, **kwargs)
