"""Hand-written, per-environment explanations of the reward function and
termination condition, plus a heuristic for classifying a finished test
episode as a success or a failure with a plain-English reason.

Gymnasium doesn't expose these as structured data — they're written into
each env's docs/source. We keep a curated table for the featured envs and
fall back to a generic length/return-based heuristic for everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EnvDoc:
    reward_function: str
    termination: str
    success_fn: Callable[[float, int, bool, bool, Optional[int]], tuple[Optional[bool], str]]


def _cartpole_success(total_reward, length, terminated, truncated, max_steps):
    if truncated and not terminated:
        return True, f"Reached the {max_steps}-step time limit — the episode was truncated, not ended by failure. The pole never fell."
    return False, f"Terminated early at step {length}: the pole angle or cart position crossed the failure threshold before the {max_steps}-step limit."


def _mountaincar_success(total_reward, length, terminated, truncated, max_steps):
    if terminated and not truncated:
        return True, f"Reached the goal flag at step {length}, before the {max_steps}-step limit — that's what ends the episode successfully here."
    return False, f"Never reached the goal within {max_steps} steps (reward accumulates -1/step, so total reward near -{max_steps} means it ran out of time)."


def _pendulum_success(total_reward, length, terminated, truncated, max_steps):
    avg = total_reward / max(length, 1)
    if avg > -2.0:
        return True, f"Average reward per step was {avg:.2f} — close to 0 means the pendulum stayed near upright with little control effort."
    return False, f"Average reward per step was {avg:.2f} — well below 0 means it spent most of the episode away from upright or using large actions."


def _acrobot_success(total_reward, length, terminated, truncated, max_steps):
    if terminated and not truncated:
        return True, f"Swung the foot above the target height at step {length}, ending the episode early — that's the goal condition."
    return False, f"Never reached the target height within {max_steps} steps."


def _lunarlander_success(total_reward, length, terminated, truncated, max_steps):
    if total_reward >= 200:
        return True, f"Total reward {total_reward:.0f} ≥ 200, SB3/Gym's usual bar for 'solved' — a controlled landing between the flags."
    if terminated and total_reward < 0:
        return False, f"Terminated with total reward {total_reward:.0f} — most likely a crash (crashing gives a large one-off penalty)."
    return False, f"Total reward {total_reward:.0f} — landed but not cleanly, or ran out of fuel/time without a clean landing."


def _frozenlake_success(total_reward, length, terminated, truncated, max_steps):
    if terminated and total_reward > 0:
        return True, f"Reached the goal tile at step {length} (reward is 0 everywhere except +1 for the goal)."
    if terminated:
        return False, f"Fell into a hole at step {length} — the episode terminates immediately with 0 reward."
    return False, f"Ran out of the {max_steps}-step limit without reaching the goal."


def _taxi_success(total_reward, length, terminated, truncated, max_steps):
    if terminated and total_reward > 0:
        return True, f"Completed a pickup + drop-off at step {length} (a +20 reward is only given on a successful drop-off)."
    return False, f"Ran out of the {max_steps}-step limit (each step costs -1, illegal pickup/drop-off costs -10) without finishing the trip."


def _cliffwalking_success(total_reward, length, terminated, truncated, max_steps):
    if terminated:
        return True, f"Reached the goal cell at step {length}."
    return False, f"Ran out of the {max_steps}-step limit without reaching the goal (falling off the cliff costs -100 but resets to start rather than ending the episode)."


def _bipedalwalker_success(total_reward, length, terminated, truncated, max_steps):
    if terminated and total_reward < 0:
        return False, f"Terminated with total reward {total_reward:.0f} — the walker fell (falling gives a one-off -100 penalty)."
    if total_reward > 250:
        return True, f"Total reward {total_reward:.0f} — walked the course without falling and made good forward progress."
    return False, f"Total reward {total_reward:.0f} — stayed upright but made limited forward progress, or the episode was cut short."


ENV_DOCS: dict[str, EnvDoc] = {
    "CartPole-v1": EnvDoc(
        reward_function="+1 for every timestep the pole stays upright and the cart stays on the track.",
        termination="Terminates (fails) if the pole tilts past ±12°, the cart moves past ±2.4 units from center, "
        "or the pole falls. Reaching 500 steps truncates the episode instead — that's a success, not a failure.",
        success_fn=_cartpole_success,
    ),
    "MountainCar-v0": EnvDoc(
        reward_function="-1 for every timestep, until the goal flag is reached. There's no reward shaping — "
        "the agent has to discover that swinging back before accelerating forward is worth the extra steps.",
        termination="Terminates when the car's position reaches ≥ 0.5 (the flag). Truncates at 200 steps if the goal is never reached.",
        success_fn=_mountaincar_success,
    ),
    "MountainCarContinuous-v0": EnvDoc(
        reward_function="A small penalty proportional to action magnitude each step, plus +100 on reaching the goal.",
        termination="Terminates on reaching position ≥ 0.45. Truncates at 999 steps.",
        success_fn=_mountaincar_success,
    ),
    "Pendulum-v1": EnvDoc(
        reward_function="-(θ² + 0.1·θ̇² + 0.001·action²) every step — a continuous penalty for being away from "
        "upright, moving fast, or using large torque. There's no terminal 'goal' state.",
        termination="Never terminates early — always truncates at 200 steps.",
        success_fn=_pendulum_success,
    ),
    "Acrobot-v1": EnvDoc(
        reward_function="-1 for every timestep until the free end swings above a target height.",
        termination="Terminates once the end-effector height exceeds the target. Truncates at 500 steps.",
        success_fn=_acrobot_success,
    ),
    "LunarLander-v3": EnvDoc(
        reward_function="Shaped reward based on distance/speed/angle to the pad, plus large one-off bonuses/penalties: "
        "≈+100 for a safe landing, -100 for crashing, +10 per leg touching the ground, small fuel-use penalties.",
        termination="Terminates on crashing or coming to rest on the pad. Truncates at 1000 steps.",
        success_fn=_lunarlander_success,
    ),
    "LunarLanderContinuous-v3": EnvDoc(
        reward_function="Same shaping as LunarLander-v3, with continuous thrust actions instead of discrete engines.",
        termination="Terminates on crashing or landing. Truncates at 1000 steps.",
        success_fn=_lunarlander_success,
    ),
    "BipedalWalker-v3": EnvDoc(
        reward_function="Reward for forward progress (proportional to hull horizontal speed), small penalties for "
        "motor torque used, and a one-off -100 penalty if the hull touches the ground.",
        termination="Terminates if the hull hits the ground or reaches the end of the course. Truncates at 1600 steps.",
        success_fn=_bipedalwalker_success,
    ),
    "FrozenLake-v1": EnvDoc(
        reward_function="0 everywhere except +1 for reaching the goal tile.",
        termination="Terminates immediately on falling into a hole (reward 0) or reaching the goal (reward 1). Truncates at 100 steps.",
        success_fn=_frozenlake_success,
    ),
    "Taxi-v3": EnvDoc(
        reward_function="-1 per step (fuel cost), -10 for an illegal pickup/drop-off, +20 for a correct drop-off.",
        termination="Terminates on a successful passenger drop-off. Truncates at 200 steps.",
        success_fn=_taxi_success,
    ),
    "CliffWalking-v0": EnvDoc(
        reward_function="-1 per step, -100 for stepping into the cliff region (which also resets you to the start, without ending the episode).",
        termination="Terminates on reaching the goal cell. Truncates at 200 steps.",
        success_fn=_cliffwalking_success,
    ),
}


def get_env_doc(env_id: str) -> Optional[EnvDoc]:
    return ENV_DOCS.get(env_id)


def classify_episode(env_id: str, total_reward: float, length: int, terminated: bool, truncated: bool, max_steps: Optional[int]):
    doc = ENV_DOCS.get(env_id)
    if doc is not None:
        return doc.success_fn(total_reward, length, terminated, truncated, max_steps)

    # Generic fallback for envs we haven't hand-documented.
    if truncated and not terminated:
        return True, "Reached the environment's time limit without hitting a terminal failure condition."
    if terminated:
        near_limit = max_steps is not None and length >= 0.95 * max_steps
        if near_limit:
            return True, f"Terminated at step {length}, right near the {max_steps}-step limit — likely a goal/success state rather than a failure."
        return False, f"Terminated early, at step {length} of an episode that could have run to ~{max_steps or '?'} — likely hit a failure condition."
    return None, "Episode ended without a clear terminated/truncated signal."
