"""Look inside a trained SB3 model: describe its MLP architecture, and run
one real forward pass with hooks so every intermediate number shown on the
Report page came from an actual `torch` computation, not a mock-up.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces


def _policy_sequential(model, algo):
    """The Sequential stack that actually produces the agent's decision."""
    if algo == "DQN":
        return model.policy.q_net.q_net, "q_net.q_net"
    return model.policy.mlp_extractor.policy_net, "mlp_extractor.policy_net"


def architecture_summary(model, algo: str) -> dict:
    seq, prefix = _policy_sequential(model, algo)
    layers = []
    for i, module in enumerate(seq):
        cls = module.__class__.__name__
        entry = {"name": f"{prefix}.{i}", "type": cls}
        if cls == "Linear":
            entry["in_features"] = module.in_features
            entry["out_features"] = module.out_features
        layers.append(entry)

    if algo != "DQN":
        head = model.policy.action_net
        layers.append(
            {
                "name": "action_net",
                "type": "Linear",
                "in_features": head.in_features,
                "out_features": head.out_features,
                "role": "action head",
            }
        )
    else:
        last_linear = [l for l in layers if l["type"] == "Linear"][-1]
        last_linear["role"] = "Q-value head"

    return {
        "layers": layers,
        "total_parameters": sum(p.numel() for p in model.policy.parameters()),
        "activation": "ReLU" if algo == "DQN" else "Tanh",
    }


def trace_forward(model, algo: str, obs) -> dict:
    """Run a single real forward pass through the policy network, hooking
    every Linear/activation layer on the decision-making path so we can
    show the exact numbers flowing through it.
    """
    device = model.device
    obs_arr = np.asarray(obs, dtype=np.float32)
    obs_tensor = torch.as_tensor(obs_arr).float().unsqueeze(0).to(device)

    if algo == "DQN":
        prefixes = ("q_net.q_net",)
    else:
        prefixes = ("features_extractor", "mlp_extractor.policy_net", "action_net")

    activations: list[dict] = []

    def make_hook(name):
        def hook(module, inp, out):
            arr = out.detach().cpu().numpy().reshape(-1)
            activations.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "size": int(arr.size),
                    "sample": np.round(arr[:12], 4).tolist(),
                    "in_features": getattr(module, "in_features", None),
                    "out_features": getattr(module, "out_features", None),
                }
            )

        return hook

    handles = []
    for name, module in model.policy.named_modules():
        if module.__class__.__name__ not in ("Linear", "Tanh", "ReLU", "Flatten"):
            continue
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            handles.append(module.register_forward_hook(make_hook(name)))

    action_space = model.policy.action_space
    result = {
        "input": np.round(obs_arr, 4).tolist(),
        "activations": None,
        "output": None,
        "action_space_type": action_space.__class__.__name__,
    }

    with torch.no_grad():
        if algo == "DQN":
            q_values = model.policy.q_net(obs_tensor).cpu().numpy().flatten()
            result["output"] = {
                "kind": "q_values",
                "values": np.round(q_values, 4).tolist(),
                "chosen_action": int(np.argmax(q_values)),
            }
        else:
            actions, values, log_prob = model.policy.forward(obs_tensor, deterministic=True)
            if isinstance(action_space, spaces.Discrete):
                logits_full = np.array(next(a["sample"] for a in activations if a["name"] == "action_net"))
                probs = torch.softmax(torch.as_tensor(logits_full), dim=0).numpy()
                result["output"] = {
                    "kind": "action_probs",
                    "logits": np.round(logits_full, 4).tolist(),
                    "probs": np.round(probs, 4).tolist(),
                    "chosen_action": int(actions.cpu().numpy().flatten()[0]),
                }
            else:
                mean_action = actions.cpu().numpy().flatten()
                result["output"] = {
                    "kind": "continuous_action",
                    "chosen_action": np.round(mean_action, 4).tolist(),
                }
            result["state_value"] = float(values.cpu().numpy().flatten()[0])

    for h in handles:
        h.remove()

    result["activations"] = activations
    return result


PPO_HYPERPARAM_KEYS = ["learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef"]
A2C_HYPERPARAM_KEYS = ["learning_rate", "n_steps", "gamma", "gae_lambda", "ent_coef", "vf_coef"]
DQN_HYPERPARAM_KEYS = ["learning_rate", "buffer_size", "learning_starts", "batch_size", "gamma", "exploration_fraction", "exploration_final_eps", "target_update_interval", "train_freq"]

HYPERPARAM_KEYS = {"PPO": PPO_HYPERPARAM_KEYS, "A2C": A2C_HYPERPARAM_KEYS, "DQN": DQN_HYPERPARAM_KEYS}


def get_hyperparams(model, algo: str) -> dict:
    out = {}
    for key in HYPERPARAM_KEYS.get(algo, []):
        val = getattr(model, key, None)
        if callable(val):
            try:
                val = val(1.0)  # SB3 schedules are functions of "progress remaining"
            except Exception:
                val = str(val)
        if hasattr(val, "item"):
            val = val.item()
        if not isinstance(val, (int, float, str, bool, type(None))):
            val = str(val)
        out[key] = val
    return out
