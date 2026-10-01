"""rl_dashboard — a local, visual training dashboard for Stable-Baselines3.

Wraps any Gymnasium environment + SB3 algorithm and streams live training
internals (rewards, losses, gradient norms, policy network weights, and
rendered frames) to a browser dashboard over a websocket, alongside an
interactive "Math Lab" explaining the Bellman equations, the policy
gradient theorem, and gradient descent itself.
"""

__version__ = "0.1.0"
