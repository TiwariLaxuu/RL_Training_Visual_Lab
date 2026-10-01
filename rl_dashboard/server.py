"""Local FastAPI dashboard: pick any Gymnasium env + SB3 algorithm from the
browser, watch it train live (rewards, losses, gradients, weights, frames),
and explore the RL math (Bellman equations, policy gradients, gradient
descent) in an interactive "Math Lab".

Run with: python run_dashboard.py
"""

from __future__ import annotations

import asyncio
import queue
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rl_dashboard.env_docs import get_env_doc
from rl_dashboard.envs import list_envs, make_env
from rl_dashboard.gradient_lab import run_1d_descent, run_gradient_descent
from rl_dashboard.introspect import architecture_summary, get_hyperparams, trace_forward
from rl_dashboard.trainer import VisualTrainer, compatible_algos

app = FastAPI(title="RL Dashboard")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Single-job state: this is a local, one-user tool, so one training run
# at a time is enough — no job queue, no auth, no persistence.
current_job: dict[str, VisualTrainer | None] = {"trainer": None}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/envs")
def api_list_envs():
    return [e.__dict__ for e in list_envs()]


@app.get("/api/algos")
def api_algos(env_id: str):
    try:
        env = make_env(env_id, render=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Couldn't create '{env_id}': {exc}") from exc
    try:
        algos = compatible_algos(env.action_space)
        return {
            "algos": algos,
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
        }
    finally:
        env.close()


class TrainRequest(BaseModel):
    env_id: str
    algo: str = "PPO"
    total_timesteps: int = 20_000
    learning_rate: float = 3e-4
    seed: int | None = None


@app.post("/api/train/start")
def api_train_start(req: TrainRequest):
    existing = current_job["trainer"]
    if existing is not None and existing.is_alive():
        raise HTTPException(status_code=409, detail="A training job is already running.")

    trainer = VisualTrainer(
        env_id=req.env_id,
        algo=req.algo,
        total_timesteps=req.total_timesteps,
        learning_rate=req.learning_rate,
        seed=req.seed,
    )
    current_job["trainer"] = trainer
    trainer.start()
    return {"status": "started"}


@app.post("/api/train/stop")
def api_train_stop():
    trainer = current_job["trainer"]
    if trainer is None:
        raise HTTPException(status_code=400, detail="No active training job.")
    trainer.stop()
    return {"status": "stopping"}


@app.get("/api/train/status")
def api_train_status():
    trainer = current_job["trainer"]
    if trainer is None:
        return {"active": False}
    return {"active": trainer.is_alive(), "env_id": trainer.env_id, "algo": trainer.algo}


class GradientLab2DRequest(BaseModel):
    learning_rate: float = 0.05
    steps: int = 40
    start_w: float = -4.0
    start_b: float = 4.0
    n_points: int = 30
    true_w: float = 2.0
    true_b: float = -1.0
    noise: float = 1.0
    seed: int = 0


@app.post("/api/gradient-lab/2d")
def api_gradient_lab_2d(req: GradientLab2DRequest):
    return run_gradient_descent(**req.model_dump())


class GradientLab1DRequest(BaseModel):
    learning_rate: float = 0.1
    steps: int = 30
    start_x: float = 8.0


@app.post("/api/gradient-lab/1d")
def api_gradient_lab_1d(req: GradientLab1DRequest):
    return run_1d_descent(**req.model_dump())


@app.get("/api/report")
def api_report():
    trainer = current_job["trainer"]
    if trainer is None or trainer.model is None:
        return {"available": False}

    model, algo = trainer.model, trainer.algo

    sample_env = make_env(trainer.env_id, render=False)
    obs, _ = sample_env.reset()
    try:
        forward_trace = trace_forward(model, algo, obs)
    finally:
        sample_env.close()

    doc = get_env_doc(trainer.env_id)

    return {
        "available": True,
        "env_id": trainer.env_id,
        "algo": algo,
        "total_timesteps_requested": trainer.total_timesteps,
        "trained_at_step": int(model.num_timesteps),
        "still_training": trainer.is_alive(),
        "architecture": architecture_summary(model, algo),
        "hyperparams": get_hyperparams(model, algo),
        "forward_trace": forward_trace,
        "env_doc": {"reward_function": doc.reward_function, "termination": doc.termination} if doc else None,
    }


class TestRunRequest(BaseModel):
    n_episodes: int = 10
    deterministic: bool = True


@app.post("/api/test/run")
def api_test_run(req: TestRunRequest):
    trainer = current_job["trainer"]
    if trainer is None or trainer.model is None:
        raise HTTPException(status_code=400, detail="No trained model yet — train first.")
    if trainer.is_alive():
        raise HTTPException(status_code=409, detail="Training is still running — stop or wait for it to finish before testing.")

    n = max(1, min(req.n_episodes, 50))
    results = trainer.run_test_episodes(n_episodes=n, deterministic=req.deterministic)

    successes = [r for r in results if r["success"] is True]
    return {
        "results": results,
        "summary": {
            "n_episodes": len(results),
            "n_success": len(successes),
            "success_rate": len(successes) / len(results) if results else 0,
            "avg_reward": sum(r["reward"] for r in results) / len(results) if results else 0,
            "avg_length": sum(r["length"] for r in results) / len(results) if results else 0,
        },
    }


@app.websocket("/ws/train")
async def ws_train(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            trainer = current_job["trainer"]
            if trainer is None:
                await asyncio.sleep(0.2)
                continue
            try:
                event = trainer.event_queue.get_nowait()
                await websocket.send_json(event)
            except queue.Empty:
                await asyncio.sleep(0.03)
    except WebSocketDisconnect:
        return
