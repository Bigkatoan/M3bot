"""Train M3bot RL agent with RSL-RL (PPO).

Usage examples:
    # Headless training (fastest, no display):
    ./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Reach-v0 --headless

    # Headless with video recording every 2000 steps:
    ./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Lift-v0 --headless --video --video_interval 2000

    # Livestream to browser (WebRTC at http://localhost:8211):
    ./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Push-v0 --livestream 2

    # Resume from checkpoint:
    ./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-Lift-v0 --headless --resume

    # Custom num_envs and iterations:
    ./IsaacLab/isaaclab.sh -p train.py --task Isaac-M3-PickPlace-v0 --headless --num_envs 2048 --max_iterations 5000

Available tasks (state-based):
    Isaac-M3-Reach-v0
    Isaac-M3-Lift-v0
    Isaac-M3-Push-v0
    Isaac-M3-PickPlace-v0

Available tasks (vision — adds 128x128 RGB side camera):
    Isaac-M3-Reach-Vision-v0
    Isaac-M3-Lift-Vision-v0
    Isaac-M3-Push-Vision-v0
    Isaac-M3-PickPlace-Vision-v0
"""

"""AppLauncher MUST be created before any IsaacSim/IsaacLab imports."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train M3bot RL agent with RSL-RL PPO.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    choices=[
        "Isaac-M3-Reach-v0",
        "Isaac-M3-Lift-v0",
        "Isaac-M3-Push-v0",
        "Isaac-M3-PickPlace-v0",
        "Isaac-M3-Reach-Vision-v0",
        "Isaac-M3-Lift-Vision-v0",
        "Isaac-M3-Push-Vision-v0",
        "Isaac-M3-PickPlace-Vision-v0",
    ],
    help="Name of the Gymnasium task to train.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max training iterations.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training (enables cameras)."
)
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between video recordings.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from latest checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint to resume from.")
parser.add_argument(
    "--logger",
    type=str,
    default="tensorboard",
    choices=["tensorboard", "wandb", "neptune"],
    help="Logging backend.",
)
parser.add_argument("--experiment_name", type=str, default=None, help="Override experiment name for logs.")
parser.add_argument("--run_name", type=str, default=None, help="Suffix appended to run directory name.")

# AppLauncher args: --headless, --livestream, --device, etc.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras when recording video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports (after AppLauncher)
# ---------------------------------------------------------------------------
import importlib.metadata as metadata
import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import get_checkpoint_path

# --- Patch rsl_rl check_nan to handle nested TensorDicts (vision tasks) ---
import rsl_rl.utils.utils as _rsl_utils
from tensordict import TensorDict as _TensorDict

def _check_nan_patched(obs: _TensorDict, rewards: torch.Tensor, dones: torch.Tensor) -> None:
    """check_nan that recurses into nested TensorDicts (needed for vision obs groups)."""
    def _check_td(td: _TensorDict, prefix: str = "") -> None:
        for key, val in td.items():
            if isinstance(val, _TensorDict):
                _check_td(val, prefix=f"{prefix}{key}/")
            elif torch.is_tensor(val) and val.is_floating_point():
                if torch.isnan(val).any():
                    raise ValueError(
                        f"The observation '{prefix}{key}' contains NaN values. "
                        "This usually indicates a bug in the environment's step() or reset()."
                    )
    _check_td(obs)
    if torch.isnan(rewards).any():
        raise ValueError("The rewards contain NaN values.")
    if torch.isnan(dones.float()).any():
        raise ValueError("The dones contain NaN values.")

_rsl_utils.check_nan = _check_nan_patched
# Also patch the reference inside on_policy_runner if it already imported it
try:
    import rsl_rl.runners.on_policy_runner as _runner_mod
    _runner_mod.check_nan = _check_nan_patched
except Exception:
    pass

# --- M3bot imports ---
import m3_tasks  # registers all gym envs   # noqa: F401
from m3_tasks.agents.rsl_rl_ppo_cfg import (
    M3LiftPPOCfg,
    M3LiftVisionPPOCfg,
    M3PickPlacePPOCfg,
    M3PickPlaceVisionPPOCfg,
    M3PushPPOCfg,
    M3PushVisionPPOCfg,
    M3ReachPPOCfg,
    M3ReachVisionPPOCfg,
)
from m3_tasks.m3_lift.m3_lift_env_cfg import M3LiftEnvCfg
from m3_tasks.m3_lift.m3_lift_vision_env_cfg import M3LiftVisionEnvCfg
from m3_tasks.m3_pick_place.m3_pick_place_env_cfg import M3PickPlaceEnvCfg
from m3_tasks.m3_pick_place.m3_pick_place_vision_env_cfg import M3PickPlaceVisionEnvCfg
from m3_tasks.m3_push.m3_push_env_cfg import M3PushEnvCfg
from m3_tasks.m3_push.m3_push_vision_env_cfg import M3PushVisionEnvCfg
from m3_tasks.m3_reach.m3_reach_env_cfg import M3ReachEnvCfg
from m3_tasks.m3_reach.m3_reach_vision_env_cfg import M3ReachVisionEnvCfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task registry: task_id → (EnvCfg class, AgentCfg class)
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "Isaac-M3-Reach-v0":         (M3ReachEnvCfg,      M3ReachPPOCfg),
    "Isaac-M3-Lift-v0":          (M3LiftEnvCfg,       M3LiftPPOCfg),
    "Isaac-M3-Push-v0":          (M3PushEnvCfg,       M3PushPPOCfg),
    "Isaac-M3-PickPlace-v0":     (M3PickPlaceEnvCfg,  M3PickPlacePPOCfg),
    # Vision tasks use CNN actor/critic with obs_groups routing state+image
    "Isaac-M3-Reach-Vision-v0":     (M3ReachVisionEnvCfg,     M3ReachVisionPPOCfg),
    "Isaac-M3-Lift-Vision-v0":      (M3LiftVisionEnvCfg,      M3LiftVisionPPOCfg),
    "Isaac-M3-Push-Vision-v0":      (M3PushVisionEnvCfg,      M3PushVisionPPOCfg),
    "Isaac-M3-PickPlace-Vision-v0": (M3PickPlaceVisionEnvCfg, M3PickPlaceVisionPPOCfg),
}

# Minimum supported rsl-rl version
RSL_RL_MIN_VERSION = "3.0.1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # --- Check rsl-rl version ---
    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) < version.parse(RSL_RL_MIN_VERSION):
        raise RuntimeError(
            f"rsl-rl-lib {installed_version} is installed but >={RSL_RL_MIN_VERSION} is required.\n"
            "Install: ./IsaacLab/isaaclab.sh -p -m pip install rsl-rl-lib"
        )
    print(f"[INFO] rsl-rl-lib version: {installed_version}")

    # --- Build env_cfg and agent_cfg ---
    EnvCfgCls, AgentCfgCls = TASK_REGISTRY[args_cli.task]
    env_cfg: ManagerBasedRLEnvCfg = EnvCfgCls()
    agent_cfg = AgentCfgCls()

    # CLI overrides
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    agent_cfg.seed = args_cli.seed
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # --- Logging directory ---
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging to: {log_dir}")

    env_cfg.log_dir = log_dir

    # --- Handle deprecated rsl-rl config ---
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # --- Create environment ---
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # --- Video wrapper ---
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording training videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # --- RSL-RL wrapper ---
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Resume from checkpoint ---
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")

    # --- Create runner ---
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        runner.load(resume_path)

    # --- Save configs ---
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # --- Train ---
    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    elapsed = round(time.time() - start_time, 2)
    print(f"[INFO] Training complete in {elapsed}s. Logs: {log_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
