"""Play/evaluate a trained M3bot RL agent with RSL-RL.

Loads a checkpoint, runs the policy, and optionally records video.
Policy is also exported to JIT (.pt) and ONNX format for deployment.

Usage examples:
    # Visualize latest checkpoint for a task (opens Isaac Sim GUI):
    ./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Reach-v0

    # Specify exact checkpoint:
    ./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Lift-v0 \\
        --checkpoint logs/rsl_rl/m3_lift/2025-01-01_12-00-00/model_1000.pt

    # Record a video and exit:
    ./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Lift-v0 --video --video_length 300

    # Headless + video (fastest):
    ./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Push-v0 --headless --video

    # Livestream to browser:
    ./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Reach-v0 --livestream 2

    # Real-time playback (slows sim to match real time):
    ./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Reach-v0 --real_time

    # Run with fewer environments:
    ./IsaacLab/isaaclab.sh -p play.py --task Isaac-M3-Lift-v0 --num_envs 4
"""

"""AppLauncher MUST be created before any IsaacSim/IsaacLab imports."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play/evaluate a trained M3bot RL agent.")
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
    help="Name of the task to evaluate.",
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments for evaluation.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file. Auto-detects latest if omitted.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record a video and exit."
)
parser.add_argument("--video_length", type=int, default=300, help="Video length in steps.")
parser.add_argument(
    "--real_time", action="store_true", default=False, help="Slow simulation to real-time speed."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric (use USD I/O ops instead)."
)
parser.add_argument(
    "--export_policy", action="store_true", default=True, help="Export policy to JIT and ONNX after loading."
)

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
import os
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)

from isaaclab_tasks.utils import get_checkpoint_path

# --- M3bot imports ---
import m3_tasks  # registers all gym envs  # noqa: F401
from m3_tasks.agents.rsl_rl_ppo_cfg import (
    M3LiftPPOCfg,
    M3PickPlacePPOCfg,
    M3PushPPOCfg,
    M3ReachPPOCfg,
)
from m3_tasks.m3_lift.m3_lift_env_cfg import M3LiftEnvCfg
from m3_tasks.m3_lift.m3_lift_vision_env_cfg import M3LiftVisionEnvCfg
from m3_tasks.m3_pick_place.m3_pick_place_env_cfg import M3PickPlaceEnvCfg
from m3_tasks.m3_pick_place.m3_pick_place_vision_env_cfg import M3PickPlaceVisionEnvCfg
from m3_tasks.m3_push.m3_push_env_cfg import M3PushEnvCfg
from m3_tasks.m3_push.m3_push_vision_env_cfg import M3PushVisionEnvCfg
from m3_tasks.m3_reach.m3_reach_env_cfg import M3ReachEnvCfg
from m3_tasks.m3_reach.m3_reach_vision_env_cfg import M3ReachVisionEnvCfg

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "Isaac-M3-Reach-v0":         (M3ReachEnvCfg,      M3ReachPPOCfg),
    "Isaac-M3-Lift-v0":          (M3LiftEnvCfg,       M3LiftPPOCfg),
    "Isaac-M3-Push-v0":          (M3PushEnvCfg,       M3PushPPOCfg),
    "Isaac-M3-PickPlace-v0":     (M3PickPlaceEnvCfg,  M3PickPlacePPOCfg),
    "Isaac-M3-Reach-Vision-v0":     (M3ReachVisionEnvCfg,     M3ReachPPOCfg),
    "Isaac-M3-Lift-Vision-v0":      (M3LiftVisionEnvCfg,      M3LiftPPOCfg),
    "Isaac-M3-Push-Vision-v0":      (M3PushVisionEnvCfg,      M3PushPPOCfg),
    "Isaac-M3-PickPlace-Vision-v0": (M3PickPlaceVisionEnvCfg, M3PickPlacePPOCfg),
}

# Log base paths per task (matches experiment_name in PPO configs)
EXPERIMENT_NAMES = {
    "Isaac-M3-Reach-v0":            "m3_reach",
    "Isaac-M3-Lift-v0":             "m3_lift",
    "Isaac-M3-Push-v0":             "m3_push",
    "Isaac-M3-PickPlace-v0":        "m3_pick_place",
    "Isaac-M3-Reach-Vision-v0":     "m3_reach",
    "Isaac-M3-Lift-Vision-v0":      "m3_lift",
    "Isaac-M3-Push-Vision-v0":      "m3_push",
    "Isaac-M3-PickPlace-Vision-v0": "m3_pick_place",
}


def main():
    installed_version = metadata.version("rsl-rl-lib")
    print(f"[INFO] rsl-rl-lib version: {installed_version}")

    # --- Build configs ---
    EnvCfgCls, AgentCfgCls = TASK_REGISTRY[args_cli.task]
    env_cfg: ManagerBasedRLEnvCfg = EnvCfgCls()
    agent_cfg = AgentCfgCls()

    # Play settings: fewer envs, no noise
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False

    agent_cfg.seed = args_cli.seed

    # --- Handle deprecated rsl-rl config ---
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # --- Find checkpoint ---
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", EXPERIMENT_NAMES[args_cli.task]))
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # --- Create environment ---
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # --- Video wrapper ---
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video ({args_cli.video_length} steps) to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # --- RSL-RL wrapper ---
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Load runner and checkpoint ---
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # --- Export policy ---
    if args_cli.export_policy:
        export_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
        else:
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            normalizer = None
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer

            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
        print(f"[INFO] Policy exported to: {export_dir}")

    # --- Get inference policy ---
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- Simulation loop ---
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    print("[INFO] Starting evaluation. Press Ctrl+C to stop.")
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # Reset recurrent states for terminated episodes (rsl-rl >= 4.0.0)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                if version.parse(installed_version) >= version.parse("2.3.0"):
                    runner.alg.policy.reset(dones)
                else:
                    runner.alg.actor_critic.reset(dones)

        timestep += 1

        # Exit after video recording
        if args_cli.video and timestep >= args_cli.video_length:
            print(f"[INFO] Video recorded ({args_cli.video_length} steps). Exiting.")
            break

        # Real-time delay
        if args_cli.real_time:
            elapsed = time.time() - start_time
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
