"""M3bot tasks package — registers all Gymnasium environments.

Usage:
    import gymnasium as gym
    import m3_tasks  # registers all envs

    env = gym.make("Isaac-M3-Reach-v0")
    env = gym.make("Isaac-M3-Reach-Vision-v0")
    env = gym.make("Isaac-M3-Lift-v0")
    # ...

Environments:
    State-based (flat obs):
        Isaac-M3-Reach-v0       — reach EE to commanded 3D pose
        Isaac-M3-Lift-v0        — pick up cube and move to 3D goal
        Isaac-M3-Push-v0        — push cube to XY goal on ground
        Isaac-M3-PickPlace-v0   — grasp cube and place at 3D goal

    Vision-augmented (state + 128×128 RGB side camera):
        Isaac-M3-Reach-Vision-v0
        Isaac-M3-Lift-Vision-v0
        Isaac-M3-Push-Vision-v0
        Isaac-M3-PickPlace-Vision-v0
"""

import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv

from m3_tasks.m3_reach.m3_reach_env_cfg import M3ReachEnvCfg
from m3_tasks.m3_reach.m3_reach_vision_env_cfg import M3ReachVisionEnvCfg

from m3_tasks.m3_lift.m3_lift_env_cfg import M3LiftEnvCfg
from m3_tasks.m3_lift.m3_lift_vision_env_cfg import M3LiftVisionEnvCfg

from m3_tasks.m3_push.m3_push_env_cfg import M3PushEnvCfg
from m3_tasks.m3_push.m3_push_vision_env_cfg import M3PushVisionEnvCfg

from m3_tasks.m3_pick_place.m3_pick_place_env_cfg import M3PickPlaceEnvCfg
from m3_tasks.m3_pick_place.m3_pick_place_vision_env_cfg import M3PickPlaceVisionEnvCfg

##
# Register environments
##

# --- Reach ---
gym.register(
    id="Isaac-M3-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": M3ReachEnvCfg()},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-M3-Reach-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": M3ReachVisionEnvCfg()},
    disable_env_checker=True,
)

# --- Lift ---
gym.register(
    id="Isaac-M3-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": M3LiftEnvCfg()},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-M3-Lift-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": M3LiftVisionEnvCfg()},
    disable_env_checker=True,
)

# --- Push ---
gym.register(
    id="Isaac-M3-Push-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": M3PushEnvCfg()},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-M3-Push-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": M3PushVisionEnvCfg()},
    disable_env_checker=True,
)

# --- Pick & Place ---
gym.register(
    id="Isaac-M3-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": M3PickPlaceEnvCfg()},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-M3-PickPlace-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": M3PickPlaceVisionEnvCfg()},
    disable_env_checker=True,
)
