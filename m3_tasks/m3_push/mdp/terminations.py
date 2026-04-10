"""MDP terminations for M3bot push task."""
from __future__ import annotations
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms


def object_reached_goal_xy(
    env,
    command_name: str = "object_pose",
    threshold: float = 0.03,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when object XY position is within threshold of goal."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance_xy = torch.norm(des_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    return distance_xy < threshold
