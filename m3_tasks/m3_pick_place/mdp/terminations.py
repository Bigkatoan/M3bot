"""MDP terminations for M3bot pick-and-place task."""
from __future__ import annotations
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms


def object_out_of_reach(
    env,
    max_reach: float = 0.19,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when object XY-distance from robot base exceeds arm reach.

    Args:
        max_reach: Maximum allowed XY distance in metres (default 0.19 = 190 mm).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    xy_dist = torch.norm(
        object.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1
    )
    return xy_dist > max_reach


def object_behind_robot(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when object drifts behind the robot base (world +X side).

    The arm extends in the world -X direction; anything with world_x >= robot world_x
    is considered behind the robot and unreachable.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    # arm extends in world +X direction; behind = world_x <= robot base world_x
    return object.data.root_pos_w[:, 0] <= robot.data.root_pos_w[:, 0]


def object_reached_goal(
    env,
    command_name: str = "object_pose",
    threshold: float = 0.03,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when object is within threshold distance of goal."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return distance < threshold
