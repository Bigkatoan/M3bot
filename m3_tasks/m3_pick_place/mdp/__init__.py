"""MDP module for M3bot pick-and-place task."""

from isaaclab.envs.mdp import (  # noqa: F401
    joint_pos_rel,
    joint_vel_rel,
    generated_commands,
    last_action,
    action_rate_l2,
    joint_vel_l2,
    time_out,
    reset_scene_to_default,
    reset_root_state_uniform,
    root_height_below_minimum,
    modify_reward_weight,
)
from m3_tasks.m3_lift.mdp.observations import object_position_in_robot_root_frame  # noqa: F401
from isaaclab.envs.mdp.actions.actions_cfg import (  # noqa: F401
    JointPositionActionCfg,
    BinaryJointPositionActionCfg,
)
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg  # noqa: F401

from .rewards import (  # noqa: F401
    object_is_lifted,
    object_ee_distance,
    object_goal_distance,
    object_placed_at_goal,
)
from .terminations import object_reached_goal  # noqa: F401
