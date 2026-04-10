"""M3bot Reach task environment configuration.

Task: Command the EE (g1 palm) to reach random 3D positions in the robot workspace.
Action space: 4 joint positions (joint0-3).
Observation space: joint_pos(4) + joint_vel(4) + ee_pose_command(7) + last_action(4).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import m3_tasks.m3_reach.mdp as mdp

from m3_tasks.m3_robot_cfg import M3_ARM_CFG, M3_EE_BODY

##
# Scene
##


@configclass
class M3ReachSceneCfg(InteractiveSceneCfg):
    """Scene with robot, ground plane, and lighting."""

    robot = M3_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# Commands
##


@configclass
class M3ReachCommandsCfg:
    """Uniform pose command for EE target."""

    ee_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name=M3_EE_BODY,
        resampling_time_range=(12.0, 12.0),  # match episode_length_s → goal fixed for whole episode
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            # Robot spawned with 180° Z rotation → body +X = world -X.
            # Arm extends in world +X (body -X), so use NEGATIVE pos_x
            # to place targets in front of the arm.
            #   body pos_x=(-0.18, -0.05) → world X = (0.30, 0.43)
            # Adjust bounds after observing arm reach in sim.
            pos_x=(-0.18, -0.05),
            pos_y=(-0.08, 0.08),      # symmetric side-to-side (body ±Y → world ∓Y)
            pos_z=(0.05, 0.20),       # 5–20 cm height
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),         # locked: 4-DOF cannot achieve arbitrary orientation
            yaw=(0.0, 0.0),
        ),
    )


##
# Actions
##


@configclass
class M3ReachActionsCfg:
    """Joint position action for arm joints only."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint0", "joint1", "joint2", "joint3"],
        scale=0.5,
        use_default_offset=True,
    )


##
# Observations
##


@configclass
class M3ReachObservationsCfg:
    """State-based observations for reach task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_pose_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Events
##


@configclass
class M3ReachEventsCfg:
    """Reset events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


##
# Rewards
##


@configclass
class M3ReachRewardsCfg:
    """Reward shaping for reach task."""

    # Primary: minimize distance to goal (positive reward via tanh)
    reach_reward = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[M3_EE_BODY]),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )

    # Secondary: penalize large position error
    position_penalty = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[M3_EE_BODY]),
            "command_name": "ee_pose",
        },
    )

    # Smoothness penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Terminations
##


@configclass
class M3ReachTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Curriculum
##


@configclass
class M3ReachCurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 10000},
    )


##
# Environment
##


@configclass
class M3ReachEnvCfg(ManagerBasedRLEnvCfg):
    """M3bot reach environment — state-based observations."""

    scene: M3ReachSceneCfg = M3ReachSceneCfg(num_envs=4096, env_spacing=1.5)
    observations: M3ReachObservationsCfg = M3ReachObservationsCfg()
    actions: M3ReachActionsCfg = M3ReachActionsCfg()
    commands: M3ReachCommandsCfg = M3ReachCommandsCfg()
    rewards: M3ReachRewardsCfg = M3ReachRewardsCfg()
    terminations: M3ReachTerminationsCfg = M3ReachTerminationsCfg()
    events: M3ReachEventsCfg = M3ReachEventsCfg()
    curriculum: M3ReachCurriculumCfg = M3ReachCurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 60.0
        self.viewer.eye = (1.0, 1.0, 1.0)
        self.viewer.lookat = (0.3, -0.25, 0.1)
