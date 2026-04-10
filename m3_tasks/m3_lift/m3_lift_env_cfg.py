"""M3bot Lift task environment configuration.

Task: Pick up a small cube (5×5×5 cm) from the ground and bring it to a goal position.
Actions: 4 arm joint positions + 1 binary gripper command (open/close).
Observations:
  - joint_pos (6: joint0-3 + fingers)
  - joint_vel (6)
  - object position in robot root frame (3)
  - goal command (7)
  - last actions (5: arm + gripper)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
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

import m3_tasks.m3_lift.mdp as mdp

from m3_tasks.m3_robot_cfg import M3_ARM_CFG, M3_EE_BODY, M3_GRIPPER_CLOSE, M3_GRIPPER_OPEN

##
# Scene
##


@configclass
class M3LiftSceneCfg(InteractiveSceneCfg):
    """Scene with robot, cube object, ground and lighting."""

    robot = M3_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Cube to pick up — red 1 cm cube on the ground
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=2.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, -0.25, 0.005),  # On ground, in robot workspace
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

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
class M3LiftCommandsCfg:
    """Random goal pose for object to reach."""

    object_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name=M3_EE_BODY,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(-0.35, -0.10),
            pos_y=(-0.08, 0.08),
            pos_z=(0.05, 0.20),   # Target in the air — must be lifted
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


##
# Actions
##


@configclass
class M3LiftActionsCfg:
    """4 arm joints + binary gripper."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint0", "joint1", "joint2", "joint3"],
        scale=0.5,
        use_default_offset=True,
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_finger", "right_finger"],
        open_command_expr=M3_GRIPPER_OPEN,
        close_command_expr=M3_GRIPPER_CLOSE,
    )


##
# Observations
##


@configclass
class M3LiftObservationsCfg:
    """State-based observations for lift task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Events
##


@configclass
class M3LiftEventsCfg:
    """Randomise object spawn position on reset."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


##
# Rewards
##


@configclass
class M3LiftRewardsCfg:
    """Reward shaping for lift task: approach → grasp → lift → place."""

    # Stage 1: approach the object
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object"), "ee_frame_cfg": SceneEntityCfg("robot", body_names=[M3_EE_BODY])},
    )

    # Stage 2: lift the object off the ground
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=15.0,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("object")},
    )

    # Stage 3: move object toward goal (coarse)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        weight=16.0,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    )

    # Stage 3: fine-grained goal tracking
    object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_distance,
        weight=5.0,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    )

    # Penalties
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
class M3LiftTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


##
# Curriculum
##


@configclass
class M3LiftCurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


##
# Environment
##


@configclass
class M3LiftEnvCfg(ManagerBasedRLEnvCfg):
    """M3bot lift environment — state-based observations."""

    scene: M3LiftSceneCfg = M3LiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: M3LiftObservationsCfg = M3LiftObservationsCfg()
    actions: M3LiftActionsCfg = M3LiftActionsCfg()
    commands: M3LiftCommandsCfg = M3LiftCommandsCfg()
    rewards: M3LiftRewardsCfg = M3LiftRewardsCfg()
    terminations: M3LiftTerminationsCfg = M3LiftTerminationsCfg()
    events: M3LiftEventsCfg = M3LiftEventsCfg()
    curriculum: M3LiftCurriculumCfg = M3LiftCurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 60.0
        self.viewer.eye = (1.0, 1.0, 1.0)
        self.viewer.lookat = (0.3, -0.25, 0.1)
