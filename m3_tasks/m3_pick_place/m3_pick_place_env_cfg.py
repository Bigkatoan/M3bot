"""M3bot Pick-and-Place task environment configuration.

Task: Grasp a cube from its spawn location and place it at a random goal position.
      More complex than lift: object must reach a specific 3D target position.
Actions: 4 arm joints + 1 binary gripper.
Observations:
  - joint_pos (6)
  - joint_vel (6)
  - object position in robot frame (3)
  - goal command (7)
  - last actions (5)
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

import m3_tasks.m3_pick_place.mdp as mdp

from m3_tasks.m3_robot_cfg import M3_ARM_CFG, M3_EE_BODY, M3_GRIPPER_CLOSE, M3_GRIPPER_OPEN

##
# Scene
##


@configclass
class M3PickPlaceSceneCfg(InteractiveSceneCfg):
    robot = M3_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Green cube to pick up and place
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=2.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Robot base at world (0.25,-0.25,0). Arm extends in world +X (180° Z rotation).
            # init at 0.37 → dist from base = 0.12m = 120mm in front.
            pos=(0.37, -0.25, 0.01),
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
class M3PickPlaceCommandsCfg:
    """Random 3D goal for the object (can be anywhere in reachable space)."""

    object_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name=M3_EE_BODY,
        resampling_time_range=(15.0, 15.0),  # match episode_length_s → goal fixed for whole episode
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            # In robot frame; arm faces -X. Front = neg robot_x.
            # 60 mm min → 180 mm max; lateral ±60 mm; height 0→150 mm.
            # Worst-case 3D distance: sqrt(0.18²+0.06²+0.15²) ≈ 0.238 m — keep
            # z low so arm can actually reach; XY capped at 60-180 mm.
            pos_x=(-0.18, -0.06),
            pos_y=(-0.06, 0.06),
            pos_z=(0.005, 0.15),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


##
# Actions
##


@configclass
class M3PickPlaceActionsCfg:
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
class M3PickPlaceObservationsCfg:
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
class M3PickPlaceEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Offsets added to init_state.pos=(0.37,-0.25,0.01). Robot base=(0.25,-0.25,0).
            # world_x ∈ (0.31, 0.43) → dist_x ∈ (0.06, 0.18) — all in FRONT (+X) ✓
            # world_y ∈ (-0.31,-0.19) → dist_y ≤ 0.06
            # max XY = sqrt(0.18²+0.06²) ≈ 0.189 m < 0.19 m ✓  min = 0.06 m ✓
            "pose_range": {"x": (-0.06, 0.06), "y": (-0.06, 0.06), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


##
# Rewards
##


@configclass
class M3PickPlaceRewardsCfg:
    # Stage 1: move EE toward object
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object"), "ee_frame_cfg": SceneEntityCfg("robot", body_names=[M3_EE_BODY])},
    )

    # Stage 2: lift off ground
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=15.0,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("object")},
    )

    # Stage 3: transport to goal (coarse) — only active when lifted
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        weight=16.0,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    )

    # Stage 3: transport (fine-grained) — only active when lifted
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
class M3PickPlaceTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    # Fail immediately if cube escapes the arm's reach (190 mm).
    object_out_of_reach = DoneTerm(
        func=mdp.object_out_of_reach,
        params={"max_reach": 0.19},
    )

    # Fail immediately if cube drifts behind the robot base.
    object_behind_robot = DoneTerm(
        func=mdp.object_behind_robot,
    )


##
# Curriculum
##


@configclass
class M3PickPlaceCurriculumCfg:
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
class M3PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """M3bot pick-and-place environment — state-based observations."""

    scene: M3PickPlaceSceneCfg = M3PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: M3PickPlaceObservationsCfg = M3PickPlaceObservationsCfg()
    actions: M3PickPlaceActionsCfg = M3PickPlaceActionsCfg()
    commands: M3PickPlaceCommandsCfg = M3PickPlaceCommandsCfg()
    rewards: M3PickPlaceRewardsCfg = M3PickPlaceRewardsCfg()
    terminations: M3PickPlaceTerminationsCfg = M3PickPlaceTerminationsCfg()
    events: M3PickPlaceEventsCfg = M3PickPlaceEventsCfg()
    curriculum: M3PickPlaceCurriculumCfg = M3PickPlaceCurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0  # Longer for pick+place
        self.sim.dt = 1.0 / 60.0
        self.viewer.eye = (1.0, 1.0, 1.0)
        self.viewer.lookat = (0.3, -0.25, 0.1)
