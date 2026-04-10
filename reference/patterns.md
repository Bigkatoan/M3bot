# =============================================================
# ISAACLAB v2.3.2 KEY PATTERNS FOR M3BOT TASKS
# =============================================================
#
# IMPORTS:
#   from isaaclab.envs import ManagerBasedRLEnvCfg
#   from isaaclab.utils import configclass
#   from dataclasses import MISSING
#
# CONFIGCLASS PATTERN:
#   @configclass                          - mandatory decorator
#   @configclass inherits another:        - uses dataclass inheritance
#   MISSING                               - must be set by child class
#   __post_init__()                       - called after dataclass init
#
# SCENE ASSETS:
#   ArticulationCfg for robots
#   RigidObjectCfg for objects (cube, box)
#   AssetBaseCfg for static objects (table, plane, lights)
#   FrameTransformerCfg for EE tracking (lift, push, pick_place)
#   CameraCfg for vision
#
# SPAWNERS (no Nucleus needed):
#   sim_utils.UsdFileCfg(usd_path="relative/path.usd")
#   sim_utils.CuboidCfg(size=(0.05,0.05,0.05))    ← cube object
#   sim_utils.SphereCfg(radius=0.03)               ← sphere object
#   sim_utils.GroundPlaneCfg()                     ← infinite ground
#   sim_utils.DomeLightCfg(color=..., intensity=3000.0) ← lighting
#
# ACTUATOR PATTERN:
#   ImplicitActuatorCfg(
#       joint_names_expr=["joint.*"],
#       effort_limit_sim=<N·m or N>,
#       velocity_limit_sim=<rad/s or m/s>,
#       stiffness=<Nm/rad>,
#       damping=<Nm·s/rad>,
#   )
#
# M3BOT SERVO SPECS (verified):
#   MG996R (joint0,2,3): stiffness=20.0, damping=0.5, effort_limit=1.08, vel_limit=6.2
#   20kg servo (joint1): stiffness=40.0, damping=1.0, effort_limit=1.96, vel_limit=6.5
#   MG90S (gripper):     stiffness=200.0(N/m), damping=5.0, effort_limit=5.0(N), vel_limit=0.05(m/s)
#
# M3BOT ROBOT INFO:
#   USD path:          "ARM/M3bot.usd"
#   spawn pos:         (0.25, -0.25, 0.0)
#   arm joints:        joint0, joint1, joint2, joint3
#   gripper joints:    left_finger (prismatic, 0→0.008m), right_finger (prismatic, 0→-0.008m)
#   EE body:           "g1" (palm link)
#   camera pos:        (0.683, -0.224, 0.209)  ← robot_spawn + camera_joint_xyz
#   camera quat:       (0.001, -0.218, 0.000, 0.976) ← from URDF camera_joint RPY
#
# GRIPPER BINARY ACTION:
#   BinaryJointPositionActionCfg(
#       asset_name="robot",
#       joint_names=["left_finger", "right_finger"],
#       open_command_expr={"left_finger": 0.0,   "right_finger": 0.0},
#       close_command_expr={"left_finger": 0.008, "right_finger": -0.008},
#   )
#
# EE FRAME TRANSFORMER (for lift/push/pick_place):
#   FrameTransformerCfg(
#       prim_path="{ENV_REGEX_NS}/Robot",
#       debug_vis=False,
#       target_frames=[FrameTransformerCfg.FrameCfg(
#           prim_path="{ENV_REGEX_NS}/Robot/g1",
#           name="end_effector",
#       )],
#   )
#
# OBJECT SPAWN (no Nucleus):
#   RigidObjectCfg(
#       prim_path="{ENV_REGEX_NS}/Object",
#       spawn=sim_utils.CuboidCfg(
#           size=(0.05, 0.05, 0.05),
#           rigid_props=sim_utils.RigidBodyPropertiesCfg(max_linear_velocity=2.0),
#           mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
#           collision_props=sim_utils.CollisionPropertiesCfg(),
#           visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
#       ),
#       init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.25, 0.055)),
#   )
#
# COMMAND TYPES:
#   Reach:      UniformPoseCommandCfg   (position + orientation of EE)
#   Lift:       UniformPoseCommandCfg   (position of object target)
#   Push:       UniformPose2dCommandCfg (xy target on table)
#   PickPlace:  UniformPoseCommandCfg   (position of object target)
#
# OBSERVATION concatenate_terms:
#   True  → all terms concatenated into flat vector (state-based policy)
#   False → each term is separate dict entry (required for images!)
#
# ENV SPACING:
#   Single task, no objects:  env_spacing=1.5
#   With objects on table:    env_spacing=2.5
#   Vision:                   env_spacing=2.5 (cameras per env)
#
# GYM IDs (register in __init__.py):
#   "Isaac-M3-Reach-v0"
#   "Isaac-M3-Reach-Vision-v0"
#   "Isaac-M3-Lift-v0"
#   "Isaac-M3-Lift-Vision-v0"
#   "Isaac-M3-Push-v0"
#   "Isaac-M3-Push-Vision-v0"
#   "Isaac-M3-PickPlace-v0"
#   "Isaac-M3-PickPlace-Vision-v0"
