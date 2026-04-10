"""M3bot robot configuration for IsaacLab tasks.

Robot: M3bot (4-DOF arm + binary gripper)
USD:   ARM/M3bot.usd
URDF:  robot.urdf

Joint layout (from robot.urdf):
  joint0: revolute, mb1→b2, axis=(0,1,0),  limits=[-π/2, π/2]   MG996R
  joint1: revolute, b1→m1,  axis=(0,0,-1), limits=[-π/2, π/2]   20kg servo
  joint2: revolute, m1→f2,  axis=(0,0,-1), limits=[-π/2, π/2]   MG996R
  joint3: revolute, f2→g1,  axis=(0,0,1),  limits=[-π/2, π/2]   MG996R
  left_finger:  prismatic, g1→g3, limits=[-0.01, 0.0] (open=0.0, close=0.008)
  right_finger: prismatic, g1→g4, limits=[0.0,  0.01] (open=0.0, close=-0.008)

EE body: link4 (palm — USD renames URDF 'g1' → 'link4')
Camera (standalone prim, not in URDF, placed at env level):
  Side-view at env pos (0.05, -0.80, 0.50), looking at scene center (-0.05, -0.10, 0.15)
  Covers: X -0.49..+0.59 (robot base +0.25 ✓, workspace -0.40 ✓), Z -0.16..+0.42 ✓
  Intrinsics: RealSense D435 RGB — H-FOV=69°, focal_length=15.24mm
  Rotation (w,x,y,z) = (0.52612, -0.84745, -0.06023, 0.03739) for convention="ros"

Servo specs (actual hardware):
  MG996R:     stall torque 1.08 N·m, speed 6.2 rad/s  @ 6V
  20kg servo: stall torque 1.96 N·m, speed 6.5 rad/s
  MG90S:      stall torque 0.245 N·m → 5.0 N linear via gear, speed 0.05 m/s
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg

##
# Convenience constants
##

M3_EE_BODY = "end_effector"  # Xform fixed-jointed to link4/g1 in USD
M3_ARM_JOINT_EXPR = ["joint[0-3]"]
M3_GRIPPER_JOINT_EXPR = ["left_finger", "right_finger"]

# Gripper commands (open / close)
M3_GRIPPER_OPEN = {"left_finger": 0.0, "right_finger": 0.0}
M3_GRIPPER_CLOSE = {"left_finger": 0.008, "right_finger": -0.008}

# Camera pose taken from URDF `camera_joint` (child link `camera_link`).
# URDF origin: xyz=(0.433, 0.026, 0.209), rpy=(0.000, 0.439, 3.140)
# World position = robot spawn (0.25, -0.25, 0.0) + urdf xyz = (0.683, -0.224, 0.209)
# Quaternion computed from URDF RPY (roll,pitch,yaw) following the URDF/ROS convention
# and left as (w,x,y,z) here. We interpret this rotation using CameraCfg's
# "world" convention (forward=+X, up=+Z) so the camera faces the robot as in URDF.
M3_CAMERA_POS = (0.683, -0.224, 0.209)
M3_CAMERA_ROT = (0.000777, -0.217742, 0.000173, 0.976006)

##
# Articulation config
##

M3_ARM_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="ARM/M3bot.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8, # default 4 is not enough for stable stacking with high-friction gripper
            solver_velocity_iteration_count=0, # default 1, but 0 seems to reduce high-frequency oscillations from contact instability
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.25, -0.25, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),  # 180° around Z — USD default faces backward
        joint_pos={
            "joint0": 0.0,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "left_finger": 0.0,
            "right_finger": 0.0,
        },
    ),
    actuators={
        # Arm joints — high stiffness/damping for RL stability (real servo specs too weak to hold against gravity)
        "mg996r_act": ImplicitActuatorCfg(
            joint_names_expr=["joint0", "joint2", "joint3"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=6.2,
            stiffness=10000.0,
            damping=1000.0,
        ),
        # joint1 (shoulder) — carries most of arm weight
        "mg20kg_act": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=6.5,
            stiffness=100000.0,
            damping=400.0,
        ),
        # Gripper fingers — matched exactly to M3bot_Simple.py values.
        # velocity_limit=0.05 caused binary-action oscillation (too slow to reach 0.008m target).
        # left_finger: closes at +0.008 m, right_finger: closes at -0.008 m (opposing directions).
        "mg90s_left": ImplicitActuatorCfg(
            joint_names_expr=["left_finger"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "mg90s_right": ImplicitActuatorCfg(
            joint_names_expr=["right_finger"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)

##
##
# Camera config (standalone side-view camera)
##

M3_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/side_cam",
    update_period=0.0,  # render every physics step — guarantees fresh data on every obs collection
    height=480,   # D435 standard: 640×480 landscape
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=15.24,        # D435 RGB H-FOV=69°: f = aperture/(2*tan(34.5°))
        focus_distance=400.0,
        horizontal_aperture=20.955,  # standard sensor width; H-FOV = 2*atan(20.955/30.48)=69°
        # Far clip 1.8 m: all workspace content is ≤1.2 m from camera,
        # nearest adjacent env is ≥2.07 m away → safely excluded.
        clipping_range=(0.05, 1.8),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=M3_CAMERA_POS,
        rot=M3_CAMERA_ROT,
        convention="world",
    ),
)
