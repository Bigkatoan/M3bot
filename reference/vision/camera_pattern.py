# ============================================================
# VISION / CAMERA PATTERN IN ISAACLAB v2.3.2
# SOURCE: stack_ik_rel_visuomotor_env_cfg.py
# ============================================================
#
# KEY RULES:
#   1. CameraCfg goes in scene setup
#   2. Image observation group MUST have concatenate_terms=False
#   3. mdp.image() func with data_type="rgb" or "distance_to_image_plane"
#   4. convention="ros" → camera +Z is forward (optical axis)
#   5. rot=(w, x, y, z) quaternion in WORLD frame
#
# QUATERNION REFERENCE:
#   table_cam at (1.0, 0.0, 0.4) looking at workspace origin:
#   rot=(0.35355, -0.61237, -0.61237, 0.35355) → looks -X and slightly down
#
#   M3bot side camera from robot.urdf camera_joint:
#   URDF xyz=(0.433, 0.026, 0.209) RPY=(0.0, 0.439, 3.14)
#   World pos = robot_spawn(0.25,-0.25,0) + (0.433,0.026,0.209) = (0.683,-0.224,0.209)
#   Quaternion (w,x,y,z) = (0.001, -0.218, 0.000, 0.976)
#
# ============================================================

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp import image
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# 1. Add camera to scene (inside your SceneCfg class):
#
#     side_cam = CameraCfg(
#         prim_path="{ENV_REGEX_NS}/side_cam",   # free-floating prim
#         update_period=0.0,
#         height=128,
#         width=128,
#         data_types=["rgb"],   # add "distance_to_image_plane" for depth
#         spawn=sim_utils.PinholeCameraCfg(
#             focal_length=24.0,
#             focus_distance=400.0,
#             horizontal_aperture=20.955,
#             clipping_range=(0.05, 3.0),
#         ),
#         offset=CameraCfg.OffsetCfg(
#             pos=(0.683, -0.224, 0.209),       # camera_joint xyz + robot spawn
#             rot=(0.001, -0.218, 0.000, 0.976), # from URDF RPY=(0, 0.439, 3.14)
#             convention="ros",
#         ),
#     )

# 2. Vision observation group (concatenate_terms MUST be False):
#
#     @configclass
#     class VisionPolicyCfg(ObsGroup):
#         rgb = ObsTerm(
#             func=image,
#             params={"sensor_cfg": SceneEntityCfg("side_cam"), "data_type": "rgb", "normalize": False}
#         )
#         # Optional depth:
#         # depth = ObsTerm(
#         #     func=image,
#         #     params={"sensor_cfg": SceneEntityCfg("side_cam"),
#         #             "data_type": "distance_to_image_plane", "normalize": False}
#         # )
#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = False   # ← REQUIRED for image obs
#
#     vision_policy: VisionPolicyCfg = VisionPolicyCfg()

# 3. Extra sim settings for rendering (optional but recommended):
#
#     self.num_rerenders_on_reset = 3
#     self.sim.render.antialiasing_mode = "DLAA"
