"""M3bot Lift Vision environment configuration.

Inherits from M3LiftEnvCfg; adds side camera (128×128 RGB) observation group.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


from m3_tasks.m3_robot_cfg import M3_CAMERA_CFG
from m3_tasks.m3_lift.m3_lift_env_cfg import M3LiftEnvCfg, M3LiftSceneCfg, M3LiftObservationsCfg
from m3_tasks.vision_utils import set_vis_markers_guide_purpose, image_nchw


@configclass
class M3LiftVisionSceneCfg(M3LiftSceneCfg):
    side_cam: CameraCfg = M3_CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/side_cam", height=128, width=128)


@configclass
class M3LiftVisionObservationsCfg(M3LiftObservationsCfg):
    @configclass
    class VisionPolicyCfg(ObsGroup):
        rgb = ObsTerm(
            func=image_nchw,
            params={"sensor_cfg": SceneEntityCfg("side_cam"), "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # returns raw tensor (B,C,H,W) — required by RSL-RL CNNModel

    vision_policy: VisionPolicyCfg = VisionPolicyCfg()


@configclass
class M3LiftVisionEnvCfg(M3LiftEnvCfg):
    """M3bot lift environment with RGB camera observation."""

    scene: M3LiftVisionSceneCfg = M3LiftVisionSceneCfg(num_envs=32, env_spacing=2.5)
    observations: M3LiftVisionObservationsCfg = M3LiftVisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.num_rerenders_on_reset = 2
        self.sim.render.antialiasing_mode = "FXAA"
        self.scene.num_envs = 32
        # Keep debug_vis=True (markers visible in viewport/LiveStream).
        # Startup event sets their USD purpose to 'guide' so CameraSensor skips them.
        self.events.set_vis_guide = EventTerm(
            func=set_vis_markers_guide_purpose,
            mode="startup",
        )
