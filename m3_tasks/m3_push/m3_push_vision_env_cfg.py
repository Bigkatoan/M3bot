"""M3bot Push Vision environment configuration."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


from m3_tasks.m3_robot_cfg import M3_CAMERA_CFG
from m3_tasks.m3_push.m3_push_env_cfg import M3PushEnvCfg, M3PushSceneCfg, M3PushObservationsCfg
from m3_tasks.vision_utils import set_vis_markers_guide_purpose, image_nchw


@configclass
class M3PushVisionSceneCfg(M3PushSceneCfg):
    side_cam: CameraCfg = M3_CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/side_cam", height=128, width=128)


@configclass
class M3PushVisionObservationsCfg(M3PushObservationsCfg):
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
class M3PushVisionEnvCfg(M3PushEnvCfg):
    scene: M3PushVisionSceneCfg = M3PushVisionSceneCfg(num_envs=32, env_spacing=2.5)
    observations: M3PushVisionObservationsCfg = M3PushVisionObservationsCfg()

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
