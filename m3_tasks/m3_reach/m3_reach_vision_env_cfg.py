"""M3bot Reach Vision environment configuration.

Vision variant: adds side camera (128×128 RGB) as a separate observation group.
Policy receives: state_obs (flat vector) + rgb image (128×128×3).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from m3_tasks.m3_robot_cfg import M3_CAMERA_CFG
from m3_tasks.m3_reach.m3_reach_env_cfg import M3ReachEnvCfg, M3ReachSceneCfg, M3ReachObservationsCfg
from m3_tasks.vision_utils import set_vis_markers_guide_purpose


@configclass
class M3ReachVisionSceneCfg(M3ReachSceneCfg):
    """Reach scene with side camera added."""

    side_cam: CameraCfg = M3_CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/side_cam")


@configclass
class M3ReachVisionObservationsCfg(M3ReachObservationsCfg):
    """Observations: state policy group + separate vision policy group."""

    @configclass
    class VisionPolicyCfg(ObsGroup):
        """RGB image from side camera. concatenate_terms MUST be False for images."""

        rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("side_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # Required — images cannot be concatenated

    vision_policy: VisionPolicyCfg = VisionPolicyCfg()


@configclass
class M3ReachVisionEnvCfg(M3ReachEnvCfg):
    """M3bot reach environment with RGB camera observation."""

    scene: M3ReachVisionSceneCfg = M3ReachVisionSceneCfg(num_envs=32, env_spacing=2.5)
    observations: M3ReachVisionObservationsCfg = M3ReachVisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Re-render on reset to avoid stale images
        self.num_rerenders_on_reset = 2
        self.sim.render.antialiasing_mode = "FXAA"
        # Fewer envs by default for vision (GPU descriptor limit)
        self.scene.num_envs = 32
        # Markers remain visible (debug_vis=True from base) for the 3D viewport / LiveStream.
        # A startup event changes their USD purpose to 'guide' so the CameraSensor
        # (Replicator) skips them — they don't exist on real hardware.
        self.events.set_vis_guide = EventTerm(
            func=set_vis_markers_guide_purpose,
            mode="startup",
        )
