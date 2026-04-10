"""RSL-RL PPO agent configurations for all M3bot tasks.

State-based tasks use legacy RslRlPpoActorCriticCfg (migrated automatically to
actor/critic MLPModelCfg by handle_deprecated_rsl_rl_cfg).

Vision tasks use the new-style API: explicit actor/critic RslRlCNNModelCfg +
obs_groups to route both `policy` (flat state) and `vision_policy` (64×64 RGB
image as (B,3,64,64)) into a CNN+MLP network.  Actor and critic share their CNN
encoders (share_cnn_encoders=True) to halve GPU memory for the conv layers.

CNN architecture for 64×64 input:
  Conv(32, k=8, s=4) → 15×15
  Conv(64, k=4, s=2) → 6×6
  Conv(64, k=3, s=1) → 4×4
  Flatten → 1024 dims
  Concat with state obs → MLP [256, 128] → output
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

# Shared PPO algorithm parameters (same across all tasks)
_PPO_ALG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=8,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class M3ReachPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO for M3-Reach. Obs: 4+4+7+4=19 dims."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "m3_reach"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 64, 64],
        critic_hidden_dims=[128, 64, 64],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = _PPO_ALG


@configclass
class M3LiftPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO for M3-Lift. Obs: 6+6+3+7+6=28 dims."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "m3_lift"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = _PPO_ALG


@configclass
class M3PushPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO for M3-Push. Obs: 4+4+3+7+4=22 dims."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "m3_push"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 64, 64],
        critic_hidden_dims=[128, 64, 64],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = _PPO_ALG


@configclass
class M3PickPlacePPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO for M3-PickPlace. Obs: 6+6+3+7+5=27 dims."""

    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "m3_pick_place"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = _PPO_ALG


##############################################################################
# Vision PPO configurations
# obs_groups routes both `policy` (flat state) and `vision_policy` (B,3,64,64)
# to actor and critic.  CNNs are shared between actor/critic via
# share_cnn_encoders=True in the algorithm config.
##############################################################################

# Shared CNN config for 128×128 RGB input
# Conv(32,k=8,s=4) → 31×31, Conv(64,k=4,s=2) → 14×14, Conv(64,k=3,s=2) → 6×6, Flatten → 2304 dims
_CNN_CFG = RslRlCNNModelCfg.CNNCfg(
    output_channels=[32, 64, 64],
    kernel_size=[8, 4, 3],
    stride=[4, 2, 2],
    activation="elu",
)

# Algorithm config with CNN sharing enabled
_PPO_ALG_VISION = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=4,   # fewer epochs: vision mini-batches are larger
    num_mini_batches=4,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    share_cnn_encoders=True,  # actor & critic share CNN weights → half GPU memory
)


@configclass
class M3ReachVisionPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO+CNN for M3-Reach-Vision. State(19) + RGB(3×64×64) → CNN+MLP."""

    num_steps_per_env = 16   # smaller rollout buffer for vision (GPU memory)
    max_iterations = 3000
    save_interval = 100
    experiment_name = "m3_reach_vision"
    obs_groups = {"actor": ["policy", "vision_policy"], "critic": ["policy", "vision_policy"]}
    actor = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        cnn_cfg=_CNN_CFG,
    )
    critic = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        cnn_cfg=_CNN_CFG,
    )
    algorithm = _PPO_ALG_VISION


@configclass
class M3LiftVisionPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO+CNN for M3-Lift-Vision. State(28) + RGB(3×64×64) → CNN+MLP."""

    num_steps_per_env = 16
    max_iterations = 5000
    save_interval = 100
    experiment_name = "m3_lift_vision"
    obs_groups = {"actor": ["policy", "vision_policy"], "critic": ["policy", "vision_policy"]}
    actor = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        cnn_cfg=_CNN_CFG,
    )
    critic = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        cnn_cfg=_CNN_CFG,
    )
    algorithm = _PPO_ALG_VISION


@configclass
class M3PushVisionPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO+CNN for M3-Push-Vision. State(22) + RGB(3×64×64) → CNN+MLP."""

    num_steps_per_env = 16
    max_iterations = 4000
    save_interval = 100
    experiment_name = "m3_push_vision"
    obs_groups = {"actor": ["policy", "vision_policy"], "critic": ["policy", "vision_policy"]}
    actor = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        cnn_cfg=_CNN_CFG,
    )
    critic = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        cnn_cfg=_CNN_CFG,
    )
    algorithm = _PPO_ALG_VISION


@configclass
class M3PickPlaceVisionPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO+CNN for M3-PickPlace-Vision. State(27) + RGB(3×64×64) → CNN+MLP."""

    num_steps_per_env = 16
    max_iterations = 8000
    save_interval = 200
    experiment_name = "m3_pick_place_vision"
    obs_groups = {"actor": ["policy", "vision_policy"], "critic": ["policy", "vision_policy"]}
    actor = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=1.0,
        cnn_cfg=_CNN_CFG,
    )
    critic = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=False,
        init_noise_std=1.0,
        cnn_cfg=_CNN_CFG,
    )
    algorithm = _PPO_ALG_VISION
