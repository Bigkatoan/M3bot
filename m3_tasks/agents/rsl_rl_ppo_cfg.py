"""RSL-RL PPO agent configurations for all M3bot tasks.

Uses the legacy RslRlPpoActorCriticCfg (compatible with rsl-rl >= 3.0.1).
Network sizes are tuned for M3bot's small observation spaces.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

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
