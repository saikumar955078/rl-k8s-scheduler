# rl_scheduler/agent/train_final.py
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from rl_scheduler.env.k8s_multi_cloud_env import K8sMultiCloudEnv

config = (
    PPOConfig()
    .environment(K8sMultiCloudEnv)
    .rollouts(num_rollout_workers=6, num_envs_per_worker=4)
    .framework("torch")
    .training(
        train_batch_size=8000,
        sgd_minibatch_size=512,
        num_sgd_iter=15,
        lr=5e-4,
        gamma=0.995,
    )
    .resources(num_gpus=0)
    .evaluation(evaluation_interval=5, evaluation_duration=20)
)

tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={"training_iteration": 80},      # ~3 hours max on M2 Air
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=10,
            num_to_keep=5,
            checkpoint_at_end=True,
        ),
        name="FINAL_PPO_AWS_AZURE",
        verbose=2,
    ),
).fit()

print("Training finished! Best model saved in ~/ray_results/FINAL_PPO_AWS_AZURE")