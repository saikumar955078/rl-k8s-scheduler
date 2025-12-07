from ray.rllib.algorithms.ppo import PPO, PPOConfig
from rl_scheduler.env.k8s_multi_cloud_env import K8sMultiCloudEnv
import os

# Optional: Apple Silicon GPU (MPS)
os.environ["RAY_ENABLE_MACOS_METAL"] = "1"

# ------------------- PPO Configuration -------------------
config = (
    PPOConfig()
    .environment(K8sMultiCloudEnv)
    .framework("torch")
    .rollouts(num_rollout_workers=1)   # collects experience
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=256,
        num_sgd_iter=10,
        lr=3e-4,
        gamma=0.99,
    )
)

agent = PPO(config=config)

# ------------------- Train for a few iterations -------------------
NUM_ITERATIONS = 5  # change this to run more/fewer iterations

for i in range(NUM_ITERATIONS):
    result = agent.train()
    print(f"Iteration {i+1}: reward_mean={result['episode_reward_mean']}")
    checkpoint = agent.save()
    print(f"Saved checkpoint to {checkpoint}")

print("Training completed!")