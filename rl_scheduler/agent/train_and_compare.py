import os
import logging
import matplotlib.pyplot as plt
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from gymnasium.wrappers import TimeLimit
from rl_scheduler.env.k8s_multi_cloud_env import K8sMultiCloudEnv

# ------------------- Setup -------------------
os.environ["RAY_ENABLE_MACOS_METAL"] = "1"  # Optional: Apple Silicon GPU

# Suppress Ray and environment logs for cleaner output
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("rl_scheduler.env").setLevel(logging.WARNING)

# ------------------- Environment Wrapper -------------------
def env_creator(config):
    return TimeLimit(K8sMultiCloudEnv(), max_episode_steps=100)

tune.register_env("K8sMultiCloudEnv-v0", env_creator)

# ------------------- PPO Configuration -------------------
config = (
    PPOConfig()
    .environment("K8sMultiCloudEnv-v0")
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=256,
        num_sgd_iter=10,
        lr=3e-4,
        gamma=0.99,
    )
)

agent = PPO(config=config)

# ------------------- Train RL Agent -------------------
NUM_ITERATIONS = 5
rl_rewards = []

print("=== Training RL Agent ===")
for i in range(NUM_ITERATIONS):
    result = agent.train()
    mean_reward = result.get("episode_reward_mean", 0)
    rl_rewards.append(mean_reward)
    print(f"Iteration {i+1}: RL reward mean = {mean_reward:.2f}", flush=True)
    checkpoint = agent.save()
    print(f"Saved checkpoint to {checkpoint}", flush=True)

# ------------------- Run Baseline Scheduler -------------------
print("\n=== Running Baseline Scheduler ===")
env = TimeLimit(K8sMultiCloudEnv(), max_episode_steps=100)
baseline_rewards = []

for episode in range(NUM_ITERATIONS):
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    while not (done or truncated):
        # Simple round-robin: 0=AWS, 1=Azure
        action = 0 if env.current_step % 2 == 0 else 1

        # Unpack 5-tuple returned by step()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    baseline_rewards.append(total_reward)
    print(f"Episode {episode+1}: baseline reward = {total_reward:.2f}", flush=True)

# ------------------- Side-by-Side Comparison -------------------
print("\n=== Side-by-Side Comparison ===")
for i in range(NUM_ITERATIONS):
    rl_r = rl_rewards[i]
    baseline_r = baseline_rewards[i]
    print(f"Iteration {i+1}: RL = {rl_r:.2f} | Baseline = {baseline_r:.2f}", flush=True)

# ------------------- Optional: Plot Metrics -------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, NUM_ITERATIONS + 1), rl_rewards, marker='o', label="RL Agent")
plt.plot(range(1, NUM_ITERATIONS + 1), baseline_rewards, marker='x', label="Baseline Scheduler")
plt.xlabel("Iteration")
plt.ylabel("Total Reward")
plt.title("RL Agent vs Baseline Scheduler Rewards")
plt.legend()
plt.grid(True)
plt.show()
