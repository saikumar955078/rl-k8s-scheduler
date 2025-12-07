# rl_scheduler/agent/eval_ppo.py
"""
Evaluation script for trained RL Kubernetes multi-cloud scheduler.
Loads a PPO checkpoint and runs a few steps to see AWS vs Azure decisions.
"""

import os
import numpy as np
from ray.rllib.algorithms.ppo import PPO
from rl_scheduler.env.k8s_multi_cloud_env import K8sMultiCloudEnv

# ------------------- CONFIG -------------------
CHECKPOINT_PATH = "ray_results/k8s_multi_cloud_ppo_final/checkpoint_000120/checkpoint-120"  # adjust if needed
os.environ["RAY_ENABLE_MACOS_METAL"] = "1"

# ------------------- LOAD AGENT -------------------
agent = PPO.from_checkpoint(CHECKPOINT_PATH)
env = K8sMultiCloudEnv()

obs, _ = env.reset()
done = False
step = 0

print(f"Evaluating PPO agent on {env.max_steps} steps of multi-cloud environment...\n")

while not done and step < 20:   # just 20 steps for quick evaluation
    action = agent.compute_single_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {step+1:02d}: Scheduled pod to {info['chosen_cloud'].upper()} | Reward={reward:.4f} | CPU_obs={[round(obs[4],3), round(obs[5],3)]}")
    step += 1

print("\nEvaluation complete.")
env.close()
