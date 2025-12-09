# rl_scheduler/agent/final_evaluation.py
import glob
import os
from pathlib import Path

from ray.rllib.algorithms.ppo import PPO
from rl_scheduler.env.k8s_multi_cloud_env import K8sMultiCloudEnv
import numpy as np

# ------------------------------------------------------------------
# AUTOMATICALLY FIND THE BEST (i.e. latest) CHECKPOINT – 100% reliable
# ------------------------------------------------------------------
ray_results = Path.home() / "ray_results" / "FINAL_PPO_AWS_AZURE"

# Look for all checkpoint_XXXXX folders (Ray 2.x style
candidates = list(ray_results.rglob("checkpoint_*[0-9]"))  # ends with a number

if not candidates:
    print("No checkpoint found! Did training actually finish?")
    print("Checked in:", ray_results)
    print("Available folders:", [p.name for p in ray_results.iterdir()] if ray_results.exists() else "folder missing")
    exit(1)

# Pick the one with the highest number (= latest/most trained)
checkpoint_path = max(candidates, key=lambda p: int(p.name.split("_")[-1]))
print(f"Loading best checkpoint: {checkpoint_path}")
print(f"Training steps ≈ {checkpoint_path.name.split('_')[-1]}\n")

# ------------------------------------------------------------------
# Load algorithm & environment
# ------------------------------------------------------------------
algo = PPO.from_checkpoint(str(checkpoint_path))   # new RLlib API (Ray ≥2.8)
# algo = PPO.checkpoint.load(str(checkpoint_path)) # old API – also kept for compatibility
env = K8sMultiCloudEnv(fast_mode=True)

# ------------------------------------------------------------------
# Run 100 full episodes with the trained agent (no exploration)
# ------------------------------------------------------------------
rewards = []
choices = {"AWS": 0, "Azure": 0}

for ep in range(1, 101):
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        choices["AWS" if action == 0 else "Azure"] += 1
    rewards.append(ep_reward)

    if ep % 20 == 0:
        print(f"Episode {ep:3d} → cost = ${-ep_reward:6.3f}")

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
costs = [-r for r in rewards]

avg_cost = np.mean(costs)

print("\n" + "="*60)
print("FINAL EVALUATION RESULTS (100 episodes)")
print("="*60)
print(f"Average cost per episode       : ${avg_cost:.4f}")
print(f"Total decisions total           : {sum(choices.values())}")
print(f"Agent chose AWS                : {choices['AWS']:4d} times ({choices['AWS']/sum(choices.values()):.1%})")
print(f"Agent chose Azure              : {choices['Azure']:4d} times ({choices['Azure']/sum(choices.values()):.1%})")

# Baseline from your dataset (cost-only greedy scheduler)
baseline_cost = 4.765
improvement = 100 * (baseline_cost - avg_cost) / baseline_cost

print(f"\nImprovement vs greedy baseline (${baseline_cost:.3f}): {improvement:5.1f}% better")
print("="*60)

# Bonus: if you want to save the results
with open("../results/final_evaluation_summary.txt", "w") as f:
    f.write(f"Avg cost: ${avg_cost:.4f} | Improvement: {improvement:.1f}%\n")
    f.write(f"AWS choices: {choices['AWS']} | Azure choices: {choices['Azure']}\n")

print("\nSummary also saved to results/final_evaluation_summary.txt")