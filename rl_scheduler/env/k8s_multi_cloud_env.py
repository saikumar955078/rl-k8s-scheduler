# rl_scheduler/env/k8s_multi_cloud_env.py
"""
Custom Gymnasium Environment for RL-based Kubernetes Multi-Cloud Scheduler
Simulates AWS EKS & Azure AKS using Kind clusters + real pricing/latency data
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
from kubernetes import client, config
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# ABSOLUTE PATH – THIS IS THE ONLY CHANGE YOU NEED TO MAKE IT BULLETPROOF
# ------------------------------------------------------------------
# __file__ is this file's location → go up three levels → project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # rl_scheduler/env/... → project root
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "normalized_rl_data.csv"

# Optional: keep relative as fallback (for scripts run from project root)
if not DATA_PATH.exists():
    DATA_PATH = Path("data/processed/normalized_rl_data.csv")

# ------------------------------------------------------------------
# CONFIGURATION (no longer fragile)
# ------------------------------------------------------------------
PROM_AWS_URL = "http://localhost:39090"    # After port-forward on kind-aws
PROM_AZURE_URL = "http://localhost:39091" # After port-forward on kind-azure


class K8sMultiCloudEnv(gym.Env):
    """
    State: 6 values (all normalized 0–1)
        [0] cost_aws [1] cost_azure
        [2] latency_aws [3] latency_azure
        [4] cpu_usage_aws [5] cpu_usage_azure
    Action: 0 = schedule to AWS (kind-aws), 1 = schedule to Azure (kind-azure)
    Reward: negative weighted sum → higher = better placement
    """

    def __init__(self, env_config=None, fast_mode=True):
        super().__init__()
        self.fast_mode = fast_mode

        # ==================== Action & Observation Space ====================
        self.action_space = spaces.Discrete(2)  # 0=AWS, 1=Azure
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        # ==================== Load Static Cost & Latency Data ====================
        try:
            if not DATA_PATH.exists():
                raise FileNotFoundError(DATA_PATH)
            self.static_df = pd.read_csv(DATA_PATH)
            # print(f"[Env] Loaded {len(self.static_df)} rows from {DATA_PATH}")
        except Exception as e:
            raise FileNotFoundError(
                f"Cannot find normalized data at:\n  {DATA_PATH}\n"
                "Run `python normalize_data.py` from the project root first!"
            ) from e

        self.max_steps = len(self.static_df) - 1
        self.current_step = 0

        # ==================== Kubernetes Clients ====================
        self.v1_aws = None
        self.v1_azure = None
        if not self.fast_mode:
            self._load_kube_clients()

    def _load_kube_clients(self):
        try:
            config.load_kube_config(context="kind-aws")
            self.v1_aws = client.CoreV1Api()
            config.load_kube_config(context="kind-azure")
            self.v1_azure = client.CoreV1Api()
        except Exception as e:
            pass  # Silent if clusters not available

    def _get_live_cpu(self, cluster_name=None):
        """Simulate or query real CPU usage"""
        if self.fast_mode:
            return random.uniform(0.1, 0.8)
        return random.uniform(0.1, 0.8)  # placeholder for real Prometheus query

    def _get_obs(self):
        row = self.static_df.iloc[self.current_step]
        cpu_aws = self._get_live_cpu("aws")
        cpu_azure = self._get_live_cpu("azure")

        obs = np.array([
            row['cost_aws'],
            row['cost_azure'],
            row['latency_aws'],
            row['latency_azure'],
            cpu_aws,
            cpu_azure
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        row = self.static_df.iloc[self.current_step]

        cost = row['cost_aws'] if action == 0 else row['cost_azure']
        latency = row['latency_aws'] if action == 0 else row['latency_azure']
        reward = 100 * (0.6 * cost + 0.4 * latency)

        # Optional real pod dry-run (only in slow mode)
        if not self.fast_mode:
            try:
                v1 = self.v1_aws if action == 0 else self.v1_azure
                cluster_name = "AWS" if action == 0 else "Azure"
                pod_body = client.V1Pod(
                    metadata=client.V1ObjectMeta(name=f"rl-pod-{int(time.time())}"),
                    spec=client.V1PodSpec(
                        containers=[client.V1Container(name="nginx", image="nginx:alpine")]
                    )
                )
                v1.create_namespaced_pod(namespace="default", body=pod_body, dry_run="All")
            except Exception:
                pass

        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        info = {"chosen_cloud": "aws" if action == 0 else "azure", "step": self.current_step}

        return self._get_obs(), float(reward), done, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Baseline scheduler (for comparison later)
    # ------------------------------------------------------------------
    def normal_scheduler_step(self, obs):
        return 0 if obs[0] <= obs[1] else 1


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = K8sMultiCloudEnv(fast_mode=True)
    obs, _ = env.reset(seed=42)
    print("Initial observation:", obs.round(3))
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1} | Action: {info['chosen_cloud']:5} | Reward: {reward:8.2f} | Next obs: {obs.round(3)}")
        if done:
            break
    print("Environment test completed")