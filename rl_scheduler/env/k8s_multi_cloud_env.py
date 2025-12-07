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
# from prometheus_api_client import PrometheusConnect  # Optional, disabled for fast training
import time
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# CONFIGURATION – UPDATE THESE PATHS IF NEEDED
# ------------------------------------------------------------------
DATA_PATH = "data/processed/normalized_rl_data.csv"  # From your normalize_data.py
PROM_AWS_URL = "http://localhost:39090"              # After port-forward on kind-aws
PROM_AZURE_URL = "http://localhost:39091"            # After port-forward on kind-azure

# ------------------------------------------------------------------
class K8sMultiCloudEnv(gym.Env):
    """
    State: 6 values (all normalized 0–1)
        [0] cost_aws          [1] cost_azure
        [2] latency_aws       [3] latency_azure
        [4] cpu_usage_aws     [5] cpu_usage_azure

    Action: 0 = schedule to AWS (kind-aws), 1 = schedule to Azure (kind-azure)
    Reward: negative weighted sum → higher = better placement
    """
    
    def __init__(self, env_config=None, fast_mode=True):
        super().__init__()

        self.fast_mode = fast_mode

        # ==================== Action & Observation Space ====================
        self.action_space = spaces.Discrete(2)  # 0=AWS, 1=Azure
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # ==================== Load Static Cost & Latency Data ====================
        try:
            self.static_df = pd.read_csv(DATA_PATH)
            # print(f"[Env] Loaded {len(self.static_df)} rows of real AWS/Azure data")
        except Exception as e:
            raise FileNotFoundError(f"Cannot find {DATA_PATH}. Run normalize_data.py first!") from e

        self.max_steps = len(self.static_df) - 1
        self.current_step = 0

        # ==================== Kubernetes Clients (for dry-run placement) ====================
        self.v1_aws = None
        self.v1_azure = None
        if not self.fast_mode:
            self._load_kube_clients()

        # ==================== Prometheus Clients (for live CPU metrics) ====================
        # Disabled in fast_mode
        # if not self.fast_mode:
        #     self.prom_aws = PrometheusConnect(url=PROM_AWS_URL, disable_ssl=True)
        #     self.prom_azure = PrometheusConnect(url=PROM_AZURE_URL, disable_ssl=True)

    def _load_kube_clients(self):
        try:
            config.load_kube_config(context="kind-aws")
            self.v1_aws = client.CoreV1Api()
            config.load_kube_config(context="kind-azure")
            self.v1_azure = client.CoreV1Api()
            # print("[Env] Kubernetes clients connected to kind-aws & kind-azure")
        except Exception as e:
            # print("[Env] Warning: Could not connect to clusters (normal if running outside K8s context):", e)
            pass

    def _get_live_cpu(self, prom_client=None, cluster_name=None):
        """Query average CPU usage across all nodes in the last 2 minutes"""
        if self.fast_mode:
            # Random realistic CPU usage for fast training
            return random.uniform(0.1, 0.8)
        else:
            # Query Prometheus if not in fast mode
            query = 'avg(rate(container_cpu_usage_seconds_total{namespace="default"}[2m]))'
            try:
                result = prom_client.custom_query(query)
                if result:
                    value = float(result[0]['value'][1])
                    return min(value / 2.0, 1.0)  # normalize (2 vCPU max per node)
            except Exception:
                pass
            return random.uniform(0.1, 0.8)

    def _get_obs(self):
        row = self.static_df.iloc[self.current_step]
        cpu_aws = self._get_live_cpu(None, "aws")
        cpu_azure = self._get_live_cpu(None, "azure")
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
        random.seed(seed)
        np.random.seed(seed)
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        row = self.static_df.iloc[self.current_step]

        # Reward = negative weighted sum
        cost = row['cost_aws'] if action == 0 else row['cost_azure']
        latency = row['latency_aws'] if action == 0 else row['latency_azure']
        reward = - (0.6 * cost + 0.4 * latency)

        # Optional: simulate pod placement (skip in fast mode)
        if not self.fast_mode:
            try:
                v1 = self.v1_aws if action == 0 else self.v1_azure
                cluster = "AWS" if action == 0 else "Azure"
                pod_body = client.V1Pod(
                    metadata=client.V1ObjectMeta(name=f"rl-pod-{int(time.time())}"),
                    spec=client.V1PodSpec(
                        containers=[client.V1Container(name="nginx", image="nginx:alpine")]
                    )
                )
                v1.create_namespaced_pod(namespace="default", body=pod_body, dry_run="All")
                # print(f"[Env] RL agent placed pod on simulated {cluster}")
            except Exception as e:
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
    # Baseline Scheduler for comparison
    # ------------------------------------------------------------------
    def normal_scheduler_step(self, obs):
        """
        Simple baseline scheduler:
        - Chooses the cheaper cloud at each step
        """
        cost_aws = obs[0]
        cost_azure = obs[1]
        return 0 if cost_aws <= cost_azure else 1


# ----------------------------------------------------------------------
# Quick test when you run the file directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = K8sMultiCloudEnv(fast_mode=True)
    obs, _ = env.reset()
    print("Initial observation:", obs)

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {info['chosen_cloud']} | Reward: {reward:.4f} | Obs: {obs.round(3)}")
        if done:
            break
    print("Environment test completed")
