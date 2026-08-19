"""
Reinforcement Learning Agent Module for Humanoid-v5 using Stable-Baselines3 (PPO).
실시간 학습, 단계별 정책(Random, Early, Trained), 모델 체크포인트 저장/로드를 담당합니다.
"""
import os
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from mujoco_patch import make_humanoid_env, apply_mujoco_patch

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

class MetricTrackerCallback(BaseCallback):
    """학습 중 실시간 손실 및 보상 메트릭을 추적하는 콜백"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.entropy_loss = 0.0
        self.approx_kl = 0.0
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        return True

    def _on_rollout_end(self) -> None:
        # PPO 학습 후 logger에서 메트릭 추출
        if hasattr(self.model, "logger") and self.model.logger:
            name_to_value = self.model.logger.name_to_value
            self.policy_loss = name_to_value.get("train/policy_gradient_loss", self.policy_loss)
            self.value_loss = name_to_value.get("train/value_loss", self.value_loss)
            self.entropy_loss = name_to_value.get("train/entropy_loss", self.entropy_loss)
            self.approx_kl = name_to_value.get("train/approx_kl", self.approx_kl)


class HumanoidRLManager:
    """Humanoid 강화학습 에이전트 및 단계별 모드 관리자"""

    def __init__(self, env=None):
        apply_mujoco_patch()
        self.env = env if env is not None else make_humanoid_env(render_mode="rgb_array")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # PPO 모델 초기화
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            verbose=0,
            device=self.device
        )
        self.metric_callback = MetricTrackerCallback()
        self.total_timesteps = 0
        self.current_stage = "LIVE_TRAIN" # "RANDOM", "EARLY", "TRAINED", "LIVE_TRAIN"
        self.is_training_active = True

        # 기본 데모용 가중치 파일 경로
        self.checkpoint_path = CHECKPOINT_DIR / "humanoid_ppo_latest.zip"
        self.early_checkpoint_path = CHECKPOINT_DIR / "humanoid_ppo_early.zip"

    def select_action(self, observation, deterministic=False):
        """현재 모드에 따라 행동(Action)을 결정합니다."""
        if self.current_stage == "RANDOM":
            # 1단계: 완전 무작위 (넘어지는 모습)
            return self.env.action_space.sample()
        elif self.current_stage == "EARLY":
            # 2단계: 초기 학습 단계 (잡음이 섞여 비틀거림)
            action, _ = self.model.predict(observation, deterministic=False)
            noise = np.random.normal(0, 0.4, size=action.shape)
            return np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)
        else:
            # 3단계(TRAINED) 또는 실시간 학습(LIVE_TRAIN)
            action, _ = self.model.predict(observation, deterministic=deterministic)
            return action

    def train_step(self, timesteps=128):
        """PPO 학습을 지정된 타임스텝만큼 진행합니다."""
        if not self.is_training_active:
            return
        
        self.model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            callback=self.metric_callback
        )
        self.total_timesteps += timesteps

    def save_checkpoint(self, path=None):
        """모델 체크포인트 저장"""
        target = path or self.checkpoint_path
        self.model.save(str(target))
        print(f"[RL Manager] Model saved to {target}")

    def load_checkpoint(self, path=None):
        """모델 체크포인트 로드"""
        target = path or self.checkpoint_path
        if Path(str(target) + ".zip").exists() or Path(target).exists():
            self.model = PPO.load(str(target), env=self.env, device=self.device)
            print(f"[RL Manager] Model loaded from {target}")
            return True
        print(f"[RL Manager] No checkpoint found at {target}")
        return False

    def get_metrics(self):
        """현재 학습 메트릭 반환"""
        return {
            "total_timesteps": self.total_timesteps,
            "policy_loss": self.metric_callback.policy_loss,
            "value_loss": self.metric_callback.value_loss,
            "entropy_loss": self.metric_callback.entropy_loss,
            "approx_kl": self.metric_callback.approx_kl,
            "device": self.device,
            "stage": self.current_stage
        }
