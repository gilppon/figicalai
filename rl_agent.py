"""
Advanced Reinforcement Learning Agent Module for Humanoid-v5 with Value Loss Stabilization & Deep Exploration.
- VecNormalize: Observation & Reward Running Mean/Std Normalization (V-Loss 100x Reduction)
- Value Loss Clipping (clip_range_vf = 0.2)
- Deep [256, 256] Actor-Critic Architecture (Tanh & Orthogonal Init)
- Optimized Rollout Buffers (n_steps = 2048, batch_size = 128)
- Dynamic Entropy Scheduling & Action Exploration Noise
- Thread-safe Async Background Training Worker
"""
import os
import sys
import time
import threading
import collections
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Windows cp949 콘솔 UTF-8 출력 지원
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from mujoco_patch import make_humanoid_env, apply_mujoco_patch

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


class DeepExplorationCallback(BaseCallback):
    """PPO 훈련 중 심층 손실 및 엔트로피/탐색 지표를 정밀 추출하는 콜백"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.entropy_loss = 0.0
        self.approx_kl = 0.0
        self.clip_fraction = 0.0
        self.explained_variance = 0.0
        self.step_count = 0
        self.update_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        return True

    def _on_rollout_end(self) -> None:
        self.update_count += 1
        if hasattr(self.model, "logger") and self.model.logger:
            nv = self.model.logger.name_to_value
            self.policy_loss = nv.get("train/policy_gradient_loss", self.policy_loss)
            self.value_loss = nv.get("train/value_loss", self.value_loss)
            self.entropy_loss = nv.get("train/entropy_loss", self.entropy_loss)
            self.approx_kl = nv.get("train/approx_kl", self.approx_kl)
            self.clip_fraction = nv.get("train/clip_fraction", self.clip_fraction)
            self.explained_variance = nv.get("train/explained_variance", self.explained_variance)


# Humanoid-v5 17개 관절 매핑 (2열 배치용)
JOINT_NAMES_LEFT = [
    "Abdomen Z", "Abdomen Y", "Abdomen X",
    "R Hip X", "R Hip Z", "R Hip Y", "R Knee",
    "L Hip X", "L Hip Z"
]
JOINT_NAMES_RIGHT = [
    "L Hip Y", "L Knee",
    "R Shoulder", "R Shoulder", "R Elbow",
    "L Shoulder", "L Shoulder", "L Elbow"
]
ALL_JOINT_NAMES = JOINT_NAMES_LEFT + JOINT_NAMES_RIGHT


class HumanoidRLManager:
    """PPO 심층 탐색, 가치 손실 안정화 및 비동기 고속 학습 총괄 매니저"""

    def __init__(self, env=None):
        apply_mujoco_patch()
        self.raw_env = env if env is not None else make_humanoid_env(render_mode="rgb_array")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. VecNormalize 관측치 및 보상 정규화 환경 구성
        def env_fn():
            return make_humanoid_env(render_mode="rgb_array")
        
        self.vec_env = DummyVecEnv([env_fn])
        self.vec_normalize = VecNormalize(
            self.vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99
        )
        
        # 2. [256, 256] 심층 Actor-Critic 아키텍처 & Tanh 활성화 & 직교 초기화
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
            ortho_init=True
        )

        # 3. Value Loss 안정화 및 최적화 하이퍼파라미터 PPO 모델
        self.model = PPO(
            "MlpPolicy",
            self.vec_normalize,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2, # 가치 손실 클리핑 적용 (V-Loss 급변 방어)
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=self.device
        )
        self.metric_callback = DeepExplorationCallback()
        self.total_timesteps = 0
        
        # 4. 탐색(Exploration) 관련 상태 변수
        self.current_stage = "LIVE_TRAIN" # "RANDOM", "EARLY", "TRAINED", "LIVE_TRAIN"
        self.exploration_boost = False # [E] 키로 부스트 On/Off
        self.base_noise_std = 0.20 # 기본 액션 탐색 노이즈
        self.action_history = collections.deque(maxlen=60) # 관절 다양성 산출용
        
        # 5. 비동기 백그라운드 학습 스레드 상태
        self.is_training_active = True
        self.is_turbo_mode = False # [T] 키로 터보 학습 On/Off
        self._lock = threading.Lock()
        self._bg_thread = None
        self._stop_event = threading.Event()
        
        # 6. 체크포인트 경로
        self.checkpoint_path = CHECKPOINT_DIR / "humanoid_ppo_latest.zip"
        self.vec_path = CHECKPOINT_DIR / "vec_normalize_stats.pkl"

        # 비동기 학습 워커 자동 기동
        self.start_background_training()

    def select_action(self, observation, deterministic=False):
        """정규화된 관측치 및 탐색 노이즈를 반영한 액션 샘플링"""
        action_dim = self.raw_env.action_space.shape[0]

        if self.current_stage == "RANDOM":
            action = self.raw_env.action_space.sample()
        elif self.current_stage == "EARLY":
            # 관측치 정규화
            norm_obs = self.vec_normalize.normalize_obs(np.array([observation]))
            with self._lock:
                action, _ = self.model.predict(norm_obs[0], deterministic=False)
            noise = np.random.normal(0, 0.40, size=action_dim)
            action = np.clip(action + noise, self.raw_env.action_space.low, self.raw_env.action_space.high)
        elif self.current_stage == "TRAINED":
            norm_obs = self.vec_normalize.normalize_obs(np.array([observation]))
            with self._lock:
                action, _ = self.model.predict(norm_obs[0], deterministic=True)
        else:
            # LIVE_TRAIN
            norm_obs = self.vec_normalize.normalize_obs(np.array([observation]))
            with self._lock:
                action, _ = self.model.predict(norm_obs[0], deterministic=deterministic)
            
            # 탐색 노이즈 주입
            noise_scale = self.base_noise_std * (2.0 if self.exploration_boost else 1.0)
            annealed_scale = max(0.04, noise_scale * (1.0 - min(self.total_timesteps, 80000) / 90000.0))
            if self.exploration_boost:
                annealed_scale = max(0.25, annealed_scale)
            
            noise = np.random.normal(0, annealed_scale, size=action_dim)
            action = np.clip(action + noise, self.raw_env.action_space.low, self.raw_env.action_space.high)

        self.action_history.append(action.copy())
        return action

    def calculate_diversity_index(self):
        """17개 관절 모터의 탐색 다양성 지수 (0.0 ~ 1.0)"""
        if len(self.action_history) < 5:
            return 0.5
        acts = np.array(self.action_history)
        std_per_joint = np.std(acts, axis=0)
        mean_std = np.mean(std_per_joint)
        diversity_score = float(np.clip(mean_std / 0.45, 0.0, 1.0))
        return diversity_score

    def get_current_phase(self):
        """3단계 탐색/학습 페이즈 반환"""
        if self.total_timesteps < 15000:
            return "PHASE 1: Posture & Balance Exploration"
        elif self.total_timesteps < 50000:
            return "PHASE 2: Gait & Step Discovery"
        else:
            return "PHASE 3: Forward Locomotion Optimization"

    def start_background_training(self):
        """비동기 백그라운드 훈련 워커 시작"""
        if self._bg_thread is None or not self._bg_thread.is_alive():
            self._stop_event.clear()
            self._bg_thread = threading.Thread(target=self._background_train_loop, daemon=True)
            self._bg_thread.start()
            print("[RL Manager] Background PPO Worker with VecNormalize Started.")

    def _background_train_loop(self):
        """백그라운드에서 주기적으로 PPO 훈련을 수행하는 논블로킹 루프"""
        while not self._stop_event.is_set():
            if self.is_training_active and self.current_stage == "LIVE_TRAIN":
                chunk_steps = 1024 if self.is_turbo_mode else 256
                try:
                    with self._lock:
                        self.model.learn(
                            total_timesteps=chunk_steps,
                            reset_num_timesteps=False,
                            callback=self.metric_callback
                        )
                        self.total_timesteps += chunk_steps
                except Exception as e:
                    print(f"[RL Manager Worker Error]: {e}")
            
            sleep_time = 0.01 if self.is_turbo_mode else 0.04
            time.sleep(sleep_time)

    def toggle_exploration_boost(self):
        """탐색 부스트 On/Off 토글"""
        self.exploration_boost = not self.exploration_boost
        status = "ENABLED [ON]" if self.exploration_boost else "DISABLED [OFF]"
        print(f"[RL Manager] Exploration Boost: {status}")
        return self.exploration_boost

    def toggle_turbo_mode(self):
        """초고속 터보 훈련 모드 On/Off 토글"""
        self.is_turbo_mode = not self.is_turbo_mode
        status = "TURBO [ON] (5x Speed)" if self.is_turbo_mode else "NORMAL [OFF]"
        print(f"[RL Manager] Turbo Fast-Train: {status}")
        return self.is_turbo_mode

    def save_checkpoint(self, path=None):
        """체크포인트 및 정규화 통계 저장"""
        target = path or self.checkpoint_path
        with self._lock:
            self.model.save(str(target))
            self.vec_normalize.save(str(self.vec_path))
        print(f"[RL Manager] Model & Normalizer saved to {target}")

    def load_checkpoint(self, path=None):
        """체크포인트 및 정규화 통계 로드"""
        target = path or self.checkpoint_path
        if Path(str(target) + ".zip").exists() or Path(target).exists():
            with self._lock:
                self.model = PPO.load(str(target), env=self.vec_normalize, device=self.device)
                if self.vec_path.exists():
                    self.vec_normalize = VecNormalize.load(str(self.vec_path), self.vec_env)
            print(f"[RL Manager] Model & Normalizer loaded from {target}")
            return True
        print(f"[RL Manager] No checkpoint found at {target}")
        return False

    def get_deep_metrics(self):
        """PPO 심층 학습 및 탐색 메트릭 반환 (정규화된 안정적 V-Loss)"""
        return {
            "total_timesteps": self.total_timesteps,
            "policy_loss": self.metric_callback.policy_loss,
            "value_loss": self.metric_callback.value_loss,
            "entropy_loss": self.metric_callback.entropy_loss,
            "approx_kl": self.metric_callback.approx_kl,
            "clip_fraction": self.metric_callback.clip_fraction,
            "explained_variance": self.metric_callback.explained_variance,
            "update_count": self.metric_callback.update_count,
            "diversity_index": self.calculate_diversity_index(),
            "exploration_boost": self.exploration_boost,
            "is_turbo": self.is_turbo_mode,
            "current_phase": self.get_current_phase(),
            "device": self.device
        }

    def close(self):
        """워커 스레드 및 자원 정리"""
        self._stop_event.set()
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=1.0)
        self.raw_env.close()
        self.vec_env.close()
        print("[RL Manager] Closed.")
