"""
Comprehensive Verification Gate v1.1 for Deep Exploration & PPO Async System.
"""
import os
import sys
import time

# Windows cp949 콘솔 UTF-8 출력 지원
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pygame
from PIL import Image

from mujoco_patch import make_humanoid_env, apply_mujoco_patch
from rl_agent import HumanoidRLManager
from visual_humanoid_app import VisualHumanoidApp

def run_verification():
    print("=" * 65)
    print("[Verification Gate v1.1] Starting Deep Exploration & PPO Check...")
    print("=" * 65)

    # 1. MuJoCo Patch & Environment Check
    print("\n[Step 1] Verifying MuJoCo Windows UTF-8 Patch...")
    apply_mujoco_patch()
    env = make_humanoid_env(render_mode="rgb_array", width=640, height=480)
    obs, info = env.reset()
    assert obs is not None, "Observation should not be None"
    print(f"[OK] Environment initialized. Observation shape: {obs.shape}")

    frame = env.render()
    assert frame is not None and frame.shape == (480, 640, 3), f"Invalid frame shape: {frame.shape if frame is not None else None}"
    print(f"[OK] MuJoCo 3D Rendering verified. Frame shape: {frame.shape}")

    # 2. Deep Exploration & RL Manager Check
    print("\n[Step 2] Verifying RL Deep Exploration Manager & Noise Injection...")
    rl_mgr = HumanoidRLManager(env=env)
    
    # 2-1) 액션 샘플링 및 다양성 지수 검증
    for _ in range(10):
        a = rl_mgr.select_action(obs)
    
    diversity = rl_mgr.calculate_diversity_index()
    print(f"[OK] Joint Action Diversity Score: {diversity:.3f} (0.0~1.0)")
    assert 0.0 <= diversity <= 1.0, "Diversity index out of range"

    # 2-2) 탐색 부스트 토글 검증
    boost_state = rl_mgr.toggle_exploration_boost()
    assert boost_state == True, "Exploration boost toggle failed"
    print("[OK] Exploration Boost Mode Enabled Successfully.")

    # 2-3) 백그라운드 비동기 학습 동작 검증 (1초 대기)
    print("[Step 2-3] Verifying Background PPO Training Loop...")
    time.sleep(1.2)
    deep_metrics = rl_mgr.get_deep_metrics()
    print(f"[OK] Background Timesteps: {deep_metrics['total_timesteps']:,}, Current Phase: {deep_metrics['current_phase']}")
    assert deep_metrics['total_timesteps'] > 0, "Background training did not advance timesteps"

    # 3. Deep Exploration Pygame HUD Drawing Pipeline
    print("\n[Step 3] Verifying Deep Exploration Pygame HUD Surface Pipeline...")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    
    app = VisualHumanoidApp(win_width=1320, win_height=740, render_size=(760, 700))
    
    # 15 스텝 시뮬레이션 및 렌더링 검증
    for _ in range(15):
        app.step_simulation()
    
    app.render_ui()
    
    # Surface -> Image 저장
    surf_array = pygame.surfarray.array3d(app.screen)
    img_data = np.transpose(surf_array, (1, 0, 2))
    img = Image.fromarray(img_data.astype(np.uint8))
    screenshot_path = "verification_screenshot.png"
    img.save(screenshot_path)
    print(f"[OK] Deep Exploration HUD Screenshot saved to {screenshot_path}")

    app.cleanup()
    rl_mgr.close()

    print("\n" + "=" * 65)
    print("[Verification Gate v1.1] ALL GATES PASSED 100% WITH DEEP EXPLORATION!")
    print("=" * 65)

if __name__ == "__main__":
    run_verification()
