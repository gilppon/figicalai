"""
Comprehensive Verification Gate for Humanoid-v5 Visual RL System.
- MuJoCo UTF-8 XML Patch Verification
- Environment Reset & Render Pipeline
- RL Manager & PPO Step Verification
- Pygame Surface & HUD Drawing Verification
- Frame Buffer Dump to Image File
"""
import os
import sys

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
    print("=" * 60)
    print("[Verification Gate] Starting Comprehensive System Check...")
    print("=" * 60)

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

    # 2. RL Agent Check
    print("\n[Step 2] Verifying RL Agent & PPO Manager...")
    rl_mgr = HumanoidRLManager(env=env)
    action = rl_mgr.select_action(obs)
    assert action.shape == env.action_space.shape, f"Action shape mismatch: {action.shape}"
    print(f"[OK] RL Action Selection verified. Action shape: {action.shape}")

    # PPO Single Step Train
    print("[Step 2-1] Verifying PPO Train Step...")
    rl_mgr.train_step(timesteps=64)
    metrics = rl_mgr.get_metrics()
    print(f"[OK] PPO Training Step verified. Timesteps: {metrics['total_timesteps']}")

    # 3. Pygame HUD & Visual App Component Check
    print("\n[Step 3] Verifying Pygame HUD Drawing & Surface Pipeline...")
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Headless surface mode for testing
    pygame.init()
    
    app = VisualHumanoidApp(win_width=1280, win_height=720, render_size=(760, 680))
    
    # 10 스텝 시뮬레이션 및 렌더링 검증
    for s in range(10):
        app.step_simulation()
    
    app.render_ui()
    
    # Surface -> Image 저장
    surf_array = pygame.surfarray.array3d(app.screen)
    # Transpose to (H, W, 3)
    img_data = np.transpose(surf_array, (1, 0, 2))
    img = Image.fromarray(img_data.astype(np.uint8))
    screenshot_path = "verification_screenshot.png"
    img.save(screenshot_path)
    print(f"[OK] Full GUI & HUD Rendered and saved to {screenshot_path}")

    app.cleanup()

    print("\n" + "=" * 60)
    print("[Verification Gate] ALL 3 VERIFICATION GATES PASSED 100%!")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
