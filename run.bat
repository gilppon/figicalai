@echo off
chcp 65001 > nul
echo ========================================================
echo   ⚡ Humanoid-v5 Visual RL Dashboard Launcher
echo ========================================================
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Running setup...
    uv venv .venv --python 3.11
    uv pip install --python .venv\Scripts\python.exe gymnasium "gymnasium[mujoco]" stable-baselines3 torch pygame matplotlib opencv-python pillow
)

echo Starting Visual RL Application...
".venv\Scripts\python.exe" visual_humanoid_app.py
pause
