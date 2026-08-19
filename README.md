# ⚡ FigicalAI - Humanoid-v5 Deep Exploration RL Dashboard

> **Real-time 3D Physical Simulation, Deep Exploration (Entropy Scheduling & Action Noise), and Async PPO Telemetry HUD Dashboard**

![Deep Exploration Screenshot](verification_screenshot.png)

---

## 🌟 Key Features

1. **🎮 60 FPS Real-Time 3D Physics Simulation**
   - Gymnasium & MuJoCo `Humanoid-v5` environment.
   - Built-in Windows non-ASCII (UTF-8) XML Path compatibility patch.

2. **🎲 Deep Exploration Engine (PPO)**
   - **Dynamic Entropy Scheduling**: Ensures sufficient early-stage postural exploration.
   - **Action Exploration Noise**: Scheduled Gaussian noise injection to discover complex motor coordination.
   - **Joint Action Diversity Index**: Real-time variance scoring (0~100%) across all 17 humanoid joint actuators.
   - **🎯 3-Stage Learning Phase Tracking**:
     - `PHASE 1`: Posture & Balance Exploration
     - `PHASE 2`: Gait & Step Discovery
     - `PHASE 3`: Forward Locomotion Optimization

3. **⚡ Thread-Safe Async Background Training Worker**
   - Decoupled rendering and training threads for a smooth 60 FPS UI experience.
   - High-speed PPO optimization running asynchronously in the background.
   - Live neural network weight hot-swapping into the visual agent.

4. **📊 Cyberpunk Telemetry HUD Dashboard**
   - **Live Metrics**: Episode count, Step count, Step Reward, Current Return, Best Record.
   - **Exploration Gauges**: Joint Diversity Index & Policy Entropy meters.
   - **Deep PPO Telemetry**: Policy Loss, Value Loss, Approx KL Divergence, Clip Fraction.
   - **📈 Reward Trend Line Chart**: Real-time learning curve visualization of the last 60 episodes.
   - **🎛️ Joint Actuators**: Real-time torque heatmaps for all 17 humanoid motor joints.

---

## 🎮 Controls & Hotkeys

| Key | Action |
| :--- | :--- |
| **`[SPACE]`** | Pause / Resume simulation |
| **`[E]`** | **Toggle Exploration Boost** (2x Noise injection for creative postures) |
| **`[T]`** | **Toggle Turbo Fast-Training** (Ultra-high speed PPO optimization) |
| **`[1]`** | Switch to **Random Mode** (Untrained fall) |
| **`[2]`** | Switch to **Early Stage Mode** (Wobble & step) |
| **`[3]`** | Switch to **Live Training Mode** (Async PPO with Live Exploration) |
| **`[4]`** | Switch to **Trained Expert Mode** |
| **`[+]` / `[-]`** | Speed multiplier (`1x`, `2x`, `4x`, `8x`) |
| **`[R]`** | Reset episode immediately |
| **`[S]` / `[L]`** | Save / Load model checkpoint |
| **`[ESC]` / `[Q]`** | Quit application |

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+ (or `uv` package manager)

### 2. Installation & Run
```bash
# Clone the repository
git clone https://github.com/gilppon/figicalai.git
cd figicalai

# Run via uv (Automatic venv setup)
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe gymnasium "gymnasium[mujoco]" stable-baselines3 torch pygame matplotlib opencv-python pillow

# Start Dashboard
.\.venv\Scripts\python.exe visual_humanoid_app.py
```

Or simply double-click `run.bat` on Windows!

---

## 🏗️ Project Architecture

```
figicalai/
├── mujoco_patch.py         # MuJoCo UTF-8 XML path compatibility patch & env factory
├── rl_agent.py             # PPO RL agent with deep exploration & async training worker
├── visual_humanoid_app.py  # Pygame 3D viewer & Cyberpunk Deep Exploration HUD
├── untitled2.py            # Main entry point launcher
├── verify.py               # 3-Gate automated system verification test
├── run.bat                 # One-click Windows launcher
└── verification_screenshot.png # Real-time dashboard screenshot
```

---

## 📜 License
MIT License
