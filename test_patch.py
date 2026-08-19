import gymnasium as gym
import mujoco
from pathlib import Path
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

# Windows 한글 경로 인코딩 호환성 패치
orig_init_sim = MujocoEnv._initialize_simulation

def patched_init_sim(self):
    xml_str = Path(self.fullpath).read_text(encoding='utf-8')
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    return model, data

MujocoEnv._initialize_simulation = patched_init_sim

env = gym.make('Humanoid-v5', render_mode='rgb_array')
obs, info = env.reset()
frame = env.render()
print(f"Rendered Frame Shape: {frame.shape}")
for _ in range(10):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc:
        obs, info = env.reset()
env.close()
print("MuJoCo Humanoid-v5 Patch Test Success!")
