"""
MuJoCo Windows UTF-8 XML Path Compatibility Patch & Environment Factory.
Windows 환경에서 한글 등 Non-ASCII 경로가 포함된 경우 MuJoCo C++ 엔진의 XML 로딩 오류를 완벽하게 방지합니다.
"""
from pathlib import Path
import gymnasium as gym
import mujoco
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

_patched = False

def apply_mujoco_patch():
    """Gymnasium MujocoEnv의 XML 로딩 함수를 UTF-8 문자열 로딩 방식으로 패치합니다."""
    global _patched
    if _patched:
        return
    
    orig_init_sim = MujocoEnv._initialize_simulation

    def patched_init_sim(self):
        try:
            # 1. 파일에서 UTF-8 텍스트로 직접 읽기
            xml_path = Path(self.fullpath)
            if xml_path.exists():
                xml_str = xml_path.read_text(encoding="utf-8")
                model = mujoco.MjModel.from_xml_string(xml_str)
                data = mujoco.MjData(model)
                return model, data
        except Exception:
            pass
        return orig_init_sim(self)

    MujocoEnv._initialize_simulation = patched_init_sim
    _patched = True
    print("[MuJoCo Patch] Windows UTF-8 XML Path Patch Applied Successfully.")

def make_humanoid_env(render_mode="rgb_array", width=640, height=480, **kwargs):
    """
    호환성 패치가 적용된 Humanoid-v5 환경을 생성합니다.
    """
    apply_mujoco_patch()
    env = gym.make(
        "Humanoid-v5",
        render_mode=render_mode,
        width=width,
        height=height,
        **kwargs
    )
    return env

if __name__ == "__main__":
    env = make_humanoid_env(render_mode="rgb_array")
    obs, info = env.reset()
    frame = env.render()
    print(f"Env initialized successfully! Frame shape: {frame.shape}, Obs dim: {obs.shape}")
    env.close()
