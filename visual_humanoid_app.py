"""
Real-time 3D Humanoid RL Visual Learning & Telemetry HUD Dashboard.
Pygame 기반의 고성능 실시간 렌더러와 사이버펑크 스타일 HUD를 통해
휴머노이드 에이전트의 물리 동작과 신경망 학습 과정을 실시간으로 모니터링합니다.
"""
import sys
import time
import collections
from pathlib import Path

# Windows cp949 콘솔 UTF-8 출력 지원
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pygame
import gymnasium as gym

from mujoco_patch import make_humanoid_env, apply_mujoco_patch
from rl_agent import HumanoidRLManager

# ==========================================
# 🎨 UI & 테마 색상 팔레트 (Cyberpunk Dark)
# ==========================================
BG_DARK = (13, 16, 23)
PANEL_BG = (22, 27, 34)
PANEL_BORDER = (48, 54, 61)
ACCENT_CYAN = (0, 229, 255)
ACCENT_GREEN = (0, 230, 118)
ACCENT_ORANGE = (255, 145, 0)
ACCENT_RED = (255, 61, 0)
ACCENT_PURPLE = (213, 0, 249)
TEXT_WHITE = (240, 246, 252)
TEXT_MUTED = (139, 148, 158)
BAR_BG = (33, 38, 45)
GRAPH_GRID = (30, 35, 45)


class VisualHumanoidApp:
    def __init__(self, win_width=1280, win_height=720, render_size=(760, 680)):
        pygame.init()
        pygame.display.set_caption("⚡ Humanoid-v5 Real-time Visual RL Dashboard | Powered by Antigravity")

        self.win_width = win_width
        self.win_height = win_height
        self.render_width, self.render_height = render_size
        self.screen = pygame.display.set_mode((win_width, win_height))
        self.clock = pygame.time.Clock()

        # 폰트 초기화 (Consolas -> Arial -> 기본 폰트 순서 폴백)
        try:
            self.font_title = pygame.font.SysFont("consolas", 20, bold=True)
            self.font_main = pygame.font.SysFont("consolas", 14, bold=True)
            self.font_sub = pygame.font.SysFont("consolas", 12)
            self.font_badge = pygame.font.SysFont("consolas", 13, bold=True)
        except Exception:
            self.font_title = pygame.font.Font(None, 24)
            self.font_main = pygame.font.Font(None, 18)
            self.font_sub = pygame.font.Font(None, 15)
            self.font_badge = pygame.font.Font(None, 16)

        # MuJoCo 환경 및 RL 매니저 초기화
        print("[App] Initializing MuJoCo Humanoid-v5 Environment...")
        self.env = make_humanoid_env(
            render_mode="rgb_array",
            width=self.render_width,
            height=self.render_height
        )
        self.rl_manager = HumanoidRLManager(env=self.env)

        # 에피소드 및 텔레메트리 변수
        self.obs, self.info = self.env.reset()
        self.episode_count = 1
        self.current_step = 0
        self.episode_reward = 0.0
        self.best_reward = -float("inf")
        self.last_rewards = collections.deque(maxlen=60)
        self.current_action = np.zeros(self.env.action_space.shape)
        self.forward_velocity = 0.0
        self.torso_height = 1.4 # 초기 높이
        self.step_reward = 0.0

        # 시뮬레이션 제어 변수
        self.is_paused = False
        self.speed_multiplier = 1 # 1x, 2x, 4x, 8x
        self.is_running = True
        self.fps = 60
        self.train_steps_per_cycle = 4 # 라이브 훈련 스텝

    def run(self):
        """메인 애플리케이션 루프"""
        while self.is_running:
            self.handle_events()
            
            # 시뮬레이션 진행 (배속에 따라 여러 스텝 실행)
            if not self.is_paused:
                for _ in range(self.speed_multiplier):
                    self.step_simulation()

            # 화면 렌더링
            self.render_ui()
            self.clock.tick(self.fps)

        self.cleanup()

    def handle_events(self):
        """키보드 및 윈도우 이벤트 처리"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.is_running = False
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_1:
                    self.rl_manager.current_stage = "RANDOM"
                elif event.key == pygame.K_2:
                    self.rl_manager.current_stage = "EARLY"
                elif event.key == pygame.K_3:
                    self.rl_manager.current_stage = "LIVE_TRAIN"
                elif event.key == pygame.K_4:
                    self.rl_manager.current_stage = "TRAINED"
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.speed_multiplier = min(8, self.speed_multiplier * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    self.speed_multiplier = max(1, self.speed_multiplier // 2)
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_s:
                    self.rl_manager.save_checkpoint()
                elif event.key == pygame.K_l:
                    self.rl_manager.load_checkpoint()

    def step_simulation(self):
        """1 스텝 시뮬레이션 및 학습 진행"""
        # 행동 결정
        action = self.rl_manager.select_action(self.obs)
        self.current_action = action

        # 환경 스텝
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_reward = reward
        self.episode_reward += reward
        self.current_step += 1

        # 텔레메트리 추출
        # Humanoid 관측치: x_velocity는 관측치 인덱스 또는 info에서 추출
        self.forward_velocity = info.get("x_velocity", 0.0)
        # z-coordinate (torso height)
        if len(next_obs) > 0:
            self.torso_height = next_obs[0] if isinstance(next_obs, np.ndarray) else 1.4

        # 라이브 트레이닝 진행 (LIVE_TRAIN 모드일 때)
        if self.rl_manager.current_stage == "LIVE_TRAIN" and self.current_step % 16 == 0:
            # 경량 온라인 업데이트
            self.rl_manager.train_step(timesteps=self.train_steps_per_cycle)

        self.obs = next_obs

        # 에피소드 종료 조건
        if terminated or truncated:
            self.reset_episode()

    def reset_episode(self):
        """에피소드 리셋 및 통계 갱신"""
        self.last_rewards.append(self.episode_reward)
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
        
        self.obs, self.info = self.env.reset()
        self.episode_count += 1
        self.current_step = 0
        self.episode_reward = 0.0

    def render_ui(self):
        """전체 UI 화면 렌더링"""
        self.screen.fill(BG_DARK)

        # 1. 3D 물리 시뮬레이션 렌더링
        frame = self.env.render()
        if frame is not None:
            # frame shape: (H, W, 3) -> Pygame Surface (W, H)
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self.screen.blit(surf, (20, 20))
            # 렌더링 뷰 테두리
            pygame.draw.rect(self.screen, PANEL_BORDER, (20, 20, self.render_width, self.render_height), 2)

        # 2. 우측 텔레메트리 HUD 패널
        panel_x = self.render_width + 40
        panel_y = 20
        panel_w = self.win_width - panel_x - 20
        panel_h = self.render_height

        self.draw_hud_panel(panel_x, panel_y, panel_w, panel_h)

        pygame.display.flip()

    def draw_hud_panel(self, x, y, w, h):
        """사이버펑크 HUD 텔레메트리 패널 그리기"""
        # 패널 배경 및 외곽선
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x, y, w, h), 2, border_radius=8)

        cy = y + 15
        cx = x + 15
        inner_w = w - 30

        # 헤더: 타이틀
        title_surf = self.font_title.render("⚡ HUMANOID-V5 AGENT", True, ACCENT_CYAN)
        self.screen.blit(title_surf, (cx, cy))
        
        # 실시간 FPS & 상태 배지
        stage_colors = {
            "RANDOM": (ACCENT_RED, "[1] RANDOM (FALL)"),
            "EARLY": (ACCENT_ORANGE, "[2] EARLY STAGE"),
            "LIVE_TRAIN": (ACCENT_GREEN, "[3] LIVE TRAINING"),
            "TRAINED": (ACCENT_PURPLE, "[4] TRAINED EXPERT")
        }
        color, stage_name = stage_colors.get(self.rl_manager.current_stage, (TEXT_WHITE, "UNKNOWN"))
        badge_surf = self.font_badge.render(f"MODE: {stage_name}", True, color)
        self.screen.blit(badge_surf, (cx, cy + 28))

        # 속도/일시정지 상태
        status_text = f"SPEED: {self.speed_multiplier}x | {'PAUSED ⏸' if self.is_paused else 'RUNNING ▶'}"
        status_surf = self.font_sub.render(status_text, True, TEXT_MUTED)
        self.screen.blit(status_surf, (cx + inner_w - status_surf.get_width(), cy + 28))

        cy += 58
        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 12

        # 📊 에피소드 & 보상 메트릭
        avg_reward = np.mean(self.last_rewards) if self.last_rewards else 0.0
        metrics = [
            ("Episode", f"#{self.episode_count}", TEXT_WHITE),
            ("Step / Time", f"{self.current_step} steps", TEXT_WHITE),
            ("Step Reward", f"{self.step_reward:+.2f}", ACCENT_CYAN if self.step_reward > 0 else ACCENT_RED),
            ("Current Return", f"{self.episode_reward:+.1f}", ACCENT_GREEN),
            ("Best Return", f"{self.best_reward:+.1f}" if self.best_reward > -1e5 else "0.0", ACCENT_PURPLE),
            ("Avg (Last 60)", f"{avg_reward:+.1f}", ACCENT_ORANGE),
        ]

        for i, (label, val, val_color) in enumerate(metrics):
            row = i // 2
            col = i % 2
            col_x = cx + col * (inner_w // 2)
            row_y = cy + row * 24
            
            lbl_s = self.font_sub.render(f"{label}:", True, TEXT_MUTED)
            val_s = self.font_main.render(val, True, val_color)
            self.screen.blit(lbl_s, (col_x, row_y))
            self.screen.blit(val_s, (col_x + 95, row_y - 2))

        cy += 78
        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 10

        # 📈 실시간 보상 학습 곡선 그래프
        lbl_graph = self.font_main.render("📈 REWARD HISTORY (LAST 60 EPS)", True, TEXT_WHITE)
        self.screen.blit(lbl_graph, (cx, cy))
        cy += 20
        self.draw_reward_graph(cx, cy, inner_w, 80)
        cy += 90

        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 10

        # 🏃 신체 텔레메트리 (속도 & 높이 게이지)
        lbl_tele = self.font_main.render("🏃 BODY TELEMETRY & GAUGES", True, TEXT_WHITE)
        self.screen.blit(lbl_tele, (cx, cy))
        cy += 22

        # 1) 전진 속도 게이지
        self.draw_gauge(cx, cy, inner_w, "Forward Velocity (Vx)", self.forward_velocity, -1.0, 5.0, "m/s", ACCENT_CYAN)
        cy += 32
        # 2) 상체 높이 게이지 (넘어짐 감지)
        self.draw_gauge(cx, cy, inner_w, "Torso Z-Height", self.torso_height, 0.5, 1.6, "m", ACCENT_GREEN)
        cy += 38

        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 10

        # 🎛️ 17개 관절 모터 출력 히트맵 (Actuator Torques)
        lbl_act = self.font_main.render("🎛️ 17 JOINT ACTUATOR OUTPUTS", True, TEXT_WHITE)
        self.screen.blit(lbl_act, (cx, cy))
        cy += 20
        self.draw_actuator_bars(cx, cy, inner_w, 40)
        cy += 50

        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 10

        # 🎮 키보드 단축키 안내 패널
        shortcuts = [
            "[SPACE] Pause/Play", "[1~4] Mode Switch", "[+/-] Speed 1x~8x",
            "[R] Reset Env", "[S] Save Model", "[L] Load Model", "[ESC/Q] Quit"
        ]
        sc_text = "  |  ".join(shortcuts[:4])
        sc_text2 = "  |  ".join(shortcuts[4:])
        self.screen.blit(self.font_sub.render(sc_text, True, TEXT_MUTED), (cx, cy))
        self.screen.blit(self.font_sub.render(sc_text2, True, TEXT_MUTED), (cx, cy + 16))

    def draw_reward_graph(self, x, y, w, h):
        """실시간 보상 추이 미니 그래프 렌더링"""
        pygame.draw.rect(self.screen, BAR_BG, (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, GRAPH_GRID, (x, y, w, h), 1, border_radius=4)

        if len(self.last_rewards) < 2:
            no_data = self.font_sub.render("Collecting episode data...", True, TEXT_MUTED)
            self.screen.blit(no_data, (x + w // 2 - no_data.get_width() // 2, y + h // 2 - 8))
            return

        rewards = list(self.last_rewards)
        min_r = min(min(rewards), 0)
        max_r = max(max(rewards), 500)
        rng = max_r - min_r if max_r != min_r else 1.0

        points = []
        step_x = w / (len(rewards) - 1)
        for idx, r in enumerate(rewards):
            px = x + idx * step_x
            py = y + h - ((r - min_r) / rng) * (h - 10) - 5
            points.append((px, py))

        # 그래프 선 그리기
        if len(points) >= 2:
            pygame.draw.lines(self.screen, ACCENT_CYAN, False, points, 2)
            # 마지막 점 강조
            pygame.draw.circle(self.screen, ACCENT_GREEN, (int(points[-1][0]), int(points[-1][1])), 4)

    def draw_gauge(self, x, y, w, label, val, min_v, max_v, unit, color):
        """수평 프로그레스 바 게이지 렌더링"""
        lbl_s = self.font_sub.render(f"{label}: {val:+.2f} {unit}", True, TEXT_MUTED)
        self.screen.blit(lbl_s, (x, y))

        bar_y = y + 16
        bar_h = 8
        pygame.draw.rect(self.screen, BAR_BG, (x, bar_y, w, bar_h), border_radius=4)

        norm_val = np.clip((val - min_v) / (max_v - min_v), 0.0, 1.0)
        fill_w = int(w * norm_val)
        if fill_w > 0:
            pygame.draw.rect(self.screen, color, (x, bar_y, fill_w, bar_h), border_radius=4)

    def draw_actuator_bars(self, x, y, w, h):
        """17개 관절 모터의 실시간 출력 토크 막대 차트"""
        n_act = len(self.current_action)
        bar_w = (w - (n_act - 1) * 2) // n_act
        center_y = y + h // 2

        # 0선 가이드
        pygame.draw.line(self.screen, PANEL_BORDER, (x, center_y), (x + w, center_y), 1)

        for i, val in enumerate(self.current_action):
            bx = x + i * (bar_w + 2)
            # val 범위: -1.0 ~ +1.0
            bh = int(abs(val) * (h // 2 - 2))
            bar_color = ACCENT_CYAN if val >= 0 else ACCENT_ORANGE
            
            if val >= 0:
                rect = pygame.Rect(bx, center_y - bh, bar_w, bh)
            else:
                rect = pygame.Rect(bx, center_y, bar_w, bh)
            
            pygame.draw.rect(self.screen, bar_color, rect, border_radius=2)

    def cleanup(self):
        """종료 시 자원 해제"""
        print("[App] Cleaning up and closing environment...")
        self.env.close()
        pygame.quit()


def main():
    app = VisualHumanoidApp()
    app.run()


if __name__ == "__main__":
    main()
