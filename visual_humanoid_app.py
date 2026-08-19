"""
Real-time 3D Humanoid Deep Exploration & PPO Telemetry HUD Dashboard.
실시간 3D 물리 시뮬레이션 화면과 함께 강화학습 에이전트의 심층 탐색(Exploration),
엔트로피 스케줄링, 관절 다양성 지수, PPO 신경망 손실 지표를 실시간으로 모니터링합니다.
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
# 🎨 UI & 테마 색상 팔레트 (Cyberpunk Deep Tech)
# ==========================================
BG_DARK = (10, 14, 20)
PANEL_BG = (18, 24, 34)
PANEL_BORDER = (42, 52, 68)
ACCENT_CYAN = (0, 229, 255)
ACCENT_GREEN = (0, 230, 118)
ACCENT_ORANGE = (255, 145, 0)
ACCENT_RED = (255, 61, 0)
ACCENT_PURPLE = (213, 0, 249)
ACCENT_YELLOW = (255, 214, 0)
TEXT_WHITE = (240, 246, 252)
TEXT_MUTED = (139, 148, 158)
BAR_BG = (28, 36, 48)
GRAPH_GRID = (25, 32, 44)


class VisualHumanoidApp:
    def __init__(self, win_width=1320, win_height=740, render_size=(760, 700)):
        pygame.init()
        pygame.display.set_caption("⚡ Humanoid-v5 Deep Exploration RL Dashboard | PPO Engine")

        self.win_width = win_width
        self.win_height = win_height
        self.render_width, self.render_height = render_size
        self.screen = pygame.display.set_mode((win_width, win_height))
        self.clock = pygame.time.Clock()

        # 폰트 초기화 (Consolas -> Arial -> 기본 폰트 순서 폴백)
        try:
            self.font_title = pygame.font.SysFont("consolas", 18, bold=True)
            self.font_main = pygame.font.SysFont("consolas", 13, bold=True)
            self.font_sub = pygame.font.SysFont("consolas", 11)
            self.font_badge = pygame.font.SysFont("consolas", 12, bold=True)
            self.font_phase = pygame.font.SysFont("consolas", 12, bold=True)
        except Exception:
            self.font_title = pygame.font.Font(None, 22)
            self.font_main = pygame.font.Font(None, 16)
            self.font_sub = pygame.font.Font(None, 13)
            self.font_badge = pygame.font.Font(None, 14)
            self.font_phase = pygame.font.Font(None, 14)

        # MuJoCo 환경 및 심층 RL 매니저 초기화
        print("[App] Initializing MuJoCo Humanoid-v5 Environment & PPO Deep Exploration Manager...")
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
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.is_running = False
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_e:
                    # [E] 탐색 부스트 토글
                    self.rl_manager.toggle_exploration_boost()
                elif event.key == pygame.K_t:
                    # [T] 초고속 터보 학습 토글
                    self.rl_manager.toggle_turbo_mode()
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
        """1 스텝 시뮬레이션 및 메트릭 갱신"""
        # 행동 결정 (탐색 노이즈 반영)
        action = self.rl_manager.select_action(self.obs)
        self.current_action = action

        # 환경 스텝 진행
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_reward = reward
        self.episode_reward += reward
        self.current_step += 1

        # 텔레메트리 추출
        self.forward_velocity = info.get("x_velocity", 0.0)
        if len(next_obs) > 0:
            self.torso_height = float(next_obs[0]) if isinstance(next_obs, np.ndarray) else 1.4

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
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self.screen.blit(surf, (20, 20))
            # 렌더링 뷰 테두리
            pygame.draw.rect(self.screen, PANEL_BORDER, (20, 20, self.render_width, self.render_height), 2)

        # 2. 우측 텔레메트리 HUD 패널
        panel_x = self.render_width + 35
        panel_y = 20
        panel_w = self.win_width - panel_x - 20
        panel_h = self.render_height

        self.draw_hud_panel(panel_x, panel_y, panel_w, panel_h)

        pygame.display.flip()

    def draw_hud_panel(self, x, y, w, h):
        """사이버펑크 Deep Exploration HUD 패널 그리기"""
        # 패널 배경 및 외곽선
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x, y, w, h), 2, border_radius=8)

        cy = y + 12
        cx = x + 14
        inner_w = w - 28

        metrics_deep = self.rl_manager.get_deep_metrics()

        # 1. 헤더: 타이틀
        title_surf = self.font_title.render("⚡ HUMANOID DEEP EXPLORATION", True, ACCENT_CYAN)
        self.screen.blit(title_surf, (cx, cy))
        cy += 24

        # 2. 현재 학습 페이즈 배너
        phase_text = f"🎯 {metrics_deep['current_phase']}"
        phase_surf = self.font_phase.render(phase_text, True, ACCENT_YELLOW)
        pygame.draw.rect(self.screen, BAR_BG, (cx, cy, inner_w, 22), border_radius=4)
        pygame.draw.rect(self.screen, ACCENT_YELLOW, (cx, cy, inner_w, 22), 1, border_radius=4)
        self.screen.blit(phase_surf, (cx + 8, cy + 4))
        cy += 28

        # 3. 모드 & 탐색 상태 배지
        stage_colors = {
            "RANDOM": (ACCENT_RED, "[1] RANDOM (FALL)"),
            "EARLY": (ACCENT_ORANGE, "[2] EARLY STAGE"),
            "LIVE_TRAIN": (ACCENT_GREEN, "[3] LIVE TRAINING"),
            "TRAINED": (ACCENT_PURPLE, "[4] TRAINED EXPERT")
        }
        color, stage_name = stage_colors.get(self.rl_manager.current_stage, (TEXT_WHITE, "UNKNOWN"))
        badge_surf = self.font_badge.render(f"MODE: {stage_name}", True, color)
        self.screen.blit(badge_surf, (cx, cy))

        # 부스트 & 터보 상태 태그
        boost_str = "BOOST: ON 🔥" if metrics_deep["exploration_boost"] else "BOOST: OFF"
        boost_col = ACCENT_ORANGE if metrics_deep["exploration_boost"] else TEXT_MUTED
        boost_surf = self.font_sub.render(boost_str, True, boost_col)
        self.screen.blit(boost_surf, (cx + inner_w - 170, cy + 1))

        turbo_str = "TURBO ⚡" if metrics_deep["is_turbo"] else f"SPD: {self.speed_multiplier}x"
        turbo_col = ACCENT_YELLOW if metrics_deep["is_turbo"] else TEXT_MUTED
        turbo_surf = self.font_sub.render(turbo_str, True, turbo_col)
        self.screen.blit(turbo_surf, (cx + inner_w - 65, cy + 1))

        cy += 20
        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 8

        # 4. 🎲 심층 탐색 텔레메트리 (Exploration Gauges)
        lbl_exp = self.font_main.render("🎲 EXPLORATION TELEMETRY", True, ACCENT_CYAN)
        self.screen.blit(lbl_exp, (cx, cy))
        cy += 18

        # 4-1) 관절 탐색 다양성 지수 (Diversity Score)
        div_score = metrics_deep["diversity_index"]
        self.draw_gauge(cx, cy, inner_w, "Joint Diversity Index", div_score * 100, 0.0, 100.0, "%", ACCENT_PURPLE)
        cy += 28

        # 4-2) 정책 엔트로피 / 탐색 수준
        ent_val = abs(metrics_deep["entropy_loss"])
        self.draw_gauge(cx, cy, inner_w, "Policy Entropy (Action Randomness)", ent_val, 0.0, 30.0, "", ACCENT_ORANGE)
        cy += 32

        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 8

        # 5. 📊 에피소드 & 보상 메트릭
        avg_reward = np.mean(self.last_rewards) if self.last_rewards else 0.0
        metrics = [
            ("Episode", f"#{self.episode_count}", TEXT_WHITE),
            ("Step Count", f"{self.current_step}", TEXT_WHITE),
            ("Step Reward", f"{self.step_reward:+.2f}", ACCENT_CYAN if self.step_reward > 0 else ACCENT_RED),
            ("Current Return", f"{self.episode_reward:+.1f}", ACCENT_GREEN),
            ("Best Return", f"{self.best_reward:+.1f}" if self.best_reward > -1e5 else "0.0", ACCENT_PURPLE),
            ("Avg (Last 60)", f"{avg_reward:+.1f}", ACCENT_YELLOW),
        ]

        for i, (label, val, val_color) in enumerate(metrics):
            row = i // 2
            col = i % 2
            col_x = cx + col * (inner_w // 2)
            row_y = cy + row * 20
            
            lbl_s = self.font_sub.render(f"{label}:", True, TEXT_MUTED)
            val_s = self.font_main.render(val, True, val_color)
            self.screen.blit(lbl_s, (col_x, row_y))
            self.screen.blit(val_s, (col_x + 95, row_y - 2))

        cy += 65
        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 8

        # 6. 📈 실시간 보상 학습 곡선 그래프
        lbl_graph = self.font_main.render("📈 REWARD HISTORY (LAST 60 EPS)", True, TEXT_WHITE)
        self.screen.blit(lbl_graph, (cx, cy))
        cy += 18
        self.draw_reward_graph(cx, cy, inner_w, 65)
        cy += 74

        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 8

        # 7. 🧠 PPO 신경망 손실 텔레메트리 (Deep Neural Losses)
        lbl_nn = self.font_main.render("🧠 PPO NEURAL TRAINING TELEMETRY", True, ACCENT_GREEN)
        self.screen.blit(lbl_nn, (cx, cy))
        cy += 18

        nn_metrics = [
            ("Timesteps", f"{metrics_deep['total_timesteps']:,}", ACCENT_CYAN),
            ("Policy Loss", f"{metrics_deep['policy_loss']:.4f}", TEXT_WHITE),
            ("Value Loss", f"{metrics_deep['value_loss']:.4f}", TEXT_WHITE),
            ("Approx KL", f"{metrics_deep['approx_kl']:.5f}", ACCENT_YELLOW),
            ("Clip Ratio", f"{metrics_deep['clip_fraction']:.3f}", TEXT_WHITE),
            ("Device", f"{metrics_deep['device'].upper()}", ACCENT_PURPLE),
        ]

        for i, (label, val, val_color) in enumerate(nn_metrics):
            row = i // 2
            col = i % 2
            col_x = cx + col * (inner_w // 2)
            row_y = cy + row * 18
            
            lbl_s = self.font_sub.render(f"{label}:", True, TEXT_MUTED)
            val_s = self.font_sub.render(val, True, val_color)
            self.screen.blit(lbl_s, (col_x, row_y))
            self.screen.blit(val_s, (col_x + 95, row_y))

        cy += 58
        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 8

        # 8. 🎛️ 17개 관절 모터 출력 히트맵 (Actuator Torques)
        lbl_act = self.font_main.render("🎛️ 17 JOINT ACTUATOR OUTPUTS", True, TEXT_WHITE)
        self.screen.blit(lbl_act, (cx, cy))
        cy += 16
        self.draw_actuator_bars(cx, cy, inner_w, 32)
        cy += 40

        pygame.draw.line(self.screen, PANEL_BORDER, (cx, cy), (cx + inner_w, cy), 1)
        cy += 8

        # 9. 🎮 인터랙티브 단축키 안내 패널
        sc1 = "[SPACE] Pause  [E] Expl-Boost  [T] Turbo-Train  [1~4] Modes"
        sc2 = "[+/-] Speed    [R] Reset Env   [S/L] Save/Load  [ESC/Q] Quit"
        self.screen.blit(self.font_sub.render(sc1, True, ACCENT_CYAN), (cx, cy))
        self.screen.blit(self.font_sub.render(sc2, True, TEXT_MUTED), (cx, cy + 14))

    def draw_reward_graph(self, x, y, w, h):
        """실시간 보상 추이 미니 그래프 렌더링"""
        pygame.draw.rect(self.screen, BAR_BG, (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, GRAPH_GRID, (x, y, w, h), 1, border_radius=4)

        if len(self.last_rewards) < 2:
            no_data = self.font_sub.render("Collecting exploration & reward data...", True, TEXT_MUTED)
            self.screen.blit(no_data, (x + w // 2 - no_data.get_width() // 2, y + h // 2 - 7))
            return

        rewards = list(self.last_rewards)
        min_r = min(min(rewards), 0)
        max_r = max(max(rewards), 500)
        rng = max_r - min_r if max_r != min_r else 1.0

        points = []
        step_x = w / (len(rewards) - 1)
        for idx, r in enumerate(rewards):
            px = x + idx * step_x
            py = y + h - ((r - min_r) / rng) * (h - 8) - 4
            points.append((px, py))

        # 그래프 선 그리기
        if len(points) >= 2:
            pygame.draw.lines(self.screen, ACCENT_CYAN, False, points, 2)
            pygame.draw.circle(self.screen, ACCENT_GREEN, (int(points[-1][0]), int(points[-1][1])), 3)

    def draw_gauge(self, x, y, w, label, val, min_v, max_v, unit, color):
        """수평 프로그레스 바 게이지 렌더링"""
        lbl_s = self.font_sub.render(f"{label}: {val:.1f} {unit}".strip(), True, TEXT_MUTED)
        self.screen.blit(lbl_s, (x, y))

        bar_y = y + 14
        bar_h = 6
        pygame.draw.rect(self.screen, BAR_BG, (x, bar_y, w, bar_h), border_radius=3)

        norm_val = np.clip((val - min_v) / (max_v - min_v), 0.0, 1.0)
        fill_w = int(w * norm_val)
        if fill_w > 0:
            pygame.draw.rect(self.screen, color, (x, bar_y, fill_w, bar_h), border_radius=3)

    def draw_actuator_bars(self, x, y, w, h):
        """17개 관절 모터의 실시간 출력 토크 막대 차트"""
        n_act = len(self.current_action)
        bar_w = (w - (n_act - 1) * 2) // n_act
        center_y = y + h // 2

        pygame.draw.line(self.screen, PANEL_BORDER, (x, center_y), (x + w, center_y), 1)

        for i, val in enumerate(self.current_action):
            bx = x + i * (bar_w + 2)
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
        self.rl_manager.close()
        pygame.quit()


def main():
    app = VisualHumanoidApp()
    app.run()


if __name__ == "__main__":
    main()
