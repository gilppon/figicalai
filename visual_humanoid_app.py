"""
FIGICAL AI ROBOT LAB - Humanoid-v5 Real-time Deep Exploration PPO Dashboard.
타인 GUI(ROBOT AI LAB)의 세련된 4단 분할(3D 뷰 오버레이 카드, 2x2 빅넘버, 17 DOF 2열 바, 하단 와이드 차트 + 실시간 콘솔 롤링 로그)을
100% 완벽 흡수하고, 한글 폰트 무결성 및 60FPS 비동기 PPO 엔진을 장착한 초격차 강화학습 대시보드입니다.
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
from rl_agent import HumanoidRLManager, JOINT_NAMES_LEFT, JOINT_NAMES_RIGHT, ALL_JOINT_NAMES

# ========================================================
# 🎨 UI & 테마 색상 팔레트 (Cyberpunk Deep Tech Slate)
# ========================================================
BG_DARK = (11, 15, 23)
HEADER_BG = (15, 21, 32)
PANEL_BG = (17, 23, 34)
PANEL_INNER_BG = (22, 30, 44)
PANEL_BORDER = (38, 48, 66)
OVERLAY_CARD_BG = (15, 21, 32, 210) # 반투명 오버레이

ACCENT_CYAN = (0, 229, 255)
ACCENT_MINT = (0, 230, 118)
ACCENT_GOLD = (255, 196, 0)
ACCENT_ORANGE = (255, 109, 0)
ACCENT_RED = (255, 45, 85)
ACCENT_MAGENTA = (255, 64, 129)
ACCENT_PURPLE = (124, 77, 255)
ACCENT_BLUE = (68, 138, 255)

TEXT_WHITE = (245, 247, 250)
TEXT_MUTED = (130, 142, 160)
TEXT_DIM = (90, 102, 120)
BAR_TRACK = (28, 38, 54)
GRAPH_GRID = (26, 36, 50)


def get_font(size, bold=False):
    """Windows/Linux/Mac에서 한글이 깨지지 않는 시스템 폰트를 안전하게 로드합니다."""
    font_candidates = ["malgungothic", "nanumgothic", "applegothic", "segoeui", "consolas", "arial"]
    return pygame.font.SysFont(font_candidates, size, bold=bold)


class VisualHumanoidApp:
    def __init__(self, win_width=1360, win_height=780, render_size=(790, 475)):
        pygame.init()
        pygame.display.set_caption("FIGICAL AI ROBOT LAB | PPO Deep Reinforcement Learning (Humanoid-v5)")

        self.win_width = win_width
        self.win_height = win_height
        self.render_width, self.render_height = render_size
        self.screen = pygame.display.set_mode((win_width, win_height))
        self.clock = pygame.time.Clock()

        # 폰트 로드 (한글 깨짐 없는 고해상도 폰트 체계)
        self.font_header_title = get_font(18, bold=True)
        self.font_header_sub = get_font(12, bold=False)
        self.font_card_title = get_font(11, bold=True)
        self.font_card_val_large = get_font(22, bold=True)
        self.font_section_title = get_font(12, bold=True)
        self.font_body = get_font(11, bold=False)
        self.font_body_bold = get_font(11, bold=True)
        self.font_console = get_font(11, bold=False)
        self.font_overlay_label = get_font(10, bold=True)
        self.font_overlay_val = get_font(16, bold=True)

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
        self.alive_step = 0
        self.episode_reward = 0.0
        self.best_reward = -float("inf")
        self.last_rewards = collections.deque(maxlen=100)
        self.current_action = np.zeros(self.env.action_space.shape)
        self.step_reward = 0.0

        # 하단 롤링 터미널 콘솔 로그 버퍼 (최대 100개 저장, 화면에 최근 6개 표시)
        self.console_logs = collections.deque(maxlen=100)
        self.console_logs.append("[SYSTEM] FIGICAL AI ROBOT LAB 초기화 완료 - PPO 엔진 준비됨.")
        self.console_logs.append("[SYSTEM] MuJoCo Humanoid-v5 3D 물리 시뮬레이션 연결 성공.")

        # 제어 변수
        self.is_paused = False
        self.speed_multiplier = 1 # 1x, 2x, 4x, 8x
        self.is_running = True
        self.fps = 60

    def run(self):
        """메인 애플리케이션 루프"""
        while self.is_running:
            self.handle_events()
            
            if not self.is_paused:
                for _ in range(self.speed_multiplier):
                    self.step_simulation()

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
                elif event.key in (pygame.K_f, pygame.K_PLUS, pygame.K_EQUALS):
                    # [F] 키로 배속 토글 1x -> 2x -> 4x -> 8x -> 1x
                    speeds = [1, 2, 4, 8]
                    curr_idx = speeds.index(self.speed_multiplier) if self.speed_multiplier in speeds else 0
                    self.speed_multiplier = speeds[(curr_idx + 1) % len(speeds)]
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
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_s:
                    self.rl_manager.save_checkpoint()
                    self.console_logs.append(f"[CHECKPOINT] 모델 체크포인트 저장 완료 (Episode #{self.episode_count})")
                elif event.key == pygame.K_l:
                    if self.rl_manager.load_checkpoint():
                        self.console_logs.append("[CHECKPOINT] 모델 체크포인트 로드 성공.")

    def step_simulation(self):
        """1 스텝 시뮬레이션 및 메트릭 갱신"""
        action = self.rl_manager.select_action(self.obs)
        self.current_action = action

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_reward = reward
        self.episode_reward += reward
        self.alive_step += 1

        self.obs = next_obs

        if terminated or truncated:
            self.reset_episode()

    def reset_episode(self):
        """에피소드 리셋 및 롤링 로그 기록"""
        self.last_rewards.append(self.episode_reward)
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
        
        # 롤링 터미널 로그 추가
        log_msg = f"[INFO] 에피소드 {self.episode_count:04d} 완료: 보상 {self.episode_reward:+.1f} ({self.alive_step} 스텝)"
        self.console_logs.append(log_msg)

        self.obs, self.info = self.env.reset()
        self.episode_count += 1
        self.alive_step = 0
        self.episode_reward = 0.0

    def render_ui(self):
        """전체 4단 분할 UI 화면 렌더링"""
        self.screen.fill(BG_DARK)

        # 1. 상단 네온 헤더 바 렌더링
        self.draw_header()

        # 2. 중앙 좌측: 3D 물리 시뮬레이션 뷰 + 상단 3종 오버레이 HUD 카드
        self.draw_3d_viewport(20, 50, self.render_width, self.render_height)

        # 3. 중앙 우측: 텔레메트리 3단 카드 패널 (2x2 빅넘버, PPO 진단, 17 DOF 모터 바)
        right_panel_x = 20 + self.render_width + 15
        right_panel_w = self.win_width - right_panel_x - 20
        self.draw_right_telemetry_panel(right_panel_x, 50, right_panel_w, self.render_height)

        # 4. 하단 영역 (와이드 보상 곡선 + 실시간 콘솔 롤링 로그 박스)
        bottom_y = 50 + self.render_height + 15
        bottom_h = self.win_height - bottom_y - 15
        bottom_chart_w = 490
        bottom_console_x = 20 + bottom_chart_w + 15
        bottom_console_w = self.win_width - bottom_console_x - 20

        self.draw_bottom_reward_chart(20, bottom_y, bottom_chart_w, bottom_h)
        self.draw_bottom_console_logs(bottom_console_x, bottom_y, bottom_console_w, bottom_h)

        pygame.display.flip()

    def draw_header(self):
        """상단 네온 헤더 바 (브랜딩 타이틀 + 상태 배지 + 배속 인디케이터)"""
        # 헤더 배경
        pygame.draw.rect(self.screen, HEADER_BG, (0, 0, self.win_width, 42))
        pygame.draw.line(self.screen, PANEL_BORDER, (0, 42), (self.win_width, 42), 1)

        # 좌측 Cyan Accent Bar & Title
        pygame.draw.rect(self.screen, ACCENT_CYAN, (20, 10, 4, 22), border_radius=2)
        title_surf = self.font_header_title.render("FIGICAL AI ROBOT LAB", True, TEXT_WHITE)
        self.screen.blit(title_surf, (32, 11))

        # 서브타이틀
        sub_surf = self.font_header_sub.render("PPO Deep Reinforcement Learning | Humanoid-v5 Continuous Control", True, TEXT_MUTED)
        self.screen.blit(sub_surf, (250, 15))

        # 우측 상태 배지
        deep_m = self.rl_manager.get_deep_metrics()
        status_text = "■ PPO 학습중 (Active)" if not self.is_paused else "⏸ 일시정지 (Paused)"
        status_col = ACCENT_MINT if not self.is_paused else ACCENT_ORANGE
        status_surf = self.font_header_sub.render(status_text, True, status_col)
        self.screen.blit(status_surf, (self.win_width - 240, 14))

        # 배속 인디케이터
        speed_text = f"배속: {self.speed_multiplier}x [F]"
        speed_surf = self.font_header_sub.render(speed_text, True, ACCENT_CYAN)
        self.screen.blit(speed_surf, (self.win_width - 100, 14))

    def draw_3d_viewport(self, x, y, w, h):
        """3D 물리 시뮬레이션 화면 + 상단 3종 반투명 오버레이 HUD 카드"""
        # 1. 3D 물리 시뮬레이션 프레임
        frame = self.env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self.screen.blit(surf, (x, y))

        # 외곽선 테두리
        pygame.draw.rect(self.screen, PANEL_BORDER, (x, y, w, h), 2, border_radius=6)

        # 2. 상단 3종 반투명 오버레이 HUD 카드
        deep_m = self.rl_manager.get_deep_metrics()
        noise_val = 0.60 if deep_m["exploration_boost"] else 0.30

        overlay_cards = [
            ("EPISODE", f"{self.episode_count:03d}", TEXT_WHITE, 110),
            ("ALIVE STEP", f"{self.alive_step}", ACCENT_MINT, 110),
            ("EXPLORATION NOISE", f"σ = {noise_val:.2f}", ACCENT_GOLD, 170)
        ]

        card_x = x + 15
        card_y = y + 15
        card_h = 44

        for label, val, val_color, card_w in overlay_cards:
            # 반투명 배경 서피스
            overlay_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            overlay_surf.fill(OVERLAY_CARD_BG)
            pygame.draw.rect(overlay_surf, PANEL_BORDER, (0, 0, card_w, card_h), 1, border_radius=4)
            self.screen.blit(overlay_surf, (card_x, card_y))

            # 라벨 및 수치 렌더링
            lbl_s = self.font_overlay_label.render(label, True, TEXT_MUTED)
            val_s = self.font_overlay_val.render(val, True, val_color)
            self.screen.blit(lbl_s, (card_x + 10, card_y + 5))
            self.screen.blit(val_s, (card_x + 10, card_y + 20))

            card_x += card_w + 12

    def draw_right_telemetry_panel(self, x, y, w, h):
        """우측 텔레메트리 패널 (2x2 빅넘버 + PPO 진단 + 17 DOF 모터 바)"""
        # 패널 배경
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x, y, w, h), 1, border_radius=6)

        inner_x = x + 12
        inner_w = w - 24
        cy = y + 12

        # ----------------------------------------------------
        # 1. 2x2 빅 넘버 메트릭 카드
        # ----------------------------------------------------
        card_h = 108
        pygame.draw.rect(self.screen, PANEL_INNER_BG, (inner_x, cy, inner_w, card_h), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (inner_x, cy, inner_w, card_h), 1, border_radius=4)

        avg_20 = np.mean(list(self.last_rewards)[-20:]) if self.last_rewards else 0.0
        best_val = self.best_reward if self.best_reward > -1e5 else 0.0
        deep_m = self.rl_manager.get_deep_metrics()

        cards_2x2 = [
            ("현재 에피소드 보상", f"{self.episode_reward:+.1f}", ACCENT_CYAN, 0, 0),
            ("누적 최고 보상", f"{best_val:.1f}", ACCENT_GOLD, 1, 0),
            ("20-Ep 평균 보상", f"{avg_20:.1f}", ACCENT_MINT, 0, 1),
            ("총 학습량 (스텝)", f"{deep_m['total_timesteps']:,}", ACCENT_BLUE, 1, 1),
        ]

        col_w = inner_w // 2
        row_h = card_h // 2

        for label, val, color, col, row in cards_2x2:
            bx = inner_x + col * col_w + 12
            by = cy + row * row_h + 6
            lbl_s = self.font_card_title.render(label, True, TEXT_MUTED)
            val_s = self.font_card_val_large.render(val, True, color)
            self.screen.blit(lbl_s, (bx, by))
            self.screen.blit(val_s, (bx, by + 16))

        cy += card_h + 12

        # ----------------------------------------------------
        # 2. PPO 학습 진단 (Learning Diagnostics)
        # ----------------------------------------------------
        diag_h = 96
        pygame.draw.rect(self.screen, PANEL_INNER_BG, (inner_x, cy, inner_w, diag_h), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (inner_x, cy, inner_w, diag_h), 1, border_radius=4)

        diag_title = self.font_section_title.render("PPO 학습 진단 (Learning Diagnostics)", True, TEXT_WHITE)
        self.screen.blit(diag_title, (inner_x + 12, cy + 8))

        p_loss = deep_m["policy_loss"]
        v_loss = deep_m["value_loss"]
        ent = abs(deep_m["entropy_loss"])
        updates = deep_m["update_count"]

        # 손실 및 진단 지표 출력
        l1 = f"PPO Policy Loss : {p_loss:+.4f}"
        l2 = f"Value Func Loss : {v_loss:.4f}"
        l3 = f"Entropy (탐색률) : {ent:.3f} | Updates : #{updates}"

        self.screen.blit(self.font_body.render(l1, True, ACCENT_ORANGE if p_loss >= 0 else ACCENT_RED), (inner_x + 12, cy + 30))
        self.screen.blit(self.font_body.render(l2, True, ACCENT_GOLD), (inner_x + 12, cy + 48))
        self.screen.blit(self.font_body.render(l3, True, ACCENT_MINT), (inner_x + 12, cy + 68))

        cy += diag_h + 12

        # ----------------------------------------------------
        # 3. 액추에이터 토크 - 17 DOF (Actuator Torques - 17 DOF)
        # ----------------------------------------------------
        act_h = h - (cy - y) - 12
        pygame.draw.rect(self.screen, PANEL_INNER_BG, (inner_x, cy, inner_w, act_h), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (inner_x, cy, inner_w, act_h), 1, border_radius=4)

        act_title = self.font_section_title.render("액추에이터 토크 (Actuator Torques - 17 DOF)", True, TEXT_WHITE)
        self.screen.blit(act_title, (inner_x + 12, cy + 8))

        # 좌측 9개, 우측 8개 2열 렌더링
        start_y = cy + 30
        line_spacing = 18
        col_width = (inner_w - 30) // 2

        # 좌측 열 (0~8)
        for i, name in enumerate(JOINT_NAMES_LEFT):
            val = self.current_action[i] if i < len(self.current_action) else 0.0
            row_y = start_y + i * line_spacing
            self.draw_joint_torque_row(inner_x + 10, row_y, col_width, name, val)

        # 우측 열 (9~16)
        for j, name in enumerate(JOINT_NAMES_RIGHT):
            idx = 9 + j
            val = self.current_action[idx] if idx < len(self.current_action) else 0.0
            row_y = start_y + j * line_spacing
            self.draw_joint_torque_row(inner_x + 10 + col_width + 10, row_y, col_width, name, val)

    def draw_joint_torque_row(self, x, y, w, name, val):
        """관절명 + 양방향 토크 바 (Cyan = 양수, Magenta = 음수)"""
        # 관절 라벨
        lbl_s = self.font_body.render(name, True, TEXT_MUTED)
        self.screen.blit(lbl_s, (x, y))

        # 토크 바 영역
        bar_x = x + 85
        bar_w = w - 90
        bar_h = 6
        bar_y = y + 4

        # 배경 트랙
        pygame.draw.rect(self.screen, BAR_TRACK, (bar_x, bar_y, bar_w, bar_h), border_radius=2)

        # 중앙 기준 (0)
        center_x = bar_x + bar_w // 2
        norm_val = np.clip(val, -1.0, 1.0)
        fill_len = int(abs(norm_val) * (bar_w // 2))

        if norm_val >= 0:
            # 양의 토크 (Cyan -> 오른쪽으로)
            rect = pygame.Rect(center_x, bar_y, fill_len, bar_h)
            color = ACCENT_CYAN
        else:
            # 음의 토크 (Magenta -> 왼쪽으로)
            rect = pygame.Rect(center_x - fill_len, bar_y, fill_len, bar_h)
            color = ACCENT_MAGENTA

        if fill_len > 0:
            pygame.draw.rect(self.screen, color, rect, border_radius=2)

    def draw_bottom_reward_chart(self, x, y, w, h):
        """하단 좌측: 실시간 보상 추이 곡선 (Reward Curve)"""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x, y, w, h), 1, border_radius=6)

        # 타이틀
        title = self.font_section_title.render("실시간 보상 추이 곡선 (Reward Curve)", True, TEXT_WHITE)
        self.screen.blit(title, (x + 14, y + 10))

        # 우측 상단 최고 보상 마커
        max_r_val = max(self.last_rewards) if self.last_rewards else 0.0
        max_lbl = self.font_body_bold.render(f"최고: {max_r_val:.1f}", True, ACCENT_CYAN)
        self.screen.blit(max_lbl, (x + w - max_lbl.get_width() - 14, y + 10))

        # 차트 캔버스 영역
        gx = x + 14
        gy = y + 32
        gw = w - 28
        gh = h - 44

        pygame.draw.rect(self.screen, PANEL_INNER_BG, (gx, gy, gw, gh), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (gx, gy, gw, gh), 1, border_radius=4)

        # 수평 가이드 라인 3개
        for step_i in range(1, 4):
            line_y = gy + step_i * (gh // 4)
            pygame.draw.line(self.screen, GRAPH_GRID, (gx, line_y), (gx + gw, line_y), 1)

        if len(self.last_rewards) < 2:
            no_data = self.font_body.render("에피소드 데이터 수집 중...", True, TEXT_MUTED)
            self.screen.blit(no_data, (gx + gw // 2 - no_data.get_width() // 2, gy + gh // 2 - 8))
            return

        rewards = list(self.last_rewards)
        min_r = min(min(rewards), 0)
        max_r = max(max(rewards), 300)
        rng = max_r - min_r if max_r != min_r else 1.0

        points = []
        step_x = gw / (len(rewards) - 1)
        for idx, r in enumerate(rewards):
            px = gx + idx * step_x
            py = gy + gh - ((r - min_r) / rng) * (gh - 12) - 6
            points.append((px, py))

        # 라인 차트 렌더링
        if len(points) >= 2:
            pygame.draw.lines(self.screen, ACCENT_CYAN, False, points, 2)
            # 마지막 점 강조 마커
            last_pt = (int(points[-1][0]), int(points[-1][1]))
            pygame.draw.circle(self.screen, ACCENT_MINT, last_pt, 4)

    def draw_bottom_console_logs(self, x, y, w, h):
        """하단 우측: 실시간 롤링 터미널 콘솔 로그 박스"""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x, y, w, h), 1, border_radius=6)

        # 타이틀
        title = self.font_section_title.render("PPO 에피소드 완료 로그 (Live Console)", True, TEXT_WHITE)
        self.screen.blit(title, (x + 14, y + 10))

        # 단축키 안내 뱃지
        keys_hint = self.font_body.render("[SPACE] 일시정지  [E] 탐색부스트  [T] 터보  [F] 배속  [1~4] 모드", True, TEXT_MUTED)
        self.screen.blit(keys_hint, (x + w - keys_hint.get_width() - 14, y + 10))

        # 로그 박스 내부 캔버스
        lx = x + 14
        ly = y + 32
        lw = w - 28
        lh = h - 44

        pygame.draw.rect(self.screen, PANEL_INNER_BG, (lx, ly, lw, lh), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (lx, ly, lw, lh), 1, border_radius=4)

        # 최근 로그 7줄 렌더링
        recent_logs = list(self.console_logs)[-7:]
        log_start_y = ly + 8
        for i, log_text in enumerate(recent_logs):
            # 로그 유형별 색상
            if "[SYSTEM]" in log_text:
                color = ACCENT_CYAN
            elif "[CHECKPOINT]" in log_text:
                color = ACCENT_GOLD
            elif "보상 +" in log_text or "보상 2" in log_text:
                color = ACCENT_MINT
            else:
                color = TEXT_WHITE

            log_surf = self.font_console.render(log_text, True, color)
            self.screen.blit(log_surf, (lx + 10, log_start_y + i * 20))

    def cleanup(self):
        """종료 시 자원 정리"""
        print("[App] Cleaning up and closing environment...")
        self.rl_manager.close()
        pygame.quit()


def main():
    app = VisualHumanoidApp()
    app.run()


if __name__ == "__main__":
    main()
