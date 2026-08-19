# -*- coding: utf-8 -*-
"""
Humanoid-v5 Real-time Visual Reinforcement Learning System
대표님의 지시로 구축된 휴머노이드 강화학습 시각화 대시보드 메인 진입점입니다.
"""
import sys
from mujoco_patch import apply_mujoco_patch, make_humanoid_env
from visual_humanoid_app import VisualHumanoidApp

def main():
    print("=" * 65)
    print("⚡ Humanoid-v5 실시간 시각적 강화학습 대시보드를 시작합니다!")
    print("=" * 65)
    print(" [조작 가이드]")
    print("  - [SPACE]       : 시뮬레이션 일시정지 / 재생")
    print("  - [1]           : 모드 1 - 완전 무작위 (넘어지는 모습)")
    print("  - [2]           : 모드 2 - 초기 학습 단계 (비틀거리며 걸음)")
    print("  - [3]           : 모드 3 - 실시간 PPO 학습 진행")
    print("  - [4]           : 모드 4 - 숙련자 모드 (Trained Expert)")
    print("  - [+] / [-]     : 시뮬레이션 배속 (1x, 2x, 4x, 8x)")
    print("  - [R]           : 에피소드 즉시 리셋")
    print("  - [S] / [L]     : 모델 체크포인트 저장 / 로드")
    print("  - [ESC / Q]     : 종료")
    print("=" * 65)

    # 3D 비주얼 GUI 앱 기동
    app = VisualHumanoidApp(win_width=1280, win_height=720, render_size=(760, 680))
    app.run()

if __name__ == "__main__":
    main()