#!/usr/bin/env python3
"""
test_u_shape.py — Chạy test robot trên map hình chữ U
======================================================
Mục đích: Khảo sát khả năng:
  1. Robot đi vào lòng chữ U từ miệng
  2. Giảm tốc khi tiếp cận cánh tường
  3. Tránh tường và đến đích ở đáy chữ U

Chạy:
    cd ~/rl_nav
    python3 SAC_Sparse_HER/test_u_shape.py              # DWA thuần
    python3 SAC_Sparse_HER/test_u_shape.py --sac        # DWA + SAC
    python3 SAC_Sparse_HER/test_u_shape.py --no-render  # Headless (nhanh hơn)

Output:
    - MuJoCo viewer (nếu render=True)
    - Plot quỹ đạo + velocity profile
    - In log từng bước ra terminal
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR   not in sys.path: sys.path.insert(0, _SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from gymnasium.wrappers import TimeLimit
from gym_bunker_env import BunkerEnv

# ══════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════
XML_PATH     = os.path.join(_PROJECT_ROOT, "assets", "worlds",
                             "test", "world_u_tunnel.xml")

# Pose cố định — robot tại miệng U, goal ở đáy U
ROBOT_START  = [1.1, 9.0, -1.5708]   # [x, y, theta] — hướng xuống (−π/2)
GOAL_POS     = [4.9, 9.0,  0.0]      # goal ở cánh PHẢI đường hầm

MAX_STEPS    = 600
MAX_GOAL_DIST = 15.0                  # đủ rộng để sample pose thủ công

# DWA weights mặc định (khi không dùng SAC)
DWA_WEIGHTS  = (0.55, 0.30, 0.15)    # (alpha, beta, gamma)

# SAC model
RUN_NAME     = "run_10"
CKPT_PATH    = os.path.join(_SCRIPT_DIR, "log", RUN_NAME,
                             "best_model", "best_model.zip")
# ══════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sac",       action="store_true",
                   help="Dùng SAC policy thay vì DWA weights cố định")
    p.add_argument("--no-render", action="store_true",
                   help="Tắt MuJoCo viewer (chạy headless)")
    p.add_argument("--no-plot",   action="store_true",
                   help="Không vẽ đồ thị sau khi chạy")
    p.add_argument("--run",       default=RUN_NAME,
                   help=f"Tên run SAC (default: {RUN_NAME})")
    return p.parse_args()


def load_sac_model(env, run_name):
    """Load SAC model — HER yêu cầu truyền env vào."""
    from stable_baselines3 import SAC
    ckpt = os.path.join(_SCRIPT_DIR, "log", run_name,
                        "best_model", "best_model.zip")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Không tìm thấy model: {ckpt}")
    from stable_baselines3.common.vec_env import DummyVecEnv
    _called = [False]
    def _factory():
        if _called[0]:
            raise RuntimeError("Chỉ gọi 1 lần")
        _called[0] = True
        return env
    vec_env = DummyVecEnv([_factory])
    model   = SAC.load(ckpt, env=vec_env, device="auto")
    print(f"  ✓ SAC model loaded: {ckpt}")
    return model


def run_episode(env, policy_fn):
    """
    Chạy 1 episode với pose cố định và thu thập log.
    policy_fn(obs) → action np.ndarray (3,)
    """
    env.reset()
    raw = env.unwrapped
    dt  = raw.model.opt.timestep * raw.frame_skip

    # Set pose thủ công — bypass random sampling
    print(f"\n  Start : ({ROBOT_START[0]}, {ROBOT_START[1]}, θ={ROBOT_START[2]:.2f} rad)")
    print(f"  Goal  : ({GOAL_POS[0]}, {GOAL_POS[1]})")

    # Dùng _set_qpos_pose trực tiếp
    raw._goal = np.array(GOAL_POS, dtype=np.float32)
    raw._set_qpos_pose(ROBOT_START[0], ROBOT_START[1], ROBOT_START[2])
    import mujoco
    mujoco.mj_forward(raw.model, raw.data)
    raw.data.qvel[:] = 0.0
    raw._set_goal_marker_position(raw._goal, is_final_goal=True)
    obs = raw._get_obs()

    # Log
    log = {
        "t":         [],
        "x":         [],
        "y":         [],
        "v":         [],
        "w":         [],
        "dist":      [],
        "alpha":     [],
        "beta":      [],
        "gamma":     [],
        "min_range": [],
    }

    print()
    print(f"  {'Step':>5}  {'x':>6}  {'y':>6}  {'θ°':>6}  "
          f"{'v':>6}  {'ω':>6}  {'dist':>6}  {'min_r':>6}  "
          f"{'α':>5}  {'β':>5}  {'γ':>5}")
    print("  " + "-"*85)

    outcome = "TIMEOUT"
    for step in range(MAX_STEPS):
        action = policy_fn(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        ag      = obs["achieved_goal"]
        x, y    = float(ag[0]), float(ag[1])
        qw, qz  = float(raw.data.qpos[3]), float(raw.data.qpos[6])
        theta   = 2.0 * np.arctan2(qz, qw)

        lin, ang   = raw.get_robot_velocities()
        v_cur      = float(lin[0])
        w_cur      = float(ang[1])

        dist_goal = float(np.linalg.norm(ag[:2] - raw._goal[:2]))

        # Min LiDAR range
        lidar   = obs["observation"][:raw.n_lidar*3].reshape(-1, 3)
        d_norm  = lidar[:, 2]
        d_real  = (d_norm + 1.0) / 2.0 * raw.lidar_max_range
        min_r   = float(d_real.min())

        a, b, g = float(action[0]), float(action[1]), float(action[2])

        t = step * dt
        log["t"].append(t)
        log["x"].append(x)
        log["y"].append(y)
        log["v"].append(v_cur)
        log["w"].append(w_cur)
        log["dist"].append(dist_goal)
        log["alpha"].append(a)
        log["beta"].append(b)
        log["gamma"].append(g)
        log["min_range"].append(min_r)

        # Print mỗi 10 step
        if step % 10 == 0 or terminated or truncated:
            print(f"  {step:>5}  {x:>6.2f}  {y:>6.2f}  {np.degrees(theta):>6.1f}  "
                  f"{v_cur:>+6.3f}  {w_cur:>+6.3f}  {dist_goal:>6.3f}  {min_r:>6.2f}  "
                  f"{a:>5.2f}  {b:>5.2f}  {g:>5.2f}")

        if info.get("is_success", False):
            outcome = "SUCCESS"; break
        if info.get("collision", False):
            outcome = "COLLISION"; break
        if truncated:
            break

    steps_taken = step + 1
    return outcome, steps_taken, log


def plot_results(log, outcome, steps, method_name, out_path):
    """Vẽ 5 subplot: trajectory + v + ω + dist + min_range."""
    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35)

    color_map = {"SUCCESS": "green", "COLLISION": "red", "TIMEOUT": "orange"}
    title_color = color_map.get(outcome, "black")

    # ── 1. Trajectory (lớn nhất) ───────────────────────────────────────
    ax_traj = fig.add_subplot(gs[:, 0])
    wall_color = '#8B4513'

    # Vẽ tường chữ U
    for cy in [1.0, 3.0, 5.0, 7.0]:
        ax_traj.add_patch(mpatches.Rectangle((-1.2, cy-1.0), 0.4, 2.0,
                          fc=wall_color, ec='black', lw=0.5, alpha=0.85))
        ax_traj.add_patch(mpatches.Rectangle((6.8,  cy-1.0), 0.4, 2.0,
                          fc=wall_color, ec='black', lw=0.5, alpha=0.85))
    for cx in [0.0, 2.0, 4.0, 6.0]:
        ax_traj.add_patch(mpatches.Rectangle((cx-1.0, -0.4), 2.0, 0.4,
                          fc=wall_color, ec='black', lw=0.5, alpha=0.85))

    # Quỹ đạo — tô màu theo tốc độ
    xs = np.array(log["x"])
    ys = np.array(log["y"])
    vs = np.array(log["v"])

    sc = ax_traj.scatter(xs, ys, c=vs, cmap='RdYlGn',
                          vmin=-0.2, vmax=0.5, s=12, zorder=4)
    plt.colorbar(sc, ax=ax_traj, label='v (m/s)', shrink=0.7)

    ax_traj.scatter(*ROBOT_START[:2], s=200, c='limegreen',
                    marker='o', edgecolors='k', lw=1.5, zorder=8, label='Start')
    ax_traj.scatter(*GOAL_POS[:2],   s=250, c='red',
                    marker='*', edgecolors='k', lw=1.0, zorder=8, label='Goal')
    ax_traj.add_patch(mpatches.Circle(GOAL_POS[:2], 0.2,
                      fill=False, ec='red', ls='--', lw=1.5))

    ax_traj.set_xlim(-2.5, 8.5); ax_traj.set_ylim(-1.5, 11)
    ax_traj.set_aspect('equal'); ax_traj.grid(alpha=0.3)
    ax_traj.legend(fontsize=9, loc='upper right')
    ax_traj.set_title(f"Quỹ đạo robot\n{method_name}  →  {outcome}  ({steps} steps)",
                      fontsize=11, fontweight='bold', color=title_color)
    ax_traj.set_xlabel('x (m)'); ax_traj.set_ylabel('y (m)')

    t = np.array(log["t"])

    # ── 2. Vận tốc tuyến tính ─────────────────────────────────────────
    ax_v = fig.add_subplot(gs[0, 1])
    ax_v.plot(t, log["v"], 'b-', lw=1.5, label='v (m/s)')
    ax_v.axhline(0, color='k', lw=0.8, ls='--')
    ax_v.fill_between(t, log["v"], 0,
                       where=np.array(log["v"]) > 0, alpha=0.15, color='blue',
                       label='Tiến')
    ax_v.fill_between(t, log["v"], 0,
                       where=np.array(log["v"]) < 0, alpha=0.15, color='red',
                       label='Lùi')
    ax_v.set_ylabel('v (m/s)'); ax_v.set_xlabel('t (s)')
    ax_v.set_title('Vận tốc tuyến tính  (giảm khi gần tường?)')
    ax_v.legend(fontsize=8); ax_v.grid(alpha=0.3)
    ax_v.set_ylim(-0.25, 0.6)

    # ── 3. Vận tốc góc ────────────────────────────────────────────────
    ax_w = fig.add_subplot(gs[0, 2])
    ax_w.plot(t, log["w"], 'purple', lw=1.5)
    ax_w.axhline(0, color='k', lw=0.8, ls='--')
    ax_w.fill_between(t, log["w"], 0, alpha=0.15, color='purple')
    ax_w.set_ylabel('ω (rad/s)'); ax_w.set_xlabel('t (s)')
    ax_w.set_title('Vận tốc góc  (tăng khi rẽ tránh tường?)')
    ax_w.grid(alpha=0.3)

    # ── 4. Khoảng cách đến goal + min LiDAR ───────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(t, log["dist"],      'navy',  lw=1.5, label='Dist to goal (m)')
    ax_d.plot(t, log["min_range"], 'darkorange', lw=1.5,
              label='Min LiDAR range (m)', ls='--')
    ax_d.axhline(0.2, color='red', lw=1.0, ls=':', label='Goal threshold (0.2m)')
    ax_d.axhline(0.4, color='gray', lw=1.0, ls=':', label='Robot radius (0.4m)')
    ax_d.set_ylabel('m'); ax_d.set_xlabel('t (s)')
    ax_d.set_title('Khoảng cách đến goal và LiDAR tối thiểu')
    ax_d.legend(fontsize=8); ax_d.grid(alpha=0.3)

    # ── 5. DWA weights theo thời gian ─────────────────────────────────
    ax_wt = fig.add_subplot(gs[1, 2])
    ax_wt.plot(t, log["alpha"], 'royalblue', lw=1.5, label='α (heading)')
    ax_wt.plot(t, log["beta"],  'darkorange', lw=1.5, label='β (clearance)')
    ax_wt.plot(t, log["gamma"], 'green',      lw=1.5, label='γ (velocity)')
    ax_wt.set_ylim(0, 1.05)
    ax_wt.set_ylabel('Weight value')
    ax_wt.set_xlabel('t (s)')
    ax_wt.set_title('DWA weights SAC output theo thời gian\n(thay đổi = adaptive)')
    ax_wt.legend(fontsize=8); ax_wt.grid(alpha=0.3)

    fig.suptitle(
        f"Test U-Tunnel  —  {method_name}\n"
        f"Outcome: {outcome}  |  Steps: {steps}  |  "
        f"SPL: {GOAL_POS[1] / max(GOAL_POS[1], sum(abs(np.diff(log['x'])) + abs(np.diff(log['y'])))):.3f}",
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n  📊 Plot saved: {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    render_mode = None if args.no_render else "human"
    method      = "DWA + SAC" if args.sac else "DWA thuần"

    print("=" * 60)
    print(f"  TEST MAP CHỮ U  —  {method}")
    print(f"  World: {XML_PATH}")
    print(f"  Render: {'OFF (headless)' if args.no_render else 'ON'}")
    print("=" * 60)

    if not os.path.exists(XML_PATH):
        print(f"\n❌ Không tìm thấy world file: {XML_PATH}")
        print("   Hãy copy world_u_shape.xml vào assets/worlds/test/")
        return

    # ── Tạo env ───────────────────────────────────────────────────────
    env = TimeLimit(
        BunkerEnv(xml_path=XML_PATH, render_mode=render_mode,
                  max_goal_sampling_distance=MAX_GOAL_DIST),
        max_episode_steps=MAX_STEPS
    )

    # ── Tạo policy function ───────────────────────────────────────────
    if args.sac:
        print(f"\n  Loading SAC model ({args.run})...")
        try:
            sac_model = load_sac_model(env.unwrapped, args.run)
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
            env.close(); return

        def policy_fn(obs):
            obs_b = {k: np.array(v)[np.newaxis, :] for k, v in obs.items()}
            action, _ = sac_model.predict(obs_b, deterministic=True)
            return action[0].astype(np.float32)
    else:
        # DWA thuần — weights cố định
        fixed_action = np.array(DWA_WEIGHTS, dtype=np.float32)
        def policy_fn(obs):
            return fixed_action
        print(f"\n  DWA weights: α={DWA_WEIGHTS[0]} β={DWA_WEIGHTS[1]} γ={DWA_WEIGHTS[2]}")

    # ── Chạy episode ──────────────────────────────────────────────────
    t0 = time.time()
    outcome, steps, log = run_episode(env, policy_fn)
    elapsed = time.time() - t0

    env.close()

    # ── Tóm tắt ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Outcome    : {outcome}")
    print(f"  Steps      : {steps} / {MAX_STEPS}")
    print(f"  Wall time  : {elapsed:.1f}s")
    print(f"  Mean v     : {np.mean(log['v']):.3f} m/s")
    print(f"  Mean |ω|   : {np.mean(np.abs(log['w'])):.3f} rad/s")
    print(f"  Min LiDAR  : {min(log['min_range']):.3f} m  (thấp nhất ghi nhận)")
    print(f"  Mean α     : {np.mean(log['alpha']):.3f}")
    print(f"  Mean β     : {np.mean(log['beta']):.3f}")
    print(f"  Mean γ     : {np.mean(log['gamma']):.3f}")
    print("=" * 60)

    # ── Vẽ kết quả ───────────────────────────────────────────────────
    if not args.no_plot:
        tag      = "sac" if args.sac else "dwa"
        out_dir  = os.path.join(_SCRIPT_DIR, "comparison_results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"u_shape_test_{tag}.png")
        plot_results(log, outcome, steps, method, out_path)


if __name__ == "__main__":
    main()
