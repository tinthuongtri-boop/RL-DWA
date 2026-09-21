#!/usr/bin/env python3
"""
dwa_standalone_demo.py
----------------------------------------------------------------
Chạy robot Bunker trong MuJoCo CHỈ với DWA (không dùng SAC/RL).

FIX QUAN TRỌNG trong phiên bản này:
  - width_track: dùng trục có khoảng cách lớn nhất (auto-detect)
    thay vì cứng index [2] (trục Z = chiều cao → thường = 0).
  - Thêm diagnostic block để in ra giá trị debug khi khởi động.
"""

import os, sys, time, argparse
import numpy as np

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

if _SCRIPT_DIR   not in sys.path: sys.path.insert(0, _SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

import mujoco
import mujoco.viewer
from DWA import DWASolver
from sensor_utils import get_robot_state_and_lidar
from lib.bunker_velocity_controller import BunkerVelocityController


# =================================================================
# HELPERS
# =================================================================

def setup_velocity_controller(model, data, verbose=True):
    vc = BunkerVelocityController()

    id_rr = model.body("w_rr").id
    id_lr = model.body("w_lr").id
    pos_rr = model.body_pos[id_rr]
    pos_lr = model.body_pos[id_lr]
    diff   = pos_rr - pos_lr

    # ── FIX: auto-detect trục track width (trục có khoảng cách lớn nhất) ──
    best_axis  = int(np.argmax(np.abs(diff)))
    width_track = float(abs(diff[best_axis]))

    if verbose:
        print(f"\n[DIAG] w_rr pos = {pos_rr}")
        print(f"[DIAG] w_lr pos = {pos_lr}")
        print(f"[DIAG] diff     = {diff}")
        print(f"[DIAG] → Track width = {width_track:.4f} m  (trục {['X','Y','Z'][best_axis]})")

    vc.w_track = width_track

    id_geom    = model.geom("w_rr_geom").id
    vc.r_wheel = float(model.geom_size[id_geom][0])   # radius only
    if verbose:
        print(f"[DIAG] Wheel radius = {vc.r_wheel:.4f} m")

    vc.act_r = [model.actuator(n).id for n in ("w_rr","w_rc","w_rf")]
    vc.act_l = [model.actuator(n).id for n in ("w_lr","w_lc","w_lf")]

    mujoco.set_mjcb_control(vc)
    return vc


def set_robot_pose(data, x, y, theta, height=0.25):
    data.qpos[:] = 0.0
    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[2] = height
    c, s = np.cos(theta/2), np.sin(theta/2)
    data.qpos[3] = c;  data.qpos[4] = 0.0
    data.qpos[5] = 0.0; data.qpos[6] = s
    data.qvel[:] = 0.0


def is_collision(model, data):
    for i in range(data.ncon):
        g1 = model.geom(data.contact[i].geom1).name
        g2 = model.geom(data.contact[i].geom2).name
        if g1 == "floor" or g2 == "floor":
            continue
        return True
    return False


def set_goal_marker(model, data, pos, is_final=True):
    name = "final_goal_marker" if is_final else "mid_goal_marker"
    try:
        bid = model.body(name).id
        mid = int(model.body_mocapid[bid])
        data.mocap_pos[mid] = [pos[0], pos[1], 0.1]
    except Exception:
        pass


def find_free_pose(model, data, world_min, world_max, seed=100):
    rng  = np.random.default_rng(seed)
    wmin = np.array(world_min, dtype=np.float32)
    wmax = np.array(world_max, dtype=np.float32)
    for attempt in range(5000):
        sx, sy = rng.uniform(wmin, wmax)
        sth    = rng.uniform(-np.pi, np.pi)
        set_robot_pose(data, float(sx), float(sy), float(sth))
        mujoco.mj_forward(model, data)
        if not is_collision(model, data):
            return float(sx), float(sy), float(sth), attempt+1
    return None, None, None, 5000


def plot_trajectory_2d(traj_log, obstacle_cloud, start_pos, goal_pos,
                       goal_threshold, outcome, out_path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("[warn] matplotlib chưa cài: pip install matplotlib"); return

    traj = np.array(traj_log)
    fig, ax = plt.subplots(figsize=(10,10))

    if len(obstacle_cloud) > 0:
        obs = np.array(obstacle_cloud)
        ax.scatter(obs[:,0], obs[:,1], s=2, c='gray', alpha=0.3,
                   label=f'LiDAR hits ({len(obs)} pts)')

    ax.plot(traj[:,0], traj[:,1], 'b-', lw=1.8, alpha=0.8,
            label='Quỹ đạo robot', zorder=3)

    step = max(1, len(traj)//15)
    for i in range(0, len(traj), step):
        x, y, th = traj[i,0], traj[i,1], traj[i,2]
        ax.arrow(x, y, .35*np.cos(th), .35*np.sin(th),
                 head_width=.18, head_length=.12,
                 fc='blue', ec='blue', alpha=0.6, zorder=4)

    ax.scatter(*start_pos[:2], s=250, c='limegreen', marker='o',
               edgecolors='black', lw=2, label='Start', zorder=5)
    ax.scatter(goal_pos[0], goal_pos[1], s=300, c='red', marker='*',
               edgecolors='black', lw=2, label='Goal', zorder=5)
    ax.add_patch(patches.Circle((goal_pos[0], goal_pos[1]), goal_threshold,
                                fill=False, edgecolor='red',
                                linestyle='--', lw=1.5))

    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12); ax.set_ylabel('Y (m)', fontsize=12)
    color = {'SUCCESS':'green','COLLISION':'red','TIMEOUT':'orange'}.get(outcome,'black')
    ax.set_title(f'DWA Trajectory — Outcome: {outcome}  ({len(traj)} steps)',
                 fontsize=13, color=color, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {out_path}")
    plt.show()


# =================================================================
# MAIN
# =================================================================

def run_demo(args):
    print(f"📂 Loading XML: {args.xml}")
    if not os.path.exists(args.xml):
        print(f"❌ File không tồn tại: {args.xml}"); return

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    vc    = setup_velocity_controller(model, data, verbose=True)
    dwa   = DWASolver()

    # ── Tìm / đặt pose ────────────────────────────────────────
    if args.auto_start:
        print(f"\n🔍 Tìm start pose trong {args.world_min} → {args.world_max}...")
        sx, sy, sth, n = find_free_pose(model, data, args.world_min,
                                        args.world_max, seed=args.seed)
        if sx is None:
            print("❌ Không tìm được pose trống!"); return
        args.start = [sx, sy, sth]
        print(f"✓  Tìm được sau {n} lần: ({sx:.2f}, {sy:.2f}, θ={np.degrees(sth):.0f}°)")
    else:
        set_robot_pose(data, *args.start)
        mujoco.mj_forward(model, data)
        if is_collision(model, data):
            print("❌ Vị trí ban đầu va chạm! Dùng --auto-start hoặc đổi --start"); return

    goal    = np.array(args.goal, dtype=np.float32)
    weights = (args.alpha, args.beta, args.gamma)
    set_goal_marker(model, data, goal)

    print(f"\n🎯 Start  : ({args.start[0]:.2f}, {args.start[1]:.2f}, θ={np.degrees(args.start[2]):.0f}°)")
    print(f"🎯 Goal   : ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"⚖️  Weights : α={args.alpha}  β={args.beta}  γ={args.gamma}")
    print(f"🔧 frame_skip={args.frame_skip}, max_steps={args.max_steps}\n")

    viewer_ctx = mujoco.viewer.launch_passive(model, data) if args.render else None

    try:
        step_count, outcome = 0, "TIMEOUT"
        traj_log, obstacle_cloud = [], []
        t0 = time.time()

        while step_count < args.max_steps:
            if viewer_ctx and not viewer_ctx.is_running():
                outcome = "USER_CLOSED"; break

            loop_t0 = time.time()

            # Đọc state + LiDAR
            state_pos, state_vel, ranges, obstacles = get_robot_state_and_lidar(
                model, data, num_rays=args.n_lidar, max_range=20.0)

            # DWA plan
            v_cmd, w_cmd = dwa.plan(
                state_pos, state_vel, goal, obstacles, weights)

            # Gửi lệnh
            vc.set_cmd(float(v_cmd), float(w_cmd))

            # Step physics
            for _ in range(args.frame_skip):
                mujoco.mj_step(model, data)
            if viewer_ctx:
                viewer_ctx.sync()

            step_count += 1
            dist = float(np.linalg.norm(np.array([state_pos[0], state_pos[1]]) - goal))
            traj_log.append([state_pos[0], state_pos[1], state_pos[2],
                             v_cmd, w_cmd, dist])

            if len(obstacles) > 0 and step_count % 5 == 0:
                obstacle_cloud.extend(obstacles.tolist())

            if step_count % 10 == 0:
                print(f"[{step_count:4d}] "
                      f"pos=({state_pos[0]:+6.2f},{state_pos[1]:+6.2f}) "
                      f"θ={np.degrees(state_pos[2]):+6.1f}°  "
                      f"v={state_vel[0]:+.3f} ω={state_vel[1]:+.3f}  │  "
                      f"DWA v={v_cmd:+.2f} ω={w_cmd:+.2f}  │  "
                      f"dist={dist:5.2f}  obs={len(obstacles)}")

            if dist < args.goal_threshold:
                outcome = "SUCCESS"
                print(f"\n✅ SUCCESS tại step {step_count}"); break
            if is_collision(model, data):
                outcome = "COLLISION"
                print(f"\n💥 COLLISION tại step {step_count}"); break

            if args.render:
                elapsed = time.time() - loop_t0
                target  = args.frame_skip * model.opt.timestep
                if elapsed < target:
                    time.sleep(target - elapsed)

        elapsed = time.time() - t0
        print(f"\n{'─'*40}")
        print(f"Outcome    : {outcome}")
        print(f"Steps      : {step_count}/{args.max_steps}")
        print(f"Wall time  : {elapsed:.1f}s  ({step_count/elapsed:.1f} steps/s)")

        if args.save_log and traj_log:
            p = os.path.join(_SCRIPT_DIR, "dwa_demo_log.csv")
            np.savetxt(p, np.array(traj_log),
                       header="x,y,theta,v_cmd,w_cmd,dist_to_goal",
                       delimiter=",", comments="")
            print(f"📝 Log: {p}")

        if args.plot and traj_log:
            plot_trajectory_2d(traj_log, obstacle_cloud, args.start, goal,
                               args.goal_threshold, outcome,
                               os.path.join(_SCRIPT_DIR, "dwa_trajectory.png"))

        if viewer_ctx and viewer_ctx.is_running():
            print("\nĐóng cửa sổ viewer để thoát.")
            while viewer_ctx.is_running():
                viewer_ctx.sync(); time.sleep(0.05)

    finally:
        mujoco.set_mjcb_control(None)
        if viewer_ctx:
            viewer_ctx.close()


# =================================================================
# CLI
# =================================================================

def parse_args():
    p = argparse.ArgumentParser()
    default_xml = os.path.join(_PROJECT_ROOT, "assets", "worlds",
                               "test", "World_Medium.xml")
    p.add_argument("--xml",    default=default_xml)
    p.add_argument("--start",  nargs=3, type=float, default=[0.,0.,0.],
                   metavar=("X","Y","THETA"))
    p.add_argument("--goal",   nargs=2, type=float, default=[5.,5.],
                   metavar=("X","Y"))
    p.add_argument("--alpha",  type=float, default=0.7)
    p.add_argument("--beta",   type=float, default=0.2)
    p.add_argument("--gamma",  type=float, default=0.1)
    p.add_argument("--frame_skip",     type=int,   default=10)
    p.add_argument("--max_steps",      type=int,   default=600)
    p.add_argument("--n_lidar",        type=int,   default=449)
    p.add_argument("--goal_threshold", type=float, default=0.2)
    p.add_argument("--no-render", dest="render", action="store_false")
    p.set_defaults(render=True)
    p.add_argument("--save_log",  action="store_true")
    p.add_argument("--plot",      action="store_true")
    p.add_argument("--auto-start", dest="auto_start", action="store_true")
    p.set_defaults(auto_start=False)
    p.add_argument("--world_min", nargs=2, type=float, default=[-4., -4.])
    p.add_argument("--world_max", nargs=2, type=float, default=[10., 10.])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())