"""
comparison_common.py — Helpers dùng chung cho 2 script so sánh
---------------------------------------------------------------
Dùng seeds cố định để đảm bảo DWA thuần và DWA-SAC gặp cùng start/goal.
"""

import os
import sys
import json
import numpy as np
# Import thuật toán A* (Giả sử bạn lưu code A* ở file astar_planner.py)
from astar_planner import AStarPlanner

# Dynamic obstacle controller (chỉ dùng khi ENV_NAME chứa 'very_hard')
try:
    from dynamic_obstacles_controller import DynamicObstacleController
    _HAS_DYN_CTRL = True
except ImportError:
    _HAS_DYN_CTRL = False

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR   not in sys.path: sys.path.insert(0, _SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════
# CẤU HÌNH CHUNG — cả 2 script đều dùng
# ══════════════════════════════════════════════════════════════════════
ENV_NAME      = "test/world_66_medium"
MAX_GOAL_DIST = 10
N_EPISODES    = 30
MAX_EP_STEPS  = 600
SEED_BASE     = 1000          # seeds = [1000, 1001, ..., 1000+N-1]

# Baseline DWA weights (cố định cho method 1)
BASELINE_WEIGHTS = (0.7, 0.2, 0.1)

# SAC model để test (cho method 2)  
RUN_NAME = "run_13"

# Folder lưu kết quả
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "comparison_results")
# ══════════════════════════════════════════════════════════════════════


def get_xml_path() -> str:
    return os.path.join(_PROJECT_ROOT, "assets", "worlds", f"{ENV_NAME}.xml")


def get_seeds() -> list:
    return [SEED_BASE + i for i in range(N_EPISODES)]


def get_results_path(method_name: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = ENV_NAME.replace("/", "_")
    return os.path.join(RESULTS_DIR, f"results_{method_name}_{tag}.json")


def run_single_episode(env, policy_fn, seed: int) -> dict:
    """
    Chạy 1 episode với policy_fn(obs) -> action (3 values).
    Trả về dict metrics + trajectory.

    policy_fn : callable
        nhận obs dict, trả về np.ndarray shape (3,) = [alpha, beta, gamma]
    """
    obs, info = env.reset(seed=seed)
    raw_env = env.unwrapped
    # ---------------------------------------------------------
    # # FIX CỨNG TỌA ĐỘ Ở ĐÂY (Format: [x, y, góc_yaw_radian])
    # # ---------------------------------------------------------
    # my_start = [2.0, 0.0, 0.1]  # Ví dụ: Bắt đầu ở x=0, y=-2, hướng mặt về góc 0
    # my_goal  = [15.0, 0.0, 0.1]   # Ví dụ: Đích đến ở x=7.5, y=4.0

    # try:
    #     # Ghi đè tọa độ random bằng tọa độ fix cứng của bạn
    #     obs = raw_env.set_manual_pose(initial_pose=my_start, goal_pose=my_goal)
    #     info = raw_env._get_reset_info() # Cập nhật lại info
    # except ValueError as e:
    #     print(f"\n[Lỗi tọa độ] {e}")
    #     print("Vui lòng chọn tọa độ không bị dính vào vật cản!")
    #     return None # Thoát nếu bạn lỡ nhập tọa độ đè lên tường
    # # ---------------------------------------------------------

    start     = obs["achieved_goal"][:2].copy()
    goal      = obs["desired_goal"][:2].copy()
    init_dist = float(np.linalg.norm(start - goal))

    traj           = [obs["achieved_goal"].copy().tolist()]
    obstacle_cloud = []
    weights_log    = []
    clearance_log  = []

    path_length    = 0.0
    ep_ang_smooth  = 0.0
    ep_lin_smooth  = 0.0
    last_v = 0.0
    last_w = 0.0

    dt      = raw_env.model.opt.timestep * raw_env.frame_skip

    # Khởi tạo dynamic obstacle controller nếu map có vật cản động
    dyn_ctrl = None
    if _HAS_DYN_CTRL and "very_hard" in ENV_NAME:
        dyn_ctrl = DynamicObstacleController(raw_env.model)
        dyn_ctrl.reset(raw_env.data)

    outcome = "TIMEOUT"
    steps   = 0

    for step in range(MAX_EP_STEPS):
        # Cập nhật vị trí vật cản động theo thời gian (nếu có)
        if dyn_ctrl is not None:
            dyn_ctrl.update(t=step * dt, data=raw_env.data)

        action = policy_fn(obs)
        weights_log.append(action.tolist())

        obs, reward, terminated, truncated, info = env.step(action)
        steps = step + 1
        traj.append(obs["achieved_goal"].copy().tolist())

        # LiDAR hits cho plot (subsample)
        if step % 5 == 0:
            lidar  = obs["observation"][:raw_env.n_lidar*3].reshape(-1, 3)
            d_norm = lidar[:, 2]
            d_real = (d_norm + 1.0) / 2.0 * raw_env.lidar_max_range
            hits   = d_real < raw_env.lidar_max_range * 0.95
            if hits.any():
                x0, y0, th0 = obs["achieved_goal"]
                angles = raw_env.lidar_angles[hits]
                dists  = d_real[hits]
                hx = x0 + dists * np.cos(th0 + angles)
                hy = y0 + dists * np.sin(th0 + angles)
                obstacle_cloud.extend(list(zip(hx.tolist(), hy.tolist())))
            clearance_log.append(float(d_real.min()))

        # Smoothness + path length
        lin_body, ang_body = raw_env.get_robot_velocities()
        v = float(lin_body[0]); w = float(ang_body[1])
        path_length   += abs(v) * dt
        ep_ang_smooth += ((w - last_w) / dt) ** 2
        ep_lin_smooth += ((v - last_v) / dt) ** 2
        last_v, last_w = v, w

        if info.get("is_success", False):
            outcome = "SUCCESS"; break
        if info.get("collision", False):
            outcome = "COLLISION"; break
        if truncated:
            outcome = "TIMEOUT"; break

    spl = init_dist / max(init_dist, path_length) if outcome == "SUCCESS" else 0.0

    return {
        "seed":        seed,
        "outcome":     outcome,
        "steps":       steps,
        "start":       start.tolist(),
        "goal":        goal.tolist(),
        "init_dist":   init_dist,
        "path_length": path_length,
        "spl":         spl,
        "traj":        traj,
        "obstacles":   obstacle_cloud,
        "weights":     weights_log,
        "clearance":   float(np.mean(clearance_log)) if clearance_log else 0.0,
        "ang_smooth":  ep_ang_smooth / max(steps, 1),
        "lin_smooth":  ep_lin_smooth / max(steps, 1),
    }

def run_single_episode_with_astar(env, policy_fn, seed: int) -> dict:
    """
    Chạy 1 episode kết hợp Global Planner (A*) và Local Planner (SAC-DWA).
    Ghi nhận metrics giống hệt hàm gốc để so sánh công bằng.
    """
    

    obs, info = env.reset(seed=seed)
    raw_env = env.unwrapped
    dt = raw_env.model.opt.timestep * raw_env.frame_skip

    
    start = obs["achieved_goal"][:2].copy()
    final_goal = obs["desired_goal"][:2].copy()
    z_val = obs["desired_goal"][2] # Lấy chiều cao Z gốc của môi trường
    init_dist = float(np.linalg.norm(start - final_goal))
    
    # ---------------------------------------------------------
    # 1. PHA TÌM ĐƯỜNG TOÀN CỤC (GLOBAL PLANNING)
    # ---------------------------------------------------------
    grid_map = raw_env.build_occupancy_grid(resolution=0.1, inflation_radius= 0.7)
    planner = AStarPlanner(grid_map)

    start_grid = raw_env.world_to_grid(start[0], start[1], resolution=0.1)
    goal_grid = raw_env.world_to_grid(final_goal[0], final_goal[1], resolution=0.1)

    grid_path = planner.plan(start_grid, goal_grid)

# Rút gọn Waypoints (Cứ 20 ô lưới = 2.0 mét thì cắm 1 mốc)
    world_waypoints = []
    if grid_path:
        for i in range(0, len(grid_path), 12):
            wp_2d = raw_env.grid_to_world(grid_path[i][0], grid_path[i][1], resolution=0.1)
            # Ép thêm trục Z vào để thành mảng 3D
            world_waypoints.append(np.array([wp_2d[0], wp_2d[1], z_val])) 
        
        # Đảm bảo điểm cuối cùng luôn là đích thực sự
        last_wp_2d = raw_env.grid_to_world(grid_path[-1][0], grid_path[-1][1], resolution=0.1)
        last_wp = np.array([last_wp_2d[0], last_wp_2d[1], z_val])
        
        if len(world_waypoints) == 0 or np.linalg.norm(world_waypoints[-1][:2] - last_wp[:2]) > 0.1:
            world_waypoints.append(last_wp)
    else:
        print(f"  [A* Warning] Seed {seed}: Không tìm thấy đường, ép đi thẳng!")
        world_waypoints = [obs["desired_goal"].copy()]

    # Cập nhật mục tiêu đầu tiên cho Local Planner
    current_wp_idx = 1 if len(world_waypoints) > 1 else 0
    raw_env.set_inference_goal(world_waypoints[current_wp_idx])

    # ---------------------------------------------------------
    # 2. KHỞI TẠO METRICS (Giống hệt hàm cũ)
    # ---------------------------------------------------------
    traj           = [obs["achieved_goal"].copy().tolist()]
    obstacle_cloud = []
    weights_log    = []
    clearance_log  = []

    path_length    = 0.0
    ep_ang_smooth  = 0.0
    ep_lin_smooth  = 0.0
    last_v, last_w = 0.0, 0.0

    # Khởi tạo dynamic obstacle controller (nếu có)
    dyn_ctrl = None
    if _HAS_DYN_CTRL and "very_hard" in ENV_NAME:
        dyn_ctrl = DynamicObstacleController(raw_env.model)
        dyn_ctrl.reset(raw_env.data)

    outcome = "TIMEOUT"
    steps   = 0

    # ---------------------------------------------------------
    # 3. PHA ĐIỀU KHIỂN CỤC BỘ (LOCAL PLANNING LOP)
    # ---------------------------------------------------------
    for step in range(MAX_EP_STEPS):
        if dyn_ctrl is not None:
            dyn_ctrl.update(t=step * dt, data=raw_env.data)

        # SAC tính toán hành động dựa trên local_goal hiện tại
        action = policy_fn(obs)
        weights_log.append(action.tolist())

        obs, reward, terminated, truncated, info = env.step(action)
        steps = step + 1
        robot_pos = obs["achieved_goal"][:2]
        traj.append(robot_pos.tolist())

        # ------------------------------------------------------------------
        # NEW: GLOBAL REPLANNING (VẼ LẠI ĐƯỜNG KHI CÓ VẬT CẢN ĐỘT NGỘT)
        # ------------------------------------------------------------------
        # info.get("stuck_counter") là biến bạn đã định nghĩa trong môi trường
        # Nếu xe đứng yên quá 30 steps (chứng tỏ bị vật cản mới chặn đứng, SAC không lách được)
        if info.get("stuck_counter", 0) > 30:
            print(f"  [Replanning] Kẹt vật cản đột ngột tại step {steps}! Đang vẽ lại đường...")
            
            # 1. Cập nhật lại Bản đồ (Quét các vật cản mới vừa xuất hiện)
            new_grid_map = raw_env.build_occupancy_grid(resolution=0.1, inflation_radius= 0.7)
            planner = AStarPlanner(new_grid_map)
            
            # 2. Định vị lại tọa độ Lưới
            start_grid = raw_env.world_to_grid(robot_pos[0], robot_pos[1], resolution=0.1)
            # goal_grid giữ nguyên là đích cuối
            
            # 3. Chạy A* tìm đường mới
            new_grid_path = planner.plan(start_grid, goal_grid)
            
            if new_grid_path:
                # Cắm lại các mốc waypoint mới
                world_waypoints = []
                for i in range(0, len(new_grid_path), 12):
                    wp_2d = raw_env.grid_to_world(new_grid_path[i][0], new_grid_path[i][1], resolution=0.1)
                    world_waypoints.append(np.array([wp_2d[0], wp_2d[1], z_val]))
                
                last_wp_2d = raw_env.grid_to_world(new_grid_path[-1][0], new_grid_path[-1][1], resolution=0.1)
                last_wp = np.array([last_wp_2d[0], last_wp_2d[1], z_val])
                if len(world_waypoints) == 0 or np.linalg.norm(world_waypoints[-1][:2] - last_wp[:2]) > 0.1:
                    world_waypoints.append(last_wp)
                
                # Reset lại trạm mục tiêu
                current_wp_idx = 1 if len(world_waypoints) > 1 else 0
                raw_env.set_inference_goal(world_waypoints[current_wp_idx])
                obs["desired_goal"] = world_waypoints[current_wp_idx].copy()
                
                # Reset stuck_counter trong môi trường để cho xe chạy tiếp
                raw_env._stuck_counter = 0 
            else:
                print("  [Replanning Failed] Đường đã bị bịt kín hoàn toàn!")
        # ------------------------------------------------------------------

        # Subsample LiDAR cho biểu đồ (giống hàm cũ)
        if step % 5 == 0:
            lidar = obs["observation"][:raw_env.n_lidar*3].reshape(-1, 3)
            d_real = (lidar[:, 2] + 1.0) / 2.0 * raw_env.lidar_max_range
            hits = d_real < raw_env.lidar_max_range * 0.95
            if hits.any():
                x0, y0, th0 = obs["achieved_goal"]
                angles = raw_env.lidar_angles[hits]
                dists = d_real[hits]
                hx = x0 + dists * np.cos(th0 + angles)
                hy = y0 + dists * np.sin(th0 + angles)
                obstacle_cloud.extend(list(zip(hx.tolist(), hy.tolist())))
            clearance_log.append(float(d_real.min()))

        # Tính toán Smoothness & Path Length
        lin_body, ang_body = raw_env.get_robot_velocities()
        v = float(lin_body[0]); w = float(ang_body[1])
        path_length   += abs(v) * dt
        ep_ang_smooth += ((w - last_w) / dt) ** 2
        ep_lin_smooth += ((v - last_v) / dt) ** 2
        last_v, last_w = v, w

        # --- LOGIC CHUYỂN TRẠM (WAYPOINT SWITCHING) ---
        # Chỉ tính khoảng cách 2D [x, y]
        dist_to_wp = np.linalg.norm(robot_pos - world_waypoints[current_wp_idx][:2]) 
        if dist_to_wp < 0.6 and current_wp_idx < len(world_waypoints) - 1:
            current_wp_idx += 1
            raw_env.set_inference_goal(world_waypoints[current_wp_idx])
            # Cập nhật toàn bộ mảng 3D [x, y, z] cho SAC
            obs["desired_goal"] = world_waypoints[current_wp_idx].copy()

        # --- KIỂM TRA KẾT THÚC ---
        # Kiểm tra tới đích CHÍNH (final_goal)
        if np.linalg.norm(robot_pos - final_goal) < 0.4 or info.get("is_success", False):
            outcome = "SUCCESS"
            break
        if info.get("collision", False):
            outcome = "COLLISION"
            break
        if truncated:
            outcome = "TIMEOUT"
            break

    spl = init_dist / max(init_dist, path_length) if outcome == "SUCCESS" else 0.0

    return {
        "seed":        seed,
        "outcome":     outcome,
        "steps":       steps,
        "start":       start.tolist(),
        "goal":        final_goal.tolist(),
        "init_dist":   init_dist,
        "path_length": path_length,
        "spl":         spl,
        "traj":        traj,
        "obstacles":   obstacle_cloud,
        "weights":     weights_log,
        "clearance":   float(np.mean(clearance_log)) if clearance_log else 0.0,
        "ang_smooth":  ep_ang_smooth / max(steps, 1),
        "lin_smooth":  ep_lin_smooth / max(steps, 1),
    }
def save_results(results: list, method_name: str, extra_meta: dict = None):
    """Lưu danh sách results ra JSON."""
    path = get_results_path(method_name)
    meta = {
        "method":          method_name,
        "env_name":        ENV_NAME,
        "n_episodes":      len(results),
        "max_goal_dist":   MAX_GOAL_DIST,
        "max_ep_steps":    MAX_EP_STEPS,
        "seed_base":       SEED_BASE,
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(path, "w") as f:
        json.dump({"meta": meta, "episodes": results}, f, indent=2)

    print(f"💾 Results saved: {path}")
    return path


def load_results(method_name: str) -> dict:
    with open(get_results_path(method_name)) as f:
        return json.load(f)


def print_summary(method_name: str, results: list):
    n   = len(results)
    sr  = sum(r['outcome']=='SUCCESS'   for r in results)
    cr  = sum(r['outcome']=='COLLISION' for r in results)
    tr  = sum(r['outcome']=='TIMEOUT'   for r in results)
    spl = np.mean([r['spl']        for r in results])
    clr = np.mean([r['clearance']  for r in results])
    ang = np.mean([r['ang_smooth'] for r in results])
    lin = np.mean([r['lin_smooth'] for r in results])

    print(f"\n{'─'*60}")
    print(f"  {method_name}   —   {n} episodes")
    print(f"{'─'*60}")
    print(f"  Success     : {sr/n*100:5.1f}%  ({sr}/{n})")
    print(f"  Collision   : {cr/n*100:5.1f}%  ({cr}/{n})")
    print(f"  Timeout     : {tr/n*100:5.1f}%  ({tr}/{n})")
    print(f"  SPL         : {spl:.3f}")
    print(f"  Clearance   : {clr:.2f} m")
    print(f"  Ang.Smooth  : {ang:.4f}")
    print(f"  Lin.Smooth  : {lin:.4f}")
    print(f"{'─'*60}\n")