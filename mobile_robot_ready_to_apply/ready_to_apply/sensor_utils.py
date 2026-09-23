"""
sensor_utils.py — Đọc trạng thái robot + raycast LiDAR từ MuJoCo
------------------------------------------------------------------
Đọc velocity ĐÚNG theo cách BunkerEnv.get_robot_velocities() làm:
    linear_vel_body  = R.T @ qvel[0:3]  →  v = body[0]  (forward)
    angular_vel_body = R.T @ qvel[3:6]  →  ω = body[1]  (yaw cho Bunker)

KHÔNG dùng data.qvel[5] raw vì freejoint qvel[3:6] cần được chiếu vào
body frame đúng cách tùy theo orientation của robot trong XML.
"""

import math
import numpy as np
import mujoco


def get_robot_state_and_lidar(model, data,
                              num_rays: int = 449,
                              max_range: float = 14.5,
                              sensor_height: float = 0.3488,
                              exclude_body_name: str = "mobile_base"):
    """
    Đọc trạng thái robot và quét LiDAR 360° bằng mj_ray.

    Returns
    -------
    state_pos  : (x, y, theta)           — vị trí world frame
    state_vel  : (v, omega)              — velocity body frame (giống BunkerEnv)
    ranges     : np.ndarray (num_rays,)  — khoảng cách mỗi tia (m)
    obstacles  : np.ndarray (K, 2)       — tọa độ world của điểm trúng
    """

    # ── 1. Vị trí (freejoint: qpos = [x,y,z, qw,qx,qy,qz]) ──────────
    x = float(data.qpos[0])
    y = float(data.qpos[1])

    qw = float(data.qpos[3])
    qx = float(data.qpos[4])
    qy = float(data.qpos[5])
    qz = float(data.qpos[6])
    theta = math.atan2(2.0 * (qw * qz + qx * qy),
                       1.0 - 2.0 * (qy * qy + qz * qz))

    # ── 2. Velocity — KHỚP với BunkerEnv.get_robot_velocities() ──────
    try:
        body_id = model.body('mobile_base').id
        R = data.xmat[body_id].reshape(3, 3)          # world←body rotation
        lin_world = data.qvel[0:3].copy()
        ang_world = data.qvel[3:6].copy()
        lin_body  = R.T @ lin_world                    # project to body frame
        ang_body  = R.T @ ang_world
        v     = float(lin_body[0])                     # forward velocity
        omega = float(ang_body[1])                     # yaw rate (BunkerEnv convention)
    except Exception:
        v     = float(data.qvel[0])
        omega = float(data.qvel[5])

    # ── 3. LiDAR 360° raycast ─────────────────────────────────────────
    pnt = np.array([x, y, sensor_height], dtype=np.float64)
    angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)

    try:
        exclude_id = int(model.body(exclude_body_name).id)
    except Exception:
        exclude_id = -1

    # FIX "LIDAR TỰ VA CHẠM VỚI CHÍNH ROBOT": bodyexclude của mj_ray chỉ
    # nhận DUY NHẤT 1 body id. Robot vi sai mới có hình học nằm trên
    # NHIỀU body (mobile_base, base_link, laser, imu, left/right_wheel),
    # nên chỉ loại trừ "mobile_base" là không đủ -> mọi tia tự va vào
    # chassis (đo được 449/449 tia hit ở ~0.126m ngay cả phòng trống).
    # Sửa: lọc theo geomgroup (không phụ thuộc cây body) -- mọi geom
    # thuộc về robot được gán group=2 trong mobile_body.xml, ở đây tắt
    # group 2 khi raycast. Giữ lại bodyexclude cho mobile_base để an
    # toàn nếu sau này có geom mới gắn trực tiếp lên body đó.
    geomgroup = np.ones(mujoco.mjNGROUP, dtype=np.uint8)
    geomgroup[2] = 0   # tắt group 2 = "hình học của chính robot"

    ranges        = np.full(num_rays, max_range, dtype=np.float32)
    obstacle_list = []
    geomid_out    = np.zeros(1, dtype=np.int32)

    for i, a in enumerate(angles):
        global_angle = theta + a
        vec = np.array([math.cos(global_angle),
                        math.sin(global_angle),
                        0.0], dtype=np.float64)

        dist = mujoco.mj_ray(model, data, pnt, vec,
                             geomgroup, 1, exclude_id, geomid_out)

        if 0.0 < dist <= max_range:
            ranges[i] = float(dist)
            obstacle_list.append([x + dist * math.cos(global_angle),
                                   y + dist * math.sin(global_angle)])
        # else: giữ max_range (không thêm vào obstacles — tia không trúng)

    obstacles = (np.asarray(obstacle_list, dtype=np.float32)
                 if obstacle_list
                 else np.zeros((0, 2), dtype=np.float32))

    return (x, y, theta), (v, omega), ranges, obstacles
