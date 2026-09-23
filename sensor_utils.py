"""
sensor_utils.py — Doc trang thai robot + raycast LiDAR tu MuJoCo
------------------------------------------------------------------
Doc velocity DUNG theo cach BunkerEnv.get_robot_velocities() lam:
    linear_vel_body  = R.T @ qvel[0:3]  ->  v = body[0]  (forward)
    angular_vel_body = R.T @ qvel[3:6]  ->  omega = body[1]  (yaw)

KHONG dung data.qvel[5] raw vi freejoint qvel[3:6] can duoc chieu vao
body frame dung cach tuy theo orientation cua robot trong XML.

FIX (mobile robot support):
    mj_ray() truoc day goi voi geomgroup=None (khong loc group nao ca).
    bodyexclude=exclude_id chi loai tru 1 BODY duy nhat ("mobile_base"),
    khong loai tru cac body con nhu "base_link", "left_wheel", "right_wheel"
    -- voi robot Bunker (than thap, lidar gan sat san) dieu nay khong
    gay van de, nhung voi robot mobile (than cao ~1.4m, lidar nam trong
    bounding-box cua base_link_geom) toan bo cac tia se tu bat trung
    chinh than robot, do khoang cach sai ngay ca trong phong trong.

    Fix dung: mobile_body.xml da gan group="2" cho toan bo geom thuoc
    ve robot (chassis, laser, imu, caster, banh xe). Ham nay gio truyen
    geomgroup loai tru group 2 cho mj_ray -- day la co che LOC THEO
    GROUP, khong phu thuoc cay body, nen hoat dong dung du robot co bao
    nhieu body con.

    Voi world Bunker (khong co geom nao gan group=2), fix nay la NO-OP
    hoan toan -- khong can sua gi them cho cac world Bunker cu.
"""

import math
import numpy as np
import mujoco


def get_robot_state_and_lidar(model, data,
                              num_rays: int = 449,
                              max_range: float = 14.5,
                              sensor_height: float = 0.25,
                              exclude_body_name: str = "mobile_base",
                              exclude_geom_group: int = 2):
    """
    Doc trang thai robot va quet LiDAR 360 do bang mj_ray.

    Parameters
    ----------
    exclude_geom_group : int
        Group index (0..mujoco.mjNGROUP-1) can loai tru khoi raycast.
        Mac dinh = 2, khop voi quy uoc "group=2 = geom cua chinh robot"
        dung trong assets/mobile/mobile_body.xml. Voi robot Bunker (khong
        dung quy uoc group nay) tham so nay khong co tac dung gi (no-op).
        Truyen -1 de tat han co che loc theo group (hanh vi cu, khong
        khuyen nghi cho robot mobile).

    Returns
    -------
    state_pos  : (x, y, theta)           -- vi tri world frame
    state_vel  : (v, omega)              -- velocity body frame (giong BunkerEnv)
    ranges     : np.ndarray (num_rays,)  -- khoang cach moi tia (m)
    obstacles  : np.ndarray (K, 2)       -- toa do world cua diem trung
    """

    # -- 1. Vi tri (freejoint: qpos = [x,y,z, qw,qx,qy,qz]) --------------
    x = float(data.qpos[0])
    y = float(data.qpos[1])

    qw = float(data.qpos[3])
    qx = float(data.qpos[4])
    qy = float(data.qpos[5])
    qz = float(data.qpos[6])
    theta = math.atan2(2.0 * (qw * qz + qx * qy),
                       1.0 - 2.0 * (qy * qy + qz * qz))

    # -- 2. Velocity -- KHOP voi BunkerEnv.get_robot_velocities() --------
    # BunkerEnv: v = linear_body[0], omega = angular_body[1]
    try:
        body_id = model.body('mobile_base').id
        R = data.xmat[body_id].reshape(3, 3)          # world<-body rotation
        lin_world = data.qvel[0:3].copy()
        ang_world = data.qvel[3:6].copy()
        lin_body  = R.T @ lin_world                    # project to body frame
        ang_body  = R.T @ ang_world
        v     = float(lin_body[0])                     # forward velocity
        omega = float(ang_body[1])                     # yaw rate (BunkerEnv convention)
    except Exception:
        # Fallback neu khong co body 'mobile_base'
        v     = float(data.qvel[0])
        omega = float(data.qvel[5])

    # -- 3. LiDAR 360 do raycast -------------------------------------------
    pnt = np.array([x, y, sensor_height], dtype=np.float64)
    angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)

    try:
        exclude_id = int(model.body(exclude_body_name).id)
    except Exception:
        exclude_id = -1

    # Xay mask geomgroup: mj_ray can mang do dai mujoco.mjNGROUP (=6),
    # 1 = raycast VOI geom thuoc group nay, 0 = BO QUA geom thuoc group nay.
    # Mac dinh tat ca group deu duoc raycast (=1), tru group cua robot.
    if 0 <= exclude_geom_group < mujoco.mjNGROUP:
        geomgroup = np.ones(mujoco.mjNGROUP, dtype=np.uint8)
        geomgroup[exclude_geom_group] = 0
    else:
        geomgroup = None  # khong loc theo group (hanh vi cu)

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
        # else: giu max_range (khong them vao obstacles -- tia khong trung)

    obstacles = (np.asarray(obstacle_list, dtype=np.float32)
                 if obstacle_list
                 else np.zeros((0, 2), dtype=np.float32))

    return (x, y, theta), (v, omega), ranges, obstacles