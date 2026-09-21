"""
dynamic_obstacles_controller.py
--------------------------------
Điều khiển sine motion cho 3 dynamic obstacles trong custom_very_hard.xml.
Tự động tìm actuator tên "dyn_act_*" trong model.

Dùng trong comparison_common.py hoặc run_dwa_sac.py:

    from dynamic_obstacles_controller import DynamicObstacleController

    env = TimeLimit(BunkerEnv(xml_path="custom_very_hard.xml", ...), ...)
    dyn_ctrl = DynamicObstacleController(env.unwrapped.model)

    # Trong run_single_episode, thêm vào đầu mỗi step:
    for step in range(MAX_EP_STEPS):
        dyn_ctrl.update(t=step * dt, data=env.unwrapped.data)
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np


class DynamicObstacleController:
    """
    Sine motion controller cho position actuators của dynamic obstacles.
    Mỗi obstacle có chu kỳ và phase khác nhau để tạo chuyển động không đồng bộ.
    """

    # (chu kỳ giây, phase radian) cho từng actuator theo thứ tự tìm thấy
    MOTION_PARAMS = [
        (6.0,  0.0),    # dyn_act_0: chu kỳ 6s, không lệch phase
        (9.0,  2.1),    # dyn_act_1: chu kỳ 9s, lệch 2.1 rad (~120°)
        (7.0,  4.2),    # dyn_act_2: chu kỳ 7s, lệch 4.2 rad (~240°)
    ]

    def __init__(self, model):
        self.act_ids  = []
        self.act_amps = []

        for i in range(model.nu):
            try:
                name = model.actuator(i).name
            except Exception:
                continue
            if name and name.startswith("dyn_act_"):
                self.act_ids.append(i)
                amp = float(model.actuator_ctrlrange[i][1])
                self.act_amps.append(amp)

        if self.act_ids:
            print(f"[DynCtrl] {len(self.act_ids)} dynamic obstacles found.")
        else:
            print("[DynCtrl] No dynamic obstacles — controller inactive.")

    def update(self, t: float, data) -> None:
        """Cập nhật ctrl cho tất cả dynamic obstacles theo thời gian t (giây)."""
        for k, act_id in enumerate(self.act_ids):
            T, phi = self.MOTION_PARAMS[k % len(self.MOTION_PARAMS)]
            amp    = self.act_amps[k]
            data.ctrl[act_id] = amp * np.sin(2 * np.pi * t / T + phi)

    def reset(self, data) -> None:
        """Đặt tất cả obstacles về vị trí giữa (ctrl = 0)."""
        for act_id in self.act_ids:
            data.ctrl[act_id] = 0.0