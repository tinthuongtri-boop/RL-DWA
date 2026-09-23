import numpy as np
import mujoco
from collections import deque

"""
Velocity vector:

data.qvel = [
    qvel[0] 'mobile_base_joint' (m/s) world frame x
    qvel[1] 'mobile_base_joint' (m/s) world frame y
    qvel[2] 'mobile_base_joint' (m/s) world frame z
    qvel[3] 'mobile_base_joint' (rad/s) world frame roll
    qvel[4] 'mobile_base_joint' (rad/s) world frame pitch
    qvel[5] 'mobile_base_joint' (rad/s) world frame yaw
    qvel[6] 'w_rr_joint' (rad/s) speed
    qvel[7] 'w_rc_joint' (rad/s) speed 
    qvel[8] 'w_rf_joint' (rad/s) speed
    qvel[9] 'w_lr_joint' (rad/s) speed
    qvel[10] 'w_lc_joint' (rad/s) speed
    qvel[11] 'w_lf_joint' (rad/s) speed
    ]
"""

class BunkerVelocityController:

    def __init__(self, v0: float = 0.0, w0: float = 0.0, a_max: float = 1.5,
                 alpha_max: float = 10.0, track_multiplier: float = 1.0):
        # Motion Control State
        self.v_cmd = self.v_cur = float(v0)
        self.w_cmd = self.w_cur = float(w0)
        self.a_max, self.alpha_max = abs(a_max), abs(alpha_max)
        # ROBOT-DEPENDENT: Bunker (skid-steer, 3 bánh/bên) cần track_width
        # "hiệu dụng" LỚN HƠN track_width hình học thật (code cũ hardcode
        # effective_w_track = w_track * 2.0, không có giải thích/đo đạc kèm
        # theo -- coi là suy đoán compensation cho ma sát skid-steer, CHƯA
        # được xác nhận). Robot mobile mới là diff-drive THẬT (2 bánh + 4
        # caster tự do), công thức kinematics chuẩn KHÔNG cần hệ số này.
        # Mặc định = 1.0 (đúng cho mobile). Muốn tái lập hành vi Bunker cũ,
        # truyền track_multiplier=2.0 khi khởi tạo.
        # CẦN KIỂM CHỨNG bằng thực nghiệm (lệnh v=0, w=const, đo yaw rate
        # thật qua qvel -- cùng kiểu test đã dùng ở Phase 0(b)) trước khi
        # tin số 1.0 này là đúng cho robot mobile.
        self.track_multiplier = float(track_multiplier)
        self._t_prev: float | None = None
        
        # Robot Physical Params
        self.w_track: float = None
        self.r_wheel: float = None
        self.act_r: list[int] = []
        self.act_l: list[int] = []

    def set_cmd(self, v: float, w: float) -> None:
        self.v_cmd, self.w_cmd = float(v), float(w) 

    def reset(self):
        self.v_cmd = self.v_cur = 0.0
        self.w_cmd = self.w_cur = 0.0
        self._t_prev = None

    def __call__(self, m: mujoco.MjModel, d: mujoco.MjData):
        dt = m.opt.timestep if self._t_prev is None else max(d.time-self._t_prev, 1e-9)
        self._t_prev = d.time
        self.v_cur = self._rate_limit(self.v_cur, self.v_cmd, self.a_max,     dt)
        self.w_cur = self._rate_limit(self.w_cur, self.w_cmd, self.alpha_max, dt)

        effective_w_track = self.w_track * self.track_multiplier
        w_r = (2*self.v_cur + self.w_cur*effective_w_track)/(2*self.r_wheel)
        w_l = (2*self.v_cur - self.w_cur*effective_w_track)/(2*self.r_wheel)
        for idx in self.act_r: d.ctrl[idx] = w_r
        for idx in self.act_l: d.ctrl[idx] = w_l

    @staticmethod
    def _rate_limit(cur, tgt, max_rate, dt):
        return cur + np.clip(tgt-cur, -max_rate*dt, max_rate*dt)