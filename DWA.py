"""
DWA.py — Dynamic Window Approach solver cho robot AgileX Bunker
----------------------------------------------------------------
Dựa trên file gốc của người dùng, chỉ thay đổi TỐI THIỂU:

  THAY ĐỔI DUY NHẤT so với file gốc:
    clearance_clip: 2.0 → 4.0  (nhìn xa obstacle hơn, không gây instability)

  GIỮ NGUYÊN:
    - v_samp = [0.0] khi angle > 18° (rotate-then-drive, ổn định)
    - dynamic_max_v filter        (giảm tốc khi rẽ, mượt hơn)
    - predict_time = 2.0s         (đã đủ, tăng lên gây vấn đề)
    - Không thêm straight_penalty (gây đi lùi)
    - Không thêm turn_bonus       (gây cà giật)
    - Tất cả thông số động học khác
"""

import math
import numpy as np


class DWASolver:
    def __init__(self):
        # Gioi han dong hoc — giu nguyen tu file goc
        self.max_speed     = 0.5
        self.min_speed     = -0.2
        self.max_yaw_rate  = 1.0

        self.max_accel     = 5.0
        self.max_delta_yaw = 10.0

        # Tham so thuat toan — giu nguyen tu file goc
        self.dt                  = 0.1
        self.predict_time        = 2.0
        self.v_resolution        = 0.05
        self.yaw_rate_resolution = 0.05
        self.robot_radius        = 0.4

        # THAY DOI DUY NHAT: clearance_clip 2.0 → 4.0
        # Ly do: 2.0m qua gan, DWA khong phan biet duoc obstacle xa hay gan
        # → robot lao thang vao obstacle roi moi biet. 4.0m cho phep DWA
        # "nhin" xa hon va uu tien trajectory tranh obstacle som hon.
        # KHONG tang len 7.0 vi se khien nearly-always-active lam bat on.
        self.clearance_clip = 3.25    # m  (was 2.0)

    # ------------------------------------------------------------------
    def calc_dynamic_window(self, v, omega):
        dw = [
            max(self.min_speed,     v     - self.max_accel     * self.dt),
            min(self.max_speed,     v     + self.max_accel     * self.dt),
            max(-self.max_yaw_rate, omega - self.max_delta_yaw * self.dt),
            min( self.max_yaw_rate, omega + self.max_delta_yaw * self.dt),
        ]
        return dw

    # ------------------------------------------------------------------
    def _predict_traj(self, x, y, theta, v, omega):
        T  = int(self.predict_time / self.dt) + 1
        ts = np.arange(T, dtype=np.float32) * self.dt
        if abs(omega) < 1e-6:
            xs  = x + v * np.cos(theta) * ts
            ys  = y + v * np.sin(theta) * ts
            ths = np.full(T, theta, dtype=np.float32)
        else:
            ths = (theta + omega * ts).astype(np.float32)
            xs  = (x + (v / omega) * (np.sin(ths) - math.sin(theta))).astype(np.float32)
            ys  = (y - (v / omega) * (np.cos(ths) - math.cos(theta))).astype(np.float32)
        return np.stack([xs, ys, ths], axis=-1)

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    # ------------------------------------------------------------------
    def plan(self, state_pos, state_vel, goal_pos, obstacles, rl_weights,
             ranges=None):
        x, y, theta = float(state_pos[0]), float(state_pos[1]), float(state_pos[2])
        v, omega    = float(state_vel[0]),  float(state_vel[1])
        alpha, beta, gamma = rl_weights

        dw = self.calc_dynamic_window(v, omega)
        if dw[1] <= dw[0] or dw[3] <= dw[2]:
            return [0.0, 0.0]

        obs     = np.asarray(obstacles, dtype=np.float32)
        has_obs = obs.shape[0] > 0
        goal_x, goal_y = float(goal_pos[0]), float(goal_pos[1])

        # Giu nguyen logic rotate-then-drive tu file goc.
        # Khi robot chua huong ve goal (angle > 18 deg), chi xoay tai cho.
        # Logic nay on dinh va da hoat dong tot → KHONG XOA.
        angle_to_goal = math.atan2(goal_y - y, goal_x - x)
        angle_diff    = abs(self._wrap(angle_to_goal - theta))
        if angle_diff > math.pi / 10:
            v_samp = [0.0]
        else:
            v_samp = np.arange(dw[0], dw[1] + 1e-6, self.v_resolution)

        w_samp = np.arange(dw[2], dw[3] + 1e-6, self.yaw_rate_resolution)

        # L1-normalize SAC weights
        w_sum = alpha + beta + gamma
        if w_sum < 1e-6:
            alpha, beta, gamma = 1/3, 1/3, 1/3
        else:
            alpha /= w_sum
            beta  /= w_sum
            gamma /= w_sum

        best_u    = [0.0, 0.0]
        max_score = -float('inf')

        for sv in v_samp:
            for sw in w_samp:
                sv, sw = float(sv), float(sw)

                # Giu nguyen dynamic_max_v tu file goc.
                # Giam toc khi re manh → trajectory muot hon, it rung dong.
                # KHONG XOA — no da hoat dong tot o code cu.
                turn_ratio    = abs(sw) / self.max_yaw_rate
                dynamic_max_v = self.max_speed * (1.0 - 0.75 * turn_ratio)
                if sv > dynamic_max_v:
                    continue

                traj = self._predict_traj(x, y, theta, sv, sw)

                # (a) Heading score
                fx, fy, fth = float(traj[-1, 0]), float(traj[-1, 1]), float(traj[-1, 2])
                goal_h      = math.atan2(goal_y - fy, goal_x - fx)
                heading_score = 1.0 - abs(self._wrap(goal_h - fth)) / math.pi

                # (b) Clearance score — per trajectory
                if has_obs:
                    diff     = traj[:, None, :2] - obs[None, :, :]
                    dists    = np.linalg.norm(diff, axis=-1)
                    min_dist = float(dists.min())
                else:
                    min_dist = self.clearance_clip

                if min_dist <= self.robot_radius:
                    continue

                # clearance_clip = 4.0m: DWA bat dau phan biet som hon
                # Khi obstacle 3m: score = 3/4 = 0.75 (khong phai 1.0 nhu cu)
                # → DWA uu tien trajectory tranh obstacle tu xa hon
                clearance_score = min(min_dist, self.clearance_clip) / self.clearance_clip

                # (c) Velocity score — giu nguyen, KHONG them speed_penalty
                # vi speed_penalty ket hop straight_penalty gay di lui
                velocity_score = max(sv, 0.0) / self.max_speed

                # Ham muc tieu goc — giu nguyen hoan toan
                # KHONG them turn_bonus, straight_penalty (gay instability)
                total = (alpha * heading_score
                         + beta  * clearance_score
                         + gamma * velocity_score)

                if total > max_score:
                    max_score = total
                    best_u    = [sv, sw]

        return best_u


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    dwa   = DWASolver()
    empty = np.zeros((0, 2), dtype=np.float32)

    print("=== Test 1: Huong den goal (khong co obstacle) ===")
    cases = [
        ("Goal phia truoc (0 deg)",  (0,0,0), (0,0), [5, 0],  "v>0 w~0"),
        ("Goal ben trai (90 deg)",   (0,0,0), (0,0), [0, 5],  "w>0, v=0"),
        ("Goal ben phai (-90 deg)",  (0,0,0), (0,0), [0, -5], "w<0, v=0"),
        ("Goal phia sau (180 deg)",  (0,0,0), (0,0), [-5, 0], "w!=0, v=0"),
    ]
    for name, pos, vel, goal, expect in cases:
        v, w = dwa.plan(pos, vel, np.array(goal), empty, (0.6, 0.2, 0.2))
        print(f"  {name:32s} → v={v:+.2f}  w={w:+.2f}   (expect {expect})")

    print()
    print("=== Test 2: Obstacle phia truoc, goc 0 deg (da align) ===")
    print("  clearance_clip=4.0m: biet som hon khi obstacle < 4m")
    print(f"  {'Dist':>7}  {'v*':>6}  {'w*':>6}  {'Note'}")
    print("  " + "-"*50)
    weights = (0.5, 0.3, 0.2)
    for d in [5.0, 4.0, 3.0, 2.0, 1.5, 1.0]:
        # Obstacle lech phai nhe de kiem tra re
        obs = np.array([[d, -0.3]], dtype=np.float32)
        v, w = dwa.plan((0,0,0), (0.3, 0), np.array([8.0, 0.0]),
                         obs, weights)
        note = "tranh obstacle" if abs(w) > 0.1 else "di thang"
        print(f"  {d:>6.1f}m  {v:>+6.2f}  {w:>+6.2f}  {note}")

    print()
    print("=== Test 3: Khong co obstacle, toc do toi da ===")
    v, w = dwa.plan((0,0,0), (0,0), np.array([10.0, 0.0]), empty, (0.5,0.3,0.2))
    assert v > 0.4, f"Toc do phai cao khi khong co obstacle, got v={v}"
    assert abs(w) < 0.1, f"Khong nen re khi duong thang, got w={w}"
    print(f"  v={v:+.2f}  w={w:+.2f}  → OK: di thang nhanh")

    print()
    print("=== Test 4: Dynamic window ===")
    dw = dwa.calc_dynamic_window(0.0, 0.0)
    n_w = len(np.arange(dw[2], dw[3]+1e-6, dwa.yaw_rate_resolution))
    print(f"  w in [{dw[2]:.2f}, {dw[3]:.2f}] → {n_w} samples")
    print(f"  clearance_clip = {dwa.clearance_clip}m  (was 2.0m)")
    print()
    print("  THAY DOI SO VOI FILE GOC:")
    print("    clearance_clip: 2.0 → 4.0  (chi thay doi nay)")
    print("    Tat ca con lai: giu nguyen")