# gym_bunker_env.py
from __future__ import annotations

import os
import numpy as np
import mujoco
import sys
from numpy.typing import NDArray
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3.common.env_checker import check_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.bunker_velocity_controller import BunkerVelocityController

# FIX: sensor_utils mới nhận (model, data) và trả về 4 giá trị
from DWA import DWASolver
from sensor_utils import get_robot_state_and_lidar


class BunkerEnv(MujocoEnv):
    """
    ## Parameters
    - n_lidar (int): Number of LiDAR rays (default: 449)

    ## Action Space  [DWA-RL hybrid]
    | Num | Action | Min | Max |
    |-----|--------|-----|-----|
    | 0   | alpha  | 0.0 | 1.0 |   weight: heading toward goal
    | 1   | beta   | 0.0 | 1.0 |   weight: obstacle clearance
    | 2   | gamma  | 0.0 | 1.0 |   weight: forward velocity

    ## Observation Space
    - `observation`   : Box(-1, 1, (n_lidar*3 + 2,))  → LiDAR [sin,cos,d_norm] + [v,ω]
    - `achieved_goal` : Box(-inf, inf, (3,))           → [x, y, yaw]
    - `desired_goal`  : Box(-inf, inf, (3,))           → [x, y, yaw]

    ## Reward
    r = 0 (success) | -100 (collision) | -1 (step penalty)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, xml_path: str, frame_skip: int = 10,
                 render_mode: str | None = None, n_lidar: int = 449,
                 inference_mode: bool = False,
                 max_goal_sampling_distance: float = 15):

        print(f"[BunkerEnv] max_goal_sampling_distance = {max_goal_sampling_distance}")

        # LiDAR
        self.n_lidar          = n_lidar
        self.lidar_angles     = np.linspace(-np.pi, np.pi, n_lidar,
                                            endpoint=False).astype(np.float32)
        self.lidar_max_range  = 20.0

        # World bounds
        self.xy_min = np.array([-4., -4.], np.float32)
        self.xy_max = np.array([11., 11.], np.float32)   # World 15x15m
        self.yaw_min, self.yaw_max = -np.pi, np.pi

        # Goal thresholds
        self.goal_xy_distance_threshold = 0.2
        self.min_goal_sampling_distance = 1.0
        self.max_goal_sampling_distance = max_goal_sampling_distance
        self.max_distance_diagonal = float(np.linalg.norm(self.xy_max - self.xy_min))

        # Velocity limits (for observation normalization)
        self.v_max, self.w_max = 0.5, 0.5
        self.vel_scale = np.array([self.v_max, self.w_max], np.float32)

        obs_dim = self.n_lidar * 3 + 2
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-1.0, 1.0, (obs_dim,),    dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, (3,),    dtype=np.float32),
            "desired_goal":  spaces.Box(-np.inf, np.inf, (3,),    dtype=np.float32),
        })
        self.max_geom = 1_800

        # ── Cache: LiDAR chỉ raycast 1 lần mỗi step ─────────────────
        # step() gọi sensor_utils → lưu vào cache → _get_obs() dùng lại
        self._cached_ranges:    np.ndarray | None = None
        self._cached_obstacles: np.ndarray | None = None

        # ── Cấu hình thưởng/phạt nâng cao ────────────────────────────
        # SUCCESS_REWARD  : thưởng lớn khi về đích (khuyến khích reach goal)
        # COLLISION_PEN   : phạt khi đụng tường (giữ nguyên ý gốc)
        # STUCK_PEN       : phạt khi đứng yên quá lâu
        # STEP_PEN        : phạt nhỏ mỗi step (khuyến khích đi nhanh)
        # NARROW_BONUS    : thưởng 1 lần khi đi qua chỗ hẹp an toàn
        # STUCK_PATIENCE  : số step cho phép đứng yên trước khi tính phạt
        # STUCK_THRESH    : ngưỡng dịch chuyển (m) để xem là 'không đứng yên'
        # NARROW_ENTER_R  : LiDAR min-range để xem là vào chỗ hẹp (m)
        # NARROW_EXIT_R   : LiDAR min-range để xem là thoát chỗ hẹp (m)
        # NARROW_MIN_DUR  : số step tối thiểu phải duy trì trong chỗ hẹp
        #                   để tránh false trigger (vd: lướt qua góc tường)
        self.SUCCESS_REWARD =  100.0
        self.COLLISION_PEN  = -80.0
        self.STUCK_PEN      = -65.0
        self.STEP_PEN       = -0.2
        self.STUCK_PATIENCE = 150    # steps

        self.STUCK_THRESH   = 0.005   # m

        # ===== NEW REWARD PARAM =====
        self.NEAR_COLLISION_DIST = 0.5
        self.SUCCESS_DIST = 0.15

        self.PROGRESS_GAIN = 17.5
        self.OBSTACLE_PEN_GAIN = 5.0
        self.STUCK_PEN_GAIN = 0.2   # phạt mỗi step khi stuck

        # reset mỗi episode
        self._prev_goal_dist = None
        
        # Trạng thái tracking đứng yên — reset mỗi episode
        self._stuck_counter   = 0
        self._last_xy: NDArray[np.float32] | None = None

        # Trạng thái tracking đi qua chỗ hẹp — reset mỗi episode
        # _in_narrow      : True nếu robot đang ở trong chỗ hẹp
        # _narrow_dur     : số step liên tiếp đã ở trong chỗ hẹp
        # _narrow_cleared : True nếu đã thưởng narrow bonus 1 lần (tránh farm)
        self._in_narrow      = False
        self._narrow_dur     = 0
        self._narrow_cleared = False

        super().__init__(model_path=xml_path, frame_skip=frame_skip,
                         observation_space=self.observation_space,
                         render_mode=render_mode, max_geom=self.max_geom)

        self._initialize_velocity_controller()

        self.mid_goal_bid   = self.model.body('mid_goal_marker').id
        self.mid_goal_mid   = int(self.model.body_mocapid[self.mid_goal_bid])
        self.final_goal_bid = self.model.body('final_goal_marker').id
        self.final_goal_mid = int(self.model.body_mocapid[self.final_goal_bid])

        self._goal:         NDArray[np.float32]
        self._is_final_goal: bool = True
        self.inference_mode  = inference_mode

        self.dwa = DWASolver()

    # ------------------------------------------------------------------
    # Velocity controller setup
    # ------------------------------------------------------------------
    def _initialize_velocity_controller(self):
        self.velocity_controller = BunkerVelocityController()

        id_rr = self.model.body("w_rr").id
        id_lr = self.model.body("w_lr").id

        # FIX BUG 1: track width phải là trục có khoảng cách lớn nhất
        # (trục Y theo body frame của Bunker) — KHÔNG phải index cứng [2]
        diff        = self.model.body_pos[id_rr] - self.model.body_pos[id_lr]
        width_track = float(np.max(np.abs(diff)))
        self.velocity_controller.w_track = width_track

        # FIX BUG 2: geom_size trả về array [radius, half_len, 0]
        # → chỉ lấy radius (scalar)
        id_geom = self.model.geom("w_rr_geom").id
        self.velocity_controller.r_wheel = float(self.model.geom_size[id_geom][0])

        self.velocity_controller.act_r = [self.model.actuator(n).id
                                           for n in ("w_rr", "w_rc", "w_rf")]
        self.velocity_controller.act_l = [self.model.actuator(n).id
                                           for n in ("w_lr", "w_lc", "w_lf")]

        self.sensor_frc_r = [self.model.sensor(n).id
                              for n in ("sf_rr", "sf_rc", "sf_rf")]
        self.sensor_frc_l = [self.model.sensor(n).id
                              for n in ("sf_lr", "sf_lc", "sf_lf")]

        mujoco.set_mjcb_control(self.velocity_controller)

    # ------------------------------------------------------------------
    # Action space override
    # ------------------------------------------------------------------
    def _set_action_space(self):
        self.action_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
        return self.action_space

    # ------------------------------------------------------------------
    # Velocity helper (đồng bộ với sensor_utils)
    # ------------------------------------------------------------------
    def get_robot_velocities(self):
        """Trả về (linear_vel_body, angular_vel_body) như cũ, cho tương thích."""
        linear_vel_world   = self.data.qvel[0:3].copy()
        angular_vel_world  = self.data.qvel[3:6].copy()
        body_id            = self.model.body('mobile_base').id
        R                  = self.data.xmat[body_id].reshape(3, 3)
        linear_vel_body    = R.T @ linear_vel_world
        angular_vel_body   = R.T @ angular_vel_world
        return linear_vel_body, angular_vel_body

    # ------------------------------------------------------------------
    # Pose helpers (dùng chung cho reset_model và set_manual_pose)
    # ------------------------------------------------------------------
    def _set_qpos_pose(self, x: float, y: float, theta: float,
                       height: float = 0.25) -> None:
        """
        Đặt freejoint qpos = [x, y, z, qw, qx, qy, qz].
        FIX BUG 3: height → qpos[2] (trục Z), KHÔNG phải qpos[1] (trục Y).
        """
        self.data.qpos[:] = 0.0
        self.data.qpos[0] = x
        self.data.qpos[1] = y
        self.data.qpos[2] = height          # ← FIX: index 2, không phải 1
        c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
        self.data.qpos[3] = c               # qw
        self.data.qpos[4] = 0.0             # qx
        self.data.qpos[5] = 0.0             # qy
        self.data.qpos[6] = s               # qz

    # ------------------------------------------------------------------
    # reset_model
    # ------------------------------------------------------------------
    def reset_model(self) -> dict[str, np.ndarray]:
        self.velocity_controller.reset()
        self._cached_ranges    = None
        self._cached_obstacles = None

        # Reset stuck-tracking khi bắt đầu episode mới
        self._stuck_counter = 0
        self._last_xy       = None

        # Reset narrow-passage tracking khi bắt đầu episode mới
        self._in_narrow         = False
        self._narrow_dur        = 0
        self._narrow_cleared    = False
        self._narrow_bonus_given = False

        self._prev_goal_dist = None

        rng = self.np_random

        # Sample collision-free robot start pose
        for attempt in range(10_000):
            x   = float(rng.uniform(self.xy_min[0], self.xy_max[0]))
            y   = float(rng.uniform(self.xy_min[1], self.xy_max[1]))
            th  = float(rng.uniform(self.yaw_min,   self.yaw_max))
            self._set_qpos_pose(x, y, th)
            mujoco.mj_forward(self.model, self.data)
            if not self._is_collision():
                ix, iy, ith = x, y, th
                break
        else:
            raise RuntimeError("Cannot find collision-free start pose after 10,000 attempts.")

        # Sample collision-free goal
        for attempt in range(10_000):
            gx  = float(rng.uniform(self.xy_min[0], self.xy_max[0]))
            gy  = float(rng.uniform(self.xy_min[1], self.xy_max[1]))
            gth = float(rng.uniform(self.yaw_min,   self.yaw_max))

            dist = np.linalg.norm([gx - ix, gy - iy])
            if not (self.min_goal_sampling_distance < dist <= self.max_goal_sampling_distance):
                continue

            # Check goal collision (move robot there temporarily)
            self._set_qpos_pose(gx, gy, gth)
            mujoco.mj_forward(self.model, self.data)
            goal_ok = not self._is_collision()

            # Restore robot to start pose
            self._set_qpos_pose(ix, iy, ith)
            mujoco.mj_forward(self.model, self.data)

            if goal_ok:
                self._goal = np.array([gx, gy, gth], dtype=np.float32)
                break
        else:
            raise RuntimeError("Cannot find collision-free goal after 10,000 attempts.")

        self.data.qvel[:] = 0.0
        self._is_final_goal = True
        self._set_goal_marker_position(self._goal, is_final_goal=True)
        return self._get_obs()

    # ------------------------------------------------------------------
    # set_manual_pose  (for inference / ghost scripts)
    # ------------------------------------------------------------------
    def set_manual_pose(self,
                        initial_pose: NDArray[np.float32] | list[float],
                        goal_pose:    NDArray[np.float32] | list[float]
                        ) -> dict[str, np.ndarray]:
        """
        Manually set robot + goal pose (used by sac_inference / ghost scripts).
        FIX BUG 4: angle is index [2] of pose, NOT [1].
        """
        self.velocity_controller.reset()
        self._cached_ranges    = None
        self._cached_obstacles = None

        initial_pose = np.asarray(initial_pose, dtype=np.float32)
        goal_pose    = np.asarray(goal_pose,    dtype=np.float32)

        # Set and check initial pose
        self._set_qpos_pose(float(initial_pose[0]), float(initial_pose[1]),
                            float(initial_pose[2]))   # FIX: theta = [2]
        mujoco.mj_forward(self.model, self.data)
        if self._is_collision():
            raise ValueError(f"Initial pose {initial_pose} results in a collision.")

        dist = float(np.linalg.norm(goal_pose[:2] - initial_pose[:2]))
        if not (self.min_goal_sampling_distance < dist <= self.max_goal_sampling_distance):
             raise ValueError(
                 f"Distance {dist:.2f}m outside valid range "
                 f"[{self.min_goal_sampling_distance}, {self.max_goal_sampling_distance}]")

        # Check goal pose collision
        self._goal = goal_pose.copy()
        self._set_qpos_pose(float(goal_pose[0]), float(goal_pose[1]),
                            float(goal_pose[2]))   # FIX: theta = [2]
        mujoco.mj_forward(self.model, self.data)
        has_collision = self._is_collision()

        # Restore to initial
        self._set_qpos_pose(float(initial_pose[0]), float(initial_pose[1]),
                            float(initial_pose[2]))
        mujoco.mj_forward(self.model, self.data)

        if has_collision:
            raise ValueError(f"Goal pose {goal_pose} results in a collision.")

        self.data.qvel[:] = 0.0
        self._is_final_goal = True
        self._set_goal_marker_position(self._goal, is_final_goal=True)
        return self._get_obs()

    # ------------------------------------------------------------------
    # reset / set_inference_goal / set_curriculum
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[dict, dict]:
        ob, _ = super().reset(seed=seed, options=options)
        return ob, self._get_reset_info()

    def set_inference_goal(self, goal: NDArray[np.float32]) -> None:
        self._goal = np.asarray(goal, dtype=np.float32)
        self._set_goal_marker_position(self._goal, is_final_goal=False)
        self._is_final_goal = False

    def set_curriculum(self, max_goal_sampling_distance: float) -> None:
        self.max_goal_sampling_distance = max_goal_sampling_distance

    # ------------------------------------------------------------------
    # step  — LiDAR raycast tại đây, cache để _get_obs() dùng lại
    # ------------------------------------------------------------------
    def step(self, action: NDArray[np.float32]):
        alpha, beta, gamma = float(action[0]), float(action[1]), float(action[2])

        # FIX BUG 5: sensor_utils mới nhận (model, data) → 4 return values
        state_pos, state_vel, ranges, obstacles = get_robot_state_and_lidar(
            self.model, self.data,
            num_rays=self.n_lidar,
            max_range=self.lidar_max_range
        )

        # Cache để _get_obs() dùng lại — tránh raycast 2 lần mỗi step
        self._cached_ranges    = ranges
        self._cached_obstacles = obstacles

        # DWA tính lệnh an toàn từ trọng số SAC đưa ra
        v_cmd, w_cmd = self.dwa.plan(
            state_pos=state_pos,
            state_vel=state_vel,
            goal_pos=self._goal,
            obstacles=obstacles,
            rl_weights=(alpha, beta, gamma)
        )

        self.velocity_controller.set_cmd(float(v_cmd), float(w_cmd))

        # ── Giữ nguyên ctrl của dynamic obstacles ────────────────────
        # BUG CŨ: np.zeros(model.nu) ghi đè TOÀN BỘ ctrl về 0
        #   → dynamic obstacle actuators nhận target=0 → không di chuyển
        #
        # FIX: dùng self.data.ctrl.copy() để bảo toàn các giá trị
        #   DynamicObstacleController.update() đã set vào data.ctrl.
        # velocity_controller là mjcb_control callback → vẫn override
        #   đúng các wheel actuators tại mỗi sub-step.
        self.do_simulation(ctrl=self.data.ctrl.copy(), n_frames=self.frame_skip)

        # Xóa cache sau khi obs được build (step tiếp theo sẽ raycast lại)
        obs       = self._get_obs()
        self._cached_ranges    = None
        self._cached_obstacles = None

        robot_pos  = obs["achieved_goal"]
        goal_pos   = obs["desired_goal"]

        cur_xy = np.array(robot_pos[:2], dtype=np.float32)

        # ===== 1. TIẾN GẦN GOAL KẾT HỢP CẢNH BÁO PHÍA TRƯỚC =====
        curr_dist = np.linalg.norm(cur_xy - self._goal[:2])
        if self._prev_goal_dist is None:
            self._prev_goal_dist = curr_dist
        raw_progress = self._prev_goal_dist - curr_dist
        self._prev_goal_dist = curr_dist

        # A. Xác định vùng "Trước mặt" (Góc nhìn nón hình quạt: -30 độ đến +30 độ)
        front_angle_limit = np.pi / 18 
        front_indices = np.where(np.abs(self.lidar_angles) <= front_angle_limit)[0]

        if ranges is not None and len(ranges) > 0:
            front_ranges = ranges[front_indices]
            front_hits = front_ranges[front_ranges < self.lidar_max_range]
            min_front_dist = float(front_hits.min()) if len(front_hits) > 0 else self.lidar_max_range
        else:
            min_front_dist = self.lidar_max_range

        # B. Phân tích vùng trước mặt để quyết định Thưởng/Phạt
        FRONT_SAFE_DIST = 1.7 # Ngưỡng nguy hiểm trước mặt (1.2 mét)
        
        if min_front_dist < FRONT_SAFE_DIST:
            # 1. Có vật cản! HỦY phần thưởng nếu nó đang cố rướn tới (chỉ giữ progress nếu nó đang lùi ra xa)
            progress = min(0.0, raw_progress)
            
            # 2. Phạt nhẹ cảnh báo nguy hiểm (Càng gần vật cản phạt càng nặng, tối đa trừ 2 điểm/step)
            front_warning_penalty = -3.0 * (1.0 - (min_front_dist / FRONT_SAFE_DIST))
        else:
            # Đường quang đãng! Thưởng tiến tới bình thường
            progress = raw_progress
            front_warning_penalty = 0.0

        reward_progress = self.PROGRESS_GAIN * progress

        # ===== 2. GẦN VẬT CẢN -> PHẠT NHẸ (Soft Obstacle Penalty) =====
        real_collision = self._is_collision()
        if ranges is not None and len(ranges) > 0:
            hits = ranges[ranges < self.lidar_max_range]
            min_r = float(hits.min()) if len(hits) > 0 else self.lidar_max_range
        else:
            min_r = self.lidar_max_range

        obstacle_penalty = 0.0
        if min_r < 1.0: # Dưới 1m mới bắt đầu phạt
            obstacle_penalty = - (1.0 - min_r) * self.OBSTACLE_PEN_GAIN

         # ===== 3. ĐỨNG YÊN -> PHẠT TĂNG DẦN (Stuck Penalty) =====
        if self._last_xy is None:
            displacement = float('inf')
        else:
            displacement = float(np.linalg.norm(cur_xy - self._last_xy))
        self._last_xy = cur_xy

        if displacement < self.STUCK_THRESH:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        is_stuck = self._stuck_counter >= self.STUCK_PATIENCE
        # Phạt tăng dần theo thời gian đứng yên
        stuck_penalty = - self.STUCK_PEN_GAIN * (self._stuck_counter / self.STUCK_PATIENCE)

        # ===== BASELINE REWARD =====
        # Cộng dồn: Thưởng tiến tới + Phạt vật cản + Phạt đứng yên + Phạt thời gian
        reward = reward_progress + obstacle_penalty + stuck_penalty + self.STEP_PEN

        # ===== 4 & 5. TERMINAL REWARDS (Ghi đè) =====
        collision = real_collision
        is_success = self._is_success(goal_pos[:2], robot_pos[:2])

        if collision:
            reward = self.COLLISION_PEN  # Phạt nặng nhất
        elif is_success:
            reward = self.SUCCESS_REWARD # Thưởng lớn nhất

        # ===== GÓI INFO LẠI CHUẨN GYMNASIUM =====
        info = {
            "is_success": is_success,
            "collision": collision,
            "min_lidar": min_r,
            "progress": progress,
            "stuck": is_stuck,
            "stuck_counter": self._stuck_counter,
            "obstacle_penalty": obstacle_penalty,
            "stuck_penalty": stuck_penalty,
            "front_warning_penalty": front_warning_penalty
        }

        # ===== TERMINATION =====
        if self.inference_mode:
            terminated = collision or (is_success and self._is_final_goal)
        else:
            # Sửa lại theo đúng ý bạn: Chỉ đụng (collision) mới terminate (hoặc success)
            # Nếu bạn muốn kẹt quá lâu cũng terminate để reset episode thì thêm `or is_stuck`
            terminated = is_success or collision or is_stuck

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # _get_obs  — dùng cached LiDAR nếu có, raycast lại nếu không
    # ------------------------------------------------------------------
    def _get_obs(self) -> dict[str, np.ndarray]:
        pts     = self._lidar_points().astype(np.float32)   # (n_lidar, 3)
        pts_flat = pts.flatten()

        v_body, w_body = self.get_robot_velocities()
        v_raw, w_raw   = float(v_body[0]), float(w_body[1])
        vel = np.clip(np.array([v_raw, w_raw], np.float32) / self.vel_scale,
                      -1.0, 1.0)

        obs = np.concatenate([pts_flat, vel]).astype(np.float32)

        # Robot pose
        x_r  = float(self.data.qpos[0])
        y_r  = float(self.data.qpos[1])
        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw_r = float(2.0 * np.arctan2(qz, qw))
        achieved_goal = np.array([x_r, y_r, yaw_r], dtype=np.float32)

        return {
            "observation":   obs,
            "achieved_goal": achieved_goal,
            "desired_goal":  self._goal.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # _lidar_points  — FIX BUG 6: dùng `ranges` (1D distances) thay vì
    #   ép `obstacles` (Kx2 world coords) làm distance array
    # ------------------------------------------------------------------
    def _lidar_points(self) -> np.ndarray:
        """Returns (n_lidar, 3): each row = [sin(angle), cos(angle), d_norm]."""

        if self._cached_ranges is not None:
            # Dùng kết quả đã raycast trong step() — không tốn thêm CPU
            ranges = self._cached_ranges
        else:
            # Reset hoặc gọi ngoài step() → raycast fresh
            _, _, ranges, _ = get_robot_state_and_lidar(
                self.model, self.data,
                num_rays=self.n_lidar,
                max_range=self.lidar_max_range
            )

        # Normalize: max_range → -1.0,  0m → +1.0
        d_clipped = np.clip(ranges, 0.0, self.lidar_max_range)
        d_norm    = (d_clipped / self.lidar_max_range) * 2.0 - 1.0  # ∈ [-1, 1]

        pts = np.stack([
            np.sin(self.lidar_angles),
            np.cos(self.lidar_angles),
            d_norm,
        ], axis=-1)   # (n_lidar, 3)

        return pts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_reset_info(self):
        return {"goal": self._goal.copy()}

    def _goal_pose_error(self) -> float:
        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw  = 2.0 * np.arctan2(qz, qw)
        diff = np.arctan2(np.sin(yaw - self._goal[2]),
                          np.cos(yaw - self._goal[2]))
        return float(np.abs(diff))

    # ── Robot body name prefixes — để lọc self-contact ──────────────
    _ROBOT_BODY_PREFIXES = (
        "mobile_base", "w_rr", "w_lr", "w_rf",
        "w_lf", "w_rc", "w_lc",
    )

    def _is_collision(self) -> bool:
        """
        Trả về True CHỈ KHI robot contact với OBSTACLE body.
        Bỏ qua:
          1. floor contacts  (robot đứng trên sàn → luôn có)
          2. robot self-contacts (xe xích Bunker có wheel/track
             contacts nội bộ → luôn có, không phải va chạm thật)

        Bug cũ: chỉ filter "floor" → self-contacts làm hàm
        luôn trả True → reset_model() fail 10,000 lần.
        """
        for i in range(self.data.ncon):
            g1_id = self.data.contact[i].geom1
            g2_id = self.data.contact[i].geom2

            # ── Filter 1: floor ───────────────────────────────────
            g1_name = self.model.geom(g1_id).name
            g2_name = self.model.geom(g2_id).name
            if g1_name == "floor" or g2_name == "floor":
                continue

            # ── Filter 2: robot self-contact ─────────────────────
            b1 = self.model.body(self.model.geom_bodyid[g1_id]).name
            b2 = self.model.body(self.model.geom_bodyid[g2_id]).name
            b1_robot = any(b1.startswith(p) for p in self._ROBOT_BODY_PREFIXES)
            b2_robot = any(b2.startswith(p) for p in self._ROBOT_BODY_PREFIXES)
            if b1_robot and b2_robot:
                continue

            # ── Collision thật: OBSTACLE tham gia ────────────────
            if b1.startswith("obstacle") or b2.startswith("obstacle"):
                return True

        return False

    def _is_success(self, robot_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        return np.linalg.norm(robot_pos - goal_pos) < self.SUCCESS_DIST

    def compute_reward(self, achieved_goal, desired_goal, info) -> np.ndarray:
        """
        Hàm phần thưởng với 5 thành phần:

          - SUCCESS_REWARD = +100 khi đến goal           (đến đích)
          - COLLISION_PEN  = -100 khi đụng vật cản       (an toàn)
          - STUCK_PEN      =  -50 khi đứng yên quá lâu   (khám phá)
          - NARROW_BONUS   =  +30 khi đi qua chỗ hẹp     (kỹ năng tinh tế)
          - STEP_PEN       =   -1 mỗi step               (đi nhanh)

        Quy tắc cộng thưởng:
          * step penalty là baseline cho mọi step
          * narrow bonus được CỘNG thêm vào step thường (không loại trừ
            step penalty) → tổng = -1 + 30 = +29 ở step đi qua chỗ hẹp
          * collision / stuck / success thì GHI ĐÈ luôn step penalty

        Thứ tự ưu tiên ghi đè (thấp → cao):
            step_penalty < stuck < collision < success
        Còn narrow_bonus thì cộng thêm vào KHI KHÔNG có collision / stuck.

        Hỗ trợ cả single transition (info: dict) và batch HER (info: list).
        """
        d = np.linalg.norm(achieved_goal[..., :2] - desired_goal[..., :2], axis=-1)
        is_success = d < self.goal_xy_distance_threshold

        # Parse info — có thể là dict (single) hoặc list of dicts (batch)
        if isinstance(info, dict):
            is_collision  = np.array(info.get("collision",     False), dtype=bool)
            is_stuck      = np.array(info.get("stuck",         False), dtype=bool)
            progress      = float(info.get("progress", 0.0))
            obs_pen       = float(info.get("obstacle_penalty", 0.0))
            stuck_pen     = float(info.get("stuck_penalty", 0.0))
            front_pen     = float(info.get("front_warning_penalty", 0.0)) # <--- THÊM DÒNG NÀY
        else:
            is_collision  = np.array([x.get("collision",     False) for x in info], dtype=bool)
            is_stuck      = np.array([x.get("stuck",         False) for x in info], dtype=bool)
            progress      = np.array([x.get("progress", 0.0) for x in info], dtype=np.float32)
            obs_pen       = np.array([x.get("obstacle_penalty", 0.0) for x in info], dtype=np.float32)
            stuck_pen     = np.array([x.get("stuck_penalty", 0.0) for x in info], dtype=np.float32)
            front_pen     = np.array([x.get("front_warning_penalty", 0.0) for x in info], dtype=np.float32) # <--- THÊM DÒNG NÀY

        # Base logic: Reward liên tục (Cộng thêm front_pen vào đây)
        reward = (self.PROGRESS_GAIN * progress) + obs_pen + stuck_pen + front_pen + self.STEP_PEN

        # Terminal conditions (ghi đè theo thứ tự)
        reward = np.where(is_stuck, self.STUCK_PEN, reward)
        reward = np.where(is_collision, self.COLLISION_PEN, reward)
        reward = np.where(is_success, self.SUCCESS_REWARD, reward)
        return reward

    def _set_goal_marker_position(self, pos: np.ndarray,
                                   is_final_goal: bool = False) -> None:
        p = np.array([pos[0], pos[1], 0.1])
        if is_final_goal:
            self.data.mocap_pos[self.final_goal_mid] = p
        else:
            self.data.mocap_pos[self.mid_goal_mid] = p

    def build_occupancy_grid(self, resolution=0.1, inflation_radius=0.7):
        """
        Trích xuất Grid Map từ MuJoCo.
        - resolution: Độ phân giải 1 ô lưới (0.1m = 10cm).
        - inflation_radius: Bán kính làm phình (Nên set = bán kính xe Bunker + 0.1m an toàn).
        """
        # 1. Tính toán kích thước ma trận 2D
        width = int(np.ceil((self.xy_max[0] - self.xy_min[0]) / resolution))
        height = int(np.ceil((self.xy_max[1] - self.xy_min[1]) / resolution))
        
        # Khởi tạo ma trận toàn số 0 (Đường trống)
        grid_map = np.zeros((width, height), dtype=int)

        # 2. Quét toàn bộ vật thể (geom) trong MuJoCo
        for i in range(self.model.ngeom):
            # Tùy thuộc vào version mujoco, bạn có thể lấy tên như sau:
            geom_name = self.model.geom(i).name
            
            # Lọc 1: Bỏ qua sàn nhà
            if geom_name == "floor":
                continue
            
            # Lọc 2: Bỏ qua chính bản thân con robot (không tự coi mình là vật cản)
            is_robot_part = any(geom_name.startswith(p) for p in getattr(self, '_ROBOT_BODY_PREFIXES', []))
            if is_robot_part:
                continue
                
            # Lọc 3: Lấy tọa độ (x, y) của vật cản
            pos = self.model.geom_pos[i]
            obs_x, obs_y = float(pos[0]), float(pos[1])
            
            # Lấy kích thước vật cản (dùng max size làm bán kính bao tròn)
            geom_size = self.model.geom_size[i]
            obs_radius = float(np.max(geom_size))
            
            # 3. Kỹ thuật INFLATION (Làm phình vật cản)
            safe_radius = obs_radius + inflation_radius
            
            # 4. Xác định vùng Bounding Box trên Lưới (Grid index)
            min_gx = int(np.floor((obs_x - safe_radius - self.xy_min[0]) / resolution))
            max_gx = int(np.ceil((obs_x + safe_radius - self.xy_min[0]) / resolution))
            min_gy = int(np.floor((obs_y - safe_radius - self.xy_min[1]) / resolution))
            max_gy = int(np.ceil((obs_y + safe_radius - self.xy_min[1]) / resolution))
            
            # Cắt xén (Clip) để index không vượt quá giới hạn mảng
            min_gx = max(0, min_gx)
            max_gx = min(width - 1, max_gx)
            min_gy = max(0, min_gy)
            max_gy = min(height - 1, max_gy)
            
            # 5. Tô màu vật cản (số 1) vào Grid Map
            for gx in range(min_gx, max_gx + 1):
                for gy in range(min_gy, max_gy + 1):
                    # Kiểm tra khoảng cách Euclid để vẽ hình tròn (chính xác hơn vẽ hình vuông)
                    cell_x = self.xy_min[0] + gx * resolution
                    cell_y = self.xy_min[1] + gy * resolution
                    if np.hypot(cell_x - obs_x, cell_y - obs_y) <= safe_radius:
                        grid_map[gx, gy] = 1
                        
        return grid_map
    
    def world_to_grid(self, x, y, resolution=0.1):
        """Đổi tọa độ MuJoCo (m) sang Tọa độ mảng (index) cho A*"""
        gx = int(np.floor((x - self.xy_min[0]) / resolution))
        gy = int(np.floor((y - self.xy_min[1]) / resolution))
        return (gx, gy)

    def grid_to_world(self, gx, gy, resolution=0.1):
        """Đổi Tọa độ mảng của A* ngược lại thành Tọa độ MuJoCo (m) cho SAC lái"""
        x = self.xy_min[0] + (gx + 0.5) * resolution # +0.5 để lấy tâm ô lưới
        y = self.xy_min[1] + (gy + 0.5) * resolution
        return np.array([x, y])


# ── Quick sanity check ───────────────────────────────────────────────
if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml  = os.path.join(root, "assets", "worlds", "empty.xml")
    env  = BunkerEnv(xml, render_mode="human")
    check_env(env, warn=True, skip_render_check=False)