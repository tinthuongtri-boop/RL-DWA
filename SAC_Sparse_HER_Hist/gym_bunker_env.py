# bunker_pid_env.py
from __future__ import annotations

import os
import numpy as np
import mujoco
import torch
import sys
from collections import deque
from numpy.typing import NDArray
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3.common.env_checker import check_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.bunker_velocity_controller import BunkerVelocityController

from SAC_Sparse_HER.DWA import DWASolver  # Hàm DWA (Giai đoạn 3)
from SAC_Sparse_HER.sensor_utils import get_robot_state_and_lidar # Hàm lấy dữ liệu (Giai đoạn 2)


class BunkerEnv(MujocoEnv):
    """
    ## Parameters
    - n_lidar (int): Number of LiDAR rays (default: 449)

    ## Action Space (ĐÃ ĐƯỢC CHỈNH SỬA CHO DWA)

| Num | Action | Control Min | Control Max |
|-----|--------|-------------|-------------|
| 0 | alpha | 0.0 | 1.0 |
| 1 | beta | 0.0 | 1.0 |
| 2 | gamma | 0.0 | 1.0 |

    ## Observation Space
    The observation space is a Dict with:
    - `observation`: `Box(-1, 1, (n_lidar*3 + k_history_window * 2), float32)` -> LiDAR + k_history_window * vel (v, w)
    - `achieved_goal`: `Box(-inf, inf, (3,), float32)` -> [x, y, yaw]
    - `desired_goal`: `Box(-inf, inf, (3,), float32)` -> [x, y, yaw]

    ## Reward Function
    r = (0 if success; -100 if crash; else -1)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, xml_path: str, frame_skip: int = 10, render_mode: str | None = None, n_lidar: int = 449, inference_mode: bool = False, 
                max_goal_sampling_distance: float = 20.0, k_history_window: int = 10):

        print(" Max goal sampling distance:", max_goal_sampling_distance)
        print(" K history window:", k_history_window)
        
        # LiDAR parameters
        self.n_lidar   = n_lidar
        self.lidar_angles = np.linspace(-np.pi, np.pi, self.n_lidar, endpoint=False).astype(np.float32)
        self.lidar_max_range = 20.0 # This has to be the same as the cutoff in the XML file and point_net_extractor.py.

        # History parameters
        self.k_history = k_history_window
        self.history = deque(maxlen=self.k_history) 

        # World bounds for random reset and goal sampling
        self.xy_min = np.array([-4., -4.], np.float32)
        self.xy_max = np.array([ 10., 10.], np.float32)

        # Robot's yaw bounds
        self.yaw_min = -np.pi
        self.yaw_max = np.pi

        # Goal success and reward thresholds
        self.goal_xy_distance_threshold = 0.2
        self.min_goal_sampling_distance = 1.0
        self.max_goal_sampling_distance = max_goal_sampling_distance

        self.max_distance_diagonal = np.linalg.norm(self.xy_max - self.xy_min)
        
        # Velocity: linear and angular velocity bounds (high-level limits)
        self.v_max, self.w_max = 0.5, 0.5
        self.vel_scale = np.array([self.v_max, self.w_max], np.float32)

        # Observation space: LiDAR + Velocities k_history*(n_lidar*3 + 2) + Goal Dict
        self.features_per_step = 2 # [v, w]
        obs_dim = self.n_lidar*3 + self.k_history * self.features_per_step
        
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        self.max_geom = 1_800

        # Call parent constructor (creates model/data/renderer)
        super().__init__(model_path=xml_path, frame_skip=frame_skip, observation_space=self.observation_space, render_mode=render_mode, max_geom=self.max_geom)

        self._initialize_velocity_controller()

        # Get the goal marker body ID and mocap ID
        self.mid_goal_bid = self.model.body('mid_goal_marker').id
        self.mid_goal_mid = int(self.model.body_mocapid[self.mid_goal_bid])
        self.final_goal_bid = self.model.body('final_goal_marker').id
        self.final_goal_mid = int(self.model.body_mocapid[self.final_goal_bid])

        # internal book-keeping
        self._goal: NDArray[np.float32]
        self._is_final_goal: bool = True  

        self.inference_mode = inference_mode

        ### CHỈNH SỬA DWA Ở ĐÂY: Khởi tạo module DWA Planner ###
        self.dwa = DWASolver()
        #########################################################

    def get_robot_velocities(self):
        """Get robot velocities body frame"""
        linear_vel_world_qvel = self.data.qvel[0:3].copy()    
        angular_vel_world_qvel = self.data.qvel[3:6].copy()    
        
        body_id = self.model.body('mobile_base').id
        rotation_matrix = self.data.xmat[body_id].reshape(3, 3)
        
        linear_vel_body = rotation_matrix.T @ linear_vel_world_qvel
        angular_vel_body = rotation_matrix.T @ angular_vel_world_qvel
        
        return linear_vel_body, angular_vel_body

    def reset_model(self) -> dict[str, np.ndarray]: # Fix return type hint for dict

        self.velocity_controller.reset()
        rng = self.np_random

        while True:
            initial_qpos_x, initial_qpos_y = rng.uniform(self.xy_min, self.xy_max)
            initial_theta = rng.uniform(self.yaw_min, self.yaw_max)

            self.data.qpos[:] = 0.
            self.data.qpos[:2] = initial_qpos_x, initial_qpos_y
            self.data.qpos[2] = 0.25 
            c, s = np.cos(initial_theta / 2), np.sin(initial_theta / 2)
            self.data.qpos[3:7] = (c, 0., 0., s)
            
            mujoco.mj_forward(self.model, self.data)
            if not self._is_collision():
                break

        while True:
            goal_qpos_x, goal_qpos_y = rng.uniform(self.xy_min, self.xy_max)
            goal_theta = rng.uniform(self.yaw_min, self.yaw_max)

            self._goal = np.array([goal_qpos_x, goal_qpos_y, goal_theta])

            distance = np.linalg.norm(self._goal[:2] - (initial_qpos_x, initial_qpos_y))

            if self.min_goal_sampling_distance < distance <= self.max_goal_sampling_distance:
                self.data.qpos[:2] = self._goal[:2]
                c, s = np.cos(goal_theta / 2), np.sin(goal_theta / 2)
                self.data.qpos[3:7] = (c, 0., 0., s)
                mujoco.mj_forward(self.model, self.data)
                has_collision = self._is_collision()

                self.data.qpos[:2] = initial_qpos_x, initial_qpos_y
                c, s = np.cos(initial_theta / 2), np.sin(initial_theta / 2)
                self.data.qpos[3:7] = (c, 0., 0., s)
                mujoco.mj_forward(self.model, self.data)
                
                if not has_collision:
                    break

        self.data.qvel[:] = 0.
        self._is_final_goal = True  

        self._set_goal_marker_position(self._goal, is_final_goal=True)

        self.history.clear()
        initial_z_t = self._get_single_step_features()
        for _ in range(self.k_history):
            self.history.append(initial_z_t)

        return self._get_obs()

    def reset(self, *, seed: int | None = None, options: dict[str, any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, any]]:
        ob, _ = super().reset(seed=seed, options=options)
        info = self._get_reset_info()
        return ob, info

    def set_manual_pose(self, initial_pose: NDArray[np.float32] | list[float], goal_pose: NDArray[np.float32] | list[float]) -> dict[str, np.ndarray]:
        self.velocity_controller.reset()

        initial_pose = np.array(initial_pose, dtype=np.float32)
        goal_pose = np.array(goal_pose, dtype=np.float32)

        if not (self.xy_min <= initial_pose <= self.xy_max and 
                self.xy_min[1] <= initial_pose[1] <= self.xy_max[1]):
            raise ValueError(f"Initial pose {initial_pose[:2]} is out of world bounds {self.xy_min} to {self.xy_max}")
        
        if not (self.xy_min <= goal_pose <= self.xy_max and 
                self.xy_min[1] <= goal_pose[1] <= self.xy_max[1]):
            raise ValueError(f"Goal pose {goal_pose[:2]} is out of world bounds {self.xy_min} to {self.xy_max}")

        self.data.qpos[:] = 0.
        self.data.qpos[:2] = initial_pose[:2]
        self.data.qpos[2] = 0.25
        c, s = np.cos(initial_pose[2] / 2), np.sin(initial_pose[2] / 2)
        self.data.qpos[3:7] = (c, 0., 0., s)

        mujoco.mj_forward(self.model, self.data)
        if self._is_collision():
            raise ValueError(f"Initial pose {initial_pose} results in a collision.")

        distance = np.linalg.norm(goal_pose[:2] - initial_pose[:2])
        if not (self.min_goal_sampling_distance < distance <= self.max_goal_sampling_distance):
            raise ValueError(f"Distance between start and goal ({distance:.2f}m) is outside the valid range "
                             f"[{self.min_goal_sampling_distance}, {self.max_goal_sampling_distance}]")

        self._goal = goal_pose.copy()
        
        self.data.qpos[:2] = self._goal[:2]
        c, s = np.cos(self._goal[2] / 2), np.sin(self._goal[2] / 2)
        self.data.qpos[3:7] = (c, 0., 0., s)
        mujoco.mj_forward(self.model, self.data)
        has_collision = self._is_collision()

        self.data.qpos[:2] = initial_pose[:2]
        c, s = np.cos(initial_pose[2] / 2), np.sin(initial_pose[2] / 2)
        self.data.qpos[3:7] = (c, 0., 0., s)
        mujoco.mj_forward(self.model, self.data)

        if has_collision:
            raise ValueError(f"Goal pose {goal_pose} results in a collision.")

        self.data.qvel[:] = 0.
        self._is_final_goal = True
        self._set_goal_marker_position(self._goal, is_final_goal=True)

        self.history.clear()
        initial_z_t = self._get_single_step_features()
        for _ in range(self.k_history):
            self.history.append(initial_z_t)

        return self._get_obs()

    def set_inference_goal(self, goal: NDArray[np.float32]) -> None:
        self._goal = np.array(goal, dtype=np.float32)
        self._set_goal_marker_position(self._goal, is_final_goal=False)
        self._is_final_goal = False  
        
    def step(self, action: NDArray[np.float32]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        
        ### CHỈNH SỬA DWA Ở ĐÂY: Can thiệp điều hướng ###
        # 1. Trích xuất 3 trọng số (alpha, beta, gamma) thay vì lệnh ga/phanh
        alpha, beta, gamma = action, action[1], action[2]
        
        # 2. Đọc trạng thái vị trí, vận tốc và quét chướng ngại vật bằng hàm tự viết
        state_pos, state_vel, obstacles = get_robot_state_and_lidar(self.data, num_rays=self.n_lidar, max_range=self.lidar_max_range)
        
        # 3. Sử dụng DWA tính toán lệnh an toàn dựa trên AI
        v_cmd, w_cmd = self.dwa.plan(
            state_pos=state_pos,
            state_vel=state_vel,
            goal_pos=self._goal,
            obstacles=obstacles,
            rl_weights=[alpha, beta, gamma]
        )
        
        # 4. Gửi lệnh truyền động mượt mà và an toàn đó xuống robot
        self.velocity_controller.set_cmd(float(v_cmd), float(w_cmd))
        ##################################################

        # integrate physics
        self.do_simulation(ctrl=np.zeros(self.model.nu), n_frames=self.frame_skip)

        obs        = self._get_obs()

        robot_pos  = obs["achieved_goal"]
        goal_pos   = obs["desired_goal"]
    
        is_success = self._is_success(goal_pos[:2], robot_pos[:2])
        collision  = self._is_collision()

        reward     = float(self.compute_reward(achieved_goal=robot_pos, desired_goal=goal_pos, info={"collision": collision}))

        info = {
            "is_success": is_success,
            "collision": collision,
            "max_goal_sampling_distance": self.max_goal_sampling_distance,
            "k_history_window": self.k_history,
        }
        
        if self.inference_mode:
            terminated = collision or (is_success and self._is_final_goal)
        else:
            terminated = is_success or collision

        truncated = False 

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _initialize_velocity_controller(self):
        self.velocity_controller = BunkerVelocityController()

        id_rr = self.model.body("w_rr").id
        id_lr = self.model.body("w_lr").id
        width_track = abs(self.model.body_pos[id_rr][1] - self.model.body_pos[id_lr][1])

        self.velocity_controller.w_track = width_track

        id_geom = self.model.geom("w_rr_geom").id
        self.velocity_controller.r_wheel = self.model.geom_size[id_geom]

        self.velocity_controller.act_r = [self.model.actuator(n).id for n in ("w_rr","w_rc","w_rf")]
        self.velocity_controller.act_l = [self.model.actuator(n).id for n in ("w_lr","w_lc","w_lf")]

        self.sensor_frc_r = [self.model.sensor(n).id for n in ("sf_rr","sf_rc","sf_rf")]
        self.sensor_frc_l = [self.model.sensor(n).id for n in ("sf_lr","sf_lc","sf_lf")]

        mujoco.set_mjcb_control(self.velocity_controller)

    def _set_action_space(self):
        ### CHỈNH SỬA DWA Ở ĐÂY: Thay đổi không gian hành động thành 3 biến (0 đến 1) ###
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        return self.action_space
        ################################################################################

    def _get_single_step_features(self) -> np.ndarray:
        v_raw, w_raw = self.get_robot_velocities()
        v_raw, w_raw = v_raw, w_raw[2]
        vel = np.clip(np.array([v_raw, w_raw], np.float32) / self.vel_scale, -1.0, 1.0) 
        return vel 

    def _get_obs(self) -> dict[str, np.ndarray]:
        pts = self._lidar_points().astype(np.float32)
        pts_flat = pts.flatten() 

        z_t = self._get_single_step_features()
        self.history.append(z_t)

        history_flat = np.array(self.history).flatten().astype(np.float32) 

        obs = np.concatenate([pts_flat, history_flat]) 

        world_robot_xy_pos = self.data.qpos[:2].copy()
        qw, qx, qy, qz = self.data.qpos[3:7].copy()
        world_robot_yaw = 2.0 * np.arctan2(qz, qw)
        achieved_goal = np.array([world_robot_xy_pos, world_robot_xy_pos[1], world_robot_yaw], dtype=np.float32)

        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self._goal.astype(np.float32)
        }

    def _get_reset_info(self): 
        return {"goal": self._goal.copy()}


    def _goal_pose_error(self) -> float:
        qw, qx, qy, qz = self.data.qpos[3:7].copy()
        yaw = 2.0 * np.arctan2(qz, qw) 
        goal_yaw = self._goal[2]

        angle_diff = yaw - goal_yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        return float(np.abs(angle_diff))

    def _is_collision(self) -> bool:
        for i in range(self.data.ncon):
            g1 = self.model.geom(self.data.contact[i].geom1).name
            g2 = self.model.geom(self.data.contact[i].geom2).name

            if g1 == "floor" or g2 == "floor":
                continue
            return True

        return False

    def _is_success(self, robot_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        if float(np.linalg.norm(robot_pos - goal_pos)) < self.goal_xy_distance_threshold:
            return True
        else:
            return False

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, any] | list[dict[str, any]]) -> np.ndarray:
        
        d = np.linalg.norm(achieved_goal[..., :2] - desired_goal[..., :2], axis=-1)
        
        is_success = d < self.goal_xy_distance_threshold
        
        if isinstance(info, dict):
            collision = info["collision"]                   
            is_collision = np.array(collision, dtype=bool)
        else:
            is_collision = np.array([x["collision"] for x in info], dtype=bool)
        
        reward = np.full_like(d, -1.0, dtype=np.float32) 
        reward = np.where(is_success, 0.0, reward)       
        reward = np.where(is_collision, -100.0, reward)  

        return reward

    def _lidar_points(self) -> np.ndarray:
        d_raw = self.data.sensordata.astype(np.float32) 
        d_raw = d_raw[6:] 

        d_raw = np.where(d_raw == -1.0, self.lidar_max_range, d_raw)

        d_clipped = np.clip(d_raw, 0.0, self.lidar_max_range)               
        d_norm    = (d_clipped / self.lidar_max_range) * 2.0 - 1.0          

        pts = np.stack([
            np.sin(self.lidar_angles),                 
            np.cos(self.lidar_angles),                 
            d_norm                                  
        ], axis=-1)

        return pts
    
    def _set_goal_marker_position(self, pos: np.ndarray, is_final_goal: bool = False):
        pos = np.array([pos, pos[1], 0.1])
        if is_final_goal:
            self.data.mocap_pos[self.final_goal_mid] = pos
        else:
            self.data.mocap_pos[self.mid_goal_mid] = pos

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml  = os.path.join(root, "assets", "worlds", "hallways.xml")
    env  = BunkerEnv(xml, render_mode="human")
    check_env(env, warn=True, skip_render_check=False)