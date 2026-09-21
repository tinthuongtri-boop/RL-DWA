"""
training_metrics_callback.py
─────────────────────────────
Custom callback ghi log đầy đủ metrics training vào TensorBoard:

  ┌─ Episode-level ──────────────────────────────
  │  rollout/ep_rew_mean         (đã có sẵn của SB3)
  │  rollout/ep_len_mean         (đã có sẵn)
  │  custom/success_rate         (mới)  — % ep đến đích
  │  custom/collision_rate       (mới)  — % ep đụng tường
  │  custom/stuck_rate           (mới)  — % ep bị stuck
  │  custom/timeout_rate         (mới)  — % ep hết thời gian
  │  custom/narrow_pass_rate     (mới)  — % ep đi qua chỗ hẹp
  │
  ├─ SAC weights ────────────────────────────────
  │  custom/weight_alpha_mean    — trung bình α qua các ep
  │  custom/weight_beta_mean     — trung bình β
  │  custom/weight_gamma_mean    — trung bình γ
  │  custom/weight_alpha_std     — độ biến thiên α (kiểm tra adaptive)
  │  custom/weight_beta_std      — std β
  │  custom/weight_gamma_std     — std γ
  │
  └─ Hành vi điều khiển ─────────────────────────
     custom/mean_v               — vận tốc trung bình
     custom/mean_abs_w           — |ω| trung bình
     custom/mean_min_clearance   — LiDAR min trung bình

Cách dùng (trong bunker_sac.py):

    from training_metrics_callback import TrainingMetricsCallback
    metrics_cb = TrainingMetricsCallback(log_freq=100)
    callbacks  = CallbackList([eval_callback, checkpoint_callback, metrics_cb])
"""

import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    """
    Ghi log custom metrics vào TensorBoard mỗi `log_freq` steps.

    Args:
        log_freq      : số bước giữa mỗi lần log (default 100).
        ep_window     : số episode gần nhất dùng để tính rolling stats (default 100).
        verbose       : 0 = tắt print, 1 = print mỗi lần log.
    """

    def __init__(self, log_freq: int = 100, ep_window: int = 100,
                 verbose: int = 0):
        super().__init__(verbose)
        self.log_freq  = log_freq
        self.ep_window = ep_window

        # Rolling buffers — chỉ giữ N episode gần nhất
        self.ep_outcomes      = deque(maxlen=ep_window)   # 'success' / 'collision' / 'stuck' / 'timeout'
        self.ep_narrow_passed = deque(maxlen=ep_window)   # bool
        self.ep_alphas        = deque(maxlen=ep_window)   # mean α của ep
        self.ep_betas         = deque(maxlen=ep_window)
        self.ep_gammas        = deque(maxlen=ep_window)
        self.ep_mean_v        = deque(maxlen=ep_window)
        self.ep_mean_w        = deque(maxlen=ep_window)
        self.ep_min_clearance = deque(maxlen=ep_window)

        # Tích lũy trong episode đang chạy (mỗi worker riêng)
        self._n_envs = None
        self._cur_alphas    = None
        self._cur_betas     = None
        self._cur_gammas    = None
        self._cur_vs        = None
        self._cur_ws        = None
        self._cur_min_clrs  = None
        self._cur_narrow    = None

    def _init_buffers_if_needed(self):
        """Khởi tạo buffer per-worker khi biết n_envs."""
        if self._n_envs is None:
            self._n_envs = self.training_env.num_envs
            self._cur_alphas   = [[] for _ in range(self._n_envs)]
            self._cur_betas    = [[] for _ in range(self._n_envs)]
            self._cur_gammas   = [[] for _ in range(self._n_envs)]
            self._cur_vs       = [[] for _ in range(self._n_envs)]
            self._cur_ws       = [[] for _ in range(self._n_envs)]
            self._cur_min_clrs = [[] for _ in range(self._n_envs)]
            self._cur_narrow   = [False] * self._n_envs

    def _on_step(self) -> bool:
        self._init_buffers_if_needed()

        # ── Lấy actions, observations, infos của step hiện tại ──────
        actions = self.locals.get("actions")     # (n_envs, 3)
        new_obs = self.locals.get("new_obs")     # dict
        infos   = self.locals.get("infos", [])
        dones   = self.locals.get("dones", [])

        if actions is None or new_obs is None:
            return True

        # ── Tích lũy weights vào buffer per-worker ──────────────────
        for i in range(self._n_envs):
            a = actions[i]
            self._cur_alphas[i].append(float(a[0]))
            self._cur_betas[i].append(float(a[1]))
            self._cur_gammas[i].append(float(a[2]))

            # Lấy v, ω, min_lidar từ observation['observation']
            # Layout: [n_lidar*3 floats LiDAR | v | ω]
            obs_vec = new_obs["observation"][i]
            n_lidar = (len(obs_vec) - 2) // 3
            lidar   = obs_vec[: n_lidar * 3].reshape(-1, 3)
            d_norm  = lidar[:, 2]
            d_real  = (d_norm + 1.0) / 2.0 * 20.0   # un-normalize (max_range=20)
            self._cur_min_clrs[i].append(float(d_real.min()))

            v_norm = float(obs_vec[n_lidar * 3])
            w_norm = float(obs_vec[n_lidar * 3 + 1])
            self._cur_vs[i].append(v_norm * 0.5)   # un-normalize v_max=0.5
            self._cur_ws[i].append(w_norm * 0.5)

            # Track narrow pass trong ep
            if i < len(infos) and infos[i].get("passed_narrow", False):
                self._cur_narrow[i] = True

            # ── Khi episode kết thúc → push vào rolling buffer ─────
            if i < len(dones) and dones[i]:
                info = infos[i] if i < len(infos) else {}

                # Outcome classification (priority: success > collision > stuck > timeout)
                if info.get("is_success", False):
                    outcome = "success"
                elif info.get("collision", False):
                    outcome = "collision"
                elif info.get("stuck", False):
                    outcome = "stuck"
                else:
                    outcome = "timeout"
                self.ep_outcomes.append(outcome)
                self.ep_narrow_passed.append(self._cur_narrow[i])

                if self._cur_alphas[i]:
                    self.ep_alphas.append(float(np.mean(self._cur_alphas[i])))
                    self.ep_betas.append( float(np.mean(self._cur_betas[i])))
                    self.ep_gammas.append(float(np.mean(self._cur_gammas[i])))
                    self.ep_mean_v.append(float(np.mean(self._cur_vs[i])))
                    self.ep_mean_w.append(float(np.mean(np.abs(self._cur_ws[i]))))
                    self.ep_min_clearance.append(float(np.mean(self._cur_min_clrs[i])))

                # Reset buffer của worker này
                self._cur_alphas[i].clear()
                self._cur_betas[i].clear()
                self._cur_gammas[i].clear()
                self._cur_vs[i].clear()
                self._cur_ws[i].clear()
                self._cur_min_clrs[i].clear()
                self._cur_narrow[i] = False

        # ── Ghi log mỗi log_freq steps ──────────────────────────────
        if self.num_timesteps % self.log_freq == 0 and len(self.ep_outcomes) > 0:
            n = len(self.ep_outcomes)

            # Outcome rates
            success_rate   = sum(o == "success"   for o in self.ep_outcomes) / n
            collision_rate = sum(o == "collision" for o in self.ep_outcomes) / n
            stuck_rate     = sum(o == "stuck"     for o in self.ep_outcomes) / n
            timeout_rate   = sum(o == "timeout"   for o in self.ep_outcomes) / n
            narrow_rate    = sum(self.ep_narrow_passed) / n

            self.logger.record("custom/success_rate",     success_rate)
            self.logger.record("custom/collision_rate",   collision_rate)
            self.logger.record("custom/stuck_rate",       stuck_rate)
            self.logger.record("custom/timeout_rate",     timeout_rate)
            self.logger.record("custom/narrow_pass_rate", narrow_rate)

            # SAC weight statistics
            if self.ep_alphas:
                self.logger.record("custom/weight_alpha_mean", float(np.mean(self.ep_alphas)))
                self.logger.record("custom/weight_beta_mean",  float(np.mean(self.ep_betas)))
                self.logger.record("custom/weight_gamma_mean", float(np.mean(self.ep_gammas)))
                self.logger.record("custom/weight_alpha_std",  float(np.std(self.ep_alphas)))
                self.logger.record("custom/weight_beta_std",   float(np.std(self.ep_betas)))
                self.logger.record("custom/weight_gamma_std",  float(np.std(self.ep_gammas)))

                # Behavior metrics
                self.logger.record("custom/mean_v",             float(np.mean(self.ep_mean_v)))
                self.logger.record("custom/mean_abs_w",         float(np.mean(self.ep_mean_w)))
                self.logger.record("custom/mean_min_clearance", float(np.mean(self.ep_min_clearance)))

            if self.verbose >= 1:
                print(f"[Metrics @ step {self.num_timesteps}] "
                      f"SR={success_rate:.2f}  CR={collision_rate:.2f}  "
                      f"Narrow={narrow_rate:.2f}  "
                      f"α={np.mean(self.ep_alphas):.2f}  "
                      f"β={np.mean(self.ep_betas):.2f}  "
                      f"γ={np.mean(self.ep_gammas):.2f}")

        return True
