#!/usr/bin/env python3
"""
bunker_sac_v2.py
----------------------------------------------------
✔ Train từ đầu / Resume checkpoint / Resume best_model
✔ Overwrite best_model (production style)
✔ Backup trước khi overwrite
✔ Fix optimizer khi resume
"""

import os
import sys
import shutil

import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CallbackList, EvalCallback, CheckpointCallback
)

from training_metrics_callback import TrainingMetricsCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
from gym_bunker_env import BunkerEnv
from feature_extractor import FeatureExtractor


# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────
def make_single_env(xml_paths, max_ep_steps, env_i):
    def _init():
        xml_path = xml_paths[env_i % len(xml_paths)]
        env = BunkerEnv(xml_path=xml_path, render_mode=None, n_lidar=449)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_ep_steps)
        return env
    return _init


# ─────────────────────────────────────────────────────────────
# RUN NAME
# ─────────────────────────────────────────────────────────────
def get_next_run_name(log_root):
    if not os.path.exists(log_root):
        return "run_1"

    existing = [
        d for d in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, d)) and d.startswith("run_")
    ]

    nums = []
    for d in existing:
        try:
            nums.append(int(d.split("_")[1]))
        except:
            pass

    return f"run_{max(nums) + 1}" if nums else "run_1"


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────
def train(train_xml_paths, val_xml_paths, total_steps,
          n_envs, max_ep_steps, log_dir,
          resume_path=None, resume_lr=None):

    os.makedirs(log_dir, exist_ok=True)

    # ── ENV ───────────────────────────────────────────────
    env = SubprocVecEnv([
        make_single_env(train_xml_paths, max_ep_steps, i)
        for i in range(n_envs)
    ])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor"))

    eval_env = SubprocVecEnv([
        make_single_env(val_xml_paths, max_ep_steps, i)
        for i in range(len(val_xml_paths))
    ])
    eval_env = VecMonitor(eval_env)

    n_eval_episodes = len(val_xml_paths) * 10

    # ── CALLBACK ───────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=1_500,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="rl_backup",
    )

    metrics_callback = TrainingMetricsCallback(
        log_freq=100, ep_window=100, verbose=0
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        metrics_callback
    ])

    # ── MODEL ─────────────────────────────────────────────
    if resume_path and os.path.exists(resume_path):

        print("\n" + "="*60)
        print(f"🔄 RESUME MODEL: {resume_path}")
        print("="*60 + "\n")

        # 🔥 Backup best_model trước khi overwrite
        best_model_path = os.path.join(log_dir, "best_model", "best_model.zip")
        if os.path.exists(best_model_path):
            backup_path = os.path.join(log_dir, "best_model_backup.zip")
            shutil.copy(best_model_path, backup_path)
            print("📦 Backup best_model created")

        model = SAC.load(resume_path, env=env, device="auto")

        # 🔥 Fix optimizer + learning rate
        if resume_lr:
            print(f"⚙️ Set learning rate: {resume_lr}")
            model.learning_rate = resume_lr
            model._setup_lr_schedule()
            model._setup_model()

        model.tensorboard_log = os.path.join(log_dir, "tb")

    else:
        print("\n" + "="*60)
        print("🚀 TRAIN FROM SCRATCH")
        print("="*60 + "\n")

        max_dist_diag = env.get_attr("max_distance_diagonal")[0]

        policy_kwargs = dict(
            features_extractor_class=FeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                n_lidar=449,
                max_distance_diagonal=max_dist_diag,
            ),
            net_arch=[128, 128],
        )

        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
                copy_info_dict=True,
            ),
            batch_size=256,
            learning_rate=3e-4,
            learning_starts=n_envs * max_ep_steps,
            ent_coef="auto_0.1",
            gamma=0.99,
            tau=0.005,
            buffer_size=300_000,
            train_freq=(1, "step"),
            gradient_steps=1,
            target_update_interval=2,
            tensorboard_log=os.path.join(log_dir, "tb"),
            verbose=1,
            device="auto",
        )

    print(f"Train worlds : {len(train_xml_paths)}")
    print(f"Val worlds   : {len(val_xml_paths)}")
    print(f"n_envs       : {n_envs}")
    print(f"total_steps  : {total_steps:,}")
    print(f"log_dir      : {log_dir}\n")

    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        tb_log_name="run",
        progress_bar=True,
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    root = os.path.dirname(os.path.abspath(__file__))

    worlds_train_dir = os.path.join(root, "..", "assets", "worlds", "train")
    worlds_val_dir   = os.path.join(root, "..", "assets", "worlds", "val")

    train_xml_pool = sorted(
        os.path.join(worlds_train_dir, f)
        for f in os.listdir(worlds_train_dir) if f.endswith(".xml")
    )

    val_xml_pool = sorted(
        os.path.join(worlds_val_dir, f)
        for f in os.listdir(worlds_val_dir) if f.endswith(".xml")
    )

    # ── HYPERPARAM ────────────────────────────────────────
    TOTAL_STEPS  = 100000
    N_ENVS       = 40
    MAX_EP_STEPS = 600

    base_log_dir = os.path.join(root, "log")

    # ─────────────────────────────────────────────────────
    # 🔥 CONFIG RESUME
    # ─────────────────────────────────────────────────────
    RESUME_RUN   = "run_13"   # None nếu train mới
    RESUME_TYPE  = "best"      # "best" | "checkpoint"
    RESUME_CKPT  = None
    RESUME_LR    = 3e-4

    if RESUME_RUN:

        log_dir = os.path.join(base_log_dir, RESUME_RUN)

        if RESUME_TYPE == "best":
            resume_path = os.path.join(log_dir, "best_model", "best_model.zip")

        elif RESUME_TYPE == "checkpoint":
            if not RESUME_CKPT:
                raise ValueError("❌ Cần RESUME_CKPT")
            resume_path = os.path.join(log_dir, "checkpoints", RESUME_CKPT)

        else:
            raise ValueError("❌ RESUME_TYPE sai")

        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"❌ Không tìm thấy: {resume_path}")

        print(f"\n▶ RESUME RUN : {RESUME_RUN}")
        print(f"▶ TYPE       : {RESUME_TYPE}\n")

    else:
        run_name = get_next_run_name(base_log_dir)
        log_dir  = os.path.join(base_log_dir, run_name)
        resume_path = None

    # ─────────────────────────────────────────────────────
    train(
        train_xml_paths=train_xml_pool,
        val_xml_paths=val_xml_pool,
        total_steps=TOTAL_STEPS,
        n_envs=N_ENVS,
        max_ep_steps=MAX_EP_STEPS,
        log_dir=log_dir,
        resume_path=resume_path,
        resume_lr=RESUME_LR,
    )