import argparse
import os
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)

from on_policy import (
    DenseRewardWrapper,
    FlattenObsWrapper,
    ASRObsWrapper,
    AugmentedObsWrapper,
    TASKS,
)

gym.register_envs(gymnasium_robotics)

# -------------------------------------------------
# Config
# -------------------------------------------------
LOG_DIR = "logs"
TB_LOG_DIR = os.path.join(LOG_DIR, "tb")
BEST_MODEL_DIR = os.path.join(LOG_DIR, "best_model")
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")

os.makedirs(TB_LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

ENV_ID = "FrankaKitchen-v1"

<<<<<<< HEAD
TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 8
=======
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 30
>>>>>>> 5637ac2b97d4560285cecdae97fa2d6aed5d367d
SEED = 42
USE_ASR = True
USE_SHAPED_REWARD = True


# -------------------------------------------------
# Success info wrapper (for EvalCallback success_rate)
# -------------------------------------------------
class KitchenSuccessInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if done and "is_success" not in info:
            completed = info.get("completed_tasks", [])
            if isinstance(completed, (list, tuple, set)):
                info["is_success"] = float("kettle" in completed)
            else:
                remaining = info.get("tasks_to_complete", [])
                if isinstance(remaining, (list, tuple, set)):
                    info["is_success"] = float("kettle" not in remaining)

        return obs, reward, terminated, truncated, info


# -------------------------------------------------
# Logging callback
# -------------------------------------------------
class InfoStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        success_vals = []
        completed_task_counts = []

        for info in infos:
            if "is_success" in info:
                success_vals.append(float(info["is_success"]))

            if "completed_tasks" in info:
                completed = info["completed_tasks"]
                if isinstance(completed, (list, tuple, set)):
                    completed_task_counts.append(float(len(completed)))
                elif isinstance(completed, (int, float, np.integer, np.floating)):
                    completed_task_counts.append(float(completed))

        if success_vals:
            self.logger.record("rollout/is_success_mean", float(np.mean(success_vals)))
        if completed_task_counts:
            self.logger.record("rollout/completed_tasks_mean", float(np.mean(completed_task_counts)))

        return True


class TrainingLogCallback(BaseCallback):
    """Logs and stores episode returns for plotting."""

    def __init__(self, log_freq: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_returns: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_returns.append(float(info["episode"]["r"]))

        if self.n_calls % self.log_freq == 0 and self.episode_returns:
            mean_ret = float(np.mean(self.episode_returns[-20:]))
            print(f"  Steps: {self.num_timesteps:>8,} | Mean Return (last 20 eps): {mean_ret:>8.3f}")
        return True


# -------------------------------------------------
# Env factory
# -------------------------------------------------
def make_env(rank: int, seed: int = 0, use_asr: bool = USE_ASR, use_shaped_reward: bool = USE_SHAPED_REWARD):
    def _init():
        env = gym.make(
            ENV_ID,
            tasks_to_complete=TASKS,
        )
        if use_shaped_reward:
            env = DenseRewardWrapper(env, tasks=TASKS)
        env = FlattenObsWrapper(env)
        if use_asr:
            env = ASRObsWrapper(env)
        env = AugmentedObsWrapper(env)
        env = KitchenSuccessInfoWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# -------------------------------------------------
# Sanity checks
# -------------------------------------------------
def sanity_check_env():
    print("\n========== SANITY CHECK ==========")
    env = make_env(rank=0, seed=SEED, use_asr=USE_ASR, use_shaped_reward=USE_SHAPED_REWARD)()

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}, dtype={obs.dtype}")
    expected_dim = 52 if USE_ASR else 74
    assert isinstance(obs, np.ndarray), "Expected flat Box observation"
    assert obs.shape == (expected_dim,), f"Expected obs dim {expected_dim}, got {obs.shape}"

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"One step reward: {reward}, terminated={terminated}, truncated={truncated}")

    print("Base env type:", type(env.unwrapped))
    print("==================================\n")
    env.close()


def train(
    run_name: str = "sac_shaped_asr_run_1",
    use_asr: bool = True,
    use_shaped_reward: bool = True,
):
    obs_label = "ASR-52dim" if use_asr else "FULL-74dim"
    rew_label = "SHAPED" if use_shaped_reward else "SPARSE"
    print(f"Run: {run_name}  |  Obs: {obs_label}  |  Reward: {rew_label}")
    print("Setting up vectorised environments...")

    best_model_dir = f"./sac_franka_best_{run_name}/"
    eval_log_dir = f"./sac_franka_eval_logs_{run_name}/"
    checkpoint_dir = f"./sac_franka_checkpoints_{run_name}/"

    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_env = SubprocVecEnv([
        make_env(rank=i, seed=SEED, use_asr=use_asr, use_shaped_reward=use_shaped_reward)
        for i in range(N_ENVS)
    ])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([
        make_env(rank=10_000, seed=SEED, use_asr=use_asr, use_shaped_reward=use_shaped_reward)
    ])
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=max(100_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(500_000 // N_ENVS, 1),
        save_path=checkpoint_dir,
        name_prefix=f"sac_{run_name}",
    )

    info_stats_callback = InfoStatsCallback()
    training_log_callback = TrainingLogCallback(log_freq=2048)

    callbacks = CallbackList([
        training_log_callback,
        info_stats_callback,
        eval_callback,
        checkpoint_callback,
    ])

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=250_000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=8,
        ent_coef="auto",
        tensorboard_log=TB_LOG_DIR,
        verbose=1,
        device="cuda",
        seed=SEED,
    )

    print(f"\nPolicy architecture:\n{model.policy}\n")
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} parallel envs...\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    save_path = f"sac_franka_kitchen_{run_name}_final"
    model.save(save_path)
    print(f"Training finished. Saved model to {save_path}.zip")

    train_env.close()
    eval_env.close()
    return training_log_callback


def plot_results(episode_returns: list[float], title_suffix: str = ""):
    if not episode_returns:
        print("No episode data to plot.")
        return
    returns = np.array(episode_returns, dtype=np.float32)
    window = min(20, len(returns))
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(returns, color="teal", linewidth=1, alpha=0.35, label="Episode Return")
    plt.plot(
        range(window - 1, len(returns)),
        smoothed,
        color="teal",
        linewidth=2.5,
        label=f"Smoothed (window={window})",
    )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"SB3 SAC — Franka Kitchen {title_suffix}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fname = f"learning_curve_sac{title_suffix.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Plot saved -> {fname}")


def evaluate(
    model_path: str = "sac_franka_final",
    num_episodes: int = 5,
    record_video: bool = True,
    use_asr: bool = True,
    use_shaped_reward: bool = True,
):
    gym.register_envs(gymnasium_robotics)
    render_mode = "rgb_array" if record_video else "human"
    video_folder = "franka_kitchen_eval_videos_sac_reduced_dim_asr" if record_video else None

    env = gym.make(
        ENV_ID,
        tasks_to_complete=TASKS,
        render_mode=render_mode,
    )
    if use_shaped_reward:
        env = DenseRewardWrapper(env, tasks=TASKS)
    env = FlattenObsWrapper(env)
    if use_asr:
        env = ASRObsWrapper(env)
    env = AugmentedObsWrapper(env)
    env = KitchenSuccessInfoWrapper(env)

    if record_video:
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda _: True)

    model = SAC.load(model_path, env=env)
    print(f"\nLoaded: {model_path}  |  ASR={use_asr}  |  Shaped={use_shaped_reward}")
    print(f"Running {num_episodes} evaluation episodes...\n")

    ep_returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_ret, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if not record_video:
                time.sleep(0.07)
            ep_ret += reward
            done = terminated or truncated
        ep_returns.append(ep_ret)
        print(f"  Episode {ep + 1}: return = {ep_ret:.3f}")

    env.close()
    print(f"\nMean return : {np.mean(ep_returns):.3f}")
    print(f"Std  return : {np.std(ep_returns):.3f}")
    if record_video:
        print(f"Videos saved -> {os.path.abspath(video_folder)}")


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_training", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument(
        "--model_path",
        type=str,
        default="sac_franka_best_sac_shaped_reduced_dim_asr_run_1/best_model.zip",
    )
    args = parser.parse_args()

    if args.run_training:
        log_cb = train(
            run_name="sac_shaped_reduced_dim_asr_run_1",
            use_asr=True,
            use_shaped_reward=True,
        )
        plot_results(
            log_cb.episode_returns,
            title_suffix="Shaped ASR Augmented with DIM Reduction",
        )
    else:
        evaluate(
            model_path=args.model_path,
            num_episodes=10,
            record_video=args.record_video,
            use_asr=True,
            use_shaped_reward=True,
        )