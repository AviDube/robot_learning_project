import gymnasium as gym
import gymnasium_robotics
import numpy as np
import os
import matplotlib.pyplot as plt

from gymnasium.wrappers import RecordVideo
from gymnasium.spaces import Box

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from kitchen_dense_reward import KitchenDenseRewardConfig, KitchenDenseRewardWrapper

# ─────────────────────────────────────────────
# OBSERVATION WRAPPER (imported)
# ─────────────────────────────────────────────
from flatten_obs_wrapper import FlattenObsWrapper


# ─────────────────────────────────────────────
# ENVIRONMENT CREATION
# SB3 expects a callable that returns a fresh env.
# ─────────────────────────────────────────────

TASKS = ["kettle"]

DENSE_REWARD_CONFIG = KitchenDenseRewardConfig(
    goal_epsilon=0.3,
    success_bonus=150.0,
    goal_distance_weight=10.0,
    goal_progress_weight=30.0,
    goal_progress_clip=0.05,
    arm_distance_weight=0.15,
    arm_distance_clip=0.7,
    arm_gate_width=0.25,
    arm_progress_weight=2.0,
    arm_progress_clip=0.05,
    action_penalty_weight=0.0,
    time_penalty_weight=0.02,
    timeout_failure_penalty=40.0,
    success_task_name="kettle",
)


class KitchenSuccessInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, target_tasks=None):
        super().__init__(env)
        tasks = target_tasks if target_tasks is not None else ["kettle"]
        self._target_tasks = set(tasks)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if done and "is_success" not in info:
            remaining = info.get("tasks_to_complete", None)
            if isinstance(remaining, (list, tuple, set)):
                info["is_success"] = float(self._target_tasks.isdisjoint(set(remaining)))
            else:
                completed = info.get("completed_tasks", None)
                if isinstance(completed, (list, tuple, set)):
                    info["is_success"] = float(self._target_tasks.issubset(set(completed)))

        return obs, reward, terminated, truncated, info


USE_FLATTEN_OBS_WRAPPER = True  # Set to False to disable flattening

def make_env(render_mode=None):
    """Creates and wraps a single Franka Kitchen env."""
    def _init():
        gym.register_envs(gymnasium_robotics)
        env = gym.make(
            'FrankaKitchen-v1',
            tasks_to_complete=TASKS,
            render_mode=render_mode,
        )
        env = KitchenDenseRewardWrapper(env, config=DENSE_REWARD_CONFIG)
        env = KitchenSuccessInfoWrapper(env, target_tasks=TASKS)
        if USE_FLATTEN_OBS_WRAPPER:
            env = FlattenObsWrapper(env)
        env = Monitor(env)
        return env
    return _init

# ─────────────────────────────────────────────
# CUSTOM CALLBACK — TensorBoard snapshot every 10k steps
# ─────────────────────────────────────────────
 
class TensorboardCallback(BaseCallback):
    """
    Logs a rich set of training metrics to TensorBoard every LOG_FREQ steps.
 
    Metrics written
    ───────────────
    <run_name>/ep_rew_mean    — mean episode return over the logging window
    <run_name>/ep_rew_min     — min episode return in the window
    <run_name>/ep_rew_max     — max episode return in the window
    <run_name>/ep_len_mean    — mean episode length over the logging window
    <run_name>/episodes_total — cumulative episode count
 
    Using run_name as the tag prefix means multiple runs appear as
    separate, labelled series in TensorBoard for direct comparison.
    """
 
    def __init__(self, log_freq: int = 10_000, run_name: str = "run"):
        super().__init__(verbose=0)
        self.log_freq        = log_freq
        self.run_name        = run_name
        self._window_returns: list[float] = []
        self._window_lengths: list[float] = []
        self._total_episodes: int         = 0
 
    def _on_step(self) -> bool:
        # Monitor injects an "episode" key into info when an episode ends
        for info in self.locals["infos"]:
            if "episode" in info:
                self._window_returns.append(float(info["episode"]["r"]))
                self._window_lengths.append(float(info["episode"]["l"]))
                self._total_episodes += 1
 
        if self.num_timesteps % self.log_freq == 0 and self._window_returns:
            rew_arr = np.array(self._window_returns)
            len_arr = np.array(self._window_lengths)
 
            self.logger.record(f"{self.run_name}/ep_rew_mean",    float(rew_arr.mean()))
            self.logger.record(f"{self.run_name}/ep_rew_min",     float(rew_arr.min()))
            self.logger.record(f"{self.run_name}/ep_rew_max",     float(rew_arr.max()))
            self.logger.record(f"{self.run_name}/ep_len_mean",    float(len_arr.mean()))
            self.logger.record(f"{self.run_name}/episodes_total", self._total_episodes)
            self.logger.dump(self.num_timesteps)  # flush to disk at this step
 
            # Reset window for next interval
            self._window_returns.clear()
            self._window_lengths.clear()
 
        return True


# ─────────────────────────────────────────────
# CUSTOM CALLBACK — live training log
# ─────────────────────────────────────────────

class TrainingLogCallback(BaseCallback):
    """
    Logs mean episode return to console every `log_freq` timesteps.
    SB3's Monitor wrapper populates `infos` with episode stats automatically.
    """
    def __init__(self, log_freq: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq     = log_freq
        self.episode_returns: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_returns.append(info["episode"]["r"])

        if self.n_calls % self.log_freq == 0 and self.episode_returns:
            mean_ret = np.mean(self.episode_returns[-20:])
            print(
                f"  Steps: {self.num_timesteps:>8,} | "
                f"Mean Return (last 20 eps): {mean_ret:>8.3f}"
            )
        return True


class InfoStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        success_vals = []
        for info in infos:
            if "is_success" in info:
                success_vals.append(float(info["is_success"]))

        if success_vals:
            self.logger.record("rollout/is_success_mean", float(np.mean(success_vals)))
        return True


class KitchenEvalCallback(EvalCallback):
    """
    Eval callback that logs reward components and saves best model by success rate.
    Tie-breaker: mean reward.
    """

    def __init__(self, *args, best_model_dir: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_model_dir = best_model_dir
        os.makedirs(self.best_model_dir, exist_ok=True)
        self.best_success_rate = -np.inf
        self.best_success_mean_reward = -np.inf
        self.best_success_model_path = os.path.join(self.best_model_dir, "best_model.zip")
        self._eval_success_terms: list[float] = []
        self._eval_goal_distance_terms: list[float] = []
        self._eval_arm_distance_terms: list[float] = []

    def _log_success_callback(self, locals_, globals_) -> None:
        super()._log_success_callback(locals_, globals_)

        info = locals_.get("info", None)
        if not isinstance(info, dict):
            return

        components = info.get("reward_components", None)
        if not isinstance(components, dict):
            return

        success_term = components.get("success", None)
        goal_distance = components.get("goal_distance", None)
        arm_distance = components.get("arm_distance", None)

        if success_term is not None:
            self._eval_success_terms.append(float(success_term))
        if goal_distance is not None:
            self._eval_goal_distance_terms.append(float(goal_distance))
        if arm_distance is not None:
            self._eval_arm_distance_terms.append(float(arm_distance))

    def _on_step(self) -> bool:
        is_eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if is_eval_step:
            self._eval_success_terms.clear()
            self._eval_goal_distance_terms.clear()
            self._eval_arm_distance_terms.clear()

        continue_training = super()._on_step()

        if is_eval_step:
            curr_success = None
            last_success_rate = getattr(self, "last_success_rate", None)
            if last_success_rate is not None:
                curr_success = float(last_success_rate)

            if curr_success is None:
                success_buffer = getattr(self, "_is_success_buffer", None)
                if isinstance(success_buffer, (list, tuple, np.ndarray)) and len(success_buffer) > 0:
                    curr_success = float(np.mean(success_buffer))

            if curr_success is None:
                evaluations_successes = getattr(self, "evaluations_successes", None)
                if evaluations_successes is not None and len(evaluations_successes) > 0:
                    last_eval_successes = np.asarray(evaluations_successes[-1], dtype=np.float32).reshape(-1)
                    if last_eval_successes.size > 0:
                        curr_success = float(np.mean(last_eval_successes))

            if curr_success is None:
                curr_success = 0.0

            curr_mean_reward = float(getattr(self, "last_mean_reward", -np.inf))
            improved = (
                curr_success > self.best_success_rate
                or (
                    np.isclose(curr_success, self.best_success_rate)
                    and curr_mean_reward > self.best_success_mean_reward
                )
            )
            if improved and self.model is not None:
                self.best_success_rate = curr_success
                self.best_success_mean_reward = curr_mean_reward
                self.model.save(self.best_success_model_path)
                print(
                    "New best model (by success): "
                    f"success={curr_success:.2%}, mean_reward={curr_mean_reward:.2f}"
                )

            success_mean = None
            goal_distance_mean = None
            arm_distance_mean = None
            if self._eval_success_terms:
                success_mean = float(np.mean(self._eval_success_terms))
                self.logger.record("eval/reward_success_mean", success_mean)
            if self._eval_goal_distance_terms:
                goal_distance_mean = float(np.mean(self._eval_goal_distance_terms))
                self.logger.record("eval/reward_goal_distance_mean", goal_distance_mean)
            if self._eval_arm_distance_terms:
                arm_distance_mean = float(np.mean(self._eval_arm_distance_terms))
                self.logger.record("eval/reward_arm_distance_mean", arm_distance_mean)

            if (
                success_mean is not None
                and goal_distance_mean is not None
                and arm_distance_mean is not None
            ):
                print(
                    "Eval reward components: "
                    f"success={success_mean:.4f}, "
                    f"goal_distance={goal_distance_mean:.4f}, "
                    f"arm_distance={arm_distance_mean:.4f}"
                )
            self.logger.dump(self.num_timesteps)

        return continue_training

TOTAL_TIMESTEPS = 2_000_000
 
# ── Parallel env scaling for RTX 5090 ────────────────────────────────────────
import os
_logical_cores = os.cpu_count() or 8
N_ENVS = 8
print(f"Auto-selected N_ENVS={N_ENVS} for {_logical_cores} logical CPU cores")
 
# Also scale the batch size up so minibatch count stays reasonable.
# PPO paper recommendation: total_steps / batch_size >= 32 minibatches/update.
_DYNAMIC_BATCH = 256


# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────

PPO_KWARGS = dict(
    # ── Core PPO ────────────────────────────────────────────────────────────
    n_steps          = 2048,    # rollout length T (steps per env before update)
    batch_size       = _DYNAMIC_BATCH,      # minibatch size for gradient updates
    n_epochs         = 10,      # passes over collected data per update
    gamma            = 0.99,    # discount factor γ
    gae_lambda       = 0.95,    # GAE λ (bias-variance trade-off)
    clip_range       = 0.2,     # PPO ε — how much the policy can change per update
    # ── Loss coefficients ───────────────────────────────────────────────────
    ent_coef         = 0.0,     # entropy bonus (raise to 0.01 if agent gets stuck)
    vf_coef          = 0.5,     # value-function loss weight
    max_grad_norm    = 0.5,     # gradient clipping norm
    # ── Optimiser ───────────────────────────────────────────────────────────
    learning_rate    = 3e-4,    # Adam LR (try 1e-4 if training is unstable)
    # ── Network architecture ────────────────────────────────────────────────
    policy_kwargs    = dict(
        net_arch = dict(pi=[256, 256], vf=[256, 256]),
    ),
    verbose          = 0,
    tensorboard_log  = "./ppo_franka_tb/",  # optional: run `tensorboard --logdir ./ppo_franka_tb`
)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train(run_name: str = "ppo_run_1"):
    """
    run_name  labels this run inside TensorBoard.
    Use a different name each time (e.g. "ppo_lr3e4", "ppo_ent01") so
    multiple runs appear as separate series for comparison.
    """
    print(f"Run: {run_name}")
    print("Setting up vectorised environments…")
 
    # SubprocVecEnv spawns each env in its own process — true parallelism.
    # Requires the script to be run under `if __name__ == '__main__'` (already done).
    vec_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)], start_method='fork')
    vec_env = VecMonitor(vec_env)  # aggregates episode stats across envs
 
    # Separate single env used by EvalCallback for unbiased evaluation
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecMonitor(eval_env)
 
    # ── Callbacks ─────────────────────────────────────────────────────────────
    log_callback = TrainingLogCallback(log_freq=2048)
    info_stats_callback = InfoStatsCallback()
 
    tb_callback  = TensorboardCallback(log_freq=10_000, run_name=run_name)
 
    eval_callback = KitchenEvalCallback(
        eval_env,
        best_model_save_path = None,
        best_model_dir       = "./ppo_franka_best/",
        log_path             = "./ppo_franka_eval_logs/",
        eval_freq            = 10_000,   # evaluate every N timesteps
        n_eval_episodes      = 5,
        deterministic        = True,     # greedy policy during eval
        render               = False,
        verbose              = 1,
    )
 
    checkpoint_callback = CheckpointCallback(
        save_freq  = 50_000,             # save a checkpoint every N timesteps
        save_path  = "./ppo_franka_checkpoints/",
        name_prefix= "ppo_franka",
        verbose    = 1,
    )
 
    # ── Model ─────────────────────────────────────────────────────────────────
    # MLP networks are small (2x256 layers) — CPU is faster than GPU here
    # because per-batch compute is trivial and GPU transfer overhead dominates.
    # The 5090 is better spent on CNN/transformer policies or SAC.
    model = PPO("MlpPolicy", vec_env, **PPO_KWARGS, device="cpu")
 
    print(f"\nPolicy architecture:\n{model.policy}\n")
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} parallel envs…\n")
 
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [log_callback, info_stats_callback, tb_callback, eval_callback, checkpoint_callback],
        progress_bar    = True,   # requires `pip install rich`
    )
 
    model.save("ppo_franka_kitchen_final")
    print("\nModel saved → ppo_franka_kitchen_final.zip")
 
    vec_env.close()
    eval_env.close()
 
    return log_callback


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_results(episode_returns: list[float]):
    if not episode_returns:
        print("No episode data to plot.")
        return

    returns = np.array(episode_returns)

    window = min(20, len(returns))
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(returns,  color='teal',  linewidth=1, alpha=0.35, label='Episode Return')
    plt.plot(range(window - 1, len(returns)), smoothed,
             color='teal', linewidth=2.5, label=f'Smoothed (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('SB3 PPO — Franka Kitchen Learning Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('ppo_sb3_learning_curve.png', dpi=150)
    plt.show()
    print("Plot saved → ppo_sb3_learning_curve.png")


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(
    model_path : str = "ppo_franka_kitchen_final",
    num_episodes: int = 5,
    record_video: bool = True,
):
    """
    Load a saved SB3 PPO model and run greedy evaluation episodes.
    Optionally records videos to franka_kitchen_eval_videos/.
    """
    gym.register_envs(gymnasium_robotics)

    render_mode  = 'rgb_array' if record_video else 'human'
    video_folder = "franka_kitchen_eval_videos_parrallel"

    env = gym.make('FrankaKitchen-v1',
                   tasks_to_complete=TASKS,
                   render_mode=render_mode)
    env = KitchenDenseRewardWrapper(env, config=DENSE_REWARD_CONFIG)
    env = FlattenObsWrapper(env)

    if record_video:
        env = RecordVideo(env, video_folder=video_folder,
                          episode_trigger=lambda _: True)

    model = PPO.load(model_path, env=env)
    print(f"\nLoaded model from {model_path}")
    print(f"Running {num_episodes} evaluation episodes…\n")

    ep_returns = []
    for ep in range(num_episodes):
        obs, _   = env.reset()
        ep_ret   = 0.0
        done     = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done    = terminated or truncated

        ep_returns.append(ep_ret)
        print(f"  Episode {ep + 1}: return = {ep_ret:.3f}")

    env.close()
    print(f"\nMean return : {np.mean(ep_returns):.3f}")
    print(f"Std  return : {np.std(ep_returns):.3f}")
    if record_video:
        print(f"Videos saved → {os.path.abspath(video_folder)}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_name = "ppo_run_1"
    log_callback = train(run_name=run_name)
    plot_results(log_callback.episode_returns)

    # Uncomment to evaluate the final model after training:
    # evaluate("ppo_franka_kitchen_final", num_episodes=5, record_video=True)
