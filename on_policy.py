import gymnasium as gym
import gymnasium_robotics
import numpy as np
import os
import matplotlib.pyplot as plt

from gymnasium.wrappers import RecordVideo
from gymnasium.spaces import Box

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# ─────────────────────────────────────────────────────────────────────────────
# FRANKA KITCHEN OBS[59] LAYOUT  (from robotics.farama.org docs)
# ─────────────────────────────────────────────────────────────────────────────
#
#  ROBOT JOINT POSITIONS  (indices 0-8)   — 9 dims
#    0-6  : panda0_joint1 … panda0_joint7   (arm angles, rad)
#    7    : r_gripper_finger_joint           (slide, m)
#    8    : l_gripper_finger_joint           (slide, m)
#
#  ROBOT JOINT VELOCITIES (indices 9-17)  — 9 dims
#
#  OBJECT / KITCHEN POSITIONS (indices 18-45) — 28 dims
#    18   knob_Joint_1   (bottom-right burner knob)
#    19   bottom_right_burner slide
#    20   knob_Joint_2   (bottom-left burner knob)
#    21   bottom_left_burner slide
#    22   knob_Joint_3   (top-right burner knob)
#    23   top_right_burner slide
#    24   knob_Joint_4   (top-left burner knob)
#    25   top_left_burner slide
#    26   light_switch slide
#    27   light_joint hinge
#    28   slide_cabinet translation
#    29   left hinge-cabinet rotation
#    30   right hinge-cabinet rotation
#    31   microwave hinge angle          ← used for microwave task
#    32   kettle x position              ← used for kettle task
#    33   kettle y position
#    34   kettle z position
#    35   kettle qw
#    36   kettle qx
#    37   kettle qy
#    38   kettle qz
#    39-45: further object joint states
#
#  OBJECT / KITCHEN VELOCITIES (indices 46-58) — 13 dims
#
# ─────────────────────────────────────────────────────────────────────────────

ASR_INDICES = list(range(0, 9)) + list(range(18, 46))   # 37 dims


# ─────────────────────────────────────────────────────────────────────────────
# TASK GOAL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# Each task specifies which RAW obs indices matter and what value(s) they
# must reach. Sourced from the Farama Franka Kitchen docs goal table.
# Used by DenseRewardWrapper to compute per-task progress distances.
# ─────────────────────────────────────────────────────────────────────────────

TASK_GOALS = {
    # obs[31] = microwave hinge — goal is 0.37 rad open
    "microwave": {
        "obs_indices": [31],
        "goal":        np.array([0.37], dtype=np.float32),
    },
    # obs[32:35] = kettle (x, y, z) — goal is top-left burner position
    # We use only xyz (ignore quaternion) for a clean Euclidean distance.
    "kettle": {
        "obs_indices": [32, 33, 34],
        "goal":        np.array([-0.23, 0.75, 1.62], dtype=np.float32),
    },
    # Additional tasks — uncomment if you add them to TASKS list
    # "bottom burner": {
    #     "obs_indices": [18, 19],
    #     "goal":        np.array([-0.88, -0.01], dtype=np.float32),
    # },
    # "top burner": {
    #     "obs_indices": [22, 23],
    #     "goal":        np.array([-0.92, -0.01], dtype=np.float32),
    # },
    # "light switch": {
    #     "obs_indices": [26, 27],
    #     "goal":        np.array([-0.69, -0.05], dtype=np.float32),
    # },
    # "slide cabinet": {
    #     "obs_indices": [28],
    #     "goal":        np.array([0.37], dtype=np.float32),
    # },
    # "hinge cabinet": {
    #     "obs_indices": [29, 30],
    #     "goal":        np.array([0.0, 1.45], dtype=np.float32),
    # },
}

# Tasks the agent must complete — must match gym.make() tasks_to_complete
TASKS = ['microwave', 'kettle']


# ─────────────────────────────────────────────────────────────────────────────
# REWARD SHAPING  —  DenseRewardWrapper
# ─────────────────────────────────────────────────────────────────────────────
#
# Replaces the sparse binary reward with a four-component signal:
#
#  1. COMPLETION BONUS  (+COMPLETION_BONUS per task, preserved from env)
#       Keeps the original signal intact so the agent still targets goals.
#
#  2. NORMALISED DISTANCE REWARD  (per task, per step)
#       r_dist = -dist / max_dist   ∈ [-1, 0]
#       Normalised so the scale doesn't depend on the task's unit (rad vs m).
#       Drives the agent toward each goal at every timestep.
#
#  3. PROGRESS REWARD  (per task, per step)
#       r_prog = W_PROGRESS * (prev_dist - curr_dist)
#       Pays out only when the agent is actively closing the gap.
#       Harder to hack than a raw distance term.
#
#  4. END-EFFECTOR PROXIMITY BONUS  (per task, once per task)
#       r_ee = W_EE_PROXIMITY  when  ee_dist < EE_THRESHOLD
#       Encourages the arm to first move near the target object before
#       trying to manipulate it — critical for long-horizon tasks.
#
# All weights are tunable at the top of the class.
# ─────────────────────────────────────────────────────────────────────────────

# Approximate resting-state distances used for normalisation.
# These are the typical distances at episode start; tune if needed.
_TASK_MAX_DIST = {
    "microwave": 0.37,   # rad — full range from closed to goal
    "kettle":    1.20,   # metres — rough start-to-goal Euclidean distance
}

# Approximate kettle and microwave handle world positions (xyz) used for
# end-effector proximity bonus. These are rough centroids — refine if you
# have access to MuJoCo site positions.
_TASK_EE_TARGET_OBS_INDICES = {
    "microwave": [31],          # 1-D hinge; we use a scalar EE target
    "kettle":    [32, 33, 34],  # xyz of kettle base
}


class DenseRewardWrapper(gym.Wrapper):
    """
    Augments Franka Kitchen's sparse binary reward with dense shaping.

    Operates on the RAW (pre-flatten, pre-ASR) Dict observation so it can
    read the full 59-dim 'observation' array directly.  The shaped reward
    is computed inside step() before the obs is passed to any downstream
    wrapper.

    Parameters
    ----------
    tasks           : list of task names matching TASK_GOALS keys
    w_distance      : weight for the normalised distance term
    w_progress      : weight for the delta-distance progress term
    w_ee_proximity  : one-off bonus when end-effector is near the object
    ee_threshold    : distance (m or rad) at which ee-proximity fires
    completion_bonus: reward added on top of env's own signal per completed task
    """

    # ── Tunable weights ───────────────────────────────────────────────────────
    W_DISTANCE      = 0.3   # dense distance signal strength
    W_PROGRESS      = 2.0   # progress (delta-dist) bonus multiplier — strong signal
    W_EE_PROXIMITY  = 0.5   # one-off bonus for moving EE near object
    EE_THRESHOLD    = 0.15  # metres — EE must be within this to earn proximity bonus
    COMPLETION_BONUS = 1.0  # stacked on top of env's own +1 for extra emphasis
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, env, tasks: list[str] = TASKS):
        super().__init__(env)
        self.tasks = [t for t in tasks if t in TASK_GOALS]
        if not self.tasks:
            raise ValueError(f"None of {tasks} found in TASK_GOALS. "
                             f"Available: {list(TASK_GOALS.keys())}")

        # Per-task state reset at each episode
        self._prev_dist:     dict[str, float] = {}
        self._ee_bonus_given: dict[str, bool]  = {}
        self._task_done:      dict[str, bool]  = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _task_distance(self, raw_obs: np.ndarray, task: str) -> float:
        """Euclidean distance between current and goal joint values."""
        cfg     = TASK_GOALS[task]
        current = raw_obs[cfg["obs_indices"]].astype(np.float32)
        return float(np.linalg.norm(current - cfg["goal"]))

    def _ee_position(self, raw_obs: np.ndarray) -> np.ndarray:
        """
        Approximate end-effector xyz from forward kinematics proxy.
        We use the last 3 arm joint angles (obs 4,5,6) as a cheap proxy.
        For higher accuracy, replace with a proper FK call or a MuJoCo
        site readout via env.unwrapped.data.site_xpos.
        """
        try:
            # Prefer the real EE site if accessible
            return self.env.unwrapped.data.site_xpos[
                self.env.unwrapped.model.site('end_effector').id
            ].copy()
        except Exception:
            # Fallback: use gripper joint positions as a rough proxy
            return raw_obs[7:9]   # r+l gripper slide positions (1-D fallback)

    def _object_position(self, raw_obs: np.ndarray, task: str) -> np.ndarray:
        """Returns the current positional state of the task's target object."""
        return raw_obs[TASK_GOALS[task]["obs_indices"]].astype(np.float32)

    # ── core override ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw_obs = obs["observation"]

        for task in self.tasks:
            self._prev_dist[task]      = self._task_distance(raw_obs, task)
            self._ee_bonus_given[task] = False
            self._task_done[task]      = False

        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        raw_obs = obs["observation"]

        shaped_reward = 0.0

        for task in self.tasks:
            if self._task_done[task]:
                continue   # don't double-count completed tasks

            curr_dist = self._task_distance(raw_obs, task)
            prev_dist = self._prev_dist[task]
            max_dist  = _TASK_MAX_DIST.get(task, 1.0)

            # ── 1. Completion bonus ──────────────────────────────────────────
            task_just_completed = (
                curr_dist < 0.05 * max_dist   # within 5% of goal range
            )
            if task_just_completed and not self._task_done[task]:
                shaped_reward       += self.COMPLETION_BONUS
                self._task_done[task] = True

            # ── 2. Normalised distance reward ────────────────────────────────
            norm_dist      = curr_dist / max(max_dist, 1e-6)
            shaped_reward += self.W_DISTANCE * (1.0 - norm_dist)  # ∈ [0, W_DIST]

            # ── 3. Progress reward (delta distance) ──────────────────────────
            progress       = prev_dist - curr_dist          # positive = improvement
            shaped_reward += self.W_PROGRESS * progress

            # ── 4. End-effector proximity bonus (one-off per task) ───────────
            if not self._ee_bonus_given[task]:
                try:
                    ee_pos  = self._ee_position(raw_obs)
                    obj_pos = self._object_position(raw_obs, task)
                    # Only compare positional dims that align (both must be xyz)
                    min_len = min(len(ee_pos), len(obj_pos))
                    ee_dist = float(np.linalg.norm(ee_pos[:min_len] - obj_pos[:min_len]))
                    if ee_dist < self.EE_THRESHOLD:
                        shaped_reward            += self.W_EE_PROXIMITY
                        self._ee_bonus_given[task] = True
                except Exception:
                    pass   # silently skip if EE readout fails

            self._prev_dist[task] = curr_dist

        # Combine: env's original sparse signal + all shaped terms
        total_reward = env_reward + shaped_reward

        # Log shaped components to info for debugging / TensorBoard
        info["shaped_reward"]   = float(shaped_reward)
        info["original_reward"] = float(env_reward)

        return obs, total_reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION WRAPPERS  (unchanged from on_policy_asr.py)
# ─────────────────────────────────────────────────────────────────────────────

class FlattenObsWrapper(gym.ObservationWrapper):
    """Flattens Franka Kitchen's Dict obs into a single flat 59-dim Box."""

    INCLUDE_GOALS = False

    def __init__(self, env):
        super().__init__(env)
        self._box_keys = [
            k for k, s in env.observation_space.spaces.items()
            if hasattr(s, 'shape') and s.shape is not None
        ]
        if not self._box_keys:
            raise RuntimeError("No valid Box spaces found in observation dict.")

        flat_dim = sum(
            int(np.prod(env.observation_space.spaces[k].shape))
            for k in self._box_keys
        )
        self._goal_dim = 0
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    @staticmethod
    def _flatten_nested(value) -> np.ndarray:
        if isinstance(value, dict):
            return np.concatenate([
                FlattenObsWrapper._flatten_nested(v) for v in value.values()
            ]).astype(np.float32)
        return np.asarray(value, dtype=np.float32).flatten()

    def observation(self, obs) -> np.ndarray:
        parts = [obs[k].flatten().astype(np.float32) for k in self._box_keys]
        return np.concatenate(parts)


class ASRObsWrapper(gym.ObservationWrapper):
    """Reduces 59-dim obs to 37-dim ASR by dropping velocity indices."""

    def __init__(self, env, asr_indices: list[int] = ASR_INDICES):
        super().__init__(env)
        self._asr_indices = np.array(asr_indices, dtype=np.int32)
        full_dim = env.observation_space.shape[0]
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(len(self._asr_indices),), dtype=np.float32,
        )
        print(
            f"[ASRObsWrapper] {full_dim}D → {len(self._asr_indices)}D  "
            f"(dropped {full_dim - len(self._asr_indices)} velocity dims)"
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[self._asr_indices]


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────
# Wrapper stacking order (innermost → outermost):
#   FrankaKitchen-v1  (raw Dict obs, sparse binary reward)
#       └─ DenseRewardWrapper   ← reads raw Dict obs, augments reward
#           └─ FlattenObsWrapper ← Dict → flat 59-dim Box
#               └─ ASRObsWrapper ← 59-dim → 37-dim (drop velocities)
#                   └─ Monitor
#
# DenseRewardWrapper MUST wrap the raw env before FlattenObsWrapper so it
# can read obs["observation"] as a plain numpy array.
# ─────────────────────────────────────────────────────────────────────────────

def make_env(render_mode=None, use_asr: bool = True, use_shaped_reward: bool = True):
    def _init():
        gym.register_envs(gymnasium_robotics)
        env = gym.make(
            'FrankaKitchen-v1',
            tasks_to_complete=TASKS,
            render_mode=render_mode,
        )
        if use_shaped_reward:
            env = DenseRewardWrapper(env, tasks=TASKS)   # ← before flatten!
        env = FlattenObsWrapper(env)
        if use_asr:
            env = ASRObsWrapper(env)
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class TensorboardCallback(BaseCallback):
    """Logs reward stats + shaped reward breakdown every log_freq steps."""

    def __init__(self, log_freq: int = 10_000, run_name: str = "run"):
        super().__init__(verbose=0)
        self.log_freq        = log_freq
        self.run_name        = run_name
        self._window_returns:  list[float] = []
        self._window_lengths:  list[float] = []
        self._window_shaped:   list[float] = []
        self._window_original: list[float] = []
        self._total_episodes:  int         = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self._window_returns.append(float(info["episode"]["r"]))
                self._window_lengths.append(float(info["episode"]["l"]))
                self._total_episodes += 1
            if "shaped_reward" in info:
                self._window_shaped.append(float(info["shaped_reward"]))
            if "original_reward" in info:
                self._window_original.append(float(info["original_reward"]))

        if self.num_timesteps % self.log_freq == 0 and self._window_returns:
            rew = np.array(self._window_returns)
            self.logger.record(f"{self.run_name}/ep_rew_mean",    float(rew.mean()))
            self.logger.record(f"{self.run_name}/ep_rew_min",     float(rew.min()))
            self.logger.record(f"{self.run_name}/ep_rew_max",     float(rew.max()))
            self.logger.record(f"{self.run_name}/ep_len_mean",    float(np.mean(self._window_lengths)))
            self.logger.record(f"{self.run_name}/episodes_total", self._total_episodes)

            if self._window_shaped:
                self.logger.record(f"{self.run_name}/shaped_rew_mean",
                                   float(np.mean(self._window_shaped)))
            if self._window_original:
                self.logger.record(f"{self.run_name}/original_rew_mean",
                                   float(np.mean(self._window_original)))

            self.logger.dump(self.num_timesteps)
            self._window_returns.clear()
            self._window_lengths.clear()
            self._window_shaped.clear()
            self._window_original.clear()

        return True


class TrainingLogCallback(BaseCallback):
    """Logs mean return to console every log_freq steps."""

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
            print(f"  Steps: {self.num_timesteps:>8,} | Mean Return (last 20 eps): {mean_ret:>8.3f}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS = 2_000_000

_logical_cores  = os.cpu_count() or 8
N_ENVS          = 8
_DYNAMIC_BATCH  = 256

print(f"Auto-selected N_ENVS={N_ENVS} for {_logical_cores} logical CPU cores")

PPO_KWARGS = dict(
    n_steps       = 2048,
    batch_size    = _DYNAMIC_BATCH,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,    # slightly raised from 0.0 → encourages exploration
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    learning_rate = 3e-4,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    verbose       = 0,
    tensorboard_log = "./ppo_franka_tb/",
)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    run_name          : str  = "ppo_shaped_asr_run_1",
    use_asr           : bool = True,
    use_shaped_reward : bool = True,
):
    obs_label = "ASR-37dim" if use_asr else "FULL-59dim"
    rew_label = "SHAPED"    if use_shaped_reward else "SPARSE"
    print(f"Run: {run_name}  |  Obs: {obs_label}  |  Reward: {rew_label}")
    print("Setting up vectorised environments…")

    vec_env = SubprocVecEnv(
        [make_env(use_asr=use_asr, use_shaped_reward=use_shaped_reward) for _ in range(N_ENVS)],
        start_method='fork',
    )
    vec_env = VecMonitor(vec_env)

    eval_env = DummyVecEnv([make_env(use_asr=use_asr, use_shaped_reward=use_shaped_reward)])
    eval_env = VecMonitor(eval_env)

    log_callback = TrainingLogCallback(log_freq=2048)
    tb_callback  = TensorboardCallback(log_freq=10_000, run_name=run_name)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = f"./ppo_franka_best_{run_name}/",
        log_path             = f"./ppo_franka_eval_logs_{run_name}/",
        eval_freq            = 10_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = f"./ppo_franka_checkpoints_{run_name}/",
        name_prefix = f"ppo_{run_name}",
        verbose     = 1,
    )

    model = PPO("MlpPolicy", vec_env, **PPO_KWARGS, device="cpu")

    print(f"\nPolicy architecture:\n{model.policy}\n")
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} parallel envs…\n")

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [log_callback, tb_callback, eval_callback, checkpoint_callback],
        progress_bar    = True,
    )

    save_path = f"ppo_franka_kitchen_{run_name}_final"
    model.save(save_path)
    print(f"\nModel saved → {save_path}.zip")

    vec_env.close()
    eval_env.close()
    return log_callback


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(episode_returns: list[float], title_suffix: str = ""):
    if not episode_returns:
        print("No episode data to plot.")
        return
    returns  = np.array(episode_returns)
    window   = min(20, len(returns))
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(returns, color='teal', linewidth=1, alpha=0.35, label='Episode Return')
    plt.plot(range(window - 1, len(returns)), smoothed,
             color='teal', linewidth=2.5, label=f'Smoothed (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'SB3 PPO — Franka Kitchen {title_suffix}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fname = f'learning_curve{title_suffix.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Plot saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model_path        : str  = "ppo_franka_kitchen_final",
    num_episodes      : int  = 5,
    record_video      : bool = True,
    use_asr           : bool = True,
    use_shaped_reward : bool = True,
):
    gym.register_envs(gymnasium_robotics)
    render_mode  = 'rgb_array' if record_video else 'human'
    video_folder = "franka_kitchen_eval_videos"

    env = gym.make('FrankaKitchen-v1', tasks_to_complete=TASKS, render_mode=render_mode)
    if use_shaped_reward:
        env = DenseRewardWrapper(env, tasks=TASKS)
    env = FlattenObsWrapper(env)
    if use_asr:
        env = ASRObsWrapper(env)
    if record_video:
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda _: True)

    model = PPO.load(model_path, env=env)
    print(f"\nLoaded: {model_path}  |  ASR={use_asr}  |  Shaped={use_shaped_reward}")
    print(f"Running {num_episodes} evaluation episodes…\n")

    ep_returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_ret, done = 0.0, False
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


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Shaped + ASR (recommended) ────────────────────────────────────────────
    log_cb = train(run_name="ppo_shaped_asr_run_1", use_asr=True, use_shaped_reward=True)
    plot_results(log_cb.episode_returns, title_suffix="Shaped ASR")

    # ── Ablation: sparse baseline — uncomment to compare ─────────────────────
    # log_cb_base = train(run_name="ppo_sparse_baseline", use_asr=True, use_shaped_reward=False)
    # plot_results(log_cb_base.episode_returns, title_suffix="Sparse Baseline")

    # ── Evaluate saved model ──────────────────────────────────────────────────
    # evaluate(
    #     model_path        = "./ppo_franka_best_ppo_shaped_asr_run_1/best_model.zip",
    #     num_episodes      = 10,
    #     record_video      = True,
    #     use_asr           = True,
    #     use_shaped_reward = True,
    # )