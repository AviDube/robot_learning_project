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
# Seven-component reward designed to prevent pushing and enforce grasping:
#
#  ── SHARED (all tasks) ───────────────────────────────────────────────────────
#  1. COMPLETION BONUS       (+2.0 on top of env's own +1 when task done)
#  2. NORMALISED DISTANCE    (per-step, ∈ [0, W_DISTANCE])
#  3. PROGRESS REWARD        (delta-distance, only pays when actively improving)
#
#  ── KETTLE-SPECIFIC (grasp enforcement) ──────────────────────────────────────
#  4. EE→KETTLE APPROACH     Dense reward for moving EE close to the kettle.
#                             Replaces the one-off proximity bonus with a
#                             continuous approach signal.
#
#  5. GRIPPER CLOSURE REWARD Rewards closing the fingers (small gripper gap)
#                             when the EE is already near the kettle.
#                             Fingers wide open → 0 reward.
#                             Fingers closed around object → full reward.
#                             Formula: W_GRASP * exp(-k * gripper_gap)
#                             where gripper_gap = r_finger + l_finger positions.
#
#  6. LIFT REWARD            Rewards the kettle's z position rising above a
#                             threshold (LIFT_Z). Transport credit (progress
#                             toward goal) is GATED: the agent only earns
#                             transport reward when the kettle is also lifted.
#                             This makes pushing useless — you can't earn
#                             transport reward by sliding along the floor.
#
#  7. PUSH PENALTY           Active penalty whenever the kettle moves
#                             horizontally but the gripper is open. This
#                             directly penalises the pushing behaviour.
#
# ─────────────────────────────────────────────────────────────────────────────

# Approximate resting-state distances used for normalisation.
_TASK_MAX_DIST = {
    "microwave": 0.37,   # rad  — full hinge range
    "kettle":    1.20,   # metres — start-to-goal Euclidean distance
}

# Kettle resting z height in world coordinates (approximate).
# The kettle starts on the bottom-left burner at roughly z=1.48 m.
# We require it to rise at least LIFT_Z metres above this to count as lifted.
_KETTLE_REST_Z   = 1.48   # metres (world frame, from env initial state)
_KETTLE_LIFT_MIN = 0.04   # metres above rest to count as "lifted off surface"

# Gripper finger obs indices (raw 59-dim):
#   obs[7]  = r_gripper_finger_joint (slide, metres, 0 = closed)
#   obs[8]  = l_gripper_finger_joint (slide, metres, 0 = closed)
# Total gap when wide open ≈ 0.08 m.
_GRIPPER_R_IDX = 7
_GRIPPER_L_IDX = 8
_GRIPPER_MAX_GAP = 0.08   # metres — normalisation constant

# Kettle xyz obs indices (raw 59-dim)
_KETTLE_XYZ_IDX = [32, 33, 34]


class DenseRewardWrapper(gym.Wrapper):
    """
    Grasp-aware dense reward wrapper for Franka Kitchen.

    Operates on the RAW Dict observation (before FlattenObsWrapper) so it
    can read obs["observation"][idx] directly.

    Key design decisions
    ─────────────────────
    • Transport reward for the kettle is GATED on lift height — the agent
      cannot earn distance-to-goal credit while the kettle is on the floor.
      This makes horizontal pushing unrewarded.

    • Gripper closure is rewarded continuously via an exponential kernel
      centred on a closed gripper. The agent must learn to close its fingers
      to collect this signal.

    • An explicit push penalty fires whenever the kettle moves horizontally
      while the gripper is open. This directly discourages the observed behaviour.

    • All terms are logged separately in `info` for TensorBoard debugging.

    Weights (all class-level, easy to tune)
    ─────────────────────────────────────────
    W_DISTANCE       — base distance-to-goal term weight
    W_PROGRESS       — delta-distance multiplier
    W_APPROACH       — EE-to-kettle continuous approach reward
    W_GRASP          — gripper-closure reward magnitude
    GRASP_K          — sharpness of the gripper closure exponential
    W_LIFT           — kettle lift reward weight
    W_TRANSPORT      — kettle-to-goal progress weight (gated by lift)
    W_PUSH_PENALTY   — penalty per step for pushing with open gripper
    COMPLETION_BONUS — extra reward stacked on env's +1 at task completion
    """

    # ── Shared weights ────────────────────────────────────────────────────────
    W_DISTANCE       = 0.2    # gentle always-on distance signal
    W_PROGRESS       = 1.0    # delta-distance (non-gated, applies to microwave too)
    COMPLETION_BONUS = 2.0    # stacked on env's +1

    # ── Kettle-specific weights ───────────────────────────────────────────────
    W_APPROACH       = 1.0    # EE→kettle continuous approach reward
    APPROACH_THRESH  = 0.20   # metres — start earning approach reward inside this
    W_GRASP          = 1.5    # gripper closure reward magnitude
    GRASP_K          = 30.0   # exponential sharpness (higher = more binary signal)
    W_LIFT           = 1.0    # reward per step kettle is above lift threshold
    W_TRANSPORT      = 3.0    # gated transport-to-goal reward (replaces W_PROGRESS for kettle)
    W_PUSH_PENALTY   = 2.0    # penalty magnitude for open-gripper horizontal motion
    PUSH_MOVE_THRESH = 0.003  # metres/step — minimum xy movement to trigger penalty
    PUSH_OPEN_THRESH = 0.04   # metres — gripper gap above which gripper counts as "open"
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, env, tasks: list[str] = TASKS):
        super().__init__(env)
        self.tasks = [t for t in tasks if t in TASK_GOALS]
        if not self.tasks:
            raise ValueError(f"None of {tasks} found in TASK_GOALS.")

        # Per-episode state
        self._prev_dist:       dict[str, float] = {}
        self._task_done:       dict[str, bool]  = {}
        self._prev_kettle_xy:  np.ndarray | None = None

    # ── MuJoCo helpers ────────────────────────────────────────────────────────

    def _ee_xyz(self) -> np.ndarray | None:
        """Read end-effector world position from MuJoCo site data."""
        try:
            uid = self.env.unwrapped.model.site('end_effector').id
            return self.env.unwrapped.data.site_xpos[uid].copy()
        except Exception:
            return None

    # ── Obs helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _gripper_gap(raw_obs: np.ndarray) -> float:
        """Total gripper opening in metres (0 = fully closed)."""
        return float(raw_obs[_GRIPPER_R_IDX] + raw_obs[_GRIPPER_L_IDX])

    @staticmethod
    def _kettle_xyz(raw_obs: np.ndarray) -> np.ndarray:
        return raw_obs[_KETTLE_XYZ_IDX].astype(np.float32)

    def _task_distance(self, raw_obs: np.ndarray, task: str) -> float:
        cfg = TASK_GOALS[task]
        return float(np.linalg.norm(
            raw_obs[cfg["obs_indices"]].astype(np.float32) - cfg["goal"]
        ))

    # ── Episode reset ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw_obs = obs["observation"]
        for task in self.tasks:
            self._prev_dist[task] = self._task_distance(raw_obs, task)
            self._task_done[task] = False
        self._prev_kettle_xy = self._kettle_xyz(raw_obs)[:2].copy()
        return obs, info

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        raw_obs = obs["observation"]

        shaped = 0.0
        log    = {}   # per-component breakdown for TensorBoard

        kettle_xyz  = self._kettle_xyz(raw_obs)
        gripper_gap = self._gripper_gap(raw_obs)
        ee_xyz      = self._ee_xyz()

        # ── Per-task shared terms (microwave + kettle) ────────────────────────
        for task in self.tasks:
            if self._task_done[task]:
                continue

            curr_dist = self._task_distance(raw_obs, task)
            prev_dist = self._prev_dist[task]
            max_dist  = _TASK_MAX_DIST.get(task, 1.0)

            # 1. Completion bonus
            if curr_dist < 0.05 * max_dist:
                shaped += self.COMPLETION_BONUS
                self._task_done[task] = True
                log[f"completion_{task}"] = self.COMPLETION_BONUS

            # 2. Normalised distance (non-gated — applies to both tasks)
            norm_dist = curr_dist / max(max_dist, 1e-6)
            r_dist    = self.W_DISTANCE * (1.0 - norm_dist)
            shaped   += r_dist
            log[f"dist_{task}"] = r_dist

            # 3. Progress (non-gated — used for microwave; kettle uses gated transport below)
            if task != "kettle":
                r_prog  = self.W_PROGRESS * (prev_dist - curr_dist)
                shaped += r_prog
                log[f"progress_{task}"] = r_prog

            self._prev_dist[task] = curr_dist

        # ── Kettle-specific grasp enforcement ─────────────────────────────────
        if "kettle" in self.tasks and not self._task_done.get("kettle", False):

            # 4. EE approach reward — continuous signal to move arm toward kettle
            r_approach = 0.0
            if ee_xyz is not None:
                ee_to_kettle = float(np.linalg.norm(ee_xyz - kettle_xyz))
                if ee_to_kettle < self.APPROACH_THRESH:
                    # Normalised: 1.0 when touching, 0.0 at threshold boundary
                    r_approach = self.W_APPROACH * (1.0 - ee_to_kettle / self.APPROACH_THRESH)
            shaped += r_approach
            log["approach"] = r_approach

            # 5. Gripper closure reward — exponential kernel, peaks at gap=0
            #    Only active when EE is near the kettle (within APPROACH_THRESH*1.5)
            r_grasp = 0.0
            if ee_xyz is not None:
                ee_to_kettle = float(np.linalg.norm(ee_xyz - kettle_xyz))
                if ee_to_kettle < self.APPROACH_THRESH * 1.5:
                    norm_gap = gripper_gap / max(_GRIPPER_MAX_GAP, 1e-6)
                    r_grasp  = self.W_GRASP * np.exp(-self.GRASP_K * norm_gap)
            shaped += r_grasp
            log["grasp_closure"] = r_grasp

            # 6. Lift reward + gated transport
            #    Transport credit only flows when the kettle is lifted off the surface.
            kettle_z   = float(kettle_xyz[2])
            is_lifted  = kettle_z > (_KETTLE_REST_Z + _KETTLE_LIFT_MIN)

            r_lift = 0.0
            if is_lifted:
                lift_above = kettle_z - (_KETTLE_REST_Z + _KETTLE_LIFT_MIN)
                r_lift = self.W_LIFT * min(lift_above / 0.10, 1.0)  # saturates at 10 cm above
            shaped += r_lift
            log["lift"] = r_lift

            # Gated transport: progress toward goal only counts while lifted
            r_transport = 0.0
            if is_lifted:
                kettle_dist      = self._prev_dist.get("kettle", 0.0)
                curr_kettle_dist = self._task_distance(raw_obs, "kettle")
                r_transport      = self.W_TRANSPORT * (kettle_dist - curr_kettle_dist)
            shaped += r_transport
            log["transport_gated"] = r_transport

            # 7. Push penalty — fires when kettle moves horizontally with open gripper
            r_push_penalty = 0.0
            if self._prev_kettle_xy is not None:
                xy_movement = float(np.linalg.norm(kettle_xyz[:2] - self._prev_kettle_xy))
                if xy_movement > self.PUSH_MOVE_THRESH and gripper_gap > self.PUSH_OPEN_THRESH:
                    r_push_penalty = -self.W_PUSH_PENALTY * (xy_movement / self.PUSH_MOVE_THRESH)
            shaped += r_push_penalty
            log["push_penalty"] = r_push_penalty

        # Update previous kettle xy for next step
        self._prev_kettle_xy = kettle_xyz[:2].copy()

        total_reward = env_reward + shaped

        # Log everything to info for TensorBoard
        info["shaped_reward"]   = float(shaped)
        info["original_reward"] = float(env_reward)
        info["reward_breakdown"] = log

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
    """
    Logs reward stats every log_freq steps, including a full per-component
    breakdown (approach, grasp_closure, lift, transport_gated, push_penalty…)
    so you can diagnose exactly which reward terms the agent is collecting.
    """

    def __init__(self, log_freq: int = 10_000, run_name: str = "run"):
        super().__init__(verbose=0)
        self.log_freq        = log_freq
        self.run_name        = run_name
        self._window_returns:    list[float]             = []
        self._window_lengths:    list[float]             = []
        self._window_shaped:     list[float]             = []
        self._window_original:   list[float]             = []
        self._component_windows: dict[str, list[float]]  = {}
        self._total_episodes:    int                     = 0

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
            # Accumulate per-component breakdown
            for component, value in info.get("reward_breakdown", {}).items():
                if component not in self._component_windows:
                    self._component_windows[component] = []
                self._component_windows[component].append(float(value))

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

            # Per-component series — visible individually in TensorBoard
            for component, values in self._component_windows.items():
                if values:
                    self.logger.record(f"{self.run_name}/rew_{component}",
                                       float(np.mean(values)))

            self.logger.dump(self.num_timesteps)
            self._window_returns.clear()
            self._window_lengths.clear()
            self._window_shaped.clear()
            self._window_original.clear()
            for v in self._component_windows.values():
                v.clear()

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
    video_folder = "franka_kitchen_eval_videos_2"

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
    log_cb = train(run_name="ppo_shaped_asr_run_2", use_asr=True, use_shaped_reward=True)
    plot_results(log_cb.episode_returns, title_suffix="Shaped ASR")

    # ── Ablation: sparse baseline — uncomment to compare ─────────────────────
    # log_cb_base = train(run_name="ppo_sparse_baseline", use_asr=True, use_shaped_reward=False)
    # plot_results(log_cb_base.episode_returns, title_suffix="Sparse Baseline")

    # ── Evaluate saved model ──────────────────────────────────────────────────
    # evaluate(
    #     model_path        = "./ppo_franka_best_ppo_shaped_asr_run_2/best_model.zip",
    #     num_episodes      = 10,
    #     record_video      = True,
    #     use_asr           = True,
    #     use_shaped_reward = True,
    # )