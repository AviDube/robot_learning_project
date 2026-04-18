import argparse

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
import os
import matplotlib.pyplot as plt

from gymnasium.wrappers import RecordVideo
from gymnasium.spaces import Box
from collections import defaultdict

from stable_baselines3.common.callbacks import BaseCallback
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
TASKS = ['kettle']


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
    W_DISTANCE       = 0.0      # Zeroed out to prevent passive hover-farming
    W_PROGRESS       = 1.0      # delta-distance (non-gated, applies to microwave too)
    COMPLETION_BONUS = 150.0    # Massive jackpot for finishing the task
    TIME_PENALTY     = -0.01    # Bleeds points every step to force speed

    # ── Kettle-specific weights ───────────────────────────────────────────────
    W_APPROACH       = 20.0     # Scaled up for the high-water mark delta progress
    APPROACH_THRESH  = 0.20     # metres — start earning approach reward inside this
    BONUS_GRASP      = 10.0     # Changed from W_GRASP: Now a one-time milestone payout
    BONUS_LIFT       = 15.0     # Changed from W_LIFT: Now a one-time milestone payout
    W_TRANSPORT      = 15.0     # Gated transport-to-goal reward (high value)
    W_PUSH_PENALTY   = 2.0      # penalty magnitude for open-gripper horizontal motion
    PUSH_MOVE_THRESH = 0.003    # metres/step — minimum xy movement to trigger penalty
    PUSH_OPEN_THRESH = 0.04     # metres — gripper gap above which counts as "open"
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, env, tasks: list[str] = TASKS):
        super().__init__(env)
        self.tasks = [t for t in tasks if t in TASK_GOALS]
        if not self.tasks:
            raise ValueError(f"None of {tasks} found in TASK_GOALS.")

        # Per-episode state
        self._prev_dist:       dict[str, float]  = {}
        self._task_done:       dict[str, bool]   = {}
        self._prev_kettle_xy:  np.ndarray | None = None
        self._prev_kettle_xyz: np.ndarray | None = None
        
        # Anti-exploit trackers
        self._min_ee_to_kettle: float = float('inf')
        self._has_grasped:      bool  = False
        self._has_lifted:       bool  = False

        _model = self.env.unwrapped.model
        self._kettle_bodies = {
            mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ('kettle', 'kettleroot')
        } - {-1}
        self._finger_bodies = {
            mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ('panda0_leftfinger', 'panda0_rightfinger')
        } - {-1}

    # ── MuJoCo & Obs helpers (unchanged) ──────────────────────────────────────

    def _ee_xyz(self) -> np.ndarray | None:
        try:
            model = self.env.unwrapped.model
            data = self.env.unwrapped.data
            
            # Use the correct DeepMind MuJoCo binding syntax to get the site ID
            uid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
            
            if uid == -1: # mj_name2id returns -1 if it can't find the name
                return None
                
            return data.site_xpos[uid].copy()
            
        except Exception as e:
            if not hasattr(self, '_warned_ee'):
                print(f"\nCRITICAL ERROR tracking EE: {e}\n")
                self._warned_ee = True
            return None

    @staticmethod
    def _gripper_gap(raw_obs: np.ndarray) -> float:
        return float(raw_obs[_GRIPPER_R_IDX] + raw_obs[_GRIPPER_L_IDX])

    @staticmethod
    def _kettle_xyz(raw_obs: np.ndarray) -> np.ndarray:
        return raw_obs[_KETTLE_XYZ_IDX].astype(np.float32)

    def _task_distance(self, raw_obs: np.ndarray, task: str) -> float:
        cfg = TASK_GOALS[task]
        return float(np.linalg.norm(
            raw_obs[cfg["obs_indices"]].astype(np.float32) - cfg["goal"]
        ))
    
    def _is_touching_kettle(self) -> bool:
        try:
            model = self.env.unwrapped.model
            data = self.env.unwrapped.data

            if not self._kettle_bodies or not self._finger_bodies:
                return False

            for i in range(data.ncon):
                contact = data.contact[i]
                body1 = model.geom_bodyid[contact.geom1]
                body2 = model.geom_bodyid[contact.geom2]
                if {body1, body2} & self._kettle_bodies and {body1, body2} & self._finger_bodies:
                    return True

            return False

        except Exception as e:
            if not hasattr(self, '_warned_contact_err'):
                print(f"\nCRITICAL ERROR in contact check: {e}\n")
                self._warned_contact_err = True
            return False
        
    def _kettle_handle_xyz(self) -> np.ndarray | None:
        """Dynamically tracks the exact 3D position of the kettle's handle."""
        try:
            model = self.env.unwrapped.model
            data = self.env.unwrapped.data
            
            # Use DeepMind MuJoCo bindings to find the kettle_site
            uid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'kettle_site')
            
            if uid == -1:
                return None
                
            return data.site_xpos[uid].copy()
            
        except Exception:
            return None

    # ── Episode reset ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw_obs = obs["observation"]
        for task in self.tasks:
            self._prev_dist[task] = self._task_distance(raw_obs, task)
            self._task_done[task] = False
            
        self._prev_kettle_xy = self._kettle_xyz(raw_obs)[:2].copy()
        self._prev_kettle_xyz = self._kettle_xyz(raw_obs).copy()
        self._initial_kettle_z = float(self._kettle_xyz(raw_obs)[2])
        
        # Reset anti-exploit trackers for the new episode
        self._min_ee_to_kettle = float('inf')
        self._has_grasped      = False
        self._has_lifted       = False         
        
        return obs, info

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        raw_obs = obs["observation"]

        shaped = 0.0
        log    = {}   

        # 0. The Time Penalty (Always active)
        shaped += self.TIME_PENALTY
        log["time_penalty"] = self.TIME_PENALTY

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

            # 2. Normalised distance (Zeroed out via W_DISTANCE = 0.0, left here for safe structure)
            if self.W_DISTANCE > 0:
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
            
            ee_to_kettle = float(np.linalg.norm(ee_xyz - kettle_xyz)) if ee_xyz is not None else float('inf')
            kettle_z = float(kettle_xyz[2])
            kettle_movement = float(np.linalg.norm(kettle_xyz - self._prev_kettle_xyz))

            # 4. Approach Reward (Targeting the newly fixed kettle_site)
            r_approach = 0.0
            handle_xyz = self._kettle_handle_xyz()
            ee_to_kettle = float(np.linalg.norm(ee_xyz - handle_xyz)) if (ee_xyz is not None and handle_xyz is not None) else float('inf')

            if ee_to_kettle < self._min_ee_to_kettle:
                if self._min_ee_to_kettle != float('inf'): 
                    progress = self._min_ee_to_kettle - ee_to_kettle
                    if progress > 0:
                        r_approach = self.W_APPROACH * progress
                self._min_ee_to_kettle = ee_to_kettle

            shaped += r_approach
            log["approach"] = r_approach
            
            r_stay_close = 0.0
            if ee_to_kettle < 0.10:
                r_stay_close = 0.5 * (0.10 - ee_to_kettle) / 0.10
            shaped += r_stay_close
            log["stay_close"] = r_stay_close

            # 4.5 Gated Gripper Closure (The Anti-Fist-Pump Logic)
            r_close = 0.0
            if not self._has_grasped and ee_to_kettle < 0.06:
                if self._is_touching_kettle():
                    r_close = 8.0 * (0.08 - gripper_gap)   # bumped from 5.0
                elif gripper_gap < 0.05:
                    r_close = -3.0 * ((0.05 - gripper_gap) / 0.05)  # scaled, up to -3.0
            
            shaped += r_close
            log["gripper_closure"] = r_close

            # 5. Grasp Reward (The 10-Point Milestone)
            r_grasp = 0.0
            if not self._has_grasped and ee_to_kettle < 0.05:
                # Gap < 0.05 allows for the 3.2cm wooden handle thickness
                if gripper_gap < 0.030 and self._is_touching_kettle():
                    r_grasp = self.BONUS_GRASP
                    self._has_grasped = True
                    
            shaped += r_grasp
            log["grasp_milestone"] = r_grasp

            # 6. Lift Reward (One-Time Milestone)
            r_lift = 0.0
            is_lifted = kettle_z > (self._initial_kettle_z + _KETTLE_LIFT_MIN)
            if not self._has_lifted and is_lifted:
                if self._has_grasped:
                    r_lift = self.BONUS_LIFT
                    self._has_lifted = True
            shaped += r_lift
            log["lift_milestone"] = r_lift

            if is_lifted and not self._has_grasped and gripper_gap > 0.03:
                r_wedge_penalty = -5.0
                shaped += r_wedge_penalty
                log["wedge_penalty"] = r_wedge_penalty

            # 7. Gated Transport (Continuous, scaled up)
            r_transport = 0.0
            if is_lifted and self._has_grasped:
                kettle_dist = self._prev_dist.get("kettle", 0.0)
                curr_kettle_dist = self._task_distance(raw_obs, "kettle")
                # Only reward positive progress to prevent wiggle farming on transport
                progress = kettle_dist - curr_kettle_dist
                if progress > 0: 
                    r_transport = self.W_TRANSPORT * progress
            shaped += r_transport
            log["transport_gated"] = r_transport

            # 8. Push Penalty 
            r_push_penalty = 0.0
            if self._prev_kettle_xy is not None:
                xy_movement = float(np.linalg.norm(kettle_xyz[:2] - self._prev_kettle_xy))
                if xy_movement > self.PUSH_MOVE_THRESH and gripper_gap > self.PUSH_OPEN_THRESH:
                    r_push_penalty = -self.W_PUSH_PENALTY * (xy_movement / self.PUSH_MOVE_THRESH)
            shaped += r_push_penalty
            log["push_penalty"] = r_push_penalty

        # Update previous trackers for next step
        self._prev_kettle_xy = kettle_xyz[:2].copy()
        self._prev_kettle_xyz = kettle_xyz.copy()

        total_reward = env_reward + shaped

        # Log everything to info for TensorBoard
        info["shaped_reward"]    = float(shaped)
        info["original_reward"]  = float(env_reward)
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
    Logs reward stats every log_freq steps.
    UPDATED: Accumulates reward components as EPISODE SUMS instead of step means,
    which is crucial for tracking one-time milestones correctly.
    """

    def __init__(self, log_freq: int = 10_000, run_name: str = "run"):
        super().__init__(verbose=0)
        self.log_freq        = log_freq
        self.run_name        = run_name
        self._window_returns:    list[float] = []
        self._window_lengths:    list[float] = []
        self._window_shaped:     list[float] = []
        self._window_original:   list[float] = []
        self._total_episodes:    int         = 0
        
        # Active accumulators for each parallel environment
        self._current_ep_components = None 
        
        # History of completed episode sums to log to TensorBoard
        self._ep_component_history = defaultdict(list)

    def _on_training_start(self) -> None:
        # Initialize an accumulator dictionary for each parallel env (e.g., 8 envs)
        n_envs = self.training_env.num_envs
        self._current_ep_components = [defaultdict(float) for _ in range(n_envs)]

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals["infos"]):
            
            # 1. Accumulate per-step breakdown into the current episode's total
            for component, value in info.get("reward_breakdown", {}).items():
                self._current_ep_components[i][component] += float(value)

            if "shaped_reward" in info:
                self._window_shaped.append(float(info["shaped_reward"]))
            if "original_reward" in info:
                self._window_original.append(float(info["original_reward"]))

            # 2. When an episode ends, push its totals to history and reset
            if "episode" in info:
                self._window_returns.append(float(info["episode"]["r"]))
                self._window_lengths.append(float(info["episode"]["l"]))
                self._total_episodes += 1
                
                for comp, val in self._current_ep_components[i].items():
                    self._ep_component_history[comp].append(val)
                self._current_ep_components[i].clear()

        # 3. Log to TensorBoard
        if self.num_timesteps % self.log_freq == 0 and self._window_returns:
            rew = np.array(self._window_returns)
            self.logger.record(f"{self.run_name}/ep_rew_mean",    float(rew.mean()))
            self.logger.record(f"{self.run_name}/ep_rew_min",     float(rew.min()))
            self.logger.record(f"{self.run_name}/ep_rew_max",     float(rew.max()))
            self.logger.record(f"{self.run_name}/ep_len_mean",    float(np.mean(self._window_lengths)))
            self.logger.record(f"{self.run_name}/episodes_total", self._total_episodes)

            # Log the MEAN of the EPISODE SUMS for each specific component
            for component, values in self._ep_component_history.items():
                if values:
                    self.logger.record(f"{self.run_name}/ep_sum_{component}", float(np.mean(values)))

            self.logger.dump(self.num_timesteps)
            
            # Clear buffers for the next logging window
            self._window_returns.clear()
            self._window_lengths.clear()
            self._window_shaped.clear()
            self._window_original.clear()
            self._ep_component_history.clear()

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
_DYNAMIC_BATCH  = 128

print(f"Auto-selected N_ENVS={N_ENVS} for {_logical_cores} logical CPU cores")

PPO_KWARGS = dict(
    n_steps       = 4096,
    batch_size    = _DYNAMIC_BATCH,
    n_epochs      = 15,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.05,    # slightly raised from 0.0 → encourages exploration
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

    ## change it to take an arg for run_training vs evaluation, and for which model to eval
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_training", action="store_true")
    parser.add_argument("--model_path", type=str, default="./ppo_franka_best_ppo_shaped_asr_run_2/best_model.zip")
    args = parser.parse_args()

    run_training = args.run_training  # Set to False to skip training and run evaluation only
    # ── Shaped + ASR (recommended) ────────────────────────────────────────────
    if run_training:
        log_cb = train(run_name="ppo_shaped_asr_run_2", use_asr=True, use_shaped_reward=True)
        plot_results(log_cb.episode_returns, title_suffix="Shaped ASR")

    # ── Ablation: sparse baseline — uncomment to compare ─────────────────────
    # log_cb_base = train(run_name="ppo_sparse_baseline", use_asr=True, use_shaped_reward=False)
    # plot_results(log_cb_base.episode_returns, title_suffix="Sparse Baseline")

    else:
        # ── Evaluate saved model ──────────────────────────────────────────────────
        evaluate(
            model_path        = args.model_path,
            num_episodes      = 10,
            record_video      = True,
            use_asr           = True,
            use_shaped_reward = True,
        )