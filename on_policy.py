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
#    31   microwave hinge angle          <- used for microwave task
#    32   kettle x position              <- used for kettle task
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

TASK_GOALS = {
    "microwave": {
        "obs_indices": [31],
        "goal":        np.array([0.37], dtype=np.float32),
    },
    "kettle": {
        "obs_indices": [32, 33, 34],
        "goal":        np.array([-0.23, 0.75, 1.62], dtype=np.float32),
    },
}

TASKS = ['kettle']


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_TASK_MAX_DIST = {
    "microwave": 0.37,
    "kettle":    1.20,
}
_KETTLE_LIFT_MIN  = 0.04
_GRIPPER_R_IDX    = 7
_GRIPPER_L_IDX    = 8
_GRIPPER_MAX_GAP  = 0.08
_HANDLE_DIAMETER  = 0.040    # 2 x capsule radius (0.032 m) from XML
_KETTLE_XYZ_IDX   = [32, 33, 34]


# ─────────────────────────────────────────────────────────────────────────────
# REWARD SHAPING  —  DenseRewardWrapper  (simplified, 3-term)
# ─────────────────────────────────────────────────────────────────────────────
#
#  The previous 14-term reward produced competing gradients that confused
#  PPO's policy update.  This version uses 3 terms only:
#
#  TERM 1 — finger_reach   : smooth exponential pulling both fingers toward
#                             the handle.  Replaces global_reach, approach,
#                             proximity, alignment, open_approach.
#
#  TERM 2 — grasp_quality  : product of proximity x opposition x closing.
#                             Zero unless ALL THREE hold simultaneously —
#                             no gates needed, no competing gradients.
#                             Replaces closure, coordination, finger_centering,
#                             premature_close, wedge_penalty.
#
#  TERM 3 — transport      : potential-based kettle-to-goal progress gated
#                             on _has_grasped.
#                             BUG FIX: _prev_kettle_dist is updated at the
#                             END of step() so it always holds the previous
#                             timestep's value (the old code overwrote
#                             _prev_dist mid-step in the per-task loop,
#                             making transport always ~0).
#
#  Plus milestones (grasp +30, lift +20, completion +150), a time penalty,
#  and a capped push penalty.
#
# ─────────────────────────────────────────────────────────────────────────────

class DenseRewardWrapper(gym.Wrapper):
    """
    Simplified 3-term dense reward wrapper for Franka Kitchen kettle task.
    """

    # ── Continuous Phase Weights (Locked per step when done) ──
    # Since these are Tanh^2 (bounded 0.0 to 1.0), the weight is the max per-step payout.
    W_FINGER_REACH   = 1.0     
    W_GRASP_QUALITY  = 1.0     
    W_LIFT           = 1.5     # The continuous upward progress term
    W_TRANSPORT      = 2.0     # Slightly higher to strongly pull toward the goal
    
    # ── One-Time Milestone Bonuses ──
    # These fire exactly once per episode to jump the value function
    BONUS_GRASP      = 15.0    
    BONUS_LIFT       = 15.0    
    COMPLETION_BONUS = 50.0    
    
    # ── Penalties ──
    TIME_PENALTY     = -0.02
    W_PUSH_PENALTY   = 0.5
    PUSH_MOVE_THRESH = 0.02
    PUSH_PENALTY_CAP = 1.0
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, env, tasks: list[str] = TASKS):
        super().__init__(env)
        self.tasks = [t for t in tasks if t in TASK_GOALS]
        if not self.tasks:
            raise ValueError(f"None of {tasks} found in TASK_GOALS.")

        self._prev_dist:        dict[str, float]  = {}
        self._task_done:        dict[str, bool]   = {}
        self._prev_kettle_xy:   np.ndarray | None = None
        self._prev_kettle_dist: float             = 0.0

        self._has_grasped:      bool  = False
        self._has_lifted:       bool  = False
        self._initial_kettle_z: float = 0.0

        _model = self.env.unwrapped.model
        self._kettle_bodies = {
            mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ('kettle', 'kettleroot')
        } - {-1}
        self._finger_bodies = {
            mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ('panda0_leftfinger', 'panda0_rightfinger')
        } - {-1}
        self._lf_id = mujoco.mj_name2id(
            _model, mujoco.mjtObj.mjOBJ_BODY, 'panda0_leftfinger')
        self._rf_id = mujoco.mj_name2id(
            _model, mujoco.mjtObj.mjOBJ_BODY, 'panda0_rightfinger')

    # ── MuJoCo helpers ────────────────────────────────────────────────────────

    def _kettle_handle_xyz(self) -> np.ndarray | None:
        try:
            model = self.env.unwrapped.model
            data  = self.env.unwrapped.data
            uid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'kettle_site')
            if uid == -1: return None
            return data.site_xpos[uid].copy()
        except Exception:
            return None

    def _finger_positions(self):
        """Returns (lf_pos, rf_pos) world coordinates, or (None, None)."""
        try:
            data = self.env.unwrapped.data
            lf   = data.xpos[self._lf_id].copy() if self._lf_id != -1 else None
            rf   = data.xpos[self._rf_id].copy() if self._rf_id != -1 else None
            return lf, rf
        except Exception:
            return None, None

    def _is_touching_kettle(self) -> bool:
        try:
            data  = self.env.unwrapped.data
            model = self.env.unwrapped.model
            if not self._kettle_bodies or not self._finger_bodies:
                return False
            for i in range(data.ncon):
                contact = data.contact[i]
                body1   = model.geom_bodyid[contact.geom1]
                body2   = model.geom_bodyid[contact.geom2]
                if ({body1, body2} & self._kettle_bodies and
                        {body1, body2} & self._finger_bodies):
                    return True
            return False
        except Exception:
            return False

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

    # ── Episode reset ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw_obs   = obs["observation"]

        for task in self.tasks:
            self._prev_dist[task] = self._task_distance(raw_obs, task)
            self._task_done[task] = False

        self._prev_kettle_xy   = self._kettle_xyz(raw_obs)[:2].copy()
        self._initial_kettle_z = float(self._kettle_xyz(raw_obs)[2])
        self._prev_kettle_dist = self._task_distance(raw_obs, "kettle")

        self._has_grasped = False
        self._has_lifted  = False

        return obs, info

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        raw_obs = obs["observation"]

        shaped = self.TIME_PENALTY
        log    = {"time_penalty": self.TIME_PENALTY}

        kettle_xyz  = self._kettle_xyz(raw_obs)
        gripper_gap = self._gripper_gap(raw_obs)
        handle_xyz  = self._kettle_handle_xyz()
        lf_pos, rf_pos = self._finger_positions()

        # ── Finger geometry — computed once, shared across all terms ──────────
        opposition = 0.0
        lf_dist    = float('inf')
        rf_dist    = float('inf')

        if lf_pos is not None and rf_pos is not None and handle_xyz is not None:
            lf_dist      = float(np.linalg.norm(lf_pos - handle_xyz))
            rf_dist      = float(np.linalg.norm(rf_pos - handle_xyz))
            lf_to_handle = handle_xyz - lf_pos
            rf_to_handle = handle_xyz - rf_pos
            lf_n = np.linalg.norm(lf_to_handle)
            rf_n = np.linalg.norm(rf_to_handle)
            if lf_n > 1e-6 and rf_n > 1e-6:
                opposition = float(
                    np.dot(lf_to_handle / lf_n, rf_to_handle / rf_n)
                )

        # ── Temperatures for Smooth Tolerance ─────────────────────────────────
        TAU_REACH = 0.05   # Scales how sharp the gradient is near the handle
        TAU_LIFT  = 0.05  # Scales lift gradient
        
        # ── TERM 1: Phase-Gated Finger Reach ──────────────────────────────────
        r_reach = 0.0
        if self._has_grasped:
            # Phase Complete: Lock in maximum continuous income
            r_reach = self.W_FINGER_REACH
        elif lf_pos is not None:
            # Active Phase: Smooth gradient to the handle
            mean_dist = (lf_dist + rf_dist) / 2.0
            tolerance = 1.0 - np.tanh(mean_dist / TAU_REACH)**2
            r_reach = self.W_FINGER_REACH * tolerance
            
        shaped += r_reach
        log["finger_reach"] = r_reach

        # ── TERM 2: Phase-Gated Grasp Quality ─────────────────────────────────
        r_grasp_quality = 0.0
        if self._has_grasped:
            r_grasp_quality = self.W_GRASP_QUALITY
        elif lf_pos is not None:
            mean_dist = (lf_dist + rf_dist) / 2.0
            if mean_dist < 0.8: 
                opposition_quality = np.clip(-opposition, 0.0, 1.0)
                closing = np.clip((0.08 - gripper_gap) / (0.08 - _HANDLE_DIAMETER), 0.0, 1.0)
                r_grasp_quality = self.W_GRASP_QUALITY * (0.5 * opposition_quality + 0.5 * closing)
                
        shaped += r_grasp_quality
        log["grasp_quality"] = r_grasp_quality

        # ── TERM 3: Grasp Milestone ───────────────────────────────────────────
        r_grasp_bonus = 0.0
        if not self._has_grasped and gripper_gap < 0.072 and self._is_touching_kettle():
            r_grasp_bonus = self.BONUS_GRASP
            self._has_grasped = True
        shaped += r_grasp_bonus
        if r_grasp_bonus > 0:
            log["grasp_milestone"] = r_grasp_bonus

        # ── TERM 4: Phase-Gated Lift ──────────────────────────────────────────
        r_lift = 0.0
        r_lift_bonus = 0.0
        kettle_z = float(kettle_xyz[2])
        z_diff = kettle_z - self._initial_kettle_z
        
        if self._has_lifted:
            # Phase Complete: Lock in max continuous lift reward
            r_lift = self.W_LIFT
        elif self._has_grasped:
            # Active Phase: Continuous progress gradient
            progress = np.clip(z_diff / _KETTLE_LIFT_MIN, 0.0, 1.0)
            tolerance = 1.0 - np.tanh((1.0 - progress) / TAU_LIFT)**2
            r_lift = self.W_LIFT * tolerance
            
            # Phase Transition: Give the massive 1-time bonus
            if z_diff > _KETTLE_LIFT_MIN:
                self._has_lifted = True
                r_lift = self.W_LIFT
                r_lift_bonus = self.BONUS_LIFT
                
        shaped += (r_lift + r_lift_bonus)
        log["lift_continuous"] = r_lift
        if r_lift_bonus > 0:
            log["lift_milestone"] = r_lift_bonus

        # ── TERM 5: Phase-Gated Transport ─────────────────────────────────────
        r_transport = 0.0
        r_completion = 0.0
        curr_kettle_dist = self._task_distance(raw_obs, "kettle")
        
        if self._task_done.get("kettle", False):
            # Phase Complete: Lock continuous transport income
            r_transport = self.W_TRANSPORT
        elif self._has_lifted:
            # Active Phase: Continuous distance gradient
            tolerance = 1.0 - np.tanh(curr_kettle_dist / 0.2)**2
            r_transport = self.W_TRANSPORT * tolerance
            
            # Phase Transition: Give massive 1-time completion bonus
            if curr_kettle_dist < 0.05 * _TASK_MAX_DIST["kettle"]:
                self._task_done["kettle"] = True
                r_transport = self.W_TRANSPORT
                r_completion = self.COMPLETION_BONUS
                
        shaped += (r_transport + r_completion)
        log["transport_continuous"] = r_transport
        if r_completion > 0:
            log["completion_bonus"] = r_completion

        # ── Push penalty (capped) ──────────────────────────────────────────────
        # r_push = 0.0
        # if self._prev_kettle_xy is not None and not self._has_grasped:
            
        #     mean_dist = float('inf')
        #     if lf_pos is not None:
        #         mean_dist = (lf_dist + rf_dist) / 2.0
                
        #     if mean_dist > 0.07: # FORGIVENESS ZONE
        #         xy_move = float(np.linalg.norm(kettle_xyz[:2] - self._prev_kettle_xy))
        #         if xy_move > self.PUSH_MOVE_THRESH:
        #             multiplier = min(xy_move / self.PUSH_MOVE_THRESH, self.PUSH_PENALTY_CAP)
        #             r_push     = -self.W_PUSH_PENALTY * multiplier
                    
        # shaped += r_push
        # log["push_penalty"] = r_push

        # ── Update trackers ────────────────────────────────────────────────────
        self._prev_kettle_xy = kettle_xyz[:2].copy()

        info["shaped_reward"]    = float(shaped)
        info["original_reward"]  = float(env_reward)
        info["reward_breakdown"] = log

        return obs, env_reward + shaped, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION WRAPPERS
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
            f"[ASRObsWrapper] {full_dim}D -> {len(self._asr_indices)}D  "
            f"(dropped {full_dim - len(self._asr_indices)} velocity dims)"
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[self._asr_indices]


class AugmentedObsWrapper(gym.ObservationWrapper):
    """
    Appends 14 task-relevant derived features to the ASR observation.

    These are exactly the signals the policy needs to solve the task —
    no more, no less.  Previous version had redundant EE-centric signals;
    this version is finger-centric and includes a direct transport vector.

    Added dims (14 total):
      [0:3]  lf_to_handle_xyz   — left  finger -> handle (3D)
      [3:6]  rf_to_handle_xyz   — right finger -> handle (3D)
      [6]    opposition         — dot product of unit finger->handle vectors
                                  -1 = perfectly opposing (ideal pre-grasp)
                                  +1 = fingers on same side (bad)
      [7]    gripper_gap        — sum of both finger joints (m)
      [8]    is_touching        — 1 if finger contacts kettle geom, else 0
      [9]    has_grasped        — 1 after grasp milestone fires, else 0
      [10:13] kettle_to_goal_xyz — vector from kettle root to goal (3D)
                                   gives transport direction directly
      [13]   kettle_dz          — kettle z minus initial z (lift progress, m)
      [14:20] orientation features (6D) — see _get_orientation_features()

    Wrapper stack (unchanged):
      FrankaKitchen-v1
          └─ DenseRewardWrapper
              └─ FlattenObsWrapper
                  └─ ASRObsWrapper
                      └─ AugmentedObsWrapper   <- here  (37 -> 51 dims)
                          └─ Monitor
    """

    N_EXTRA = 20   # 3+3+1+1+1+1+3+1+6

    def __init__(self, env):
        super().__init__(env)
        base_dim = env.observation_space.shape[0]
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + self.N_EXTRA,), dtype=np.float32,
        )
        print(f"[AugmentedObsWrapper] {base_dim}D -> {base_dim + self.N_EXTRA}D "
              f"(+{self.N_EXTRA} derived features)")

    def _get_raw_env(self):
        """Walk the wrapper stack down to the unwrapped MuJoCo env."""
        e = self.env
        while hasattr(e, 'env'):
            e = e.env
        return e.unwrapped

    def _handle_xyz(self) -> np.ndarray | None:
        try:
            raw = self._get_raw_env()
            uid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_SITE, 'kettle_site')
            return raw.data.site_xpos[uid].copy() if uid != -1 else None
        except Exception:
            print("Error getting handle position; returning None.")
            return None

    def _is_touching(self) -> bool:
        try:
            raw   = self._get_raw_env()
            model = raw.model
            data  = raw.data
            kettle_bodies = {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in ('kettle', 'kettleroot')
            } - {-1}
            finger_bodies = {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in ('panda0_leftfinger', 'panda0_rightfinger')
            } - {-1}
            for i in range(data.ncon):
                c  = data.contact[i]
                b1 = model.geom_bodyid[c.geom1]
                b2 = model.geom_bodyid[c.geom2]
                if {b1, b2} & kettle_bodies and {b1, b2} & finger_bodies:
                    return True
            return False
        except Exception:
            print("Error checking contact; assuming not touching.")
            return False
    
    def _get_orientation_features(self) -> np.ndarray:
        try:
            raw = self._get_raw_env()
            model = raw.model
            data  = raw.data
            
            # Dynamically grab the body IDs so we don't rely on __init__ variables
            lf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'panda0_leftfinger')
            rf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'panda0_rightfinger')
            
            if lf_id == -1 or rf_id == -1:
                print("WARNING: Could not find panda0_leftfinger or panda0_rightfinger bodies in XML.")
                return np.zeros(6, dtype=np.float32)

            # 1. Finger Axis Vector (3D)
            lf_pos = data.xpos[lf_id].copy()
            rf_pos = data.xpos[rf_id].copy()
            finger_vec = rf_pos - lf_pos
            finger_axis = finger_vec / (np.linalg.norm(finger_vec) + 1e-6)
            
            # 2. Palm Approach Vector (3D)
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
            if site_id != -1:
                site_xmat = data.site_xmat[site_id]
                palm_vec = np.array([site_xmat[2], site_xmat[5], site_xmat[8]])
            else:
                palm_vec = np.array([0.0, 0.0, -1.0]) 
                print("WARNING: End effector site not found; using default palm vector.")
                
            return np.concatenate([finger_axis, palm_vec]).astype(np.float32)
            
        except Exception as e:
            # THIS will finally print the actual error if it crashes again
            print(f"CRITICAL ERROR in _get_orientation_features: {e}")
            return np.zeros(6, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        raw        = self._get_raw_env()
        handle_xyz = self._handle_xyz()

        # ── [0:3] lf_to_handle   [3:6] rf_to_handle   [6] opposition ─────────
        lf_to_handle = np.zeros(3, dtype=np.float32)
        rf_to_handle = np.zeros(3, dtype=np.float32)
        opposition   = np.float32(0.0)

        if handle_xyz is not None:
            try:
                lf_id = mujoco.mj_name2id(
                    raw.model, mujoco.mjtObj.mjOBJ_BODY, 'panda0_leftfinger')
                rf_id = mujoco.mj_name2id(
                    raw.model, mujoco.mjtObj.mjOBJ_BODY, 'panda0_rightfinger')
                if lf_id != -1:
                    lf_to_handle = (handle_xyz - raw.data.xpos[lf_id]).astype(np.float32)
                if rf_id != -1:
                    rf_to_handle = (handle_xyz - raw.data.xpos[rf_id]).astype(np.float32)

                lf_n = np.linalg.norm(lf_to_handle)
                rf_n = np.linalg.norm(rf_to_handle)
                if lf_n > 1e-6 and rf_n > 1e-6:
                    opposition = np.float32(
                        np.dot(lf_to_handle / lf_n, rf_to_handle / rf_n)
                    )
            except Exception:
                pass

        # ── [7] gripper_gap ───────────────────────────────────────────────────
        # obs[7] and obs[8] are the two finger joints in ASR space
        gripper_gap = np.float32(obs[7] + obs[8])

        # ── [8] is_touching ───────────────────────────────────────────────────
        touching = np.float32(self._is_touching())

        # ── [9] has_grasped ───────────────────────────────────────────────────
        # Read the milestone flag from DenseRewardWrapper by walking the stack
        has_grasped = np.float32(0.0)
        try:
            e = self.env
            while hasattr(e, 'env'):
                if hasattr(e, '_has_grasped'):
                    has_grasped = np.float32(e._has_grasped)
                    break
                e = e.env
        except Exception:
            pass

        # ── [10:13] kettle_to_goal_xyz ────────────────────────────────────────
        # Direct vector from current kettle position to goal position.
        # ASR remaps raw indices: raw[32,33,34] -> ASR[23,24,25]
        #   raw index 32 = 9 + (32-18) = ASR index 23
        kettle_to_goal = np.zeros(3, dtype=np.float32)
        try:
            kettle_pos     = obs[23:26].astype(np.float32)
            goal_pos       = TASK_GOALS["kettle"]["goal"]
            kettle_to_goal = goal_pos - kettle_pos
        except Exception:
            pass

        # ── [13] kettle_dz ────────────────────────────────────────────────────
        # How far the kettle has been lifted above its initial z.
        # ASR[25] = raw[34] = kettle z position
        kettle_dz = np.float32(0.0)
        try:
            e = self.env
            while hasattr(e, 'env'):
                if hasattr(e, '_initial_kettle_z'):
                    kettle_dz = np.float32(obs[25]) - np.float32(e._initial_kettle_z)
                    break
                e = e.env
        except Exception:
            pass

        orientation_features = self._get_orientation_features()  # [14:20]

        extra = np.concatenate([
            lf_to_handle,    # [0:3]   left  finger -> handle vector
            rf_to_handle,    # [3:6]   right finger -> handle vector
            [opposition],    # [6]     finger opposition scalar
            [gripper_gap],   # [7]     gripper gap (m)
            [touching],      # [8]     contact flag
            [has_grasped],   # [9]     grasp milestone flag
            kettle_to_goal,  # [10:13] kettle -> goal vector
            [kettle_dz],     # [13]    lift progress (m)
            orientation_features,  # [14:20] orientation features
        ]).astype(np.float32)

        return np.concatenate([obs, extra])


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────
# Wrapper stacking order (innermost -> outermost):
#   FrankaKitchen-v1  (raw Dict obs, sparse binary reward)
#       └─ DenseRewardWrapper   <- reads raw Dict obs, augments reward
#           └─ FlattenObsWrapper <- Dict -> flat 59-dim Box
#               └─ ASRObsWrapper <- 59-dim -> 37-dim (drop velocities)
#                   └─ AugmentedObsWrapper <- 37-dim -> 51-dim (+14 derived)
#                       └─ Monitor
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
            env = DenseRewardWrapper(env, tasks=TASKS)
        env = FlattenObsWrapper(env)
        if use_asr:
            env = ASRObsWrapper(env)
        env = AugmentedObsWrapper(env)
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class TensorboardCallback(BaseCallback):
    """
    Logs reward stats every log_freq steps.
    Accumulates reward components as EPISODE SUMS instead of step means,
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

        self._current_ep_components = None
        self._ep_component_history  = defaultdict(list)

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._current_ep_components = [defaultdict(float) for _ in range(n_envs)]

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals["infos"]):

            for component, value in info.get("reward_breakdown", {}).items():
                self._current_ep_components[i][component] += float(value)

            if "shaped_reward" in info:
                self._window_shaped.append(float(info["shaped_reward"]))
            if "original_reward" in info:
                self._window_original.append(float(info["original_reward"]))

            if "episode" in info:
                self._window_returns.append(float(info["episode"]["r"]))
                self._window_lengths.append(float(info["episode"]["l"]))
                self._total_episodes += 1

                for comp, val in self._current_ep_components[i].items():
                    self._ep_component_history[comp].append(val)
                self._current_ep_components[i].clear()

        if self.num_timesteps % self.log_freq == 0 and self._window_returns:
            rew = np.array(self._window_returns)
            self.logger.record(f"{self.run_name}/ep_rew_mean",    float(rew.mean()))
            self.logger.record(f"{self.run_name}/ep_rew_min",     float(rew.min()))
            self.logger.record(f"{self.run_name}/ep_rew_max",     float(rew.max()))
            self.logger.record(f"{self.run_name}/ep_len_mean",    float(np.mean(self._window_lengths)))
            self.logger.record(f"{self.run_name}/episodes_total", self._total_episodes)

            for component, values in self._ep_component_history.items():
                if values:
                    self.logger.record(
                        f"{self.run_name}/ep_sum_{component}",
                        float(np.mean(values))
                    )

            self.logger.dump(self.num_timesteps)

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
        self.log_freq         = log_freq
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

TOTAL_TIMESTEPS = 10_000_000

_logical_cores = os.cpu_count() or 8
N_ENVS         = 30
_DYNAMIC_BATCH = 4096  # 30 envs * 2048 steps per env = 61,440 total steps per rollout

print(f"Auto-selected N_ENVS={N_ENVS} for {_logical_cores} logical CPU cores")

PPO_KWARGS = dict(
    n_steps       = 2048,    # 30 envs * 2048 = 61,440 steps per rollout
    batch_size    = _DYNAMIC_BATCH,    # 61,440 / 4096 = 15 minibatches per epoch
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.005,   # Lowered to stop the robot from vibrating/twitching
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    learning_rate = 3e-4,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    verbose       = 0,
    tensorboard_log = "./ppo_franka_tb_new/",
)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(
    run_name          : str  = "ppo_shaped_asr_run_1",
    use_asr           : bool = True,
    use_shaped_reward : bool = True,
):
    obs_label = "ASR-51dim" if use_asr else "FULL-73dim"
    rew_label = "SHAPED"    if use_shaped_reward else "SPARSE"
    print(f"Run: {run_name}  |  Obs: {obs_label}  |  Reward: {rew_label}")
    print("Setting up vectorised environments...")

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

    model = PPO("MlpPolicy", vec_env, **PPO_KWARGS, device="cuda")

    print(f"\nPolicy architecture:\n{model.policy}\n")
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} parallel envs...\n")

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [log_callback, tb_callback, eval_callback, checkpoint_callback],
        progress_bar    = True,
    )

    save_path = f"ppo_franka_kitchen_{run_name}_final"
    model.save(save_path)
    print(f"\nModel saved -> {save_path}.zip")

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
    print(f"Plot saved -> {fname}")


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
    env = AugmentedObsWrapper(env)
    if record_video:
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda _: True)

    model = PPO.load(model_path, env=env)
    print(f"\nLoaded: {model_path}  |  ASR={use_asr}  |  Shaped={use_shaped_reward}")
    print(f"Running {num_episodes} evaluation episodes...\n")

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
        print(f"Videos saved -> {os.path.abspath(video_folder)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_training", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--model_path", type=str,
                        default="./ppo_franka_best_ppo_shaped_asr_run_4/best_model.zip")
    args = parser.parse_args()

    if args.run_training:
        log_cb = train(run_name="ppo_shaped_asr_run_4", use_asr=True, use_shaped_reward=True)
        plot_results(log_cb.episode_returns, title_suffix="Shaped ASR Augmented")
    else:
        evaluate(
            model_path        = args.model_path,
            num_episodes      = 10,
            record_video      = args.record_video,
            use_asr           = True,
            use_shaped_reward = True,
        )