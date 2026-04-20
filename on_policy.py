import argparse
import time
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
# REWARD SHAPING  —  DenseRewardWrapper
# ─────────────────────────────────────────────────────────────────────────────
#
#  Changes from previous version:
#  • Lift bonus GATED on _has_grasped  — wedging under with open fingers
#    no longer earns any lift reward
#  • Transport GATED on _has_grasped   — same gate, no free transport
#  • Wedge penalty                     — active penalty for lifting with
#    open gripper, directly discourages the observed failure mode
#  • Grasp threshold tightened         — 0.045 → 0.030 m gripper gap
#  • _is_touching_kettle uses cached body IDs and checks 'kettleroot'
#    (all kettle collision geoms are parented to kettleroot, not kettle)
#
# ─────────────────────────────────────────────────────────────────────────────

_TASK_MAX_DIST = {
    "microwave": 0.37,
    "kettle":    1.20,
}
_KETTLE_LIFT_MIN = 0.04

_GRIPPER_R_IDX   = 7
_GRIPPER_L_IDX   = 8
_GRIPPER_MAX_GAP = 0.08

_KETTLE_XYZ_IDX  = [32, 33, 34]


class DenseRewardWrapper(gym.Wrapper):
    """
    Grasp-aware dense reward wrapper for Franka Kitchen — kettle task.

    Fixes applied vs. previous version
    ────────────────────────────────────
    BUG 1  — Transport stale-distance fix:
               prev_distances snapshot is taken BEFORE the per-task loop
               so the transport block reads the distance from the previous
               step, not the one that was just overwritten.

    BUG 2  — Push penalty capped at 5× threshold to prevent runaway
               negative spikes that dominated early training.

    BUG 3  — Premature-closure fix (the primary reported failure mode):
               • opposition is now computed early and promoted to a GATE
                 for both closure and coordination rewards.
               • open_approach reward: keeps gripper open while closing in.
               • premature_close penalty: fires when fingers close before
                 they are opposing each other across the handle.
               • coordination guard tightened with fingers_opposing gate.
               • closure spatial threshold widened to 0.06 m (was 0.04 m)
                 so the signal can actually fire once fingers are in place.

    Reward ladder (roughly ordered by episode timeline)
    ────────────────────────────────────────────────────
      global_reach       potential-based, always active pre-grasp
      approach           potential-based inside 0.25 m pre-grasp
      alignment          direct value, orientation toward handle
      proximity          exponential pull inside 0.15 m
      open_approach      keep gripper open while closing in       [NEW]
      finger_centering   midpoint + opposition bonus
      premature_close    penalty for closing without opposition    [NEW]
      closure            gated on fingers_opposing + touch        [FIXED]
      coordination       gated on fingers_opposing                [FIXED]
      grasp_milestone    one-time +30 on confirmed grasp
      lift_milestone     one-time +20 on grasped lift             [bumped]
      transport_gated    potential-based, gated on lift+grasp     [FIXED]
      completion         one-time +150
      wedge_penalty      per-step penalty for open-gripper lift
      push_penalty       capped per-step penalty for sliding kettle [FIXED]
      time_penalty       small constant per step
    """

    # ── Shared weights ────────────────────────────────────────────────────────
    W_DISTANCE       = 0.0
    W_PROGRESS       = 1.0
    COMPLETION_BONUS = 150.0
    TIME_PENALTY     = -0.02

    # ── Kettle-specific weights ───────────────────────────────────────────────
    W_APPROACH            = 25.0
    APPROACH_THRESH       = 0.20
    BONUS_GRASP           = 30.0
    BONUS_LIFT            = 20.0      # bumped: lift > grasp makes more sense
    W_TRANSPORT           = 15.0
    W_PUSH_PENALTY        = 1.0
    PUSH_MOVE_THRESH      = 0.003
    PUSH_PENALTY_CAP      = 5.0       # NEW: caps multiplier so penalty ≤ 5×
    W_WEDGE_PENALTY       = 1.5
    W_OPEN_APPROACH       = 2.0       # NEW: reward keeping gripper open while closing in
    W_PREMATURE_CLOSE     = 3.0       # NEW: penalty for closing without opposition
    OPPOSITION_GATE       = -0.5      # NEW: dot-product threshold to consider fingers opposing
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, env, tasks: list[str] = TASKS):
        super().__init__(env)
        self.tasks = [t for t in tasks if t in TASK_GOALS]
        if not self.tasks:
            raise ValueError(f"None of {tasks} found in TASK_GOALS.")

        self._prev_dist:       dict[str, float]  = {}
        self._task_done:       dict[str, bool]   = {}
        self._prev_kettle_xy:  np.ndarray | None = None
        self._prev_kettle_xyz: np.ndarray | None = None

        self._min_ee_to_kettle:  float       = float('inf')
        self._prev_ee_to_kettle: float | None = None
        self._has_grasped:       bool         = False
        self._has_lifted:        bool         = False

        _model = self.env.unwrapped.model
        self._kettle_bodies = {
            mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ('kettle', 'kettleroot')
        } - {-1}
        self._finger_bodies = {
            mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ('panda0_leftfinger', 'panda0_rightfinger')
        } - {-1}

        self._lf_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, 'panda0_leftfinger')
        self._rf_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, 'panda0_rightfinger')

    # ── MuJoCo helpers ────────────────────────────────────────────────────────

    def _ee_xyz(self) -> np.ndarray | None:
        try:
            model = self.env.unwrapped.model
            data  = self.env.unwrapped.data
            uid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
            if uid == -1: return None
            return data.site_xpos[uid].copy()
        except Exception:
            return None

    def _ee_xmat(self) -> np.ndarray | None:
        try:
            model = self.env.unwrapped.model
            data  = self.env.unwrapped.data
            uid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
            if uid == -1: return None
            return data.site_xmat[uid].reshape(3, 3).copy()
        except Exception:
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
            data  = self.env.unwrapped.data
            model = self.env.unwrapped.model
            if not self._kettle_bodies or not self._finger_bodies: return False
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
        """Returns (lf_pos, rf_pos) or (None, None) on failure."""
        try:
            data   = self.env.unwrapped.data
            lf_pos = data.xpos[self._lf_id].copy() if self._lf_id != -1 else None
            rf_pos = data.xpos[self._rf_id].copy() if self._rf_id != -1 else None
            return lf_pos, rf_pos
        except Exception:
            return None, None

    # ── Episode reset ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw_obs   = obs["observation"]
        for task in self.tasks:
            self._prev_dist[task] = self._task_distance(raw_obs, task)
            self._task_done[task] = False

        self._prev_kettle_xy   = self._kettle_xyz(raw_obs)[:2].copy()
        self._prev_kettle_xyz  = self._kettle_xyz(raw_obs).copy()
        self._initial_kettle_z = float(self._kettle_xyz(raw_obs)[2])

        self._min_ee_to_kettle  = float('inf')
        self._prev_ee_to_kettle = None
        self._has_grasped       = False
        self._has_lifted        = False

        return obs, info

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        raw_obs = obs["observation"]

        shaped = 0.0
        log    = {}

        shaped += self.TIME_PENALTY
        log["time_penalty"] = self.TIME_PENALTY

        kettle_xyz  = self._kettle_xyz(raw_obs)
        gripper_gap = self._gripper_gap(raw_obs)
        ee_xyz      = self._ee_xyz()
        ee_xmat     = self._ee_xmat()

        # ── BUG 1 FIX: snapshot distances BEFORE the per-task loop ────────────
        # The loop overwrites self._prev_dist[task] at the end of each
        # iteration. The transport block (step 10) needs the value from
        # the *previous* step, so we capture it here.
        prev_distances_snapshot = {t: self._prev_dist[t] for t in self.tasks}

        # ── Per-task shared terms ─────────────────────────────────────────────
        for task in self.tasks:
            if self._task_done[task]: continue

            curr_dist = self._task_distance(raw_obs, task)
            prev_dist = self._prev_dist[task]
            max_dist  = _TASK_MAX_DIST.get(task, 1.0)

            if curr_dist < 0.05 * max_dist:
                shaped += self.COMPLETION_BONUS
                self._task_done[task] = True
                log[f"completion_{task}"] = self.COMPLETION_BONUS

            if self.W_DISTANCE > 0:
                norm_dist = curr_dist / max(max_dist, 1e-6)
                r_dist    = self.W_DISTANCE * (1.0 - norm_dist)
                shaped   += r_dist
                log[f"dist_{task}"] = r_dist

            if task != "kettle":
                r_prog  = self.W_PROGRESS * (prev_dist - curr_dist)
                shaped += r_prog
                log[f"progress_{task}"] = r_prog

            self._prev_dist[task] = curr_dist   # update after reading

        # ── Kettle-specific grasp enforcement ─────────────────────────────────
        if "kettle" in self.tasks and not self._task_done.get("kettle", False):

            kettle_z   = float(kettle_xyz[2])
            handle_xyz = self._kettle_handle_xyz()

            ee_to_kettle = (
                float(np.linalg.norm(ee_xyz - handle_xyz))
                if (ee_xyz is not None and handle_xyz is not None)
                else float('inf')
            )

            # ── Finger geometry — computed once, used by multiple blocks ──────
            lf_pos, rf_pos = self._finger_positions()

            opposition      = 0.0   # dot product of unit vectors finger→handle
            fingers_opposing = False
            midpoint        = None

            if lf_pos is not None and rf_pos is not None and handle_xyz is not None:
                midpoint     = (lf_pos + rf_pos) / 2.0
                lf_to_handle = handle_xyz - lf_pos
                rf_to_handle = handle_xyz - rf_pos
                lf_norm      = np.linalg.norm(lf_to_handle)
                rf_norm      = np.linalg.norm(rf_to_handle)
                if lf_norm > 1e-6 and rf_norm > 1e-6:
                    opposition = float(
                        np.dot(lf_to_handle / lf_norm, rf_to_handle / rf_norm)
                    )
                    # opposition < OPPOSITION_GATE  →  fingers are on opposite
                    # sides of the handle, i.e. in a valid pre-grasp posture
                    fingers_opposing = opposition < self.OPPOSITION_GATE

            # ── 1. GLOBAL REACH — potential-based, no gate ────────────────────
            r_global_reach = 0.0
            if not self._has_grasped and ee_xyz is not None and handle_xyz is not None:
                if self._prev_ee_to_kettle is not None:
                    r_global_reach = 8.0 * (self._prev_ee_to_kettle - ee_to_kettle)
            shaped += r_global_reach
            log["global_reach"] = r_global_reach

            # ── 2. APPROACH — potential-based inside 0.25 m ───────────────────
            r_approach = 0.0
            if ee_to_kettle < 0.25 and not self._has_grasped:
                if self._prev_ee_to_kettle is not None:
                    r_approach = self.W_APPROACH * (self._prev_ee_to_kettle - ee_to_kettle)
            shaped += r_approach
            log["approach"] = r_approach

            # Update distance tracker
            self._prev_ee_to_kettle = ee_to_kettle

            # ── 3. ORIENTATION ALIGNMENT ──────────────────────────────────────
            r_align = 0.0
            if ee_xmat is not None and ee_xyz is not None and handle_xyz is not None:
                if ee_to_kettle < 0.25 and not self._has_grasped:
                    dir_to_handle = handle_xyz - ee_xyz
                    norm = np.linalg.norm(dir_to_handle)
                    if norm > 1e-6:
                        dir_to_handle  /= norm
                        gripper_forward = ee_xmat[:, 2]
                        alignment_val   = float(np.dot(gripper_forward, dir_to_handle))
                        if alignment_val > 0:
                            r_align = 3.0 * alignment_val * (0.25 - ee_to_kettle)
            shaped += r_align
            log["alignment"] = r_align

            # ── 4. PROXIMITY — exponential pull inside 0.15 m ────────────────
            r_proximity = 0.0
            if not self._has_grasped and ee_to_kettle < 0.15:
                r_proximity = 3.0 * (np.exp(6.0 * (0.15 - ee_to_kettle)) - 1.0)
            shaped += r_proximity
            log["proximity"] = r_proximity

            # ── 5. OPEN-APPROACH — reward keeping gripper open while closing in
            # [NEW] Prevents the agent from pre-closing before it reaches the
            # handle. Pays proportionally to both openness and proximity.
            r_open_approach = 0.0
            if not self._has_grasped and ee_to_kettle < 0.15:
                openness        = gripper_gap / _GRIPPER_MAX_GAP   # 1=open, 0=closed
                r_open_approach = self.W_OPEN_APPROACH * openness * (0.15 - ee_to_kettle)
            shaped += r_open_approach
            log["open_approach"] = r_open_approach

            # ── 6. FINGER CENTERING ───────────────────────────────────────────
            r_center = 0.0
            if not self._has_grasped and handle_xyz is not None and midpoint is not None:
                mid_to_handle = float(np.linalg.norm(midpoint - handle_xyz))
                if mid_to_handle < 0.08:
                    r_center = 1.5 * (0.08 - mid_to_handle)
                    if mid_to_handle < 0.032:
                        r_center += 0.1

                # opposition bonus
                if opposition < 0:
                    r_center += 1.5 * (-opposition)

                # ── HANDLE AXIS ALIGNMENT — NEW ───────────────────────────────
                # The handle is a horizontal capsule running along X in local frame
                # (euler="0 1.57 0"). Penalise the finger midpoint sliding toward
                # either tip of the handle (±0.1 m from center) instead of gripping
                # the middle. Only fires when fingers are already close enough to
                # matter (mid_to_handle < 0.08).
                if mid_to_handle < 0.08:
                    try:
                        data  = self.env.unwrapped.data
                        model = self.env.unwrapped.model
                        kb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'kettleroot')
                        if kb_id != -1:
                            kettle_xmat       = data.xmat[kb_id].reshape(3, 3)
                            handle_axis_world = kettle_xmat @ np.array([1.0, 0.0, 0.0])
                            axial_offset      = float(np.dot(midpoint - handle_xyz, handle_axis_world))
                            if abs(axial_offset) > 0.07:   # drifting toward handle tip
                                r_center -= 1.0 * (abs(axial_offset) - 0.07)
                    except Exception:
                        pass

            shaped += r_center
            log["finger_centering"] = r_center

            # ── 7. PREMATURE CLOSE PENALTY ────────────────────────────────────
            # [NEW] Active penalty when the gripper is closing inside approach
            # range but the fingers are NOT yet opposing each other.
            # This directly counteracts the failure mode of pre-closing.
            r_premature_close = 0.0
            if not self._has_grasped and ee_to_kettle < 0.10 and not fingers_opposing:
                if gripper_gap < 0.06:   # fingers have started closing
                    r_premature_close = -self.W_PREMATURE_CLOSE * (0.06 - gripper_gap)
            shaped += r_premature_close
            log["premature_close"] = r_premature_close

            # ── 8. CLOSURE — gated on fingers_opposing + touch ────────────────
            # [FIXED] Now requires fingers_opposing so closure reward only
            # fires when the fingers are actually straddling the handle.
            # Spatial threshold widened to 0.06 m (was 0.04 m) so the gate
            # can trigger once the fingers are properly positioned.
            r_closure = 0.0
            if not self._has_grasped and ee_to_kettle < 0.06 and fingers_opposing:
                if self._is_touching_kettle():
                    # 0 when fully open (0.08 m), 1 at handle contact width (0.064 m)
                    closure_progress = (0.08 - gripper_gap) / (0.08 - 0.064)
                    r_closure = 6.0 * np.clip(closure_progress, 0.0, 1.0)
            shaped += r_closure
            log["closure"] = r_closure

            # ── 9. COORDINATION BONUS — gated on fingers_opposing ─────────────
            # [FIXED] fingers_opposing gate added so this can't fire while
            # the gripper is pre-closing from the wrong position.
            r_coord = 0.0
            if (not self._has_grasped and ee_to_kettle < 0.07
                    and gripper_gap < 0.05 and fingers_opposing):
                r_coord = 4.0 * (0.07 - ee_to_kettle) * (0.05 - gripper_gap)
            shaped += r_coord
            log["coordination"] = r_coord

            # ── 10. GRASP MILESTONE ───────────────────────────────────────────
            r_grasp = 0.0
            if not self._has_grasped and ee_to_kettle < 0.06 and fingers_opposing:
                if gripper_gap < 0.075 and self._is_touching_kettle():
                    r_grasp           = self.BONUS_GRASP
                    self._has_grasped = True
            shaped += r_grasp
            log["grasp_milestone"] = r_grasp

            # ── 11. LIFT MILESTONE ────────────────────────────────────────────
            is_lifted = kettle_z > (self._initial_kettle_z + _KETTLE_LIFT_MIN)
            r_lift = 0.0
            if not self._has_lifted and is_lifted and self._has_grasped:
                r_lift           = self.BONUS_LIFT
                self._has_lifted = True
            shaped += r_lift
            log["lift_milestone"] = r_lift

            # ── 12. WEDGE PENALTY ─────────────────────────────────────────────
            # r_wedge = 0.0
            # if is_lifted and not self._has_grasped and gripper_gap > 0.03:
            #     r_wedge = -self.W_WEDGE_PENALTY
            # shaped += r_wedge
            # log["wedge_penalty"] = r_wedge

            # ── 13. GATED TRANSPORT ───────────────────────────────────────────
            # [FIXED] Uses prev_distances_snapshot which was captured before
            # the per-task loop updated self._prev_dist["kettle"]. This means
            # `prev_kettle_dist` is genuinely from the previous timestep.
            r_transport = 0.0
            if is_lifted and self._has_grasped:
                prev_kettle_dist = prev_distances_snapshot.get("kettle", 0.0)
                curr_kettle_dist = self._task_distance(raw_obs, "kettle")
                progress         = prev_kettle_dist - curr_kettle_dist
                if progress > 0:
                    r_transport = self.W_TRANSPORT * progress
            shaped += r_transport
            log["transport_gated"] = r_transport

            # ── 14. PUSH PENALTY — capped ─────────────────────────────────────
            # [FIXED] Raw multiplier (xy_movement / PUSH_MOVE_THRESH) can be
            # very large (e.g. 10×) and dominate early training. Capped at
            # PUSH_PENALTY_CAP to bound the per-step magnitude.
            # r_push_penalty = 0.0
            # if self._prev_kettle_xy is not None:
            #     xy_movement = float(np.linalg.norm(kettle_xyz[:2] - self._prev_kettle_xy))
            #     if xy_movement > self.PUSH_MOVE_THRESH and not self._has_grasped:
            #         if ee_to_kettle > 0.08:
            #             multiplier     = min(xy_movement / self.PUSH_MOVE_THRESH,
            #                                  self.PUSH_PENALTY_CAP)
            #             r_push_penalty = -self.W_PUSH_PENALTY * multiplier
            # shaped += r_push_penalty
            # log["push_penalty"] = r_push_penalty

        # ── Update trackers ───────────────────────────────────────────────────
        self._prev_kettle_xy  = kettle_xyz[:2].copy()
        self._prev_kettle_xyz = kettle_xyz.copy()

        total_reward = env_reward + shaped

        info["shaped_reward"]    = float(shaped)
        info["original_reward"]  = float(env_reward)
        info["reward_breakdown"] = log

        return obs, total_reward, terminated, truncated, info


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


# class ASRObsWrapper(gym.ObservationWrapper):
#     """Keeps full 59-dim obs and appends velocity dims at the end."""

#     def __init__(self, env, asr_indices: list[int] = ASR_INDICES):
#         super().__init__(env)
#         self._asr_indices = np.array(asr_indices, dtype=np.int32)
#         full_dim = env.observation_space.shape[0]
#         self._full_dim = full_dim

#         # Derive velocity indices as everything NOT in asr_indices
#         asr_set = set(asr_indices)
#         self._velocity_indices = np.array(
#             [i for i in range(full_dim) if i not in asr_set], dtype=np.int32
#         )

#         # New shape: full obs + velocity dims appended
#         new_dim = full_dim + len(self._velocity_indices)
#         self.observation_space = Box(
#             low=-np.inf, high=np.inf,
#             shape=(new_dim,), dtype=np.float32,
#         )
#         print(
#             f"[ASRObsWrapper] {full_dim}D → {new_dim}D  "
#             f"(kept all, appended {len(self._velocity_indices)} velocity dims)"
#         )

#     def observation(self, obs: np.ndarray) -> np.ndarray:
#         velocity_dims = obs[self._velocity_indices]
#         return np.concatenate([obs, velocity_dims]).astype(np.float32)


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


class AugmentedObsWrapper(gym.ObservationWrapper):
    """
    Appends task-relevant derived features to the ASR observation.

    Added dims (15 total):
      [0:3] — EE → kettle handle vector (3D)   tells the agent exactly which
               direction to move without having to discover subtraction
      [3]   — EE → handle scalar distance       salient proximity signal
      [4]   — gripper gap                        already in ASR but made salient
                                                 here so it's at the end of obs
      [5]   — is_touching_kettle (0/1)           contact signal the policy can't
                                                 observe from positions alone

    Wrapper stack order (must be outermost after ASRObsWrapper):
      FrankaKitchen-v1
          └─ DenseRewardWrapper
              └─ FlattenObsWrapper
                  └─ ASRObsWrapper
                      └─ AugmentedObsWrapper   ← here
                          └─ Monitor
    """

    N_EXTRA = 15   # number of appended dims

    def __init__(self, env):
        super().__init__(env)
        base_dim = env.observation_space.shape[0]
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + self.N_EXTRA,), dtype=np.float32,
        )
        print(f"[AugmentedObsWrapper] {base_dim}D → {base_dim + self.N_EXTRA}D "
              f"(+{self.N_EXTRA} derived features)")

    def _get_raw_env(self):
        """Walk the wrapper stack down to the unwrapped MuJoCo env."""
        e = self.env
        while hasattr(e, 'env'):
            e = e.env
        return e.unwrapped

    def _ee_xyz(self) -> np.ndarray | None:
        try:
            raw = self._get_raw_env()
            uid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
            return raw.data.site_xpos[uid].copy() if uid != -1 else None
        except Exception:
            return None

    def _handle_xyz(self) -> np.ndarray | None:
        try:
            raw = self._get_raw_env()
            uid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_SITE, 'kettle_site')
            return raw.data.site_xpos[uid].copy() if uid != -1 else None
        except Exception:
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
            return False

    def observation(self, obs: np.ndarray) -> np.ndarray:
        ee_xyz     = self._ee_xyz()
        handle_xyz = self._handle_xyz()

        if ee_xyz is not None and handle_xyz is not None:
            delta    = (handle_xyz - ee_xyz).astype(np.float32)
            distance = np.float32(np.linalg.norm(delta))
        else:
            delta    = np.zeros(3, dtype=np.float32)
            distance = np.float32(0.5)

        gripper_gap = np.float32(obs[7] + obs[8])
        touching    = np.float32(self._is_touching())

        # ── NEW: individual finger → handle deltas ────────────────────────────
        lf_delta = np.zeros(3, dtype=np.float32)
        rf_delta = np.zeros(3, dtype=np.float32)
        if handle_xyz is not None:
            try:
                raw   = self._get_raw_env()
                lf_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, 'panda0_leftfinger')
                rf_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, 'panda0_rightfinger')
                if lf_id != -1:
                    lf_delta = (handle_xyz - raw.data.xpos[lf_id]).astype(np.float32)
                if rf_id != -1:
                    rf_delta = (handle_xyz - raw.data.xpos[rf_id]).astype(np.float32)
            except Exception:
                pass

        # ── NEW: gripper forward alignment scalar ─────────────────────────────
        alignment = np.float32(0.0)
        try:
            raw = self._get_raw_env()
            uid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
            if uid != -1 and distance > 1e-6:
                gripper_forward = raw.data.site_xmat[uid].reshape(3, 3)[:, 2]
                alignment = np.float32(np.dot(gripper_forward, delta / distance))
        except Exception:
            pass

        # ── NEW: has_grasped flag ─────────────────────────────────────────────
        # Walk up to DenseRewardWrapper to read its milestone state
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

        # ── NEW: kettle delta-z from initial ─────────────────────────────────
        kettle_dz = np.float32(0.0)
        try:
            e = self.env
            while hasattr(e, 'env'):
                if hasattr(e, '_initial_kettle_z'):
                    current_z  = np.float32(obs[25])  # kettle Z in ASR
                    kettle_dz  = current_z - np.float32(e._initial_kettle_z)
                    break
                e = e.env
        except Exception:
            pass

        extra = np.concatenate([
            delta,          # 3D  — EE→handle (existing)
            [distance],     # 1D  — scalar distance (existing)
            [gripper_gap],  # 1D  — gap (existing, kept for explicitness)
            [touching],     # 1D  — contact flag (existing)
            lf_delta,       # 3D  — left finger→handle (NEW)
            rf_delta,       # 3D  — right finger→handle (NEW)
            [alignment],    # 1D  — forward alignment scalar (NEW)
            [has_grasped],  # 1D  — milestone flag (NEW)
            [kettle_dz],    # 1D  — lift progress (NEW)
        ]).astype(np.float32)

        return np.concatenate([obs, extra])


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────
# Wrapper stacking order (innermost → outermost):
#   FrankaKitchen-v1  (raw Dict obs, sparse binary reward)
#       └─ DenseRewardWrapper   ← reads raw Dict obs, augments reward
#           └─ FlattenObsWrapper ← Dict → flat 59-dim Box
#               └─ ASRObsWrapper ← 59-dim → 37-dim (drop velocities)
#                   └─ AugmentedObsWrapper ← 37-dim → 43-dim (+6 derived)
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
        env = AugmentedObsWrapper(env)   # ← NEW: appends derived features
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
    batch_size    = _DYNAMIC_BATCH,    # 61,440 / 4096 = 15 minibatches per epoch (Perfect for stability)
    n_epochs      = 10,      
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.005,   # Lowered to stop the robot from vibrating/twitching
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    learning_rate = 3e-4,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),  # larger network
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
    obs_label = "ASR-43dim" if use_asr else "FULL-65dim"
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

    model = PPO("MlpPolicy", vec_env, **PPO_KWARGS, device="cuda")

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
    video_folder = "franka_kitchen_eval_videos_dim_reduction" if record_video else None

    env = gym.make('FrankaKitchen-v1', tasks_to_complete=TASKS, render_mode=render_mode)
    # Use this to see all available camera names
    # Print only camera name
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
    print(f"Running {num_episodes} evaluation episodes…\n")

    ep_returns = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_ret, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if not record_video:
                time.sleep(0.07)  # slow down for human viewing when not recording
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_training", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--model_path", type=str,
                        default="ppo_franka_best_ppo_franka_best_ppo_shaped_asr_run_3_no_push_penalty_reduced_dim/best_model.zip",)
    args = parser.parse_args()

    if args.run_training:
        log_cb = train(run_name="ppo_franka_best_ppo_shaped_asr_run_3_no_push_penalty_reduced_dim", use_asr=True, use_shaped_reward=True)
        plot_results(log_cb.episode_returns, title_suffix="Shaped ASR Augmented DIM Reduction")
    else:
        evaluate(
            model_path        = args.model_path,
            num_episodes      = 10,
            record_video      = args.record_video,
            use_asr           = True,
            use_shaped_reward = True,
        )