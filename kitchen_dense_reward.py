from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class KitchenDenseRewardConfig:
    goal_epsilon: float = 0.3
    success_bonus: float = 150.0
    goal_distance_weight: float = 10.0
    goal_progress_weight: float = 30.0
    goal_progress_clip: float = 0.05
    arm_distance_weight: float = 0.15
    arm_distance_clip: float = 0.7
    arm_gate_width: float = 0.25
    arm_progress_weight: float = 2.0
    arm_progress_clip: float = 0.05
    action_penalty_weight: float = 0.001
    time_penalty_weight: float = 0.02
    timeout_failure_penalty: float = 40.0
    success_task_name: str = "kettle"
    ee_name_keywords: Tuple[str, ...] = (
        "end_effector",
        "eef",
        "gripper",
        "panda_hand",
        "hand",
    )
    kettle_name_keywords: Tuple[str, ...] = ("kettle",)


def _flatten_goal(goal: Any) -> np.ndarray:
    if isinstance(goal, dict):
        parts = [_flatten_goal(goal[k]) for k in sorted(goal.keys())]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)
    return np.asarray(goal, dtype=np.float32).reshape(-1)


def _goal_distance(achieved_goal: Any, desired_goal: Any) -> float:
    ag = _flatten_goal(achieved_goal)
    dg = _flatten_goal(desired_goal)
    if ag.shape != dg.shape:
        raise ValueError(
            f"Goal shapes do not match after flattening: achieved={ag.shape}, desired={dg.shape}"
        )
    return float(np.linalg.norm(ag - dg, ord=2))


def _action_penalty(action: Any) -> float:
    if action is None:
        return 0.0
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.square(a)))


def _is_batch_goal(goal: Any) -> bool:
    if isinstance(goal, np.ndarray):
        return goal.ndim >= 2
    if isinstance(goal, (list, tuple)):
        if not goal:
            return False
        first = goal[0]
        return isinstance(first, (dict, np.ndarray, list, tuple))
    return False


def _batch_size(goal: Any) -> int:
    if isinstance(goal, np.ndarray):
        return int(goal.shape[0])
    if isinstance(goal, (list, tuple)):
        return len(goal)
    raise TypeError(f"Unsupported batch goal type: {type(goal)}")


def _get_goal_at(goal: Any, idx: int) -> Any:
    if isinstance(goal, np.ndarray):
        return goal[idx]
    if isinstance(goal, (list, tuple)):
        return goal[idx]
    raise TypeError(f"Unsupported batch goal type: {type(goal)}")


def _expand_info(info: Any, batch_size: int):
    if isinstance(info, (list, tuple)):
        info_list = list(info)
        if len(info_list) == batch_size:
            return info_list
        if len(info_list) == 1:
            return info_list * batch_size
        raise ValueError(
            f"Batch reward computation got batch_size={batch_size} but len(info)={len(info_list)}."
        )
    return [info] * batch_size


class KitchenDenseRewardWrapper(gym.Wrapper):
    """
    Dense reward for Franka Kitchen with explicit kettle-centric shaping.

    Reward terms:
      + success bonus when task is solved
      - distance between achieved and desired goal states
      + progress in achieved/desired goal distance (delta over previous step)
      - gated distance between end-effector and kettle
      + progress in end-effector-to-kettle distance (small)
      - action magnitude penalty
      - small time penalty
      - timeout penalty when episode truncates without success

    HER safety:
      compute_reward() must not depend on the wrapper's live simulator state.
      It only uses (achieved_goal, desired_goal) plus transition-local info.
    """

    def __init__(self, env: gym.Env, config: KitchenDenseRewardConfig | None = None):
        super().__init__(env)
        self.config = config or KitchenDenseRewardConfig()
        self._refs_resolved = False
        self._ee_ref: Tuple[str, str] | None = None
        self._kettle_ref: Tuple[str, str] | None = None
        self._prev_goal_distance: float | None = None
        self._prev_arm_distance: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict) and "achieved_goal" in obs and "desired_goal" in obs:
            self._prev_goal_distance = _goal_distance(obs["achieved_goal"], obs["desired_goal"])
        else:
            self._prev_goal_distance = None
        self._prev_arm_distance = self._ee_to_kettle_distance()
        return obs, info

    def _get_model_data(self):
        base = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

        model = getattr(base, "model", None)
        data = getattr(base, "data", None)
        if model is not None and data is not None:
            return model, data

        sim = getattr(base, "sim", None)
        if sim is not None:
            model = getattr(sim, "model", None)
            data = getattr(sim, "data", None)
            if model is not None and data is not None:
                return model, data

        return None, None

    @staticmethod
    def _list_entity_names(model: Any, entity: str) -> list[str]:
        names: list[str] = []
        seq_names = getattr(model, f"{entity}_names", None)
        if isinstance(seq_names, (list, tuple)):
            for name in seq_names:
                if isinstance(name, str) and name:
                    names.append(name)

        n = int(getattr(model, f"n{entity}", 0))
        accessor = getattr(model, entity, None)
        if callable(accessor):
            for i in range(n):
                try:
                    name = accessor(i).name
                except Exception:
                    continue
                if isinstance(name, str) and name:
                    names.append(name)
        deduped = []
        seen = set()
        for name in names:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped

    @staticmethod
    def _match_name(names: Sequence[str], keywords: Sequence[str]) -> str | None:
        lowered = [(name, name.lower()) for name in names]
        for kw in keywords:
            kw_low = kw.lower()
            for name, lower_name in lowered:
                if kw_low in lower_name:
                    return name
        return None

    def _resolve_named_refs(self):
        if self._refs_resolved:
            return

        model, _ = self._get_model_data()
        if model is None:
            self._refs_resolved = True
            return

        site_names = self._list_entity_names(model, "site")
        body_names = self._list_entity_names(model, "body")

        ee_site = self._match_name(site_names, self.config.ee_name_keywords)
        ee_body = self._match_name(body_names, self.config.ee_name_keywords)
        kettle_site = self._match_name(site_names, self.config.kettle_name_keywords)
        kettle_body = self._match_name(body_names, self.config.kettle_name_keywords)

        if ee_site is not None:
            self._ee_ref = ("site", ee_site)
        elif ee_body is not None:
            self._ee_ref = ("body", ee_body)

        if kettle_site is not None:
            self._kettle_ref = ("site", kettle_site)
        elif kettle_body is not None:
            self._kettle_ref = ("body", kettle_body)

        self._refs_resolved = True

    @staticmethod
    def _site_position(model: Any, data: Any, name: str) -> np.ndarray | None:
        try:
            if hasattr(data, "site") and callable(data.site):
                return np.asarray(data.site(name).xpos, dtype=np.float32)
        except Exception:
            pass
        try:
            if hasattr(model, "site") and callable(model.site) and hasattr(data, "site_xpos"):
                sid = model.site(name).id
                return np.asarray(data.site_xpos[sid], dtype=np.float32)
        except Exception:
            pass
        try:
            if hasattr(model, "site_name2id") and callable(model.site_name2id) and hasattr(
                data, "site_xpos"
            ):
                sid = model.site_name2id(name)
                return np.asarray(data.site_xpos[sid], dtype=np.float32)
        except Exception:
            pass
        return None

    @staticmethod
    def _body_position(model: Any, data: Any, name: str) -> np.ndarray | None:
        try:
            if hasattr(data, "body") and callable(data.body):
                return np.asarray(data.body(name).xpos, dtype=np.float32)
        except Exception:
            pass
        try:
            if hasattr(model, "body") and callable(model.body):
                bid = model.body(name).id
                if hasattr(data, "xpos"):
                    return np.asarray(data.xpos[bid], dtype=np.float32)
                if hasattr(data, "body_xpos"):
                    return np.asarray(data.body_xpos[bid], dtype=np.float32)
        except Exception:
            pass
        try:
            if hasattr(model, "body_name2id") and callable(model.body_name2id):
                bid = model.body_name2id(name)
                if hasattr(data, "xpos"):
                    return np.asarray(data.xpos[bid], dtype=np.float32)
                if hasattr(data, "body_xpos"):
                    return np.asarray(data.body_xpos[bid], dtype=np.float32)
        except Exception:
            pass
        return None

    def _named_position(self, ref: Tuple[str, str] | None) -> np.ndarray | None:
        if ref is None:
            return None
        kind, name = ref
        model, data = self._get_model_data()
        if model is None or data is None:
            return None
        if kind == "site":
            return self._site_position(model, data, name)
        if kind == "body":
            return self._body_position(model, data, name)
        return None

    def _ee_to_kettle_distance(self) -> float | None:
        if not self._refs_resolved:
            self._resolve_named_refs()

        ee_pos = self._named_position(self._ee_ref)
        kettle_pos = self._named_position(self._kettle_ref)
        if ee_pos is None or kettle_pos is None:
            return None
        return float(np.linalg.norm(ee_pos - kettle_pos, ord=2))

    @staticmethod
    def _arm_distance_from_info(info: Any) -> float | None:
        if not isinstance(info, dict):
            return None
        if "ee_to_kettle_distance" not in info:
            return None
        try:
            return float(info["ee_to_kettle_distance"])
        except Exception:
            return None

    @staticmethod
    def _prev_goal_distance_from_info(info: Any) -> float | None:
        if not isinstance(info, dict):
            return None
        if "prev_goal_distance" not in info:
            return None
        try:
            return float(info["prev_goal_distance"])
        except Exception:
            return None

    @staticmethod
    def _prev_arm_distance_from_info(info: Any) -> float | None:
        if not isinstance(info, dict):
            return None
        if "prev_arm_distance" not in info:
            return None
        try:
            return float(info["prev_arm_distance"])
        except Exception:
            return None

    def _compute_reward_single(
        self,
        achieved_goal: Any,
        desired_goal: Any,
        info: Any,
        action: Any = None,
        allow_live_arm_distance_fallback: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self.config

        goal_dist = _goal_distance(achieved_goal, desired_goal)
        # HER-safe success: recomputed from current (achieved_goal, desired_goal).
        success_term = float(goal_dist < cfg.goal_epsilon)

        prev_goal_dist = self._prev_goal_distance_from_info(info)
        goal_progress = 0.0
        if prev_goal_dist is not None:
            goal_progress = prev_goal_dist - goal_dist
        goal_progress = float(
            np.clip(goal_progress, -float(cfg.goal_progress_clip), float(cfg.goal_progress_clip))
        )

        # HER-safe arm shaping: use transition-local info only.
        ee_kettle_dist = self._arm_distance_from_info(info)
        if ee_kettle_dist is None and allow_live_arm_distance_fallback:
            ee_kettle_dist = self._ee_to_kettle_distance()

        action_source = action
        if action_source is None and isinstance(info, dict):
            action_source = info.get("action", None)

        action_penalty = _action_penalty(action_source)
        arm_dist_raw = float(ee_kettle_dist) if ee_kettle_dist is not None else 0.0
        arm_dist_term = min(arm_dist_raw, float(cfg.arm_distance_clip))
        arm_gate_width = max(float(cfg.arm_gate_width), 1e-6)
        arm_gate = float(np.clip((goal_dist - cfg.goal_epsilon) / arm_gate_width, 0.0, 1.0))
        prev_arm_dist = self._prev_arm_distance_from_info(info)
        arm_progress = 0.0
        if prev_arm_dist is not None:
            arm_progress = prev_arm_dist - arm_dist_raw
        arm_progress = float(
            np.clip(arm_progress, -float(cfg.arm_progress_clip), float(cfg.arm_progress_clip))
        )

        timeout_failure = 0.0
        if isinstance(info, dict):
            is_timeout = bool(info.get("is_timeout", False))
            if is_timeout and success_term < 0.5:
                timeout_failure = 1.0

        reward = (
            cfg.success_bonus * float(success_term)
            - cfg.goal_distance_weight * goal_dist
            + cfg.goal_progress_weight * goal_progress
            - cfg.arm_distance_weight * arm_gate * arm_dist_term
            + cfg.arm_progress_weight * arm_progress
            - cfg.action_penalty_weight * action_penalty
            - cfg.time_penalty_weight
            - cfg.timeout_failure_penalty * timeout_failure
        )

        components = {
            "success": float(success_term),
            "goal_distance": float(goal_dist),
            "goal_progress": float(goal_progress),
            "arm_distance": float(arm_dist_term),
            "arm_distance_raw": float(arm_dist_raw),
            "arm_gate": float(arm_gate),
            "arm_progress": float(arm_progress),
            "action_penalty": float(action_penalty),
            "time_penalty": float(cfg.time_penalty_weight),
            "timeout_failure": float(timeout_failure),
            "total": float(reward),
        }
        return float(reward), components

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        if not isinstance(obs, dict) or "achieved_goal" not in obs or "desired_goal" not in obs:
            raise KeyError(
                "KitchenDenseRewardWrapper expects Dict observations with "
                "'achieved_goal' and 'desired_goal' keys."
            )

        info = dict(info)
        ee_kettle_dist = self._ee_to_kettle_distance()
        if ee_kettle_dist is not None:
            info["ee_to_kettle_distance"] = float(ee_kettle_dist)
        if self._prev_goal_distance is not None:
            info["prev_goal_distance"] = float(self._prev_goal_distance)
        if self._prev_arm_distance is not None:
            info["prev_arm_distance"] = float(self._prev_arm_distance)
        info["is_timeout"] = bool(truncated and not terminated)
        info["action"] = np.asarray(action, dtype=np.float32).copy()

        reward, components = self._compute_reward_single(
            achieved_goal=obs["achieved_goal"],
            desired_goal=obs["desired_goal"],
            info=info,
            action=action,
            allow_live_arm_distance_fallback=False,
        )

        info["reward_components"] = components
        info["dense_reward"] = reward
        info["sparse_reward"] = components["success"]

        self._prev_goal_distance = components["goal_distance"]
        self._prev_arm_distance = components["arm_distance_raw"]

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if _is_batch_goal(achieved_goal):
            batch_size = _batch_size(achieved_goal)
            info_list = _expand_info(info, batch_size)
            rewards = np.array(
                [
                    self._compute_reward_single(
                        achieved_goal=_get_goal_at(achieved_goal, i),
                        desired_goal=_get_goal_at(desired_goal, i),
                        info=info_list[i],
                        allow_live_arm_distance_fallback=False,
                    )[0]
                    for i in range(batch_size)
                ],
                dtype=np.float32,
            )
            return rewards

        reward, _ = self._compute_reward_single(
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            info=info,
            allow_live_arm_distance_fallback=False,
        )
        return float(reward)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        if hasattr(self.env, "compute_terminated"):
            return self.env.compute_terminated(achieved_goal, desired_goal, info)
        if hasattr(self.env.unwrapped, "compute_terminated"):
            return self.env.unwrapped.compute_terminated(achieved_goal, desired_goal, info)
        raise AttributeError(
            f"Underlying env of type {type(self.env)} has no compute_terminated()."
        )

    def compute_truncated(self, achieved_goal, desired_goal, info):
        if hasattr(self.env, "compute_truncated"):
            return self.env.compute_truncated(achieved_goal, desired_goal, info)
        if hasattr(self.env.unwrapped, "compute_truncated"):
            return self.env.unwrapped.compute_truncated(achieved_goal, desired_goal, info)
        raise AttributeError(
            f"Underlying env of type {type(self.env)} has no compute_truncated()."
        )
