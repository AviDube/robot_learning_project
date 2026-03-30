from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class KitchenDenseRewardConfig:
    goal_epsilon: float = 0.3
    sparse_weight: float = 1.0
    elementwise_weight: float = 1.0
    distance_weight: float = 0.2
    progress_weight: float = 0.8
    action_penalty_weight: float = 0.01
    time_penalty_weight: float = 0.01


def _flatten_goal(goal: Any) -> np.ndarray:
    if isinstance(goal, dict):
        parts = [_flatten_goal(goal[k]) for k in sorted(goal.keys())]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)
    return np.asarray(goal, dtype=np.float32).reshape(-1)


def _copy_goal(goal: Any) -> Any:
    if goal is None:
        return None
    if isinstance(goal, dict):
        return {k: _copy_goal(v) for k, v in goal.items()}
    return np.asarray(goal, dtype=np.float32).copy()


def _goal_distance(achieved_goal: Any, desired_goal: Any) -> float:
    ag = _flatten_goal(achieved_goal)
    dg = _flatten_goal(desired_goal)
    if ag.shape != dg.shape:
        raise ValueError(
            f"Goal shapes do not match after flattening: achieved={ag.shape}, desired={dg.shape}"
        )
    return float(np.linalg.norm(ag - dg, ord=2))


def _elementwise_sparse_hits(achieved_goal: Any, desired_goal: Any, epsilon: float) -> float:
    if isinstance(achieved_goal, dict) and isinstance(desired_goal, dict):
        shared = sorted(set(achieved_goal.keys()) & set(desired_goal.keys()))
        return float(
            sum(
                _elementwise_sparse_hits(achieved_goal[k], desired_goal[k], epsilon)
                for k in shared
            )
        )

    ag = np.asarray(achieved_goal, dtype=np.float32).reshape(-1)
    dg = np.asarray(desired_goal, dtype=np.float32).reshape(-1)
    if ag.shape != dg.shape:
        raise ValueError(
            f"Elementwise goal shapes do not match: achieved={ag.shape}, desired={dg.shape}"
        )
    return float(np.linalg.norm(ag - dg, ord=2) < epsilon)


def _action_penalty(action: Any) -> float:
    if action is None:
        return 0.0
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return 0.0
    return -float(np.mean(np.square(a)))


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


def _expand_info(info: Any, batch_size: int) -> List[Any]:
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
    Reward shaping wrapper for Franka Kitchen.

    Final reward:
      r_t = w_sparse * r_sparse
          + w_elem * r_elementwise
          + w_dist * (-||s-g||_2)
          + w_prog * (d_prev - d_curr)
          + w_act  * (-||a||^2_mean)
          - w_time
    """

    def __init__(
        self,
        env: gym.Env,
        config: KitchenDenseRewardConfig | None = None,
    ):
        super().__init__(env)
        self.config = config or KitchenDenseRewardConfig()
        self._prev_achieved_goal: Any = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict) and "achieved_goal" in obs:
            self._prev_achieved_goal = _copy_goal(obs["achieved_goal"])
        else:
            self._prev_achieved_goal = None
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        if not isinstance(obs, dict) or "achieved_goal" not in obs or "desired_goal" not in obs:
            raise KeyError(
                "KitchenDenseRewardWrapper expects Dict observations with "
                "'achieved_goal' and 'desired_goal' keys."
            )

        info = dict(info)
        info["prev_achieved_goal"] = _copy_goal(self._prev_achieved_goal)
        info["action"] = np.asarray(action, dtype=np.float32).copy()

        reward, components = self._compute_reward_single(
            obs["achieved_goal"],
            obs["desired_goal"],
            info,
        )
        info["reward_components"] = components
        info["dense_reward"] = reward
        info["sparse_reward"] = components["sparse"]

        self._prev_achieved_goal = _copy_goal(obs["achieved_goal"])
        return obs, reward, terminated, truncated, info

    def _compute_reward_single(
        self,
        achieved_goal: Any,
        desired_goal: Any,
        info: Any,
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self.config

        curr_dist = _goal_distance(achieved_goal, desired_goal)
        sparse_term = float(curr_dist < cfg.goal_epsilon)
        elementwise_term = _elementwise_sparse_hits(
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            epsilon=cfg.goal_epsilon,
        )

        prev_goal = None
        action = None
        if isinstance(info, dict):
            prev_goal = info.get("prev_achieved_goal", None)
            action = info.get("action", None)

        if prev_goal is not None:
            prev_dist = _goal_distance(prev_goal, desired_goal)
            progress_term = prev_dist - curr_dist
        else:
            progress_term = 0.0

        dist_term = -curr_dist
        action_term = _action_penalty(action)
        time_term = -1.0

        reward = (
            cfg.sparse_weight * sparse_term
            + cfg.elementwise_weight * elementwise_term
            + cfg.distance_weight * dist_term
            + cfg.progress_weight * progress_term
            + cfg.action_penalty_weight * action_term
            + cfg.time_penalty_weight * time_term
        )

        components = {
            "sparse": float(sparse_term),
            "elementwise": float(elementwise_term),
            "distance": float(dist_term),
            "progress": float(progress_term),
            "action": float(action_term),
            "time": float(time_term),
            "total": float(reward),
        }
        return float(reward), components

    def compute_reward(self, achieved_goal, desired_goal, info):
        if _is_batch_goal(achieved_goal):
            batch_size = _batch_size(achieved_goal)
            info_list = _expand_info(info, batch_size)

            rewards = np.array(
                [
                    self._compute_reward_single(
                        _get_goal_at(achieved_goal, i),
                        _get_goal_at(desired_goal, i),
                        info_list[i],
                    )[0]
                    for i in range(batch_size)
                ],
                dtype=np.float32,
            )
            return rewards

        reward, _ = self._compute_reward_single(achieved_goal, desired_goal, info)
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
