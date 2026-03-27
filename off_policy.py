import os
from typing import Any, Dict, Tuple

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
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
TASKS = ["kettle"]

TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 8
SEED = 42


# -------------------------------------------------
# Flatten / unflatten helpers
# -------------------------------------------------
def _flatten_dict_space(
    dict_space: spaces.Dict,
) -> Tuple[spaces.Box, Dict[str, Tuple[Tuple[int, ...], int, int]]]:
    lows = []
    highs = []
    meta = {}
    cursor = 0

    for key in sorted(dict_space.spaces.keys()):
        subspace = dict_space.spaces[key]
        if not isinstance(subspace, spaces.Box):
            raise NotImplementedError(
                f"Only Box subspaces are supported inside nested Dicts, got {type(subspace)} for key '{key}'."
            )

        flat_low = np.asarray(subspace.low, dtype=np.float32).reshape(-1)
        flat_high = np.asarray(subspace.high, dtype=np.float32).reshape(-1)

        size = flat_low.shape[0]
        meta[key] = (subspace.shape, cursor, cursor + size)
        cursor += size

        lows.append(flat_low)
        highs.append(flat_high)

    low = np.concatenate(lows, axis=0).astype(np.float32)
    high = np.concatenate(highs, axis=0).astype(np.float32)

    flat_box = spaces.Box(low=low, high=high, dtype=np.float32)
    return flat_box, meta


def _flatten_nested_value(value: dict) -> np.ndarray:
    parts = [np.asarray(value[k], dtype=np.float32).reshape(-1) for k in sorted(value.keys())]
    return np.concatenate(parts, axis=0).astype(np.float32)


def _unflatten_nested_value(
    flat_value: np.ndarray,
    meta: Dict[str, Tuple[Tuple[int, ...], int, int]],
) -> dict:
    flat_value = np.asarray(flat_value, dtype=np.float32).reshape(-1)
    out = {}
    for key in sorted(meta.keys()):
        shape, start, end = meta[key]
        out[key] = flat_value[start:end].reshape(shape).astype(np.float32)
    return out


# -------------------------------------------------
# Wrapper
# -------------------------------------------------
class FlattenNestedDictWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("FlattenNestedDictWrapper expects a Dict observation space.")

        self._nested_meta = {}
        new_spaces = {}

        for key, space in env.observation_space.spaces.items():
            if isinstance(space, spaces.Dict):
                flat_space, meta = _flatten_dict_space(space)
                new_spaces[key] = flat_space
                self._nested_meta[key] = meta
            else:
                if isinstance(space, spaces.Box):
                    new_spaces[key] = spaces.Box(
                        low=np.asarray(space.low, dtype=np.float32),
                        high=np.asarray(space.high, dtype=np.float32),
                        dtype=np.float32,
                    )
                else:
                    new_spaces[key] = space

        self.observation_space = spaces.Dict(new_spaces)
        self._goal_env = self.env.unwrapped

    def observation(self, observation):
        out = {}
        for key, value in observation.items():
            if key in self._nested_meta:
                out[key] = _flatten_nested_value(value)
            else:
                out[key] = np.asarray(value, dtype=np.float32) if isinstance(value, np.ndarray) else value
        return out

    def _restore_goal_format_single(self, value: Any, key_name: str):
        if key_name not in self._nested_meta:
            return value
        return _unflatten_nested_value(value, self._nested_meta[key_name])

    def _compute_reward_single(self, achieved_goal, desired_goal, info):
        achieved_goal_restored = self._restore_goal_format_single(achieved_goal, "achieved_goal")
        desired_goal_restored = self._restore_goal_format_single(desired_goal, "desired_goal")

        reward = self._goal_env.compute_reward(
            achieved_goal_restored,
            desired_goal_restored,
            info,
        )
        return float(reward)

    def compute_reward(self, achieved_goal, desired_goal, info):
        ag = np.asarray(achieved_goal)
        dg = np.asarray(desired_goal)

        if ag.ndim >= 2:
            batch_size = ag.shape[0]

            if isinstance(info, (list, tuple)):
                info_list = list(info)
                if len(info_list) != batch_size:
                    if len(info_list) == 1:
                        info_list = info_list * batch_size
                    else:
                        raise ValueError(
                            f"Batch reward computation got batch_size={batch_size} but len(info)={len(info_list)}."
                        )
            else:
                info_list = [info] * batch_size

            rewards = np.array(
                [
                    self._compute_reward_single(ag[i], dg[i], info_list[i])
                    for i in range(batch_size)
                ],
                dtype=np.float32,
            )
            return rewards

        return float(self._compute_reward_single(ag, dg, info))

    def compute_terminated(self, achieved_goal, desired_goal, info):
        achieved_goal_restored = self._restore_goal_format_single(achieved_goal, "achieved_goal")
        desired_goal_restored = self._restore_goal_format_single(desired_goal, "desired_goal")

        if hasattr(self._goal_env, "compute_terminated"):
            return self._goal_env.compute_terminated(
                achieved_goal_restored, desired_goal_restored, info
            )
        raise AttributeError(
            f"Underlying unwrapped env of type {type(self._goal_env)} has no compute_terminated()."
        )

    def compute_truncated(self, achieved_goal, desired_goal, info):
        achieved_goal_restored = self._restore_goal_format_single(achieved_goal, "achieved_goal")
        desired_goal_restored = self._restore_goal_format_single(desired_goal, "desired_goal")

        if hasattr(self._goal_env, "compute_truncated"):
            return self._goal_env.compute_truncated(
                achieved_goal_restored, desired_goal_restored, info
            )
        raise AttributeError(
            f"Underlying unwrapped env of type {type(self._goal_env)} has no compute_truncated()."
        )


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


# -------------------------------------------------
# Env factory
# -------------------------------------------------
def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make(
            ENV_ID,
            tasks_to_complete=TASKS,
        )
        env = KitchenSuccessInfoWrapper(env)
        env = FlattenNestedDictWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# -------------------------------------------------
# Sanity checks
# -------------------------------------------------
def sanity_check_env():
    print("\n========== SANITY CHECK ==========")
    env = make_env(rank=0, seed=SEED)()

    obs, info = env.reset()
    print("Observation keys:", list(obs.keys()))
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}")

    assert "observation" in obs, "Missing 'observation' key"
    assert "achieved_goal" in obs, "Missing 'achieved_goal' key"
    assert "desired_goal" in obs, "Missing 'desired_goal' key"

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"One step reward: {reward}, terminated={terminated}, truncated={truncated}")

    recomputed_reward = env.compute_reward(
        next_obs["achieved_goal"],
        next_obs["desired_goal"],
        info,
    )
    print(f"Recomputed single reward: {recomputed_reward}")

    batch_ag = np.stack([next_obs["achieved_goal"], next_obs["achieved_goal"]], axis=0)
    batch_dg = np.stack([next_obs["desired_goal"], next_obs["desired_goal"]], axis=0)
    batch_info = [info, info]

    batch_reward = env.compute_reward(batch_ag, batch_dg, batch_info)
    print(f"Recomputed batch reward: {batch_reward}, shape={batch_reward.shape}")

    print("Base env type:", type(env.unwrapped))
    print("==================================\n")
    env.close()


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    sanity_check_env()

    # Parallel training envs
    train_env = SubprocVecEnv([make_env(rank=i, seed=SEED) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env)

    # Single eval env
    eval_env = DummyVecEnv([make_env(rank=10_000, seed=SEED)])
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="sac_franka",
    )

    info_stats_callback = InfoStatsCallback()

    callbacks = CallbackList([
        info_stats_callback,
        eval_callback,
        checkpoint_callback,
    ])

    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=100_000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=8,
        ent_coef="auto",
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=8,
            goal_selection_strategy="future",
            copy_info_dict=True,
        ),
        tensorboard_log=TB_LOG_DIR,
        verbose=1,
        device="cuda",
        seed=SEED,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=False,
    )

    model.save("sac_franka_final")
    print("Training finished. Saved model to sac_franka_final.zip")
