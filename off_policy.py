import argparse
import os
from typing import Any, Dict, Tuple

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from kitchen_dense_reward import KitchenDenseRewardConfig, KitchenDenseRewardWrapper

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

TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 8
SEED = 42
USE_HER = False


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
        # Keep the wrapped env here so HER calls use our dense compute_reward().
        self._goal_env = self.env

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
    def __init__(self, env: gym.Env, target_tasks=None):
        super().__init__(env)
        tasks = target_tasks if target_tasks is not None else ["kettle"]
        self._target_tasks = set(tasks)

    def compute_reward(self, achieved_goal, desired_goal, info):
        if hasattr(self.env, "compute_reward"):
            return self.env.compute_reward(achieved_goal, desired_goal, info)
        if hasattr(self.env.unwrapped, "compute_reward"):
            return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)
        raise AttributeError(
            f"Underlying env of type {type(self.env)} has no compute_reward()."
        )

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


# -------------------------------------------------
# Logging callback
# -------------------------------------------------
class TrainingLogCallback(BaseCallback):
    """
    Logs mean episode return to console every `log_freq` timesteps.
    VecMonitor/Monitor injects episode stats into info.
    """

    def __init__(self, log_freq: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_returns: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_returns.append(float(info["episode"]["r"]))

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
            # Save best model based on success first, then mean reward.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAC on Franka Kitchen with dense reward and success-first model selection."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a SAC .zip checkpoint to continue training from.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help="Number of timesteps to train in this invocation.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="sac_dense_no_her",
        help="Run label shown in stdout.",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip one-step environment sanity checks at startup.",
    )
    return parser.parse_args()


# -------------------------------------------------
# Env factory
# -------------------------------------------------
def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make(
            ENV_ID,
            tasks_to_complete=TASKS,
        )
        env = KitchenDenseRewardWrapper(env, config=DENSE_REWARD_CONFIG)
        env = KitchenSuccessInfoWrapper(env, target_tasks=TASKS)
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
    args = parse_args()
    if not args.skip_sanity_check:
        sanity_check_env()

    run_name = args.run_name
    print(f"Run: {run_name}")
    print(f"HER enabled: {USE_HER}")
    print("Setting up vectorised environments…")

    # Parallel training envs
    train_env = SubprocVecEnv([make_env(rank=i, seed=SEED) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env)

    # Single eval env
    eval_env = DummyVecEnv([make_env(rank=10_000, seed=SEED)])
    eval_env = VecMonitor(eval_env)

    eval_callback = KitchenEvalCallback(
        eval_env=eval_env,
        best_model_save_path=None,
        best_model_dir=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(20_000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="sac_franka",
        save_replay_buffer=True,
    )

    log_callback = TrainingLogCallback(log_freq=2048)
    info_stats_callback = InfoStatsCallback()

    callbacks = CallbackList([
        log_callback,
        info_stats_callback,
        eval_callback,
        checkpoint_callback,
    ])

    reset_num_timesteps = True
    if args.resume_from:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")
        print(f"Resuming from checkpoint: {args.resume_from}")
        model = SAC.load(
            args.resume_from,
            env=train_env,
            device="auto",
            seed=SEED,
        )
        model.tensorboard_log = TB_LOG_DIR
        reset_num_timesteps = False

        replay_buffer_path = args.resume_from.replace(".zip", "_replay_buffer.pkl")
        if os.path.isfile(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
            print(f"Loaded replay buffer: {replay_buffer_path}")
        else:
            print("Replay buffer file not found next to checkpoint; continuing without it.")
    else:
        model = SAC(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=50_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=2,
            ent_coef="auto",
            tensorboard_log=TB_LOG_DIR,
            verbose=0,
            device="auto",
            seed=SEED,
        )

    print(f"\nPolicy architecture:\n{model.policy}\n")
    print(f"Training for {args.total_timesteps:,} timesteps across {N_ENVS} parallel envs…\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=True,  # requires `pip install rich`
    )

    model.save("sac_franka_final")
    print("Training finished. Saved model to sac_franka_final.zip")
