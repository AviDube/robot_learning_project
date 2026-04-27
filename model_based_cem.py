import argparse
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gymnasium.wrappers import RecordVideo

from on_policy import (
    ASRObsWrapper,
    AugmentedObsWrapper,
    DenseRewardWrapper,
    FlattenObsWrapper,
    TASKS,
)


gym.register_envs(gymnasium_robotics)

ENV_ID = "FrankaKitchen-v1"


@dataclass
class MBConfig:
    seed: int = 42
    total_steps: int = 10_000_000
    random_warmup_steps: int = 10_000
    model_update_every: int = 250
    model_train_epochs: int = 25
    model_batch_size: int = 1024

    ensemble_size: int = 5
    hidden_size: int = 512
    model_lr: float = 3e-4
    weight_decay: float = 1e-6

    mpc_horizon: int = 18
    mpc_population: int = 512
    mpc_elites: int = 64
    mpc_iters: int = 6
    mpc_alpha: float = 0.15
    mpc_min_std: float = 0.05
    uncertainty_coef: float = 0.2

    replay_capacity: int = 400_000
    eval_every_steps: int = 10_000
    eval_episodes: int = 5

    use_asr: bool = True
    use_shaped_reward: bool = True
    run_name: str = "mb_cem_uncertainty_run_1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(seed: int, use_asr: bool, use_shaped_reward: bool, render_mode=None):
    env = gym.make(
        ENV_ID,
        tasks_to_complete=TASKS,
        render_mode=render_mode,
    )
    if use_shaped_reward:
        env = DenseRewardWrapper(env, tasks=TASKS)
    env = FlattenObsWrapper(env)
    if use_asr:
        env = ASRObsWrapper(env)
    env = AugmentedObsWrapper(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.rews = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        idx = self.ptr
        self.obs[idx] = obs
        self.acts[idx] = act
        self.rews[idx] = rew
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.acts[idx],
            self.rews[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


class DynamicsModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EnsembleDynamics:
    """
    PETS-style ensemble that predicts [delta_obs, reward] from [obs, action].
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ensemble_size: int,
        hidden_size: int,
        lr: float,
        weight_decay: float,
        device: str,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = obs_dim + 1
        self.ensemble_size = ensemble_size
        self.device = torch.device(device)

        self.models = [
            DynamicsModel(obs_dim + act_dim, self.out_dim, hidden_size).to(self.device)
            for _ in range(ensemble_size)
        ]
        self.opts = [
            optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
            for m in self.models
        ]

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def _fit_normalizer(self, x: np.ndarray, y: np.ndarray):
        self.x_mean = x.mean(axis=0, keepdims=True)
        self.x_std = x.std(axis=0, keepdims=True) + 1e-6
        self.y_mean = y.mean(axis=0, keepdims=True)
        self.y_std = y.std(axis=0, keepdims=True) + 1e-6

    def _norm_x(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.from_numpy(self.x_mean).to(self.device)
        x_std = torch.from_numpy(self.x_std).to(self.device)
        return (x - x_mean) / x_std

    def _denorm_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        y_mean = torch.from_numpy(self.y_mean).to(self.device)
        y_std = torch.from_numpy(self.y_std).to(self.device)
        return y_norm * y_std + y_mean

    def train_models(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        rews: np.ndarray,
        next_obs: np.ndarray,
        epochs: int,
        batch_size: int,
    ):
        if len(obs) < batch_size:
            return None

        delta_obs = next_obs - obs
        x = np.concatenate([obs, acts], axis=-1).astype(np.float32)
        y = np.concatenate([delta_obs, rews], axis=-1).astype(np.float32)

        self._fit_normalizer(x, y)
        y_mean = torch.from_numpy(self.y_mean).to(self.device)
        y_std = torch.from_numpy(self.y_std).to(self.device)

        n = x.shape[0]
        losses = []
        for _ in range(epochs):
            for model, opt in zip(self.models, self.opts):
                idx = np.random.permutation(n)
                for start in range(0, n, batch_size):
                    b = idx[start : start + batch_size]
                    xb = torch.from_numpy(x[b]).to(self.device)
                    yb = torch.from_numpy(y[b]).to(self.device)

                    xb = self._norm_x(xb)
                    yb = (yb - y_mean) / y_std

                    pred = model(xb)
                    loss = nn.functional.mse_loss(pred, yb)
                    losses.append(float(loss.item()))

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

        return float(np.mean(losses)) if losses else None

    @torch.no_grad()
    def predict_all(self, obs: np.ndarray, acts: np.ndarray):
        """
        obs: [B, obs_dim], acts: [B, act_dim]
        returns:
          next_obs_all: [E, B, obs_dim]
          rew_all: [E, B]
        """
        x = np.concatenate([obs, acts], axis=-1).astype(np.float32)
        xt = torch.from_numpy(x).to(self.device)
        xt = self._norm_x(xt)

        preds = []
        for model in self.models:
            y_norm = model(xt)
            y = self._denorm_y(y_norm)
            preds.append(y)

        y_all = torch.stack(preds, dim=0)
        delta = y_all[..., : self.obs_dim]
        rew = y_all[..., self.obs_dim]

        obs_t = torch.from_numpy(obs).to(self.device)
        next_obs = obs_t.unsqueeze(0) + delta

        return next_obs.cpu().numpy(), rew.cpu().numpy()


def plot_results(
    episode_returns: list[float],
    eval_steps: list[int],
    eval_means: list[float],
    eval_stds: list[float],
    run_name: str,
):
    if not episode_returns and not eval_steps:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if episode_returns:
        returns = np.array(episode_returns, dtype=np.float32)
        window = min(20, len(returns))
        smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")

        axes[0].plot(returns, color="teal", linewidth=1, alpha=0.35, label="Episode Return")
        axes[0].plot(
            range(window - 1, len(returns)),
            smoothed,
            color="teal",
            linewidth=2.5,
            label=f"Smoothed (window={window})",
        )
        axes[0].set_title("Training Episode Returns")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Return")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()
    else:
        axes[0].set_title("Training Episode Returns")
        axes[0].text(0.5, 0.5, "No episode return data", ha="center", va="center")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

    if eval_steps:
        x = np.array(eval_steps, dtype=np.int32)
        y = np.array(eval_means, dtype=np.float32)
        s = np.array(eval_stds, dtype=np.float32)

        axes[1].plot(x, y, color="navy", linewidth=2.2, label="Eval Mean Return")
        axes[1].fill_between(x, y - s, y + s, color="cornflowerblue", alpha=0.25, label="+-1 std")
        axes[1].set_title("Evaluation Returns vs Steps")
        axes[1].set_xlabel("Environment Steps")
        axes[1].set_ylabel("Return")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()
    else:
        axes[1].set_title("Evaluation Returns vs Steps")
        axes[1].text(0.5, 0.5, "No evaluation data", ha="center", va="center")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    fig.suptitle(f"Model-Based CEM (Uncertainty) - {run_name}")
    fig.tight_layout()

    safe_name = run_name.replace(" ", "_")
    fname = f"learning_curve_mb_{safe_name}.png"
    fig.savefig(fname, dpi=150)
    plt.show()
    print(f"Plot saved -> {fname}")


class CEMPlanner:
    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int,
        population: int,
        elites: int,
        iterations: int,
        alpha: float,
        min_std: float,
        uncertainty_coef: float,
    ):
        self.action_low = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)
        self.horizon = horizon
        self.population = population
        self.elites = elites
        self.iterations = iterations
        self.alpha = alpha
        self.min_std = min_std
        self.uncertainty_coef = uncertainty_coef

        self.act_dim = action_low.shape[0]
        self.mean = np.zeros((horizon, self.act_dim), dtype=np.float32)
        self.std = np.ones((horizon, self.act_dim), dtype=np.float32) * 0.6

    def reset(self):
        self.mean.fill(0.0)
        self.std.fill(0.6)

    def _rollout_score(self, dynamics: EnsembleDynamics, obs: np.ndarray, action_seqs: np.ndarray):
        """
        action_seqs: [N, H, A]
        score = model reward - uncertainty penalty.
        """
        n = action_seqs.shape[0]
        curr = np.repeat(obs[None, :], n, axis=0)
        returns = np.zeros((n,), dtype=np.float32)

        for t in range(self.horizon):
            a_t = action_seqs[:, t, :]
            next_obs_all, rew_all = dynamics.predict_all(curr, a_t)

            mean_rew = rew_all.mean(axis=0)
            disagreement = next_obs_all.std(axis=0).mean(axis=1)

            returns += mean_rew - self.uncertainty_coef * disagreement

            model_idx = np.random.randint(0, dynamics.ensemble_size, size=n)
            curr = next_obs_all[model_idx, np.arange(n)]

        return returns

    def act(self, dynamics: EnsembleDynamics, obs: np.ndarray):
        mean = self.mean.copy()
        std = self.std.copy()

        for _ in range(self.iterations):
            noise = np.random.randn(self.population, self.horizon, self.act_dim).astype(np.float32)
            seqs = mean[None, :, :] + noise * std[None, :, :]
            seqs = np.clip(seqs, self.action_low, self.action_high)

            scores = self._rollout_score(dynamics, obs, seqs)
            elite_idx = np.argsort(scores)[-self.elites :]
            elites = seqs[elite_idx]

            new_mean = elites.mean(axis=0)
            new_std = elites.std(axis=0)
            new_std = np.maximum(new_std, self.min_std)

            mean = self.alpha * mean + (1.0 - self.alpha) * new_mean
            std = self.alpha * std + (1.0 - self.alpha) * new_std

        action = mean[0].copy()

        self.mean[:-1] = mean[1:]
        self.mean[-1] = 0.0
        self.std[:-1] = std[1:]
        self.std[-1] = std[-1]

        return np.clip(action, self.action_low, self.action_high)


def evaluate_policy(config: MBConfig, planner: CEMPlanner, dynamics: EnsembleDynamics, model_path: str | None):
    if model_path is not None:
        state = torch.load(model_path, map_location=dynamics.device)
        for i in range(dynamics.ensemble_size):
            dynamics.models[i].load_state_dict(state[f"model_{i}"])
        dynamics.x_mean = state["x_mean"]
        dynamics.x_std = state["x_std"]
        dynamics.y_mean = state["y_mean"]
        dynamics.y_std = state["y_std"]

    env = make_env(
        seed=config.seed + 123,
        use_asr=config.use_asr,
        use_shaped_reward=config.use_shaped_reward,
        render_mode="human",
    )

    returns = []
    for ep in range(config.eval_episodes):
        obs, _ = env.reset()
        planner.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = planner.act(dynamics, obs)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            done = terminated or truncated
            time.sleep(0.03)
        returns.append(ep_ret)
        print(f"Eval episode {ep + 1}: return={ep_ret:.3f}")

    env.close()
    print(f"Mean return={np.mean(returns):.3f}, std={np.std(returns):.3f}")


def train(config: MBConfig, record_video: bool = False):
    set_seed(config.seed)

    checkpoint_dir = f"./mb_franka_checkpoints_{config.run_name}/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "ensemble.pt")

    tb_log_dir = f"./mb_franka_tb/{config.run_name}"
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    env = make_env(
        seed=config.seed,
        use_asr=config.use_asr,
        use_shaped_reward=config.use_shaped_reward,
        render_mode="rgb_array" if record_video else None,
    )
    if record_video:
        env = RecordVideo(env, video_folder=f"mb_eval_videos_{config.run_name}", episode_trigger=lambda _: True)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    replay = ReplayBuffer(obs_dim, act_dim, config.replay_capacity)

    dynamics = EnsembleDynamics(
        obs_dim=obs_dim,
        act_dim=act_dim,
        ensemble_size=config.ensemble_size,
        hidden_size=config.hidden_size,
        lr=config.model_lr,
        weight_decay=config.weight_decay,
        device=device,
    )

    planner = CEMPlanner(
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        horizon=config.mpc_horizon,
        population=config.mpc_population,
        elites=config.mpc_elites,
        iterations=config.mpc_iters,
        alpha=config.mpc_alpha,
        min_std=config.mpc_min_std,
        uncertainty_coef=config.uncertainty_coef,
    )

    obs, _ = env.reset()
    planner.reset()
    ep_ret = 0.0
    ep_len = 0
    episode = 0
    episode_returns = []
    eval_step_history = []
    eval_mean_history = []
    eval_std_history = []

    print(f"Starting model-based training ({config.total_steps:,} steps)")
    print(f"Obs dim={obs_dim}, action dim={act_dim}, device={device}")
    print(f"TensorBoard logdir: {os.path.abspath(tb_log_dir)}")
    print(f"Checkpoint dir: {os.path.abspath(checkpoint_dir)}")

    for step in range(1, config.total_steps + 1):
        if step <= config.random_warmup_steps or replay.size < config.model_batch_size:
            action = env.action_space.sample()
        else:
            action = planner.act(dynamics, obs)

        next_obs, rew, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.add(obs, action, rew, next_obs, done)

        obs = next_obs
        ep_ret += rew
        ep_len += 1

        if done:
            episode += 1
            print(f"Episode {episode:4d} | step={step:7d} | len={ep_len:4d} | return={ep_ret:9.3f}")
            episode_returns.append(float(ep_ret))
            writer.add_scalar("train/episode_return", float(ep_ret), step)
            writer.add_scalar("train/episode_length", int(ep_len), step)
            obs, _ = env.reset()
            planner.reset()
            ep_ret = 0.0
            ep_len = 0

        if step % config.model_update_every == 0 and replay.size >= config.model_batch_size:
            dataset = replay.sample(min(replay.size, 80_000))
            model_loss = dynamics.train_models(
                obs=dataset[0],
                acts=dataset[1],
                rews=dataset[2],
                next_obs=dataset[3],
                epochs=config.model_train_epochs,
                batch_size=config.model_batch_size,
            )
            if model_loss is not None:
                writer.add_scalar("model/loss", model_loss, step)

        if step % config.eval_every_steps == 0 and replay.size >= config.model_batch_size:
            eval_returns = []
            eval_env = make_env(
                seed=config.seed + 1000 + step,
                use_asr=config.use_asr,
                use_shaped_reward=config.use_shaped_reward,
            )
            for _ in range(config.eval_episodes):
                e_obs, _ = eval_env.reset()
                planner.reset()
                e_done = False
                e_ret = 0.0
                while not e_done:
                    e_act = planner.act(dynamics, e_obs)
                    e_obs, e_rew, e_term, e_trunc, _ = eval_env.step(e_act)
                    e_ret += e_rew
                    e_done = e_term or e_trunc
                eval_returns.append(e_ret)
            eval_env.close()
            print(
                f"[Eval @ {step:7d}] mean={np.mean(eval_returns):8.3f} "
                f"std={np.std(eval_returns):7.3f}"
            )

            eval_mean = float(np.mean(eval_returns))
            eval_std = float(np.std(eval_returns))
            writer.add_scalar("eval/mean_return", eval_mean, step)
            writer.add_scalar("eval/std_return", eval_std, step)
            eval_step_history.append(int(step))
            eval_mean_history.append(eval_mean)
            eval_std_history.append(eval_std)

            state = {
                **{f"model_{i}": dynamics.models[i].state_dict() for i in range(dynamics.ensemble_size)},
                "x_mean": dynamics.x_mean,
                "x_std": dynamics.x_std,
                "y_mean": dynamics.y_mean,
                "y_std": dynamics.y_std,
                "config": vars(config),
            }
            torch.save(state, ckpt_path)
            print(f"Saved checkpoint -> {ckpt_path}")

    env.close()
    writer.flush()
    writer.close()
    plot_results(
        episode_returns=episode_returns,
        eval_steps=eval_step_history,
        eval_means=eval_mean_history,
        eval_stds=eval_std_history,
        run_name=config.run_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model-based RL with CEM + uncertainty ensemble")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved ensemble checkpoint")
    parser.add_argument("--total_steps", type=int, default=2_000_000)
    parser.add_argument("--run_name", type=str, default="mb_cem_uncertainty_run_1")
    parser.add_argument("--uncertainty_coef", type=float, default=0.2)
    args = parser.parse_args()

    cfg = MBConfig(
        total_steps=args.total_steps,
        run_name=args.run_name,
        uncertainty_coef=args.uncertainty_coef,
    )

    if args.train:
        train(cfg, record_video=args.record_video)
    else:
        obs_env = make_env(seed=cfg.seed, use_asr=cfg.use_asr, use_shaped_reward=cfg.use_shaped_reward)
        dynamics = EnsembleDynamics(
            obs_dim=obs_env.observation_space.shape[0],
            act_dim=obs_env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            hidden_size=cfg.hidden_size,
            lr=cfg.model_lr,
            weight_decay=cfg.weight_decay,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        planner = CEMPlanner(
            action_low=obs_env.action_space.low,
            action_high=obs_env.action_space.high,
            horizon=cfg.mpc_horizon,
            population=cfg.mpc_population,
            elites=cfg.mpc_elites,
            iterations=cfg.mpc_iters,
            alpha=cfg.mpc_alpha,
            min_std=cfg.mpc_min_std,
            uncertainty_coef=cfg.uncertainty_coef,
        )
        obs_env.close()

        if args.model_path is None:
            args.model_path = os.path.join(f"mb_franka_checkpoints_{cfg.run_name}", "ensemble.pt")
        evaluate_policy(cfg, planner, dynamics, args.model_path)