import gymnasium as gym
import gymnasium_robotics
import numpy as np
from torch.utils.tensorboard import SummaryWriter

gym.register_envs(gymnasium_robotics)

TASKS          = ['microwave', 'kettle']
TOTAL_STEPS    = 2_000_000
LOG_FREQ       = 10_000
RUN_NAME       = "random_baseline"
SAMPLE_EPISODES = 100   # number of episodes to estimate the baseline mean

# ── Step 1: estimate mean return from random policy ───────────────────────────
print(f"Estimating random baseline over {SAMPLE_EPISODES} episodes…")

env = gym.make('FrankaKitchen-v1', tasks_to_complete=TASKS)
returns = []

for _ in range(SAMPLE_EPISODES):
    obs, _         = env.reset()
    episode_return = 0.0
    done           = False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        episode_return += reward
        done = terminated or truncated
    returns.append(episode_return)

env.close()

baseline_mean = float(np.mean(returns))
baseline_std  = float(np.std(returns))
print(f"  Mean return : {baseline_mean:.4f}")
print(f"  Std  return : {baseline_std:.4f}\n")

# ── Step 2: write as a constant line across all 2M steps ─────────────────────
writer = SummaryWriter(log_dir=f"./ppo_franka_tb/{RUN_NAME}")

for step in range(LOG_FREQ, TOTAL_STEPS + LOG_FREQ, LOG_FREQ):
    writer.add_scalar(f"{RUN_NAME}/ep_rew_mean", baseline_mean, step)
    writer.add_scalar("eval/mean_reward",         baseline_mean, step)

writer.flush()
writer.close()

print(f"Constant baseline line written for {TOTAL_STEPS:,} steps.")
print("Run: tensorboard --logdir ./ppo_franka_tb")