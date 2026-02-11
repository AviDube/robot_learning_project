import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import matplotlib.pyplot as plt

# gymnasium is the reinforcment learning environment
# tells the RL environment to register the robotics environment, ex: Franka's kitchen
gym.register_envs(gymnasium_robotics)
video_folder = "franka_kitchen_videos"

# initializes the environment
# arg1 specify env[str]: Franka Kitchen
# arg2 List[str]: containing tasks to complete (open microwave door)
# arg3 Boolean: terminate task for completion (default true)
# arg4 Boolean: remove task when completed (default true)
# arg5 Float: object noise ratio (default 0.0005)
# arg6 Float: robot noise ratio (default 0.01)
# arg7 Integer: max number of steps per episode (default 280)
# render mode: gymnasium arg
env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'], render_mode='rgb_array')

# initialize to record video of episode
env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda episode_id: True)

num_episodes = 3
total_steps = 0 # total steps from all episodes in environment
step_counts = [] # total steps per episode
cumulative_returns = [] #total reward from each episode

print(f"Starting 3 episodes with a random agent...")

for episode in range(num_episodes):
    # resets environment back to start state (initial observation)
    # start of new episode
    observation, info = env.reset() # start of rollout
    episode_reward = 0 # total reward for curr episode
    terminated = False # task ended due to completion
    truncated = False # task ended due to time constraint
    
    while not (terminated or truncated):
        action = env.action_space.sample() # random agent (franka robo)
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward # sum of rewards for curr episode
        total_steps += 1 # adds to total steps for all episodes
        step_counts.append(total_steps) # adds to list keeping track of all total steps per episode (x-axis)
        cumulative_returns.append(episode_reward) # adds to list keeping track of all total rewards per episode (y-axis)

env.close()

print(f"\n--- Results ---")
print(f"Videos saved in: {os.path.abspath(video_folder)}")

plt.figure(figsize=(10, 5))
plt.plot(step_counts, cumulative_returns, label='Random Agent Return', color='teal', linewidth=2)
plt.xlabel('Environment Steps')
plt.ylabel('Cumulative Return')
plt.title('Franka Kitchen: Mean Return vs. Environment Steps')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('performance_plot.png')
plt.show()

print(f"\nMean Performance (Return): {np.mean(cumulative_returns[-1]):.4f}")