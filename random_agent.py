import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import matplotlib.pyplot as plt

gym.register_envs(gymnasium_robotics)
video_folder = "franka_kitchen_videos"
env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'], render_mode='rgb_array')
env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda episode_id: True)

num_episodes = 3
total_steps = 0
step_counts = []
cumulative_returns = []

print(f"Starting 3 episodes with a random agent...")

for episode in range(num_episodes):
    observation, info = env.reset()
    episode_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        total_steps += 1
        step_counts.append(total_steps)
        cumulative_returns.append(episode_reward)

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