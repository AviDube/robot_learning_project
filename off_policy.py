from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import gymnasium_robotics


gym.register_envs(gymnasium_robotics)


def make_env():
    env = gym.make("FrankaKitchen-v1", tasks_to_complete=["microwave", "kettle"])
    env = gym.wrappers.FlattenObservation(env)
    return env


env = DummyVecEnv([make_env])

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    verbose=1,
    device="cuda",
)

model.learn(total_timesteps=1_000_000)

model.save("sac_franka")
