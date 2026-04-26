import numpy as np
from gymnasium.spaces import Box
import gymnasium as gym

class FlattenObsWrapper(gym.ObservationWrapper):
    """
    Flattens Franka Kitchen's Dict observation into a single 1-D Box.

    The Dict space has three keys:
      'observation'   → np.ndarray of shape (59,)   ← standard Box, use this
      'achieved_goal' → nested dict of task arrays   ← shape=None, skip
      'desired_goal'  → nested dict of task arrays   ← shape=None, skip

    We use only 'observation' for the policy. If you want goal-conditioned
    learning, set INCLUDE_GOALS=True to also flatten and append the goal arrays
    from the raw obs dict (they are readable at step-time even though their
    space metadata is None).
    """

    INCLUDE_GOALS = False  # set True for goal-conditioned policy input

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
        if self.INCLUDE_GOALS:
            raw_obs, _ = env.reset()
            for key in ('achieved_goal', 'desired_goal'):
                if key in raw_obs:
                    self._goal_dim += self._flatten_nested(raw_obs[key]).shape[0]
            flat_dim += self._goal_dim

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    @staticmethod
    def _flatten_nested(value) -> np.ndarray:
        """Recursively flatten a value that may be a dict or ndarray."""
        if isinstance(value, dict):
            return np.concatenate([
                FlattenObsWrapper._flatten_nested(v) for v in value.values()
            ]).astype(np.float32)
        return np.asarray(value, dtype=np.float32).flatten()

    def observation(self, obs) -> np.ndarray:
        parts = [obs[k].flatten().astype(np.float32) for k in self._box_keys]

        if self.INCLUDE_GOALS:
            for key in ('achieved_goal', 'desired_goal'):
                if key in obs:
                    parts.append(self._flatten_nested(obs[key]))

        return np.concatenate(parts)
