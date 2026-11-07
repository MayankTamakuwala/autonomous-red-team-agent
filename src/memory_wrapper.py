"""Gives the agent a memory of everything it has observed so far in the current episode.

In partially observable mode, NASim/PenGym observations only contain what the *last* action
revealed. For example, a service scan followed by an OS scan returns an observation that shows
the OS but no longer the services. This wrapper keeps a running "knowledge grid" with the same
shape as the observation and merges each new observation into it.
"""

import gymnasium as gym


class MemoryWrapper(gym.Wrapper):
    """Accumulates observations over an episode and returns the accumulated grid, flattened.

    The wrapped env must be created with ``flat_obs=False`` so observations arrive as a 2D grid:
    one row per host, plus a final row describing the result of the last action.

    Merge rule:
      - Host rows: any non-zero value in the new observation overwrites the stored value.
        Host facts never change within an episode (services don't disappear, access never
        drops), so a newly revealed value never contradicts an earlier one.
      - Last row (action result): always replaced by the newest one.
    """

    def __init__(self, env):
        super().__init__(env)
        grid_space = env.observation_space
        assert len(grid_space.shape) == 2, "Create the env with flat_obs=False"
        self.observation_space = gym.spaces.Box(
            low=grid_space.low.flatten(),
            high=grid_space.high.flatten(),
            dtype=grid_space.dtype,
        )
        self.knowledge = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.knowledge = obs.copy()
        return self.knowledge.flatten(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        host_rows = obs[:-1]
        revealed = host_rows != 0
        self.knowledge[:-1][revealed] = host_rows[revealed]
        self.knowledge[-1] = obs[-1]

        return self.knowledge.flatten(), reward, done, truncated, info
