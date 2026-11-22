"""Makes every network look the same to the agent.

Each network describes its hosts with different columns and has a different list of actions.
This wrapper turns any network into the same shape: 15 numbers per host and 12 actions per host,
so one agent can play all of them.

Usage:
    env = CommonView(make_env(SCENARIO_DIR / "tiny.yml", seed=0))
    obs, info = env.reset(seed=0)    # shape (4, 15) on tiny, (17, 15) on medium-multi-site
    action = host * len(ACTION_TYPES) + ACTION_TYPES.index("e_ssh")
    obs, reward, done, truncated, info = env.step(action)
    env.action_mask                  # which actions this network has
"""

import gymnasium as gym
import numpy as np

# Every network uses some of these
OS = ["linux"]
SERVICES = ["ssh", "ftp", "http", "samba", "smtp"]
PROCESSES = ["tomcat", "proftpd", "cron"]
SCANS = ["service_scan", "os_scan", "subnet_scan", "process_scan"]
EXPLOITS = ["e_ssh", "e_ftp", "e_http", "e_samba", "e_smtp"]
PRIVESCS = ["pe_tomcat", "pe_daclsvc", "pe_schtask"]

ACTION_TYPES = SCANS + EXPLOITS + PRIVESCS  # 12 per host

# Each host gets 15 numbers: 6 about its status, then its OS, services, and processes
N_STATUS = 6
N_FEATURES = N_STATUS + len(OS) + len(SERVICES) + len(PROCESSES)

# The last row says how the last action went: success, or which kind of error
N_RESULT = 4


class CommonView(gym.Wrapper):
    """Gives any network the same 15 numbers per host and the same 12 actions per host.

    Wraps a MemoryWrapper.

    Observation: one row per host, then a last row with the last action's result.
    Action: host * 12 + action type. Only actions where `action_mask` is True exist in
    this network.
    """

    def __init__(self, env):
        super().__init__(env)
        scenario = env.unwrapped.scenario
        self.hosts = list(scenario.address_space)

        # Where status, OS, services, and processes start in this network's grid
        status_start = len(scenario.subnets) + max(scenario.subnets)
        os_start = status_start + N_STATUS
        services_start = os_start + len(scenario.os)
        processes_start = services_start + len(scenario.services)

        # Find each shared name in this network (None if the network doesn't have it)
        def columns(shared, own, start):
            return [start + own.index(name) if name in own else None for name in shared]

        feature_cols = (
            columns(OS, scenario.os, os_start)
            + columns(SERVICES, scenario.services, services_start)
            + columns(PROCESSES, scenario.processes, processes_start)
        )

        # Which columns of this network's grid get copied into the common view
        self.copy_from = [status_start + i for i in range(N_STATUS)]
        self.copy_to = list(range(N_STATUS))
        for i, col in enumerate(feature_cols):
            if col is not None:
                self.copy_from.append(col)
                self.copy_to.append(N_STATUS + i)

        # For each host and action type: this network's action number (-1 if it doesn't have it)
        self.action_table = np.full((len(self.hosts), len(ACTION_TYPES)), -1)
        for env_action, act in enumerate(env.unwrapped.action_space.actions):
            host = self.hosts.index(act.target)
            self.action_table[host, ACTION_TYPES.index(act.name)] = env_action
        self.action_mask = (self.action_table >= 0).flatten()

        # The new sizes
        self.action_space = gym.spaces.Discrete(self.action_table.size)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(len(self.hosts) + 1, N_FEATURES), dtype=np.float32
        )

    def common_obs(self):
        """Copy what the agent knows into the common view."""
        grid = self.env.knowledge
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[:-1, self.copy_to] = grid[:-1, self.copy_from]  # hosts
        obs[-1, :N_RESULT] = grid[-1, :N_RESULT]  # last action's result
        return obs

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self.common_obs(), info

    def step(self, action):
        # Turn the agent's choice into this network's action number
        env_action = int(self.action_table.flat[action])
        if env_action < 0:
            host, kind = divmod(int(action), len(ACTION_TYPES))
            raise ValueError(f"This network has no {ACTION_TYPES[kind]} on {self.hosts[host]}")
        _, reward, done, truncated, info = self.env.step(env_action)
        return self.common_obs(), reward, done, truncated, info
