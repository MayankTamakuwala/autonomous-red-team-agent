"""Neural network that scores every action from the agent's current observation."""

from torch import nn


class QNetwork(nn.Module):
    """Fully connected network: observation -> hidden -> hidden -> one score per action."""

    def __init__(self, obs_size, n_actions, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, obs):
        """Return one score per action for the given observation."""
        return self.layers(obs)
