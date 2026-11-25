"""Q-network that works with any number of hosts.

The network first learns a description of each host. It also builds a summary of
the whole network, so it can understand each host in the context of the others.

It then combines these two views to score the 12 possible actions for each host.

Every host uses the same network weights, so the model works with 3 hosts, 50
hosts, or any other number. It reads observations from CommonView.
"""

import torch
from torch import nn

from src.env.common_view import ACTION_TYPES, N_FEATURES, N_RESULT


class HostQNetwork(nn.Module):
    """Score every possible (host, action) pair.

    The observation contains one row per host, plus a final row containing the
    result of the previous action.

    We return one score for every action on every host, in CommonView's order:
    host * 12 + action_type.

    A single observation produces (n_hosts * 12,) scores.
    A batch produces (batch_size, n_hosts * 12) scores.
    """

    def __init__(self, hidden_size=64):
        super().__init__()

        # Turn each host's raw features into a more useful representation.
        # The same layer is used for every host.
        self.host_layer = nn.Sequential(nn.Linear(N_FEATURES, hidden_size), nn.ReLU())

        # Score a host's actions using three things:
        #   1. The host's own description.
        #   2. The average and strongest features across the whole network.
        #   3. The result of the previous action.
        self.score_layers = nn.Sequential(
            nn.Linear(3 * hidden_size + N_RESULT, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(ACTION_TYPES)),
        )

    def forward(self, obs, host_mask=None):
        """Return a score for every possible action.

        host_mask tells us which hosts are real and which are just padding.
        Padding is needed when batching networks with different numbers of hosts.
        If no mask is provided, we assume every host is real.
        """

        # PyTorch expects a batch dimension, even when we only have one observation.
        # Add one now, and remove it again before returning the scores.
        single_observation = obs.dim() == 2

        if single_observation:
            obs = obs.unsqueeze(0)

            if host_mask is not None:
                host_mask = host_mask.unsqueeze(0)

        # The last row contains the previous result.
        # Every row before it describes a host.
        hosts = obs[:, :-1]
        result = obs[:, -1, :N_RESULT]

        # If we weren't given a mask, treat every host as real.
        if host_mask is None:
            host_mask = torch.ones(hosts.shape[:2], dtype=torch.bool, device=obs.device)

        # Add a dimension so the mask can be applied to all features of each host.
        real_hosts = host_mask.unsqueeze(-1)

        # First, give every host its own learned description.
        described = self.host_layer(hosts)

        # Now build a summary of the whole network.
        #
        # The average tells us what a typical host looks like.
        # The maximum preserves strong features that might only appear on one host.
        # Ignore padding in both calculations.
        average = (described * real_hosts).sum(dim=1) / real_hosts.sum(dim=1)

        largest = described.masked_fill(~real_hosts, float("-inf")).max(dim=1).values

        # Combine the network summary with the result of the previous action.
        # This is the same context for every host in the network.
        context = torch.cat([average, largest, result], dim=1)

        # Give that same network-wide context to each host.
        n_hosts = hosts.shape[1]
        context = context.unsqueeze(1).expand(-1, n_hosts, -1)

        # Each host now has its own description AND the network-wide context.
        # Use both to score all possible actions on that host.
        host_and_context = torch.cat([described, context], dim=2)
        scores = self.score_layers(host_and_context)

        # Turn (batch, hosts, actions) into (batch, hosts * actions).
        # This keeps CommonView's ordering: host * 12 + action_type.
        scores = scores.flatten(start_dim=1)

        # If the input was a single observation, remove the batch dimension.
        if single_observation:
            return scores[0]

        return scores
