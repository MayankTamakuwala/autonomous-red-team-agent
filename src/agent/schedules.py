"""How random the agent is at each point in training.

Early on the agent knows nothing, so it should mostly explore (epsilon near 1). As it learns,
it should rely more on its Q-network (epsilon near 0), while still exploring a little.
"""


def linear_epsilon(step, start=1.0, end=0.05, decay_steps=10_000):
    """Epsilon that drops in a straight line from start to end over decay_steps, then stays at end.

    Example with the defaults: step 0 -> 1.0, step 5,000 -> 0.525, step 10,000 and after -> 0.05.
    """
    if step >= decay_steps:
        return end
    return start + (end - start) * step / decay_steps
