def compute_gae(
    buffer,
    gamma=0.99,
    gae_lambda=0.95,
):
    advantage = 0.0

    for t in reversed(range(len(buffer))):

        # Do not bootstrap after a true termination.
        bootstrap_mask = 1.0 - buffer.terminals[t]

        delta = (
            buffer.rewards[t]
            + gamma
            * buffer.next_values[t]
            * bootstrap_mask
            - buffer.values[t]
        )

        # Do not propagate GAE across an episode reset,
        # including time-limit truncations.
        continuation_mask = 1.0 - buffer.episode_ends[t]

        advantage = (
            delta
            + gamma
            * gae_lambda
            * continuation_mask
            * advantage
        )

        buffer.advantages[t] = advantage

    buffer.returns[:len(buffer)] = (
        buffer.advantages[:len(buffer)]
        + buffer.values[:len(buffer)]
    )

    return (
        buffer.advantages[:len(buffer)],
        buffer.returns[:len(buffer)],
    )