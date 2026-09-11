import torch


def compute_batch_is_deviation(
    actor,
    replay_batch,
    normalizer,
    normalize_observations=True,
    device="cpu",
):
    """
    Compute the DISC replay-batch inclusion statistic.

    Old raw observations are normalized using the CURRENT
    observation statistics before being evaluated by the
    current policy.
    """

    if normalize_observations:
        observations_np = (
            normalizer.normalize_observation(
                replay_batch.raw_observations,
                update=False,
            )
        )
    else:
        observations_np = (
            replay_batch.raw_observations
        )

    observations = torch.as_tensor(
        observations_np,
        dtype=torch.float32,
        device=device,
    )

    actions = torch.as_tensor(
        replay_batch.actions,
        dtype=torch.float32,
        device=device,
    )

    behavior_log_probs = torch.as_tensor(
        replay_batch.log_probs_per_dim,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        distribution = actor(
            observations
        )

        current_log_probs = (
            distribution.log_prob(
                actions
            )
        )

        log_ratio_per_dim = (
            current_log_probs
            - behavior_log_probs
        )

        ratio_per_dim = torch.exp(
            log_ratio_per_dim
        )

        # rho'_(t,d) = |1 - rho_(t,d)| + 1
        shifted_deviation = (
            torch.abs(
                1.0 - ratio_per_dim
            )
            + 1.0
        )

        batch_score = (
            shifted_deviation
            .mean()
            .item()
        )

    return batch_score


def select_replay_batches(
    actor,
    replay_buffer,
    normalizer,
    epsilon_b=0.1,
    normalize_observations=True,
    device="cpu",
):
    """
    Select batches satisfying:

        mean(rho'_(t,d)) < 1 + epsilon_b

    Returns:
        eligible_batches
        batch_scores

    Batches are checked newest first.
    """

    eligible_batches = []
    batch_scores = []

    threshold = 1.0 + epsilon_b

    batches = (
        replay_buffer
        .get_newest_first()
    )

    for age, batch in enumerate(
        batches
    ):

        score = compute_batch_is_deviation(
            actor=actor,
            replay_batch=batch,
            normalizer=normalizer,
            normalize_observations=(
                normalize_observations
            ),
            device=device,
        )

        accepted = (
            score < threshold
        )

        batch_scores.append(
            {
                "age": age,
                "score": score,
                "accepted": accepted,
            }
        )

        if accepted:
            eligible_batches.append(
                batch
            )

    return (
        eligible_batches,
        batch_scores,
    )