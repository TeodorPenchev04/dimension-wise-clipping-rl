import numpy as np
import torch


def compute_gae_v(
    replay_batch,
    actor,
    critic,
    normalizer,
    gamma=0.99,
    gae_lambda=0.95,
    normalize_observations=True,
    normalize_rewards=True,
    device="cpu",
):
    """
    Compute GAE-V for a replayed rollout.

    Old raw observations and rewards are transformed using
    the CURRENT normalization statistics.

    Returns:
        advantages
        returns
        values
        importance_ratios
        normalized_observations
    """

    # ==================================================
    # Current observation normalization
    # ==================================================

    if normalize_observations:

        observations_np = (
            normalizer.normalize_observation(
                replay_batch.raw_observations,
                update=False,
            )
        )

        next_observations_np = (
            normalizer.normalize_observation(
                replay_batch.raw_next_observations,
                update=False,
            )
        )

    else:

        observations_np = (
            replay_batch.raw_observations
        )

        next_observations_np = (
            replay_batch.raw_next_observations
        )

    # ==================================================
    # Current reward normalization
    # ==================================================

    if normalize_rewards:

        reward_scale = np.sqrt(
            normalizer.return_rms.var
            + normalizer.epsilon
        )

        rewards_np = (
            replay_batch.raw_rewards
            / reward_scale
        )

        rewards_np = np.clip(
            rewards_np,
            -normalizer.reward_clip,
            normalizer.reward_clip,
        ).astype(np.float32)

    else:

        rewards_np = (
            replay_batch.raw_rewards
            .astype(np.float32)
        )

    # ==================================================
    # Convert to tensors
    # ==================================================

    observations = torch.as_tensor(
        observations_np,
        dtype=torch.float32,
        device=device,
    )

    next_observations = torch.as_tensor(
        next_observations_np,
        dtype=torch.float32,
        device=device,
    )

    actions = torch.as_tensor(
        replay_batch.actions,
        dtype=torch.float32,
        device=device,
    )

    rewards = torch.as_tensor(
        rewards_np,
        dtype=torch.float32,
        device=device,
    )

    terminals = torch.as_tensor(
        replay_batch.terminals,
        dtype=torch.float32,
        device=device,
    )

    episode_ends = torch.as_tensor(
        replay_batch.episode_ends,
        dtype=torch.float32,
        device=device,
    )

    behavior_log_probs_per_dim = (
        torch.as_tensor(
            replay_batch.log_probs_per_dim,
            dtype=torch.float32,
            device=device,
        )
    )

    # ==================================================
    # Current values and policy ratios
    # ==================================================

    with torch.no_grad():

        values = critic(
            observations
        )

        next_values = critic(
            next_observations
        )

        distribution = actor(
            observations
        )

        current_log_probs_per_dim = (
            distribution.log_prob(
                actions
            )
        )

        # Joint importance ratio rho_t
        joint_log_ratio = (
            current_log_probs_per_dim
            - behavior_log_probs_per_dim
        ).sum(dim=-1)

        importance_ratios = torch.exp(
            joint_log_ratio
        )

        truncated_ratios = torch.clamp(
            importance_ratios,
            max=1.0,
        )

        # ==================================================
        # TD residual
        # ==================================================

        bootstrap_mask = (
            1.0 - terminals
        )

        deltas = (
            rewards
            + gamma
            * next_values
            * bootstrap_mask
            - values
        )

        # ==================================================
        # GAE-V
        # ==================================================
        #
        # A_t =
        # delta_t
        # + gamma * lambda
        #   * min(1, rho_(t+1))
        #   * A_(t+1)
        #
        # Stop recursion across episode boundaries.
        # ==================================================

        advantages = torch.zeros_like(
            rewards
        )

        next_advantage = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=device,
        )

        num_steps = rewards.shape[0]

        for t in reversed(
            range(num_steps)
        ):

            continuation_mask = (
                1.0
                - episode_ends[t]
            )

            if t == num_steps - 1:

                next_ratio = torch.tensor(
                    1.0,
                    dtype=torch.float32,
                    device=device,
                )

            else:

                next_ratio = (
                    truncated_ratios[t + 1]
                )

            advantage = (
                deltas[t]
                + gamma
                * gae_lambda
                * continuation_mask
                * next_ratio
                * next_advantage
            )

            advantages[t] = advantage

            next_advantage = advantage

        # ==================================================
        # V-trace target
        # ==================================================

        returns = (
            truncated_ratios
            * advantages
            + values
        )

    return (
        advantages.cpu().numpy().astype(
            np.float32
        ),
        returns.cpu().numpy().astype(
            np.float32
        ),
        values.cpu().numpy().astype(
            np.float32
        ),
        importance_ratios.cpu().numpy().astype(
            np.float32
        ),
        observations_np.astype(
            np.float32
        ),
    )