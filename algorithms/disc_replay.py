import numpy as np
import torch

from core.gae_v import compute_gae_v
from core.replay_buffer import ReplayBatch


ALPHA_IS_MIN = 2.0 ** -10
ALPHA_IS_MAX = 64.0


def rollout_to_replay_batch(current_buffer):
    """
    Convert the current rollout into the same raw-data
    representation used by old replay batches.

    This lets the current rollout and old rollouts be
    processed consistently using the CURRENT
    normalization statistics.
    """

    size = len(current_buffer)

    return ReplayBatch(
        raw_observations=(
            current_buffer
            .raw_observations[:size]
        ),
        raw_next_observations=(
            current_buffer
            .raw_next_observations[:size]
        ),
        actions=(
            current_buffer
            .actions[:size]
        ),
        raw_rewards=(
            current_buffer
            .raw_rewards[:size]
        ),
        terminals=(
            current_buffer
            .terminals[:size]
        ),
        episode_ends=(
            current_buffer
            .episode_ends[:size]
        ),
        log_probs_per_dim=(
            current_buffer
            .log_probs_per_dim[:size]
        ),
    )


def prepare_single_batch(
    replay_batch,
    actor,
    critic,
    normalizer,
    gamma,
    gae_lambda,
    normalize_observations,
    normalize_rewards,
    device,
):
    """
    Re-evaluate one rollout using the current policy,
    critic and normalization statistics.

    GAE-V reduces approximately to ordinary GAE for the
    newest on-policy rollout because rho is near one.
    """

    (
        advantages,
        returns,
        values,
        importance_ratios,
        observations,
    ) = compute_gae_v(
        replay_batch=replay_batch,
        actor=actor,
        critic=critic,
        normalizer=normalizer,
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_observations=(
            normalize_observations
        ),
        normalize_rewards=(
            normalize_rewards
        ),
        device=device,
    )

    return {
        "observations": observations,
        "actions": replay_batch.actions.copy(),
        "old_log_probs_per_dim": (
            replay_batch
            .log_probs_per_dim
            .copy()
        ),
        "old_values": values,
        "advantages": advantages,
        "returns": returns,
        "rho_now": importance_ratios,
    }


def concatenate_batches(batches):
    """
    Concatenate a list of prepared rollout batches.
    """

    if not batches:
        return None

    keys = [
        "observations",
        "actions",
        "old_log_probs_per_dim",
        "old_values",
        "advantages",
        "returns",
        "rho_now",
    ]

    return {
        key: np.concatenate(
            [
                batch[key]
                for batch in batches
            ],
            axis=0,
        )
        for key in keys
    }


def prepare_replay_data(
    current_buffer,
    old_replay_batches,
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
    Prepare Full-DISC data.

    Important details:

    1. Current and old rollouts are evaluated using the
       current normalization statistics.

    2. GAE-V is used consistently.

    3. Current on-policy data is kept separate from old
       replay data because the released DISC code samples
       64 current samples and 64 samples per accepted
       rollout batch during every gradient step.

    4. J_IS is computed only from current data.
    """

    # ==================================================
    # Current rollout
    # ==================================================

    current_batch = rollout_to_replay_batch(
        current_buffer
    )

    current_data = prepare_single_batch(
        replay_batch=current_batch,
        actor=actor,
        critic=critic,
        normalizer=normalizer,
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_observations=(
            normalize_observations
        ),
        normalize_rewards=(
            normalize_rewards
        ),
        device=device,
    )

    # ==================================================
    # Accepted old replay batches
    # ==================================================

    prepared_old_batches = []

    for replay_batch in old_replay_batches:

        batch_data = prepare_single_batch(
            replay_batch=replay_batch,
            actor=actor,
            critic=critic,
            normalizer=normalizer,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_observations=(
                normalize_observations
            ),
            normalize_rewards=(
                normalize_rewards
            ),
            device=device,
        )

        prepared_old_batches.append(
            batch_data
        )

    old_data = concatenate_batches(
        prepared_old_batches
    )

    return {
        "current": current_data,
        "old": old_data,
        "num_old_batches": (
            len(old_replay_batches)
        ),
        "num_batches": (
            1
            + len(old_replay_batches)
        ),
    }


def to_tensor(
    array,
    device,
):
    return torch.as_tensor(
        array,
        dtype=torch.float32,
        device=device,
    )


def tensorize_dataset(
    data,
    device,
):
    if data is None:
        return None

    return {
        key: to_tensor(
            value,
            device,
        )
        for key, value in data.items()
    }


def normalize_disc_advantages(
    advantages,
    rho_now,
):
    """
    Advantage normalization used in the released DISC
    implementation.

    radv = min(1, rho) * A

    A <- (
        A
        - mean(radv) / mean(min(1, rho))
    ) / std(radv)

    This differs from ordinary PPO z-score advantage
    normalization and matters for off-policy replay.
    """

    truncated_rho = torch.clamp(
        rho_now,
        max=1.0,
    )

    weighted_advantages = (
        truncated_rho
        * advantages
    )

    denominator_mean = (
        truncated_rho.mean()
        + 1e-8
    )

    centered_advantages = (
        advantages
        - weighted_advantages.mean()
        / denominator_mean
    )

    weighted_std = torch.std(
        weighted_advantages,
        correction=0,
    )

    normalized_advantages = (
        centered_advantages
        / (
            weighted_std
            + 1e-8
        )
    )

    return normalized_advantages


def disc_replay_update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    current_buffer,
    replay_data,
    alpha_is,
    jis_target=0.0001,
    clip_ratio=0.2,
    epochs=10,
    base_minibatch_size=64,
    gradient_steps_per_epoch=32,
    value_loss_coef=0.5,
    entropy_coef=0.0,
    max_grad_norm=0.5,
    normalize_advantages=True,
    device="cpu",
):
    """
    Full DISC optimization.

    Each gradient step contains:

        64 current samples

    plus

        64 samples for every accepted old rollout batch.

    Sampling is performed WITH REPLACEMENT, matching the
    released DISC implementation more closely.

    J_IS is calculated only from the current on-policy
    component.
    """

    del current_buffer

    num_old_batches = (
        replay_data["num_old_batches"]
    )

    num_batches = (
        replay_data["num_batches"]
    )

    effective_minibatch_size = (
        base_minibatch_size
        * num_batches
    )

    # ==================================================
    # Tensorize datasets
    # ==================================================

    current = tensorize_dataset(
        replay_data["current"],
        device,
    )

    old = tensorize_dataset(
        replay_data["old"],
        device,
    )

    current_size = (
        current["observations"]
        .shape[0]
    )

    expected_current_size = (
        base_minibatch_size
        * gradient_steps_per_epoch
    )

    if (
        current_size
        != expected_current_size
    ):
        raise ValueError(
            f"Expected current rollout size "
            f"{expected_current_size}, "
            f"but got {current_size}."
        )

    if old is not None:

        old_size = (
            old["observations"]
            .shape[0]
        )

        expected_old_size = (
            expected_current_size
            * num_old_batches
        )

        if (
            old_size
            != expected_old_size
        ):
            raise ValueError(
                f"Expected {expected_old_size} "
                f"old replay samples, "
                f"but got {old_size}."
            )

    # ==================================================
    # Metrics
    # ==================================================

    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "jis": [],
        "ratio_mean": [],
        "ratio_std": [],
        "dimension_clipped_fraction": [],
        "actor_grad_norm": [],
        "critic_grad_norm": [],
    }

    # ==================================================
    # Optimization
    # ==================================================

    for _ in range(epochs):

        for _ in range(
            gradient_steps_per_epoch
        ):

            # ==========================================
            # Current data
            #
            # np.random.choice in the released code
            # samples with replacement.
            # ==========================================

            current_indices = torch.randint(
                low=0,
                high=current_size,
                size=(
                    base_minibatch_size,
                ),
                device=device,
            )

            obs_parts = [
                current["observations"][
                    current_indices
                ]
            ]

            action_parts = [
                current["actions"][
                    current_indices
                ]
            ]

            old_log_prob_parts = [
                current[
                    "old_log_probs_per_dim"
                ][current_indices]
            ]

            old_value_parts = [
                current["old_values"][
                    current_indices
                ]
            ]

            advantage_parts = [
                current["advantages"][
                    current_indices
                ]
            ]

            return_parts = [
                current["returns"][
                    current_indices
                ]
            ]

            rho_parts = [
                current["rho_now"][
                    current_indices
                ]
            ]

            # ==========================================
            # Old replay data
            # ==========================================

            if (
                num_old_batches > 0
                and old is not None
            ):

                old_minibatch_size = (
                    base_minibatch_size
                    * num_old_batches
                )

                old_size = (
                    old["observations"]
                    .shape[0]
                )

                old_indices = torch.randint(
                    low=0,
                    high=old_size,
                    size=(
                        old_minibatch_size,
                    ),
                    device=device,
                )

                obs_parts.append(
                    old["observations"][
                        old_indices
                    ]
                )

                action_parts.append(
                    old["actions"][
                        old_indices
                    ]
                )

                old_log_prob_parts.append(
                    old[
                        "old_log_probs_per_dim"
                    ][old_indices]
                )

                old_value_parts.append(
                    old["old_values"][
                        old_indices
                    ]
                )

                advantage_parts.append(
                    old["advantages"][
                        old_indices
                    ]
                )

                return_parts.append(
                    old["returns"][
                        old_indices
                    ]
                )

                rho_parts.append(
                    old["rho_now"][
                        old_indices
                    ]
                )

            # ==========================================
            # Combined DISC mini-batch
            # ==========================================

            obs_batch = torch.cat(
                obs_parts,
                dim=0,
            )

            action_batch = torch.cat(
                action_parts,
                dim=0,
            )

            old_log_prob_batch = (
                torch.cat(
                    old_log_prob_parts,
                    dim=0,
                )
            )

            old_value_batch = (
                torch.cat(
                    old_value_parts,
                    dim=0,
                )
            )

            advantage_batch = (
                torch.cat(
                    advantage_parts,
                    dim=0,
                )
            )

            return_batch = torch.cat(
                return_parts,
                dim=0,
            )

            rho_now_batch = torch.cat(
                rho_parts,
                dim=0,
            )

            # ==========================================
            # DISC advantage normalization
            # ==========================================

            if normalize_advantages:

                advantage_batch = (
                    normalize_disc_advantages(
                        advantages=(
                            advantage_batch
                        ),
                        rho_now=(
                            rho_now_batch
                        ),
                    )
                )

            # ==========================================
            # Dimension-wise DISC objective
            # ==========================================

            distribution = actor(
                obs_batch
            )

            new_log_probs_per_dim = (
                distribution.log_prob(
                    action_batch
                )
            )

            log_ratio_per_dim = (
                new_log_probs_per_dim
                - old_log_prob_batch
            )

            ratio_per_dim = torch.exp(
                log_ratio_per_dim
            )

            clipped_ratio_per_dim = (
                torch.clamp(
                    ratio_per_dim,
                    1.0 - clip_ratio,
                    1.0 + clip_ratio,
                )
            )

            sign = torch.sign(
                advantage_batch
            ).unsqueeze(-1)

            clipped_components = (
                sign
                * torch.minimum(
                    sign
                    * ratio_per_dim,
                    sign
                    * clipped_ratio_per_dim,
                )
            )

            dimensionwise_ratio = (
                clipped_components.prod(
                    dim=-1
                )
            )

            ratio_normalizer = (
                dimensionwise_ratio
                .mean()
                .detach()
                + 1e-8
            )

            disc_surrogate = (
                dimensionwise_ratio
                * advantage_batch
                / ratio_normalizer
            )

            # ==========================================
            # J_IS
            #
            # CURRENT on-policy samples only.
            # ==========================================

            jis_observations = (
                current["observations"][
                    current_indices
                ]
            )

            jis_actions = (
                current["actions"][
                    current_indices
                ]
            )

            jis_old_log_probs = (
                current[
                    "old_log_probs_per_dim"
                ][current_indices]
            )

            jis_distribution = actor(
                jis_observations
            )

            jis_new_log_probs = (
                jis_distribution.log_prob(
                    jis_actions
                )
            )

            jis_joint_log_ratio = (
                jis_new_log_probs
                - jis_old_log_probs
            ).sum(dim=-1)

            jis = (
                0.5
                * (
                    jis_joint_log_ratio
                    ** 2
                ).mean()
            )

            # ==========================================
            # Actor
            # ==========================================

            policy_loss = (
                -disc_surrogate.mean()
                + alpha_is
                * jis
            )

            entropy = (
                distribution
                .entropy()
                .sum(dim=-1)
                .mean()
            )

            actor_loss = (
                policy_loss
                - entropy_coef
                * entropy
            )

            actor_optimizer.zero_grad()

            actor_loss.backward()

            actor_grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    actor.parameters(),
                    max_grad_norm,
                )
            )

            actor_optimizer.step()

            # ==========================================
            # Critic
            # ==========================================

            predicted_values = critic(
                obs_batch
            )

            value_pred_clipped = (
                old_value_batch
                + torch.clamp(
                    predicted_values
                    - old_value_batch,
                    -clip_ratio,
                    clip_ratio,
                )
            )

            value_loss_unclipped = (
                predicted_values
                - return_batch
            ) ** 2

            value_loss_clipped = (
                value_pred_clipped
                - return_batch
            ) ** 2

            value_loss = (
                0.5
                * torch.max(
                    value_loss_unclipped,
                    value_loss_clipped,
                ).mean()
            )

            critic_loss = (
                value_loss_coef
                * value_loss
            )

            critic_optimizer.zero_grad()

            critic_loss.backward()

            critic_grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    critic.parameters(),
                    max_grad_norm,
                )
            )

            critic_optimizer.step()

            # ==========================================
            # Diagnostics
            # ==========================================

            with torch.no_grad():

                advantage_expanded = (
                    advantage_batch
                    .unsqueeze(-1)
                )

                dimension_clipped = (
                    (
                        (
                            advantage_expanded
                            > 0
                        )
                        &
                        (
                            ratio_per_dim
                            > 1.0
                            + clip_ratio
                        )
                    )
                    |
                    (
                        (
                            advantage_expanded
                            < 0
                        )
                        &
                        (
                            ratio_per_dim
                            < 1.0
                            - clip_ratio
                        )
                    )
                ).float().mean()

            metrics[
                "policy_loss"
            ].append(
                policy_loss.item()
            )

            metrics[
                "value_loss"
            ].append(
                value_loss.item()
            )

            metrics[
                "entropy"
            ].append(
                entropy.item()
            )

            metrics[
                "jis"
            ].append(
                jis.item()
            )

            metrics[
                "ratio_mean"
            ].append(
                ratio_per_dim
                .mean()
                .item()
            )

            metrics[
                "ratio_std"
            ].append(
                ratio_per_dim
                .std(
                    correction=0
                )
                .item()
            )

            metrics[
                "dimension_clipped_fraction"
            ].append(
                dimension_clipped.item()
            )

            metrics[
                "actor_grad_norm"
            ].append(
                float(
                    actor_grad_norm
                )
            )

            metrics[
                "critic_grad_norm"
            ].append(
                float(
                    critic_grad_norm
                )
            )

    # ==================================================
    # Adaptive alpha_IS
    # ==================================================

    mean_jis = (
        sum(metrics["jis"])
        / len(metrics["jis"])
    )

    if (
        mean_jis
        > jis_target * 1.5
    ):
        alpha_is *= 2.0

    elif (
        mean_jis
        < jis_target / 1.5
    ):
        alpha_is /= 2.0

    alpha_is = max(
        ALPHA_IS_MIN,
        min(
            alpha_is,
            ALPHA_IS_MAX,
        ),
    )

    # ==================================================
    # Final metrics
    # ==================================================

    averaged_metrics = {
        key: (
            sum(values)
            / len(values)
        )
        for key, values
        in metrics.items()
    }

    averaged_metrics[
        "alpha_is"
    ] = alpha_is

    averaged_metrics[
        "num_batches"
    ] = num_batches

    averaged_metrics[
        "effective_minibatch_size"
    ] = effective_minibatch_size

    return (
        averaged_metrics,
        alpha_is,
    )