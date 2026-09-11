import torch


ALPHA_IS_MIN = 2.0 ** -10
ALPHA_IS_MAX = 64.0


def disc_update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    buffer,
    alpha_is,
    jis_target=0.0001,
    clip_ratio=0.2,
    epochs=10,
    minibatch_size=64,
    gradient_steps_per_epoch=32,
    value_loss_coef=0.5,
    entropy_coef=0.0,
    max_grad_norm=0.5,
    normalize_advantages=True,
    device="cpu",
):
    """
    DISC without replay.

    Features:
    - Dimension-wise importance ratios
    - Dimension-wise clipping
    - J_IS penalty
    - Adaptive alpha_IS
    - alpha_IS bounded to [2^-10, 64]
    - Standard GAE
    - No replay buffer
    """

    num_samples = len(buffer)

    expected_samples = (
        minibatch_size * gradient_steps_per_epoch
    )

    if num_samples != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, "
            f"got {num_samples}."
        )

    observations = torch.as_tensor(
        buffer.observations[:num_samples],
        dtype=torch.float32,
        device=device,
    )

    actions = torch.as_tensor(
        buffer.actions[:num_samples],
        dtype=torch.float32,
        device=device,
    )

    old_log_probs_per_dim = torch.as_tensor(
        buffer.log_probs_per_dim[:num_samples],
        dtype=torch.float32,
        device=device,
    )

    old_values = torch.as_tensor(
        buffer.values[:num_samples],
        dtype=torch.float32,
        device=device,
    )

    advantages = torch.as_tensor(
        buffer.advantages[:num_samples],
        dtype=torch.float32,
        device=device,
    )

    returns = torch.as_tensor(
        buffer.returns[:num_samples],
        dtype=torch.float32,
        device=device,
    )

    if normalize_advantages:
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

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

    for _ in range(epochs):

        permutation = torch.randperm(
            num_samples,
            device=device,
        )

        for step in range(
            gradient_steps_per_epoch
        ):
            start = step * minibatch_size
            end = start + minibatch_size

            indices = permutation[start:end]

            obs_batch = observations[indices]
            action_batch = actions[indices]

            old_log_prob_batch = (
                old_log_probs_per_dim[indices]
            )

            old_value_batch = old_values[indices]
            advantage_batch = advantages[indices]
            return_batch = returns[indices]

            # ==========================================
            # Actor
            # ==========================================

            distribution = actor(obs_batch)

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

            clipped_ratio_per_dim = torch.clamp(
                ratio_per_dim,
                1.0 - clip_ratio,
                1.0 + clip_ratio,
            )

            # Advantage sign determines which side
            # of the ratio is clipped.
            sign = torch.sign(
                advantage_batch
            ).unsqueeze(-1)

            clipped_components = (
                sign
                * torch.minimum(
                    sign * ratio_per_dim,
                    sign * clipped_ratio_per_dim,
                )
            )

            # Product across action dimensions.
            dimensionwise_ratio = (
                clipped_components.prod(
                    dim=-1
                )
            )

            # Stop-gradient normalization used by DISC.
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

            # Joint log ratio for J_IS.
            joint_log_ratio = (
                log_ratio_per_dim.sum(
                    dim=-1
                )
            )

            jis = (
                0.5
                * (
                    joint_log_ratio ** 2
                ).mean()
            )

            policy_loss = (
                -disc_surrogate.mean()
                + alpha_is * jis
            )

            entropy = (
                distribution.entropy()
                .sum(dim=-1)
                .mean()
            )

            actor_loss = (
                policy_loss
                - entropy_coef * entropy
            )

            actor_optimizer.zero_grad()

            actor_loss.backward()

            actor_grad_norm = (
                torch.nn.utils.clip_grad_norm_(
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
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(),
                    max_grad_norm,
                )
            )

            critic_optimizer.step()

            # ==========================================
            # Diagnostics
            # ==========================================

            with torch.no_grad():

                adv_expanded = (
                    advantage_batch.unsqueeze(-1)
                )

                clipped_dimensions = (
                    (
                        (adv_expanded > 0)
                        & (
                            ratio_per_dim
                            > 1.0 + clip_ratio
                        )
                    )
                    |
                    (
                        (adv_expanded < 0)
                        & (
                            ratio_per_dim
                            < 1.0 - clip_ratio
                        )
                    )
                ).float().mean()

            metrics["policy_loss"].append(
                policy_loss.item()
            )

            metrics["value_loss"].append(
                value_loss.item()
            )

            metrics["entropy"].append(
                entropy.item()
            )

            metrics["jis"].append(
                jis.item()
            )

            metrics["ratio_mean"].append(
                ratio_per_dim.mean().item()
            )

            metrics["ratio_std"].append(
                ratio_per_dim.std().item()
            )

            metrics[
                "dimension_clipped_fraction"
            ].append(
                clipped_dimensions.item()
            )

            metrics[
                "actor_grad_norm"
            ].append(
                float(actor_grad_norm)
            )

            metrics[
                "critic_grad_norm"
            ].append(
                float(critic_grad_norm)
            )

    # ==============================================
    # Adaptive alpha_IS
    # ==============================================

    mean_jis = (
        sum(metrics["jis"])
        / len(metrics["jis"])
    )

    if mean_jis < jis_target / 1.5:
        alpha_is /= 2.0

    elif mean_jis > jis_target * 1.5:
        alpha_is *= 2.0

    # Bound alpha_IS as in the released DISC code.
    alpha_is = max(
        ALPHA_IS_MIN,
        min(
            alpha_is,
            ALPHA_IS_MAX,
        ),
    )

    averaged_metrics = {
        key: sum(values) / len(values)
        for key, values in metrics.items()
    }

    averaged_metrics["alpha_is"] = alpha_is

    return averaged_metrics, alpha_is