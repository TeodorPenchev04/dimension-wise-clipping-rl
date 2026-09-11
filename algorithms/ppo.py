import torch
import torch.nn.functional as F

from core.distributions import get_log_probs


def ppo_update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    buffer,
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
    PPO update based on the OpenAI Baselines-style PPO setup
    used as the baseline in the DISC paper.

    Features:
    - Joint importance-sampling ratio
    - PPO clipped policy objective
    - Advantage normalization
    - Clipped value-function objective
    - Gradient norm clipping
    - No replay buffer
    - No J_IS penalty
    """

    num_samples = len(buffer)

    expected_samples = (
        minibatch_size * gradient_steps_per_epoch
    )

    if num_samples != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, "
            f"but buffer contains {num_samples}."
        )

    # --------------------------------------------------
    # Convert rollout data to tensors
    # --------------------------------------------------

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

    # PPO uses the joint action probability.
    old_joint_log_probs = (
        old_log_probs_per_dim.sum(dim=-1)
    )

    # --------------------------------------------------
    # Advantage normalization
    # --------------------------------------------------

    if normalize_advantages:
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "ratio_mean": [],
        "ratio_std": [],
        "ratio_outside_fraction": [],
        "zero_gradient_fraction": [],
        "actor_grad_norm": [],
        "critic_grad_norm": [],
    }

    # --------------------------------------------------
    # Optimization
    # --------------------------------------------------

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
                old_joint_log_probs[indices]
            )

            old_value_batch = old_values[indices]

            advantage_batch = advantages[indices]
            return_batch = returns[indices]

            # ==========================================
            # Actor
            # ==========================================

            (
                new_log_probs_per_dim,
                new_joint_log_probs,
            ) = get_log_probs(
                actor,
                obs_batch,
                action_batch,
            )

            log_ratio = (
                new_joint_log_probs
                - old_log_prob_batch
            )

            ratio = torch.exp(log_ratio)

            clipped_ratio = torch.clamp(
                ratio,
                1.0 - clip_ratio,
                1.0 + clip_ratio,
            )

            surrogate_1 = (
                ratio * advantage_batch
            )

            surrogate_2 = (
                clipped_ratio
                * advantage_batch
            )

            policy_loss = -torch.min(
                surrogate_1,
                surrogate_2,
            ).mean()

            distribution = actor(obs_batch)

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

            # PPO clipped value prediction
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

                approx_kl = (
                    (
                        old_log_prob_batch
                        - new_joint_log_probs
                    )
                ).mean()

                ratio_outside = (
                    (
                        ratio
                        < 1.0 - clip_ratio
                    )
                    |
                    (
                        ratio
                        > 1.0 + clip_ratio
                    )
                ).float().mean()

                # Samples whose PPO policy gradient
                # vanishes because of clipping.
                zero_gradient = (
                    (
                        (advantage_batch > 0)
                        & (
                            ratio
                            > 1.0 + clip_ratio
                        )
                    )
                    |
                    (
                        (advantage_batch < 0)
                        & (
                            ratio
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

            metrics["approx_kl"].append(
                approx_kl.item()
            )

            metrics["ratio_mean"].append(
                ratio.mean().item()
            )

            metrics["ratio_std"].append(
                ratio.std().item()
            )

            metrics[
                "ratio_outside_fraction"
            ].append(
                ratio_outside.item()
            )

            metrics[
                "zero_gradient_fraction"
            ].append(
                zero_gradient.item()
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

    return {
        key: sum(values) / len(values)
        for key, values in metrics.items()
    }