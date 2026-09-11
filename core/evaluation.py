import numpy as np
import torch


def evaluate_policy(
    env,
    actor,
    normalizer,
    num_episodes=10,
    normalize_observations=True,
    device="cpu",
):
    """
    Deterministically evaluate a Gaussian policy.

    The action is the mean of the Gaussian distribution,
    matching the deterministic evaluation procedure used
    in Han & Sung (2019).

    Observation-normalization statistics are NOT updated
    during evaluation.

    Returns:
        mean_return
        std_return
        mean_episode_length
        episode_returns
    """

    episode_returns = []
    episode_lengths = []

    actor_was_training = actor.training
    actor.eval()

    for _ in range(num_episodes):

        raw_observation, _ = env.reset()

        episode_return = 0.0
        episode_length = 0

        terminated = False
        truncated = False

        while not (
            terminated
            or truncated
        ):

            if normalize_observations:

                observation = (
                    normalizer
                    .normalize_observation(
                        raw_observation,
                        update=False,
                    )
                )

            else:

                observation = (
                    raw_observation
                    .astype(
                        np.float32
                    )
                )

            observation_tensor = (
                torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                )
            )

            with torch.no_grad():

                distribution = actor(
                    observation_tensor
                )

                # Deterministic policy:
                # use Gaussian mean.
                action = (
                    distribution.mean
                    .cpu()
                    .numpy()
                )

            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high,
            )

            (
                raw_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(
                action
            )

            episode_return += reward
            episode_length += 1

        episode_returns.append(
            episode_return
        )

        episode_lengths.append(
            episode_length
        )

    if actor_was_training:
        actor.train()

    return {
        "mean_return": float(
            np.mean(
                episode_returns
            )
        ),
        "std_return": float(
            np.std(
                episode_returns
            )
        ),
        "mean_episode_length": float(
            np.mean(
                episode_lengths
            )
        ),
        "episode_returns": (
            episode_returns
        ),
    }