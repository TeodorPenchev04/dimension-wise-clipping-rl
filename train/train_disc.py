import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from algorithms.disc import disc_update
from core.buffer import RolloutBuffer
from core.evaluation import evaluate_policy
from core.gae import compute_gae
from core.networks import GaussianActor, ValueNetwork
from core.normalization import ObservationRewardNormalizer


EVAL_EPISODES = 10


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def set_learning_rate(
    optimizer,
    learning_rate,
):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env-id",
        type=str,
        default="Hopper-v5",
    )

    parser.add_argument(
        "--total-steps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
    )

    args = parser.parse_args()

    # ==================================================
    # Configuration
    # ==================================================

    config = load_config(
        args.config
    )

    gamma = config["gamma"]

    gae_lambda = config[
        "gae_lambda"
    ]

    rollout_steps = config[
        "rollout_steps"
    ]

    epochs = config[
        "epochs"
    ]

    gradient_steps_per_epoch = config[
        "gradient_steps_per_epoch"
    ]

    minibatch_size = config[
        "minibatch_size"
    ]

    clip_ratio = config[
        "clip_ratio"
    ]

    learning_rate_initial = config[
        "learning_rate_initial"
    ]

    learning_rate_min = config[
        "learning_rate_min"
    ]

    anneal_learning_rate = config[
        "anneal_learning_rate"
    ]

    value_loss_coef = config[
        "value_loss_coef"
    ]

    entropy_coef = config[
        "entropy_coef"
    ]

    max_grad_norm = config[
        "max_grad_norm"
    ]

    adam_epsilon = config[
        "adam_epsilon"
    ]

    normalize_advantages = config[
        "normalize_advantages"
    ]

    normalize_observations = config[
        "normalize_observations"
    ]

    normalize_rewards = config[
        "normalize_rewards"
    ]

    observation_clip = config[
        "observation_clip"
    ]

    reward_clip = config[
        "reward_clip"
    ]

    normalization_epsilon = config[
        "normalization_epsilon"
    ]

    hidden_sizes = tuple(
        config["hidden_sizes"]
    )

    jis_target = config[
        "jis_target"
    ]

    alpha_is = config[
        "alpha_is_initial"
    ]

    device = torch.device(
        "cpu"
    )

    # ==================================================
    # Reproducibility
    # ==================================================

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    # ==================================================
    # Training environment
    # ==================================================

    env = gym.make(
        args.env_id
    )

    raw_observation, _ = env.reset(
        seed=args.seed
    )

    env.action_space.seed(
        args.seed
    )

    # ==================================================
    # Separate evaluation environment
    # ==================================================

    eval_env = gym.make(
        args.env_id
    )

    eval_seed = (
        args.seed + 10_000
    )

    eval_env.reset(
        seed=eval_seed
    )

    eval_env.action_space.seed(
        eval_seed
    )

    obs_dim = (
        env.observation_space
        .shape[0]
    )

    act_dim = (
        env.action_space
        .shape[0]
    )

    print(
        f"Environment: "
        f"{args.env_id}"
    )

    print(
        f"Observation dimension: "
        f"{obs_dim}"
    )

    print(
        f"Action dimension: "
        f"{act_dim}"
    )

    print(
        "Algorithm: DISC without replay"
    )

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Evaluation episodes: "
        f"{EVAL_EPISODES}"
    )

    # ==================================================
    # Normalization
    # ==================================================

    normalizer = (
        ObservationRewardNormalizer(
            obs_shape=(obs_dim,),
            gamma=gamma,
            observation_clip=(
                observation_clip
            ),
            reward_clip=(
                reward_clip
            ),
            epsilon=(
                normalization_epsilon
            ),
        )
    )

    if normalize_observations:

        observation = (
            normalizer
            .normalize_observation(
                raw_observation,
                update=True,
            )
        )

    else:

        observation = (
            raw_observation
            .astype(
                np.float32
            )
        )

    # ==================================================
    # Networks
    # ==================================================

    actor = GaussianActor(
        obs_dim,
        act_dim,
        hidden_sizes,
    ).to(
        device
    )

    critic = ValueNetwork(
        obs_dim,
        hidden_sizes,
    ).to(
        device
    )

    actor_optimizer = (
        torch.optim.Adam(
            actor.parameters(),
            lr=(
                learning_rate_initial
            ),
            eps=adam_epsilon,
        )
    )

    critic_optimizer = (
        torch.optim.Adam(
            critic.parameters(),
            lr=(
                learning_rate_initial
            ),
            eps=adam_epsilon,
        )
    )

    # ==================================================
    # Rollout buffer
    # ==================================================

    buffer = RolloutBuffer(
        rollout_steps,
        obs_dim,
        act_dim,
    )

    # ==================================================
    # Output
    # ==================================================

    env_name = (
        args.env_id
        .split("-")[0]
        .lower()
    )

    output_dir = (
        Path("experiments")
        / env_name
        / f"disc_seed{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / "metrics.csv"
    )

    with open(
        output_dir
        / "config.yaml",
        "w",
    ) as file:

        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    fieldnames = [
        "environment_steps",
        "update",
        "learning_rate",
        "mean_recent_return",
        "eval_mean_return",
        "eval_std_return",
        "eval_mean_episode_length",
        "policy_loss",
        "value_loss",
        "entropy",
        "jis",
        "alpha_is",
        "ratio_mean",
        "ratio_std",
        "dimension_clipped_fraction",
        "actor_grad_norm",
        "critic_grad_norm",
    ]

    with open(
        metrics_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

    # ==================================================
    # Training state
    # ==================================================

    global_step = 0
    update_number = 0

    episode_return = 0.0
    episode_returns = []

    # ==================================================
    # Training loop
    # ==================================================

    while (
        global_step
        < args.total_steps
    ):

        buffer.reset()

        # ==============================================
        # Learning-rate schedule
        # ==============================================

        if anneal_learning_rate:

            progress = (
                global_step
                / args.total_steps
            )

            learning_rate = max(
                learning_rate_min,
                learning_rate_initial
                * (
                    1.0
                    - progress
                ),
            )

        else:

            learning_rate = (
                learning_rate_initial
            )

        set_learning_rate(
            actor_optimizer,
            learning_rate,
        )

        set_learning_rate(
            critic_optimizer,
            learning_rate,
        )

        # ==============================================
        # Collect rollout
        # ==============================================

        for _ in range(
            rollout_steps
        ):

            obs_tensor = (
                torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                )
            )

            with torch.no_grad():

                distribution = actor(
                    obs_tensor
                )

                action_tensor = (
                    distribution.sample()
                )

                log_prob_per_dim = (
                    distribution.log_prob(
                        action_tensor
                    )
                )

                value = critic(
                    obs_tensor
                )

            action = (
                action_tensor
                .cpu()
                .numpy()
            )

            env_action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high,
            )

            (
                raw_next_observation,
                raw_reward,
                terminated,
                truncated,
                _,
            ) = env.step(
                env_action
            )

            episode_end = (
                terminated
                or truncated
            )

            # ==========================================
            # Observation normalization
            # ==========================================

            if normalize_observations:

                next_observation = (
                    normalizer
                    .normalize_observation(
                        raw_next_observation,
                        update=True,
                    )
                )

            else:

                next_observation = (
                    raw_next_observation
                    .astype(
                        np.float32
                    )
                )

            # ==========================================
            # Reward normalization
            # ==========================================

            if normalize_rewards:

                training_reward = (
                    normalizer
                    .normalize_reward(
                        raw_reward,
                        episode_end=(
                            episode_end
                        ),
                        update=True,
                    )
                )

            else:

                training_reward = (
                    raw_reward
                )

            # ==========================================
            # Next-state value
            # ==========================================

            with torch.no_grad():

                next_obs_tensor = (
                    torch.as_tensor(
                        next_observation,
                        dtype=torch.float32,
                        device=device,
                    )
                )

                next_value = critic(
                    next_obs_tensor
                ).item()

            # ==========================================
            # Store transition
            # ==========================================

            buffer.add(
                observation=(
                    observation
                ),
                next_observation=(
                    next_observation
                ),
                action=action,
                reward=(
                    training_reward
                ),
                terminal=float(
                    terminated
                ),
                episode_end=float(
                    episode_end
                ),
                value=value.item(),
                next_value=(
                    next_value
                ),
                log_prob_per_dim=(
                    log_prob_per_dim
                    .cpu()
                    .numpy()
                ),
            )

            episode_return += (
                raw_reward
            )

            global_step += 1

            # ==========================================
            # Reset / advance environment
            # ==========================================

            if episode_end:

                episode_returns.append(
                    episode_return
                )

                episode_return = 0.0

                raw_observation, _ = (
                    env.reset()
                )

                if normalize_observations:

                    observation = (
                        normalizer
                        .normalize_observation(
                            raw_observation,
                            update=True,
                        )
                    )

                else:

                    observation = (
                        raw_observation
                        .astype(
                            np.float32
                        )
                    )

            else:

                raw_observation = (
                    raw_next_observation
                )

                observation = (
                    next_observation
                )

        # ==============================================
        # Ordinary GAE
        # ==============================================

        compute_gae(
            buffer,
            gamma=gamma,
            gae_lambda=(
                gae_lambda
            ),
        )

        # ==============================================
        # DISC update without replay
        # ==============================================

        (
            metrics,
            alpha_is,
        ) = disc_update(
            actor=actor,
            critic=critic,
            actor_optimizer=(
                actor_optimizer
            ),
            critic_optimizer=(
                critic_optimizer
            ),
            buffer=buffer,
            alpha_is=(
                alpha_is
            ),
            jis_target=(
                jis_target
            ),
            clip_ratio=(
                clip_ratio
            ),
            epochs=epochs,
            minibatch_size=(
                minibatch_size
            ),
            gradient_steps_per_epoch=(
                gradient_steps_per_epoch
            ),
            value_loss_coef=(
                value_loss_coef
            ),
            entropy_coef=(
                entropy_coef
            ),
            max_grad_norm=(
                max_grad_norm
            ),
            normalize_advantages=(
                normalize_advantages
            ),
            device=device,
        )

        update_number += 1

        # ==============================================
        # Deterministic evaluation
        # ==============================================

        evaluation = evaluate_policy(
            env=eval_env,
            actor=actor,
            normalizer=normalizer,
            num_episodes=(
                EVAL_EPISODES
            ),
            normalize_observations=(
                normalize_observations
            ),
            device=device,
        )

        # ==============================================
        # Logging
        # ==============================================

        if episode_returns:

            mean_recent_return = (
                float(
                    np.mean(
                        episode_returns[
                            -10:
                        ]
                    )
                )
            )

        else:

            mean_recent_return = (
                np.nan
            )

        row = {
            "environment_steps": (
                global_step
            ),
            "update": (
                update_number
            ),
            "learning_rate": (
                learning_rate
            ),
            "mean_recent_return": (
                mean_recent_return
            ),
            "eval_mean_return": (
                evaluation[
                    "mean_return"
                ]
            ),
            "eval_std_return": (
                evaluation[
                    "std_return"
                ]
            ),
            "eval_mean_episode_length": (
                evaluation[
                    "mean_episode_length"
                ]
            ),
            **metrics,
        }

        with open(
            metrics_path,
            "a",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writerow(
                row
            )

        print(
            f"Steps: "
            f"{global_step:>8} | "
            f"Train(10): "
            f"{mean_recent_return:>8.2f} | "
            f"Eval(10): "
            f"{evaluation['mean_return']:>8.2f} | "
            f"JIS: "
            f"{metrics['jis']:.6f} | "
            f"alpha_IS: "
            f"{alpha_is:.2f} | "
            f"Dim-clip: "
            f"{metrics['dimension_clipped_fraction']:.3f}"
        )

    # ==================================================
    # Save
    # ==================================================

    torch.save(
        actor.state_dict(),
        output_dir
        / "actor.pt",
    )

    torch.save(
        critic.state_dict(),
        output_dir
        / "critic.pt",
    )

    np.savez(
        output_dir
        / "normalization_stats.npz",

        obs_mean=(
            normalizer
            .obs_rms.mean
        ),

        obs_var=(
            normalizer
            .obs_rms.var
        ),

        obs_count=(
            normalizer
            .obs_rms.count
        ),

        return_mean=(
            normalizer
            .return_rms.mean
        ),

        return_var=(
            normalizer
            .return_rms.var
        ),

        return_count=(
            normalizer
            .return_rms.count
        ),
    )

    env.close()
    eval_env.close()

    print()

    print(
        "Training complete."
    )

    print(
        f"Final alpha_IS: "
        f"{alpha_is}"
    )

    print(
        f"Results saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()