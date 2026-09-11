import numpy as np


class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.size = size

        # Normalized observations used for the current update.
        self.observations = np.zeros(
            (size, obs_dim),
            dtype=np.float32,
        )

        self.next_observations = np.zeros(
            (size, obs_dim),
            dtype=np.float32,
        )

        # Raw observations are preserved for replay.
        self.raw_observations = np.zeros(
            (size, obs_dim),
            dtype=np.float32,
        )

        self.raw_next_observations = np.zeros(
            (size, obs_dim),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (size, act_dim),
            dtype=np.float32,
        )

        # Reward used for the current training update.
        self.rewards = np.zeros(
            size,
            dtype=np.float32,
        )

        # Raw environment reward preserved for replay.
        self.raw_rewards = np.zeros(
            size,
            dtype=np.float32,
        )

        self.terminals = np.zeros(
            size,
            dtype=np.float32,
        )

        self.episode_ends = np.zeros(
            size,
            dtype=np.float32,
        )

        self.values = np.zeros(
            size,
            dtype=np.float32,
        )

        self.next_values = np.zeros(
            size,
            dtype=np.float32,
        )

        # Behavior-policy log probability
        # for every action dimension.
        self.log_probs_per_dim = np.zeros(
            (size, act_dim),
            dtype=np.float32,
        )

        self.advantages = np.zeros(
            size,
            dtype=np.float32,
        )

        self.returns = np.zeros(
            size,
            dtype=np.float32,
        )

        self.ptr = 0

    def add(
        self,
        observation,
        next_observation,
        action,
        reward,
        terminal,
        episode_end,
        value,
        next_value,
        log_prob_per_dim,
        raw_observation=None,
        raw_next_observation=None,
        raw_reward=None,
    ):
        if self.ptr >= self.size:
            raise RuntimeError(
                "Rollout buffer is full."
            )

        self.observations[self.ptr] = (
            observation
        )

        self.next_observations[self.ptr] = (
            next_observation
        )

        # Existing PPO/DISC scripts remain compatible.
        # If raw values are not supplied, fall back to
        # the provided training values.
        if raw_observation is None:
            raw_observation = observation

        if raw_next_observation is None:
            raw_next_observation = (
                next_observation
            )

        if raw_reward is None:
            raw_reward = reward

        self.raw_observations[self.ptr] = (
            raw_observation
        )

        self.raw_next_observations[self.ptr] = (
            raw_next_observation
        )

        self.actions[self.ptr] = action

        self.rewards[self.ptr] = reward
        self.raw_rewards[self.ptr] = raw_reward

        self.terminals[self.ptr] = terminal
        self.episode_ends[self.ptr] = (
            episode_end
        )

        self.values[self.ptr] = value
        self.next_values[self.ptr] = (
            next_value
        )

        self.log_probs_per_dim[self.ptr] = (
            log_prob_per_dim
        )

        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def __len__(self):
        return self.ptr