from collections import deque

import numpy as np


class ReplayBatch:
    """
    Immutable copy of one rollout batch.

    Raw observations and rewards are stored so that old
    trajectories can be normalized using the current
    normalization statistics when they are reused.
    """

    def __init__(
        self,
        raw_observations,
        raw_next_observations,
        actions,
        raw_rewards,
        terminals,
        episode_ends,
        log_probs_per_dim,
    ):
        self.raw_observations = np.array(
            raw_observations,
            dtype=np.float32,
            copy=True,
        )

        self.raw_next_observations = np.array(
            raw_next_observations,
            dtype=np.float32,
            copy=True,
        )

        self.actions = np.array(
            actions,
            dtype=np.float32,
            copy=True,
        )

        self.raw_rewards = np.array(
            raw_rewards,
            dtype=np.float32,
            copy=True,
        )

        self.terminals = np.array(
            terminals,
            dtype=np.float32,
            copy=True,
        )

        self.episode_ends = np.array(
            episode_ends,
            dtype=np.float32,
            copy=True,
        )

        # Behavior-policy probabilities from the policy
        # that originally generated this rollout.
        self.log_probs_per_dim = np.array(
            log_probs_per_dim,
            dtype=np.float32,
            copy=True,
        )

    def __len__(self):
        return len(self.raw_rewards)


class ReplayBuffer:
    """
    Stores complete rollout batches B_i, B_{i-1}, ...

    This is batch-level experience replay rather than a
    transition-level replay buffer such as SAC uses.
    """

    def __init__(self, max_batches=64):
        self.max_batches = max_batches

        self.batches = deque(
            maxlen=max_batches
        )

    def add_rollout(self, rollout_buffer):
        size = len(rollout_buffer)

        if size == 0:
            raise ValueError(
                "Cannot store an empty rollout."
            )

        batch = ReplayBatch(
            raw_observations=(
                rollout_buffer
                .raw_observations[:size]
            ),
            raw_next_observations=(
                rollout_buffer
                .raw_next_observations[:size]
            ),
            actions=(
                rollout_buffer
                .actions[:size]
            ),
            raw_rewards=(
                rollout_buffer
                .raw_rewards[:size]
            ),
            terminals=(
                rollout_buffer
                .terminals[:size]
            ),
            episode_ends=(
                rollout_buffer
                .episode_ends[:size]
            ),
            log_probs_per_dim=(
                rollout_buffer
                .log_probs_per_dim[:size]
            ),
        )

        self.batches.append(batch)

    def latest(self):
        if not self.batches:
            raise RuntimeError(
                "Replay buffer is empty."
            )

        return self.batches[-1]

    def get_all(self):
        return list(self.batches)

    def get_newest_first(self):
        return list(
            reversed(self.batches)
        )

    def clear(self):
        self.batches.clear()

    def __len__(self):
        return len(self.batches)