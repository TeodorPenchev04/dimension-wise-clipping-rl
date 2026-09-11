import numpy as np


class RunningMeanStd:
    """
    Tracks running mean and variance using numerically stable
    parallel updates.
    """

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)

        if values.ndim == self.mean.ndim:
            values = np.expand_dims(values, axis=0)

        batch_mean = np.mean(values, axis=0)
        batch_var = np.var(values, axis=0)
        batch_count = values.shape[0]

        self.update_from_moments(
            batch_mean,
            batch_var,
            batch_count,
        )

    def update_from_moments(
        self,
        batch_mean,
        batch_var,
        batch_count,
    ):
        delta = batch_mean - self.mean

        total_count = self.count + batch_count

        new_mean = (
            self.mean
            + delta * batch_count / total_count
        )

        old_variance_sum = self.var * self.count
        batch_variance_sum = batch_var * batch_count

        correction = (
            delta ** 2
            * self.count
            * batch_count
            / total_count
        )

        new_variance_sum = (
            old_variance_sum
            + batch_variance_sum
            + correction
        )

        new_var = new_variance_sum / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class ObservationRewardNormalizer:
    """
    OpenAI-Baselines-style normalization for MuJoCo PPO.

    Observations:
        (obs - mean) / sqrt(var + epsilon)

    Rewards:
        reward / sqrt(var(discounted_return) + epsilon)
    """

    def __init__(
        self,
        obs_shape,
        gamma=0.99,
        observation_clip=10.0,
        reward_clip=10.0,
        epsilon=1e-8,
    ):
        self.gamma = gamma

        self.observation_clip = observation_clip
        self.reward_clip = reward_clip
        self.epsilon = epsilon

        self.obs_rms = RunningMeanStd(
            shape=obs_shape
        )

        self.return_rms = RunningMeanStd(
            shape=()
        )

        self.discounted_return = 0.0

    def normalize_observation(
        self,
        observation,
        update=True,
    ):
        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if update:
            self.obs_rms.update(observation)

        normalized = (
            observation - self.obs_rms.mean
        ) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )

        normalized = np.clip(
            normalized,
            -self.observation_clip,
            self.observation_clip,
        )

        return normalized.astype(np.float32)

    def normalize_reward(
        self,
        reward,
        episode_end=False,
        update=True,
    ):
        """
        Reward normalization uses the running variance of
        discounted returns, as in VecNormalize.

        The raw reward should still be used when reporting
        episode return.
        """

        self.discounted_return = (
            self.gamma * self.discounted_return
            + reward
        )

        if update:
            self.return_rms.update(
                np.array(
                    self.discounted_return,
                    dtype=np.float64,
                )
            )

        normalized_reward = (
            reward
            / np.sqrt(
                self.return_rms.var
                + self.epsilon
            )
        )

        normalized_reward = np.clip(
            normalized_reward,
            -self.reward_clip,
            self.reward_clip,
        )

        if episode_end:
            self.discounted_return = 0.0

        return float(normalized_reward)

    def reset_return(self):
        self.discounted_return = 0.0