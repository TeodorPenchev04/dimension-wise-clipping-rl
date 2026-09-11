import torch


def sample_action(actor, observation):
    """
    Sample an action from the actor and return:
    - action
    - per-dimension log probability
    """

    with torch.no_grad():
        distribution = actor(observation)

        action = distribution.sample()
        log_prob_per_dim = distribution.log_prob(action)

    return action, log_prob_per_dim


def get_log_probs(actor, observations, actions):
    """
    Compute log probabilities for given actions.

    Returns:
    - per-dimension log probabilities
    - joint log probability
    """

    distribution = actor(observations)

    log_prob_per_dim = distribution.log_prob(actions)

    joint_log_prob = log_prob_per_dim.sum(dim=-1)

    return log_prob_per_dim, joint_log_prob


def get_entropy(actor, observations):
    """
    Compute mean joint entropy of the policy.
    """

    distribution = actor(observations)

    entropy_per_dim = distribution.entropy()

    joint_entropy = entropy_per_dim.sum(dim=-1)

    return joint_entropy.mean()