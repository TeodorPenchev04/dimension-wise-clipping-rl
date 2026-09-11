# Dimension-Wise Clipping for Reinforcement Learning

This repository compares **Proximal Policy Optimization (PPO)** with **Dimension-Wise Importance Sampling Weight Clipping (DISC)** on continuous-control reinforcement learning tasks.

The project is based on Han and Sung's ICML 2019 paper, *Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning*.

## Research Question

**Does dimension-wise importance sampling clipping improve learning compared with standard PPO as action dimensionality increases?**

A secondary goal is to measure how much additional benefit comes from DISC's ability to reuse previous rollout batches.

---

## Motivation

Standard PPO uses one importance sampling ratio for the complete action:

$$
\rho_t =
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
$$

For a factorized Gaussian policy:

$$
\rho_t =
\prod_{d=1}^{D} \rho_{t,d}
$$

As the action dimension increases, this product can move far from 1 even when the change in each individual action dimension is small.

PPO clips this joint ratio, which can cause the gradient of a sample to vanish.

DISC instead computes and clips each action-dimension ratio separately:

$$
\rho_{t,d} =
\frac{\pi_{\theta,d}(a_{t,d}|s_t)}
{\pi_{\theta_{\mathrm{old}},d}(a_{t,d}|s_t)}
$$

The goal is to preserve more useful policy-gradient information, especially in high-dimensional action spaces.

---

## Algorithms

### 1. PPO

- Joint importance sampling ratio
- Standard PPO clipping
- GAE
- No IS penalty
- No replay buffer

PPO objective:

$$
L^{\mathrm{PPO}} =
\min \left(
\rho_t \hat{A}_t,
\mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)
$$

---

### 2. DISC without Replay

- Dimension-wise importance ratios
- Dimension-wise clipping
- IS-control penalty $J_{IS}$
- Adaptive $\alpha_{IS}$
- Standard GAE
- No replay buffer

The IS penalty is:

$$
J_{IS} =
\frac{1}{2M}
\sum_{m=1}^{M}
(\log \rho_m)^2
$$

This variant isolates the effect of dimension-wise clipping and IS control without experience reuse.

---

### 3. Full DISC

- Dimension-wise clipping
- $J_{IS}$ penalty
- Adaptive $\alpha_{IS}$
- Replay buffer
- Replay-batch filtering
- GAE-V for off-policy advantage estimation

Previous rollout batches are stored and reused only when their importance ratios remain sufficiently close to the current policy.

---

## Experimental Setup

The main experiment uses the **same clipping ratio for all three algorithms**:

$$
\epsilon = 0.2
$$

| Algorithm | Clip Ratio | Replay | Advantage |
|---|---:|---|---|
| PPO | 0.2 | No | GAE |
| DISC | 0.2 | No | GAE |
| Full DISC | 0.2 | Yes | GAE-V |

The original DISC paper uses $\epsilon=0.2$ for PPO and $\epsilon=0.4$ for DISC.  
This project keeps the clipping ratio constant to make the comparison more controlled.

---

## Environments

| Environment | Action Dimensions |
|---|---:|
| Hopper-v5 | 3 |
| Ant-v5 | 8 |
| Humanoid-v5 | 17 |

The increasing action dimensionality allows us to test whether the benefit of dimension-wise clipping becomes larger in more complex action spaces.

---

## Main Hyperparameters

| Parameter | Value |
|---|---:|
| Discount factor $\gamma$ | 0.99 |
| GAE parameter $\lambda$ | 0.95 |
| Rollout length | 2048 |
| Update epochs | 10 |
| Clip ratio $\epsilon$ | 0.2 |
| Learning rate | 0.0003 |
| Optimizer | Adam |
| Network | 2 × 64 |
| Activation | Tanh |

DISC-specific:

| Parameter | Value |
|---|---:|
| $J_{\mathrm{targ}}$ | 0.0001 |
| Initial $\alpha_{IS}$ | 1.0 |
| Replay length | 64 batches |
| Batch threshold $\epsilon_b$ | 0.1 |

---

## Metrics

The experiments will track:

- Episode return vs. environment steps
- Final evaluation return
- Clipping fraction
- Joint importance ratio statistics
- Per-dimension ratio statistics
- Gradient norms
- Performance variance across seeds
- $J_{IS}$ and $\alpha_{IS}$
- Number of accepted replay batches

---

## Hypothesis

As action dimensionality increases:

1. PPO's joint importance ratio should deviate further from 1.
2. PPO should clip a larger fraction of samples.
3. DISC should preserve more useful gradients.
4. Replay should further improve DISC's sample efficiency.

The largest difference is expected in **Humanoid**.

---

## Repository Structure

```text
dimension-wise-clipping-rl/
├── algorithms/
│   ├── ppo.py
│   ├── disc.py
│   └── disc_replay.py
├── core/
│   ├── networks.py
│   ├── buffer.py
│   ├── distributions.py
│   ├── gae.py
│   └── gae_v.py
├── configs/
│   └── base.yaml
├── train/
├── evaluation/
├── analysis/
├── experiments/
├── results/
├── requirements.txt
└── README.md
```

---

## Reference

Seungyul Han and Youngchul Sung.  
**Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning.**  
ICML 2019, Proceedings of Machine Learning Research, Vol. 97.

Paper: https://arxiv.org/abs/1905.02363

Original implementation: https://github.com/seungyulhan/disc