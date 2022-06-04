---
Title: RL for 3DOF Quadrotor
---

## Naming conventions

1. p - Pitch angles
2. y - Yaw angles
3. r - Roll angles

## Bounds for the environment

The following observations are taken into consideration:

* The hover can move freely only about the yaw axes due to a slip ring. Hence range of yaw angles is assumed to be $360 \degree$.
* The pitch and roll axes have restricted movemement, hence the range for both these angles are set at $\pm 37.5\degree$

## Calculating the rewards

The rewards are calculated using LQR control policy. We minimize the rewards as:
$$
\[ \int_{0}^{\infty} x^TQx + u^TRu \,dt \]
$$
RL environments by default maximise rewards, hence we calculate the negative reward.

## Evaluating the environment

The episode is terminated if one of the following conditions occur:

1. The pitch and roll angles go beyond range.
2. The motor voltages fall below specified voltage levels
3. The timestep for simulation runs out
