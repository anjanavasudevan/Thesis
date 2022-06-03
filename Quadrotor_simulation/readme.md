---
Title: RL for 3DOF Quadrotor
---

## Naming conventions

1. p - Pitch angles
2. y - Yaw angles
3. r - Roll angles

## Bounds for the environment

The following observations are taken into consideration:

* The hover can move freely only about the yaw axes due to a slip ring. Hence range of yaw angles is assumed to be 360 degrees.
* The pitch and roll axes have restricted movemement, hence the range for both these angles are set at $\pm 37.5\degree$