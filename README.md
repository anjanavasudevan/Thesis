# Readme

## Objective
Non Minimum phase systems are an important class of systems in controller design due to their unique behaviour:
1. These systems are causal - output of the system depends only on the current input (not on previous or future inputs)
2. The inverse of these systems are causal yet **unstable**.

The inverse response of the system helps in controller design, and due to the unstable nature of these systems, it is difficult to design controllers for these class of systems using classical control theory methods.

Reinforcement learning, over the years has acheived considerable success in game development and robotics. The controllers for the robots and the system dynamics often tend to be of non-minimum phase, and yet the robots are highly stable. Taking cue from this approach, the thesis aims at developing controllers for all generic kinds of NMPs using RL.

## What's inside
2 environments were considered as examples of non minium phase systems:

1. `Cartpole-v0`
2. `Pendulum-v0`

The following algorithms were tested on `Cartpole-v0` environment (as the action space is discrete):

1. Deep Q learning
2. Double Deep Q learning
3. A2C (Advantage actor critic) with 32, 64 and 128 node hidden layer configuration

Out of all the 3, the A2C network performed the best of all.

The following algorithms were tested on `Pendulum-v0` environment (as the action space is continuous):

1. DDPG
2. TD3 - Deterministic and Stochastic
3. PPO

TD3 and DDPG algorithms were noisier and saturated at the score range 0 to -500. PPO, despite showing improvement, showed much noisy saturation towards the end of the time.

Hence SAC is clearly the better algorithm on the environment

## Future work
Future work will cover guaranteed stability of these systems while using a RL based controller. It is a work in progress

## Note:
The repository will be updated to ensure all the references are covered. For now the working examples are compiled.