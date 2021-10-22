# Readme

## Objective
Non Minimum phase systems are an important class of systems in controller design due to their unique behaviour:
1. These systems are causal - output of the system depends only on the current input (not on previous or future inputs)
2. The inverse of these systems are causal yet **unstable**.

The inverse response of the system helps in controller design, and due to the unstable nature of these systems, it is difficult to design controllers for these class of systems using classical control theory methods.

Reinforcement learning, over the years has acheived considerable success in game development and robotics. The controllers for the robots and the system dynamics often tend to be of non-minimum phase, and yet the robots are highly stable. Taking cue from this approach, the thesis aims at developing controllers for all generic kinds of NMPs using RL.

## What's inside
At the moment, attempts are made to develop control over a simple NMP - the cartpole.

The environment comes from the `OpenAIGym` module.

Algorithms used to control the system:
1. Q-Learning
