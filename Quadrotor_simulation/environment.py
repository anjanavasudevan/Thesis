"""
Gym environment for Quanser 3 DOF hover
Author: Anjana Vasudevan
"""
# Importing the dependencies
import gym
import numpy as np
from gym import spaces

# 1. Set values for non linear model
# a. Moment of inertia
I_xx = 0.055
I_yy = 0.055
I_zz = 0.11

# b. Force constants
K_f = 0.119
K_t = 0.0036

# 2. Linear model
A_linear = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], \
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
B_linear = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

print(A_linear.shape)

