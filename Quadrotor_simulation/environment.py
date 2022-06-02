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

# c. 3DOF body specifications
m_hover = 2.85
m_prop = m_hover / 4
g = 9.81
L = 7.75*0.0254 # Distance between pivot and motor

# d. Torque related constants for linear model
Kt_m = 0.0182 # Motor torque constant
Jm = 1.91e-6 # Motor moment of inertia
# i. Equivalent Moment of Inertia of each Propeller Section (kg.m^2)
Jeq_prop = Jm + m_prop*(L**2)
# ii. Equivalent Moment of Inertia about each Axis (kg.m^2)
Jp = 2*Jeq_prop
Jy = 4*Jeq_prop
Jr = 2*Jeq_prop

# 2. Linear model
A_linear = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], \
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
B_linear = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], \
                    [-K_t/Jy, -K_t/Jy, K_t/Jy, K_t/Jy], [L*K_f/Jp, -L*K_f/Jp, 0, 0], [0, 0, L*K_f/Jr, -L*K_f/Jr]])
C_linear = np.eye(3)
C_linear = np.hstack((C_linear, np.zeros_like(C_linear)))
D_linear = np.zeros((3, 4))

# Sanity check
print(f"Dimensions of A: {A_linear.shape}\nDimensions of B: {B_linear.shape}\nDimension of C: {C_linear.shape}\nDimensions of D: {D_linear.shape}")




