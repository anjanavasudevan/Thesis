"""
Testing LQR controller on the hover system
"""
import os
import sys

sys.path.insert(0, 'D:\IITH_AI_Docs\Thesis\Thesis-work\Thesis\Quadrotor_simulation')
from envs.hover_linear import hover_linear
from controller.lqr import LQR
import numpy as np
import control
from scipy.integrate import solve_ivp

# Create gym environment
env = hover_linear()
controller = LQR(env)

