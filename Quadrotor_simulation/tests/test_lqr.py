"""
Testing LQR controller on the hover system
"""
import os
import sys

sys.path.insert(0, 'D:\IITH_AI_Docs\Thesis\Thesis-work\Thesis\Quadrotor_simulation')
from envs.hover_linear import hover_linear
from controllers.lqr import LQR
import numpy as np
import control
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Create gym environment
env = hover_linear()
controller = LQR(env)

# Test for one episode
done = False
step = 0
state = env.reset()
states = []


while not done:
    action = controller.predict(state)
    next_state, action, reward, done = env.step(action)
    
    states.append(state)
    if done:
        break
    step += 1
    print(f"Step: {step}, State: {state}")



