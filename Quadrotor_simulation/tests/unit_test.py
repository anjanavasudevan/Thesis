"""
Unit test for the modified hover
"""

import os
import sys

sys.path.insert(0, 'D:\IITH_AI_Docs\Thesis\Thesis-work\Thesis\Quadrotor_simulation')
from envs.hover_state_space import hover_state_space
import numpy as np
import control
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

env = hover_state_space()
state = env.reset()
done = False

states = []
rewards = []
step = 0

for step in range(1000):
    action = np.array([5, 5, 5, 5])
    next_state, reward, done = env.step(action)
    
    states.append(state)
    if done:
        print(f"State: {state}, Action: {action}, Reward: {reward}, Step: {step}")
        break
    state = next_state
    #print(f"Step: {step}, State: {state}, Action: {action}, Reward: {reward}")

states = np.array(states)
t = np.arange(states.shape[0])
plt.plot(t, states[:, 0])
plt.plot(t, states[:, 2])
plt.plot(t, states[:, 4])
plt.show()