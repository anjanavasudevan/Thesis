"""
Testing LQR controller on the hover system
"""
import os
import sys

sys.path.insert(0, 'D:\IITH_AI_Docs\Thesis\Thesis-work\Thesis\Quadrotor_simulation')
from envs.hover_state_space import hover_state_space
from controllers.lqr import LQR
import numpy as np
import control
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Create gym environment
env = hover_state_space()
controller = LQR(env)

# Test for one episode
done = False
step = 0
state = env.reset()
states = []


for step in range(1000):
    action = controller.predict(state)
    next_state, reward, done = env.step(action)
    
    states.append(state)
    if done:
        print(f"State: {state}, Action: {action}, Reward: {reward}, Step: {step}")
        break
    step += 1
    state = next_state
    #print(f"Step: {step}, State: {state}, Action: {action}, Reward: {reward}")

states = np.array(states)
t = np.arange(states.shape[0])
plt.plot(t, states[:, 0])
plt.plot(t, states[:, 2])
plt.plot(t, states[:, 4])
plt.show()
