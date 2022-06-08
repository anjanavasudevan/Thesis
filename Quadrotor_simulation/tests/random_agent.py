# Using a random agent for checks
import os
import sys

sys.path.insert(0, 'D:\IITH_AI_Docs\Thesis\Thesis-work\Thesis\Quadrotor_simulation')
from envs.hover_linear import hover_linear
from controllers.lqr import LQR
import numpy as np
import matplotlib.pyplot as plt

env = hover_linear()
state = env.reset()
done = False

states = []
rewards = []
step = 0

while not done:
    action = env.action_space.sample()
    next_state, action, reward, done = env.step(action)
    
    states.append(state)
    if done:
        print(f"State: {state}, Action: {action}, Reward: {reward}, Step: {step}")
        break
    step += 1
    state = next_state
    #print(f"Step: {step}, State: {state}, Action: {action}, Reward: {reward}")

states = np.array(states)
t = np.arange(states.shape[0])
plt.plot(t, states[:, :3])
plt.show()