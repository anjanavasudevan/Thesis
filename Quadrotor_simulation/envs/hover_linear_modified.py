"""
Gym Environment for Quanser 3DOF Hover linear model with modifications in reward functions
Author: Anjana Vasudevan
"""
import os
import sys
import gym
import numpy as np
from gym.spaces import Box

class hover_state_space(gym.Env):
    """
    Hover model using the state space equations
    """
    def __init__(self):
        """
        Initialize hover environment
        """
        # Physical parameters of the hover
        self.m = 2.85
        self.m_prop = self.m / 4
        self.L = 7.75*0.0254

        # Model dynamic constants
        self.Kf = 0.119
        self.Kt = 0.0036
        self.Jm = 1.91e-6
        self.Jeq_prop = self.Jm + self.m_prop*(self.L**2)
        self.Jp = 2*self.Jeq_prop
        self.Jy = 4*self.Jeq_prop
        self.Jr = 2*self.Jeq_prop

        # State space constants
        self.A = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], \
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        self.B = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], \
                    [-self.Kt/self.Jy, -self.Kt/self.Jy, self.Kt/self.Jy, self.Kt/self.Jy], \
                        [self.L*self.Kf/self.Jp, -self.L*self.Kf/self.Jp, 0, 0], \
                        [0, 0, self.L*self.Kf/self.Jr, -self.L*self.Kf/self.Jr]])
        self.C = np.eye(3)
        self.C = np.hstack((self.C, np.zeros_like(self.C)))
        self.D = np.zeros((3, 4))

        # Define action bounds (includes bias voltage to prevent motor burn)
        self.u_min = np.array([2, 2, 2, 2])
        self.u_max = np.array([24, 24, 24, 24])

        self.action_space = Box(low=self.u_min, high=self.u_max)

        # State bounds MAKE SURE ALL ARE IN RADIANS
        self.p_max = 37.5*np.pi/180
        self.r_max = 37.5*np.pi/180
        self.y_max = np.pi

        # Maximum angular rate for all axes
        self.ang_rate_max = 60*np.pi/180

        # Specifying the observation space (format - [yaw, pitch, roll])
        self.x_max = np.array([self.y_max, self.ang_rate_max, self.p_max, \
                        self.ang_rate_max, self.r_max, self.ang_rate_max])
        self.x_min = np.array([-self.y_max, 0, -self.p_max, 0, -self.r_max])
        # self.rest = np.zeros_like(self.x_min)

        self.observation_space = Box(low=self.x_min, high=self.x_max)

        # Initial State-action
        self.state = None

        # Initialize seed
        self.seed = 0
        self.random_state = np.random.default_rng()

        # Cost function matrices
        self.Q = np.eye(6)
        costs = np.square(self.x_max)
        self.Q[np.diag_indices_from(self.Q)] = costs

        # Costs
        self.R = np.eye(4)*0.01

        # Time step
        self.dt = 0.02

    def step(self, action):
        """
        Apply control to the motor
        """
        assert action.shape == self.action_space.shape, "Check the dimensions of input action"

        # Convert the angles in [-pi, pi] range
        for idx in (0, 2, 4):
            if self.state[idx] < -np.pi:
                self.state[idx] += 2*np.pi
            if self.state[idx] > np.pi:
                self.state[idx] -= 2*np.pi

        # Get next state using Euler's method
        dxdt = self.A@self.state + self.B@action
        new_state = self.state + dxdt*self.dt

        # Calculate the rewards
        reward = -(self.state.T@Q@self.state + action.T@R@action)
        self.state = new_state

        return new_state, reward, False

    def reset(self):
        """
        Reset the environment defaults
        """
        self.state = self.random_state.uniform(low=self.x_min, high=self.x_max)
        return self.state

    
