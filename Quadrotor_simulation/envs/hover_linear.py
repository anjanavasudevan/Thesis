"""
Gym environment for Quanser 3 DOF hover linear model
Author: Anjana Vasudevan
"""
# Importing the dependencies
import os
import sys
import gym
import numpy as np
from gym.spaces import Box
from scipy.integrate import solve_ivp

# Create gym class
class hover_linear(gym.Env):
    """
    Modelling the hover using the state space linear model
    """
    def __init__(self):
        super(hover_linear, self).__init__()
        # Hover body details
        self.m = 2.85
        self.m_prop = self.m / 4

        # 
        self.g = 9.81
        self.L = 7.75*0.0254 # Distance between propeller and motor

        # Toruque related constants for the model
        self.Kf = 0.119
        self.Kt = 0.0036

        # Motor torque related
        self.Jm = 1.91e-6
        self.Jeq_prop = self.Jm + self.m_prop*(self.L**2)

        # ii. Equivalent Moment of Inertia about each Axis (kg.m^2)
        self.Jp = 2*self.Jeq_prop
        self.Jy = 4*self.Jeq_prop
        self.Jr = 2*self.Jeq_prop
        self.L = 7.75*0.0254

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
        self.x_max = np.array([self.y_max, self.p_max, self.r_max, \
                        self.ang_rate_max, self.ang_rate_max, self.ang_rate_max])
        self.x_min = np.array([-self.y_max, -self.p_max, -self.r_max, 0, 0, 0])
        # self.rest = np.zeros_like(self.x_min)

        self.observation_space = Box(low=self.x_min, high=self.x_max)

        # Set the time for simulation
        self.tstart = 0
        self.tmax = 25
        self.current_time = 0
        self.current_timestep = 0
        self.step_interval = 0.02

        # Initial State-action
        self.state = None
        self.action = np.array([2, 2, 2, 2])

        # Reward function paramters
        self.Q = np.array([[500, 0, 0, 0, 0, 0], [0, 350, 0, 0, 0, 0], [0, 0, 350, 0, 0, 0],\
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 20, 0], [0, 0, 0, 0, 0, 20]])
        self.R = np.eye(4)*0.01

        # Initialize seed
        self.seed = 0
        self.random_state = np.random.default_rng()

    def step(self, action):
        """
        Apply control to the motor
        """
        # Make sure the action is size 4 vector
        assert action.shape == self.action_space.shape

        # Determine the current time
        self.current_time = self.current_timestep*self.step_interval
        time_range = (self.current_time, self.current_time+self.step_interval)

        # Split the action vector - we ensure that episode does not terminate at the first step:
        u1, u2, u3, u4 = action

        # Convert the angles in [-pi, pi] range
        for i, angle in enumerate(self.state[:3]):
            if angle < -np.pi:
                self.state[i] += 2*np.pi
            if angle > np.pi:
                self.state[i] -= 2*np.pi
        
        # Calculate the next step
        solution = solve_ivp(self.linear_model, time_range, self.state, args=(u1, u2, u3, u4))

        # Get the next state
        next_state = solution.y[:, -1]

        # Calculate the rewards - Note: resulting reward is numpy.float64 datatype
        rewards = -(self.state.T@self.Q@self.state + self.action.T@self.R@action)
        

        # Check for doneness
        done = np.any(self.state[3:] > self.x_max[3:]) or np.any(self.state[:2] > self.x_max[:2])\
                or np.any(self.state[:2] < self.x_min[:2]) or self.current_time >= self.tmax
        
        self.current_timestep += 1
        self.state = next_state

        # Return all the observations
        return next_state, action, rewards, done

    def reset(self):
        """
        Reset to default parameters
        """
        self.state = self.random_state.uniform(low=self.x_min, high=self.x_max)
        self.current_time = 0
        self.current_timestep = 0

        return self.state

    def linear_model(self, t, x, u1=2, u2=2, u3=2, u4=2):
        """
        Evaluate using state space representation
        """
        u = np.array([u1, u2, u3, u4])
        dxdt = self.A@x + self.B@u
        return dxdt





