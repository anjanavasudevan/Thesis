"""
Gym environment for Quanser 3 DOF hover
Author: Anjana Vasudevan
"""
# Importing the dependencies
import gym
import numpy as np
from gym.spaces import Box


def linear_model(x, A, B, u):
    """
    Evaluate linear model using the state space form
    ẋ = Ax + Bu
    """
    dxdt = A@x + B@u
    return dxdt

# Create gym class
class hover(gym.Env):
    """
    Modelling the hover
    """
    def __init__(self):
        super(hover, self).__init__()
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
        self.Jeq_prop = self.Jm + self.m_prop*(self.L**2)
        self.Jm = 1.91e-6

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
        self.u_min = np.float(np.array[2, 2, 2, 2])
        self.u_max = np.float(np.array(24, 24, 24, 24))

        self.action_space = Box(low=self.u_min, high=self.u_max)

        # State bounds MAKE SURE ALL ARE IN RADIANS
        # Pitch and roll are 
        self.p_max = 37.5*np.pi/180
        self.r_max = 37.5*np.pi/180
        self.y_min = 0
        self.y_max = 2*np.pi

        # Maximum angular rate for all axes
        self.ang_rate_max = 60*np.pi/180

        # Specifying the observation space
        self.x_max = np.float(np.array[self.p_max, self.r_max, self.y_max, \
                        self.ang_rate_max, self.ang_rate_max, self.ang_rate_max])
        self.x_min = np.float(np.array([-self.p_max, -self.r_max, self.y_min, 0, 0, 0]))

        self.observation_space = Box(low=self.x_min, high=self.x_max)

    def step(self, action):
        """
        Apply control to the motor
        """
        # Make sure the action is 6*6 vector



