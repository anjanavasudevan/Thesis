"""
@author: Anjana Vasudevan
Defining a custom model environment for SISO transfer function:
-2 s + 0.5
-------------
s^2 + 5 s + 1
The above function is a non minimum phase system  with zero at 0.25
"""
# Simple NMP transfer function

# Import dependencies
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class SISO_nmp(Env):
    """
    Class environment for custom NMP.
    State space representation:
    ẋ = Ax + Bu
    y = Cx + D
    """

    def __init__(self):
        """
        Initialize the system parameters
        """
        # State space parameters:
        self.A = np.array([[-5, -1], [1, 0]])
        self.B = np.array([1, 0])
        self.C = np.array([-2, 0.25])
        self.D = np.array([0])

        # State matrix:
        self.x0 = 0
        self.x1 = 1

        # Time steps:
        self.step = 0.05

        # No. of time steps
        self.time = 25
        self.nsteps = self.time / self.step
        self.t = np.arange(0, self.time, self.step)

        # State matrix
        self.x = np.zeros([self.nsteps, 2])

        # State space (x[0] = 1 and x[1] = 1):
        self.state = np.array([1, 1])

        # Action space (the input - u)
        self.action_space = Box(low=np.float32(np.array([-5])),
                                high=np.float32(np.array([5])))

        # Observation space (the output y)
        self.observation_space = Box(low=np.float32(np.array([-5])),
                                     high=np.float(np.array([5])))

        # Set point
        self.setpoint = 0

        # Output matrix:
        self.output = []

    def process(x, t, A, B, u):
        """
        State space equation for solving.
        """
        dxdt = A@x + B*u
        return dxdt

    def
