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
        self.x = np.array([0, 1])
        self.step_size = 0.05
        self.time = 25
        self.n = int(self.time / self.step_size)

        # Counter to count the no. of steps
        self.tstep = 0

        # Track no. of steps for undershoot (Cap at 20 steps)
        self.track = 0

        # Action space (either negative input, positive input or zero input)
        self.action_space = Box(low=np.float32(np.array([-1])),
                                high=np.float32(np.array([1])))

        # State (the current value of u signal)
        self.state = np.array([random.uniform(-5, 5)])
        self.cpos = None

        # Observation space (the output y)
        self.observation_space = Box(low=np.float32(np.array([-10])),
                                     high=np.float(np.array([10])))

        # Set point
        self.setpoint = 0

        # Output matrix:
        self.output = [0]

        # Rewards:
        self.reward = None

    def process(x, t, A, B, u):
        """
        State space equation for solving: ẋ = Ax + Bu
        """
        dxdt = A@x + B*u
        return dxdt

    def get_val(u, x, C, D):
        """
        Calculate the output: y = Cx + D
        """
        y = C@x.T + D*u
        return y

    def step(self, action):
        """
        Predict the output.
        Rewards:
        For all steps (including termination): +1
        For undershoot: -5
        For opposite response: -5
        """
        # Assert the action is valid:
        error_msg = f"{action} is not a valid action."
        assert self.action_space.contains(action), error_msg

        # If action valid, get our control signal
        # Generate -ve or +ve input
        self.cpos = action*self.state

        # Get the state and output
        x_next = odeint(self.process, self.x, self.tstep*self.step_size,
                        args=(self.A, self.B, self.cpos, ))
        y = self.get_val(self.cpos, self.x, self.C, self.D)
        self.output.append(y)

        # Model the problem:
        if(self.output[self.tstep] < self.setpoint):
            # Output goes below zero
            done = True
            reward = -5
        elif(self.output[self.tstep] < self.output[self.tstep - 1] and self.track == 30):
            # Opposite response
            done = True
            reward = -5
        elif(self.tstep == self.n):
            done = True
            reward = 1
        else:
            reward = 1 + np.abs(self.output[self.tstep] - x_next[:, 1])

        return reward, done, self.output, x_next

    def render(self):
        """
        Render the environment
        """
        pass

    def reset(self):
        """
        Reset the environment back to defaults
        """
        # Reset the state
        self.state = np.array([random.uniform(-5, 5)])
        self.x = np.array([0, 1])

        self.tstep = 0
        self.track = 0

        self.output = [0]

        return self
