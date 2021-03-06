"""
LQR controller for the hover environment
Author: Anjana Vasudevan
"""
import os
import sys
import control
import numpy as np

class LQR:
    """
    Create LQR controller for the hover. Replication of the LQR function in MATLAB provide in the setup_hover.m script
    """
    def __init__(self, env):
        self.env = env
        self.A = env.A
        self.B = env.B
        self.Q = np.array([[500, 0, 0, 0, 0, 0], [0, 350, 0, 0, 0, 0], [0, 0, 350, 0, 0, 0],\
                    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 20, 0], [0, 0, 0, 0, 0, 20]])
        self.R = np.eye(4)*0.01

        # Get the optimal feedback law
        self.K, self.S, _ = control.lqr(self.A, self.B, self.Q, self.R)

    def predict(self, obs):
        """
        Get action using the feedback law
        """
        # For sanity cheks

        assert self.K.shape[1] == obs.shape[0], "Please check the dimensions of the observation"

        action = -self.K@obs
        
        # Convert all the values to +ve to prevent errors
        action = np.absolute(action)
        # Clip the action
        action = np.clip(action, a_min=2, a_max=24)
        return action