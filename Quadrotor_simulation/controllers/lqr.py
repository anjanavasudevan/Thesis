"""
LQR controller for the hover environment
Author: Anjana Vasudevan
"""

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
        self.Q = env.Q
        self.R = env.R

        # Get the optimal feedback law
        self.K, self.S, _ = control.lqr(self.A, self.B, self.Q, self.R)

    def predict(self, obs):
        """
        Get action using the feedback law
        """
        # For sanity cheks

        assert self.K.shape[1] == obs.shape[0], "Please check the dimensions of the observation"

        action = self.K@obs
        return action