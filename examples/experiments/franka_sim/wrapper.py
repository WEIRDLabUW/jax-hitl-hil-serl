import copy
import time
from franka_env.utils.rotations import euler_2_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
from pynput import keyboard
import gymnasium as gym
import mujoco


class FrankaSimEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        m = env.model
        d = env.data
        self.viewer = mujoco.viewer.launch_passive(m, d)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)


    def step(self, action):
        step_start = time.time()
        vals = super().step(action)
        self.viewer.sync()
        time_until_next_step = self.env.control_dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        return vals