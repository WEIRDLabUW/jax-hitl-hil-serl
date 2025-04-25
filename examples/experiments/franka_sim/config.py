import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    SimSpacemouseIntervention,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from experiments.config import DefaultTrainingConfig
from experiments.franka_sim.wrapper import FrankaSimEnv
# from experiments.usb_pickup_insertion.wrapper import GripperPenaltyWrapper


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front"]
    classifier_keys = ["front"]
    proprio_keys = ['panda/tcp_pos', 'panda/tcp_vel', 'panda/gripper_pos']
    # buffer_period = 1000
    # checkpoint_period = 5000
    # steps_per_update = 50
    pretraining_steps = 0 # How many steps to pre-train the model for using RLPD on offline data only.
    reward_scale = 10 # How much to scale actual rewards (not RLIF penalties) for RLIF.
    rlif_minus_one = True
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    batch_size = 64
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, *args, **kwargs):
        from franka_sim.envs import PandaPickCubeGymEnv
        env = PandaPickCubeGymEnv(
            action_scale=(0.1, 1),
            render_mode="human",
            image_obs=True,
            control_dt=0.1,
            time_limit=100,
        )
        env = FrankaSimEnv(env, action_scale=[0.1, 0.1, 0.1])
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = SimSpacemouseIntervention(env)
        return env
