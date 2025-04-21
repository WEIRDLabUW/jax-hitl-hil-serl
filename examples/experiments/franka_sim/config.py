import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    SimSpacemouseIntervention,
)

from experiments.config import DefaultTrainingConfig
from experiments.franka_sim.wrapper import FrankaSimEnv
from experiments.usb_pickup_insertion.wrapper import GripperPenaltyWrapper


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_1"]
    classifier_keys = ["side_1"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    # buffer_period = 1000
    # checkpoint_period = 5000
    # steps_per_update = 50
    pretraining_steps = 0 # How many steps to pre-train the model for using RLPD on offline data only.
    reward_scale = 1 # How much to scale actual rewards (not RLIF penalties) for RLIF.
    rlif_minus_one = False
    checkpoint_period = 4000
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
            render_mode="rgb_array",
            image_obs=True
        )
        env = FrankaSimEnv(env)
        env = SimSpacemouseIntervention(env)
        return env
