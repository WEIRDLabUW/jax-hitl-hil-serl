# %%
from absl import app, flags
import time
import numpy as np
import os
import pickle
import imageio
import cv2
import queue
from pynput import keyboard
import threading
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars
import math
from collections import defaultdict
import pandas as pd

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)

os.environ["PYTHONPATH"] = "/Users/pranavnt/jax-hitl-hil-serl"

# %%
# Updated checkpoint path to the specified directory
checkpoint_path = "/Users/pranavnt/jax-hitl-hil-serl/examples/experiments/franka_sim/sriyash_runs/hil-bc-coeff0.1-decay"

from experiments.config import DefaultTrainingConfig
class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front", "wrist"]
    classifier_keys = ["front", "wrist"]
    proprio_keys = ['panda/tcp_pos', 'panda/tcp_vel', 'panda/gripper_pos']
    # buffer_period = 1000
    # checkpoint_period = 5000
    # steps_per_update = 50
    pretraining_steps = 0 # How many steps to pre-train the model for using RLPD on offline data only.
    reward_scale = 1 # How much to scale actual rewards (not RLIF penalties) for RLIF.
    rlif_minus_one = False
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    batch_size = 64
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

exp_name = "franka_sim"
config = TrainConfig()

# %%
intervene_steps = 0  # Default number of steps between pre and post intervention states
constraint_eps = 0.1  # Default constraint epsilon

# obs_key_shapes = [('front', (1, 128, 128, 3)), ('state', (1, 7)), ('wrist', (1, 128, 128, 3))]
obs_sample = {
    'front': np.zeros((1, 128, 128, 3), dtype=np.uint8),
    'state': np.zeros((1, 7), dtype=np.float32),
    'wrist': np.zeros((1, 128, 128, 3), dtype=np.uint8),
}
action_sample = np.zeros(7, dtype=np.float32)

# Create CL config
cl_config = {
    "enabled": False,
    "constraint_coeff": 1.0,
    "constraint_eps": constraint_eps,
    "reward_coeff": 1.0,
    "enable_margin_constraint": False,
    "enable_action_constraint": False
}

agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
    seed=0,
    sample_obs=obs_sample,
    sample_action=action_sample,
    image_keys=config.image_keys,
    encoder_type=config.encoder_type,
    discount=config.discount,
    enable_cl=False,
    cl_config=cl_config,
)

# %%
ckpt_dirs = [d for d in os.listdir(checkpoint_path) if d.startswith('checkpoint_')]
if ckpt_dirs:
    latest_checkpoint = sorted(ckpt_dirs, key=lambda x: int(x.split('_')[1]))[-1]
    checkpoint_step = latest_checkpoint.split('_')[1]
    print(f"Using checkpoint: {latest_checkpoint}")

    ckpt = checkpoints.restore_checkpoint(
        os.path.join(checkpoint_path, latest_checkpoint),
        agent.state
    )
    agent = agent.replace(state=ckpt)
else:
    print("No checkpoints found. Using random initialization.")
# %%
replay_buffer_base_path = "/Users/pranavnt/jax-hitl-hil-serl/examples/experiments/franka_sim/sriyash_runs/hil-bc-coeff0.1-decay/buffer/transitions"
replay_buffer_paths = [f"{replay_buffer_base_path}_{i}.pkl" for i in range(1000, 15000, 1000)]

replay_buffer = []

for replay_buffer_path in replay_buffer_paths:
    if not os.path.exists(replay_buffer_path):
        print(f"Replay buffer path {replay_buffer_path} does not exist.")
        continue

    with open(replay_buffer_path, 'rb') as f:
        replay_buffer_part = pickle.load(f)
        replay_buffer.extend(replay_buffer_part)

print(f"Loaded {len(replay_buffer)} transitions from replay buffers")

# %%
# Update the path to the interventions directory
preference_buffer_base_path = f"{checkpoint_path}/interventions/transitions"
preference_buffer_paths = [f"{preference_buffer_base_path}_{i}.pkl" for i in range(1000, 15000, 1000)]

preference_buffer = []

for preference_buffer_path in preference_buffer_paths:
    if not os.path.exists(preference_buffer_path):
        print(f"Preference buffer path {preference_buffer_path} does not exist.")
        continue

    # Load the preference buffer
    with open(preference_buffer_path, 'rb') as f:
        preference_buffer_part = pickle.load(f)
        preference_buffer.extend(preference_buffer_part)

print(f"Loaded {len(preference_buffer)} transitions from preference buffers")

# %%
rng = jax.random.PRNGKey(0)

def get_action(obs, rng):
    rng, key = jax.random.split(rng)
    actions = agent.sample_actions(
        observations=jax.device_put(obs),
        argmax=True,
        seed=key
    )
    return actions, rng

def prepare_obses(obses):
    return {
        'front': np.array([obs['front'] for obs in obses]),
        'state': np.array([obs['state'] for obs in obses]),
        'wrist': np.array([obs['wrist'] for obs in obses]),
    }

# %%
pre_intervention_obs = [p['observations'][0] for p in preference_buffer]
intervene_actions = [p['actions'][0] for p in preference_buffer]
policy_actions = [p['policy_actions'][0] for p in preference_buffer]
post_intervention_obs = [p['observations'][-1] for p in preference_buffer]

pre_intervention_obs = {
    'front': np.array([obs['front'] for obs in pre_intervention_obs]),
    'state': np.array([obs['state'] for obs in pre_intervention_obs]),
    'wrist': np.array([obs['wrist'] for obs in pre_intervention_obs]),
}
pre_intervention_expert_action, rng = get_action(pre_intervention_obs, rng)

post_intervention_obs = {
    'front': np.array([obs['front'] for obs in post_intervention_obs]),
    'state': np.array([obs['state'] for obs in post_intervention_obs]),
    'wrist': np.array([obs['wrist'] for obs in post_intervention_obs]),
}
post_intervention_expert_action, rng = get_action(post_intervention_obs, rng)

policy_actions = np.array(policy_actions)
intervene_actions = np.array(intervene_actions)

# %%
def expert_critic(obs, action, rng):
    key, rng = jax.random.split(rng)
    q = agent.forward_critic(obs, action[:, :6], key)
    return q, rng

q_pre_expert, rng = expert_critic(pre_intervention_obs, pre_intervention_expert_action[:, :6], rng)
q_post_expert, rng = expert_critic(post_intervention_obs, post_intervention_expert_action[:, :6], rng)
q_pre_policy, rng = expert_critic(pre_intervention_obs, policy_actions[:, :6], rng)
q_pre_intervene, rng = expert_critic(pre_intervention_obs, intervene_actions[:, :6], rng)

constraint1_acc = ((q_pre_expert - q_post_expert) < 0).mean()
qvalue_based_learning_intervene = ((q_pre_policy - q_pre_intervene) < 0).mean()
qvalue_based_learning_expert = ((q_pre_policy / q_pre_expert) < 0.98).mean()
constraint2_acc = ((q_pre_intervene - q_post_expert) < 0).mean()

print(f"Constraint 1 accuracy: {constraint1_acc}")
print(f"Q-value based learning intervene accuracy: {qvalue_based_learning_intervene}")
print(f"Q-value based when2intervene accuracy (pre-intervention states): {qvalue_based_learning_expert}")
print(f"Constraint 2 accuracy: {constraint2_acc}")

# %%
# Find pre-intervention states (states right before an intervention)
pre_intervention_states = []
non_intervention_states = []

# Identify pre-intervention states (the state right before an intervention starts)
for i in range(len(replay_buffer)-1):
    cur_transition = replay_buffer[i]
    next_transition = replay_buffer[i+1]

    # Check if the current state is not an intervention but the next one is
    cur_has_intervention = 'info' in cur_transition and isinstance(cur_transition['info'], dict) and 'intervene_action' in cur_transition['info']
    next_has_intervention = 'info' in next_transition and isinstance(next_transition['info'], dict) and 'intervene_action' in next_transition['info']

    if not cur_has_intervention and next_has_intervention:
        # This is a pre-intervention state
        pre_intervention_states.append(cur_transition['observations'])
    elif not cur_has_intervention:
        # This is a regular non-intervention state
        non_intervention_states.append(cur_transition['observations'])

print(f"Found {len(pre_intervention_states)} pre-intervention states and {len(non_intervention_states)} regular non-intervention states")

# Prepare observations for evaluation
if len(pre_intervention_states) > 0 and len(non_intervention_states) > 0:
    # Sample a balanced subset if needed
    max_samples = min(len(pre_intervention_states), 1000)
    pre_intervention_sample_indices = np.random.choice(len(pre_intervention_states), min(max_samples, len(pre_intervention_states)), replace=False)
    non_intervention_sample_indices = np.random.choice(len(non_intervention_states), min(max_samples, len(non_intervention_states)), replace=False)

    pre_intervention_states_sample = [pre_intervention_states[i] for i in pre_intervention_sample_indices]
    non_intervention_states_sample = [non_intervention_states[i] for i in non_intervention_sample_indices]

    # Prepare observations
    pre_intervention_obs = prepare_obses(pre_intervention_states_sample)
    non_intervention_obs = prepare_obses(non_intervention_states_sample)

    # Get expert actions
    pre_intervention_expert_action, rng = get_action(pre_intervention_obs, rng)
    non_intervention_expert_action, rng = get_action(non_intervention_obs, rng)

    # Get policy actions (separate from expert actions)
    # Use a separate random key to ensure different actions
    rng, policy_key1 = jax.random.split(rng)
    pre_intervention_policy_action = agent.sample_actions(
        observations=jax.device_put(pre_intervention_obs),
        argmax=False,  # Sample from policy distribution instead of taking argmax
        seed=policy_key1
    )

    rng, policy_key2 = jax.random.split(rng)
    non_intervention_policy_action = agent.sample_actions(
        observations=jax.device_put(non_intervention_obs),
        argmax=False,  # Sample from policy distribution
        seed=policy_key2
    )

    # Convert to numpy arrays
    pre_intervention_policy_action = np.asarray(jax.device_get(pre_intervention_policy_action))
    non_intervention_policy_action = np.asarray(jax.device_get(non_intervention_policy_action))

    # Get Q-values for evaluation
    q_pre_intervention_expert, rng = expert_critic(pre_intervention_obs, pre_intervention_expert_action[:, :6], rng)
    q_pre_intervention_policy, rng = expert_critic(pre_intervention_obs, pre_intervention_policy_action[:, :6], rng)
    q_non_intervention_expert, rng = expert_critic(non_intervention_obs, non_intervention_expert_action[:, :6], rng)
    q_non_intervention_policy, rng = expert_critic(non_intervention_obs, non_intervention_policy_action[:, :6], rng)

    # Debug: Print Q-value statistics
    print(f"\nQ-Value Statistics:")
    print(f"Pre-Intervention Expert Q-values: min={q_pre_intervention_expert.min()}, max={q_pre_intervention_expert.max()}, mean={q_pre_intervention_expert.mean()}")
    print(f"Pre-Intervention Policy Q-values: min={q_pre_intervention_policy.min()}, max={q_pre_intervention_policy.max()}, mean={q_pre_intervention_policy.mean()}")
    print(f"Non-Intervention Expert Q-values: min={q_non_intervention_expert.min()}, max={q_non_intervention_expert.max()}, mean={q_non_intervention_expert.mean()}")
    print(f"Non-Intervention Policy Q-values: min={q_non_intervention_policy.min()}, max={q_non_intervention_policy.max()}, mean={q_non_intervention_policy.mean()}")

    # Calculate ratios (avoid division by zero)
    def safe_ratio(a, b):
        mask = (np.abs(b) > 1e-8)
        ratio = np.zeros_like(a)
        ratio[mask] = a[mask] / b[mask]
        return ratio

    pre_intervention_ratios = safe_ratio(q_pre_intervention_policy, q_pre_intervention_expert)
    non_intervention_ratios = safe_ratio(q_non_intervention_policy, q_non_intervention_expert)

    # Calculate average ratios
    avg_pre_intervention_ratio = pre_intervention_ratios.mean()
    avg_non_intervention_ratio = non_intervention_ratios.mean()

    print(f"\nRatio Analysis (Q(s, policy action) / Q(s, expert action)):")
    print(f"Average ratio at pre-intervention states: {avg_pre_intervention_ratio:.4f}")
    print(f"Average ratio at non-intervention states: {avg_non_intervention_ratio:.4f}")

    # Calculate when2intervene decision using the 0.98 threshold
    when2intervene_pre = (pre_intervention_ratios < 0.98).mean()
    when2intervene_non = (non_intervention_ratios < 0.98).mean()

    print(f"\nWhen2Intervene Method (ratio < 0.98):")
    print(f"Percentage recommending intervention at pre-intervention states: {when2intervene_pre:.4f}")
    print(f"Percentage recommending intervention at non-intervention states: {when2intervene_non:.4f}")

    # After calculating ratios and averages, add histogram visualization
    import matplotlib.pyplot as plt

    # Ensure ratios are flattened to 1D arrays
    pre_intervention_ratios_flat = pre_intervention_ratios.flatten()
    non_intervention_ratios_flat = non_intervention_ratios.flatten()

    # Calculate stats with flattened arrays
    avg_pre_intervention_ratio = pre_intervention_ratios_flat.mean()
    avg_non_intervention_ratio = non_intervention_ratios_flat.mean()
    when2intervene_pre = (pre_intervention_ratios_flat < 0.98).mean()
    when2intervene_non = (non_intervention_ratios_flat < 0.98).mean()

    # Clip extreme values for better visualization (focused on the range of interest)
    pre_intervention_ratios_clip = np.clip(pre_intervention_ratios_flat, 0.95, 1.05)
    non_intervention_ratios_clip = np.clip(non_intervention_ratios_flat, 0.95, 1.05)

    # Set up the figure and axes
    plt.figure(figsize=(12, 6))

    # Create histograms with clipped x-axis range
    bins = np.linspace(0.95, 1.05, 40)  # Focused bins between 0.95 and 1.05

    plt.hist(pre_intervention_ratios_clip, bins=bins, alpha=0.7,
             label=f'Pre-intervention states (mean={avg_pre_intervention_ratio:.3f})')
    plt.hist(non_intervention_ratios_clip, bins=bins, alpha=0.7,
             label=f'Non-intervention states (mean={avg_non_intervention_ratio:.3f})')

    # Add vertical line at 0.98 threshold
    plt.axvline(x=0.98, color='r', linestyle='--', label='When2Intervene threshold (0.98)')

    # Add labels and title
    plt.xlabel('Q-value Ratio (Q(s, policy action) / Q(s, expert action))')
    plt.ylabel('Frequency')
    plt.title('Distribution of Q-value Ratios (Clipped to 0.95-1.05 Range)')
    plt.legend()
    plt.grid(alpha=0.3)

    # Set x-axis limits explicitly
    plt.xlim(0.95, 1.05)

    # Add text annotation showing percentages below threshold
    plt.annotate(f"Pre-intervention states below threshold: {when2intervene_pre:.1%}",
                xy=(0.98, plt.gca().get_ylim()[1]*0.95),
                xytext=(0.99, plt.gca().get_ylim()[1]*0.95),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    plt.annotate(f"Non-intervention states below threshold: {when2intervene_non:.1%}",
                xy=(0.98, plt.gca().get_ylim()[1]*0.85),
                xytext=(0.99, plt.gca().get_ylim()[1]*0.85),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    # Show plot
    plt.tight_layout()
    plt.show()

    # Optional: Create a second figure with separate subplots for clearer comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pre-intervention ratios histogram
    ax1.hist(pre_intervention_ratios_clip, bins=bins, alpha=0.7)
    ax1.axvline(x=0.98, color='r', linestyle='--', label='Threshold (0.98)')
    ax1.set_title(f'Pre-intervention States\nMean: {avg_pre_intervention_ratio:.3f}, Below threshold: {when2intervene_pre:.1%}')
    ax1.set_xlabel('Q-value Ratio (clipped)')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(0.95, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Non-intervention ratios histogram
    ax2.hist(non_intervention_ratios_clip, bins=bins, alpha=0.7)
    ax2.axvline(x=0.98, color='r', linestyle='--', label='Threshold (0.98)')
    ax2.set_title(f'Non-intervention States\nMean: {avg_non_intervention_ratio:.3f}, Below threshold: {when2intervene_non:.1%}')
    ax2.set_xlabel('Q-value Ratio (clipped)')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim(0.95, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics for reference
    print("\nDetailed Statistics:")
    print(f"Pre-intervention ratio min: {pre_intervention_ratios_flat.min():.4f}")
    print(f"Pre-intervention ratio max: {pre_intervention_ratios_flat.max():.4f}")
    print(f"Non-intervention ratio min: {non_intervention_ratios_flat.min():.4f}")
    print(f"Non-intervention ratio max: {non_intervention_ratios_flat.max():.4f}")
else:
    print("Not enough pre-intervention or non-intervention states for evaluation")

# %%
# get epsiodes liek this
# for i in range(15000):
#  if replay_buffer[i]['dones']:
#    print(i)

#if a step is an intervention, it is knonw by hte structure of info.
# for a given transiiton in replay_buffer,
# dict_keys(['observations', 'actions', 'next_observations', 'rewards', 'masks', 'dones', 'grasp_penalty', 'info'])
# info here is a dict of structure
# {'success': np.False_,
#  'intervene_action': array([0.00028571, 0.        , 0.08885714, 0.        , 0.        ,
#         0.        , 0.        ]),
#  'succeed': False,
#  'grasp_penalty': 0.0,
#  'policy_action': array([ 0.8717529 , -0.62178427, -0.3692599 ,  0.7954388 , -0.6098121 ,
#         -0.9245023 , -1.        ], dtype=float32)}
# othewise if not intervention: info is: {'success': np.False_, 'succeed': False, 'grasp_penalty': -0.02}

# %%
# Visualize Q-value progression across episodes
import matplotlib.pyplot as plt
import numpy as np

# Identify episodes in the replay buffer
episode_indices = [0]  # Start of first episode
for i in range(len(replay_buffer)):
    if replay_buffer[i]['dones']:
        episode_indices.append(i+1)  # Start of next episode

# Remove the last entry if it's beyond the buffer length
if episode_indices[-1] >= len(replay_buffer):
    episode_indices.pop()

# Define how many episodes to plot (to avoid overcrowding)
max_episodes_to_plot = 10
episodes_to_plot = min(len(episode_indices)-1, max_episodes_to_plot)

# Create a colormap for distinguishing episodes
colors = plt.cm.viridis(np.linspace(0, 1, episodes_to_plot))

plt.figure(figsize=(14, 8))

# Calculate and plot Q-values for each episode
for ep_idx in range(episodes_to_plot):
    start_idx = episode_indices[ep_idx]
    end_idx = episode_indices[ep_idx+1] if ep_idx < len(episode_indices)-1 else len(replay_buffer)

    # Extract steps and observations/actions for this episode
    steps = list(range(end_idx - start_idx))
    q_values = []

    # Compute Q-values for each step in the episode
    for i in range(start_idx, end_idx):
        obs = replay_buffer[i]['observations']
        action = replay_buffer[i]['actions']

        # Prepare observation for forward_critic
        prepared_obs = {
            'front': np.expand_dims(obs['front'], axis=0),
            'state': np.expand_dims(obs['state'], axis=0),
            'wrist': np.expand_dims(obs['wrist'], axis=0),
        }

        # Reshape action to match expected dimensions (add batch dimension)
        prepared_action = np.expand_dims(action[:6], axis=0)

        # Calculate Q-value
        key, rng = jax.random.split(rng)
        q_value = float(agent.forward_critic(prepared_obs, prepared_action, key).mean())
        q_values.append(q_value)

    # Plot Q-values for this episode
    plt.plot(steps, q_values, color=colors[ep_idx],
             label=f'Episode {ep_idx+1}', alpha=0.8, linewidth=2)

# Add intervention markers if present
for ep_idx in range(episodes_to_plot):
    start_idx = episode_indices[ep_idx]
    end_idx = episode_indices[ep_idx+1] if ep_idx < len(episode_indices)-1 else len(replay_buffer)

    # Find interventions in this episode
    for step, i in enumerate(range(start_idx, end_idx)):
        if 'info' in replay_buffer[i] and isinstance(replay_buffer[i]['info'], dict) and 'intervene_action' in replay_buffer[i]['info']:
            plt.axvline(x=step, color=colors[ep_idx], linestyle='--', alpha=0.3)

# Add labels and legend
plt.xlabel('Steps within Episode', fontsize=12)
plt.ylabel('Q-value', fontsize=12)
plt.title('Q-value Progression Across Episodes', fontsize=14)
plt.grid(alpha=0.3)

# Add legend with limited items if there are many episodes
if episodes_to_plot > 5:
    legend_indices = np.linspace(0, episodes_to_plot-1, 5, dtype=int)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[i] for i in legend_indices],
               [labels[i] for i in legend_indices],
               loc='best', fontsize=10)
else:
    plt.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.show()

# Also create a plot with Q-value differences between expert and policy actions
plt.figure(figsize=(14, 8))

for ep_idx in range(episodes_to_plot):
    start_idx = episode_indices[ep_idx]
    end_idx = episode_indices[ep_idx+1] if ep_idx < len(episode_indices)-1 else len(replay_buffer)

    # Extract steps for this episode
    steps = list(range(end_idx - start_idx))
    q_diffs = []

    # Compute Q-value differences for each step
    for i in range(start_idx, end_idx):
        obs = replay_buffer[i]['observations']

        # Prepare observation for critic
        prepared_obs = {
            'front': np.expand_dims(obs['front'], axis=0),
            'state': np.expand_dims(obs['state'], axis=0),
            'wrist': np.expand_dims(obs['wrist'], axis=0),
        }

        # Get expert action and policy action
        expert_action, rng = get_action(prepared_obs, rng)
        expert_action = np.asarray(jax.device_get(expert_action))

        # Get policy action (non-optimal/exploratory)
        rng, policy_key = jax.random.split(rng)
        policy_action = agent.sample_actions(
            observations=jax.device_put(prepared_obs),
            argmax=False,  # Sample from policy distribution
            seed=policy_key
        )
        policy_action = np.asarray(jax.device_get(policy_action))

        # Reshape actions to match expected dimensions
        prepared_expert_action = np.expand_dims(expert_action[0, :6], axis=0)
        prepared_policy_action = np.expand_dims(policy_action[0, :6], axis=0)

        # Calculate Q-values
        key, rng = jax.random.split(rng)
        q_expert = float(agent.forward_critic(prepared_obs, prepared_expert_action, key).mean())

        key, rng = jax.random.split(rng)
        q_policy = float(agent.forward_critic(prepared_obs, prepared_policy_action, key).mean())

        # Calculate ratio (Q-policy / Q-expert)
        if abs(q_expert) > 1e-8:  # Avoid division by zero
            q_ratio = q_policy / q_expert
        else:
            q_ratio = 1.0  # Default to 1.0 if expert Q-value is near zero
        q_diffs.append(q_ratio)

    # Plot Q-value ratios for this episode
    plt.plot(steps, q_diffs, color=colors[ep_idx],
             label=f'Episode {ep_idx+1}', alpha=0.8, linewidth=2)

# Add the when2intervene threshold
plt.axhline(y=0.98, color='r', linestyle='--', label='When2Intervene threshold (0.98)')

# Add labels and legend
plt.xlabel('Steps within Episode', fontsize=12)
plt.ylabel('Q-value Ratio (Policy/Expert)', fontsize=12)
plt.title('Q-value Ratio Progression Across Episodes', fontsize=14)
plt.grid(alpha=0.3)

# Add legend with limited items if there are many episodes
if episodes_to_plot > 5:
    legend_indices = np.linspace(0, episodes_to_plot-1, 5, dtype=int)
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = [handles[i] for i in legend_indices] + [handles[-1]]  # Add threshold line
    new_labels = [labels[i] for i in legend_indices] + [labels[-1]]
    plt.legend(new_handles, new_labels, loc='best', fontsize=10)
else:
    plt.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.show()

# %%
# Plot histogram of Q_post/Q_pre ratios for ALL interventions in the preference buffer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Create arrays to store the pre and post Q-values
q_pre_values = []
q_post_values = []
ratios = []

# Process all intervention pairs in the preference buffer
for i in tqdm(range(len(preference_buffer)), desc="Processing interventions"):
    intervention = preference_buffer[i]

    # Get the pre-intervention state and action (first state in sequence)
    pre_obs = {
        'front': np.expand_dims(intervention['observations'][0]['front'], axis=0),
        'state': np.expand_dims(intervention['observations'][0]['state'], axis=0),
        'wrist': np.expand_dims(intervention['observations'][0]['wrist'], axis=0),
    }
    pre_action = np.expand_dims(intervention['actions'][0][:6], axis=0)

    # Get the post-intervention state and action (last state in sequence)
    post_obs = {
        'front': np.expand_dims(intervention['observations'][-1]['front'], axis=0),
        'state': np.expand_dims(intervention['observations'][-1]['state'], axis=0),
        'wrist': np.expand_dims(intervention['observations'][-1]['wrist'], axis=0),
    }
    post_action = np.expand_dims(intervention['actions'][-1][:6], axis=0)

    # Calculate Q-values
    q_pre, rng = expert_critic(pre_obs, pre_action, rng)
    q_post, rng = expert_critic(post_obs, post_action, rng)

    # Extract the values
    q_pre_value = float(q_pre.mean())
    q_post_value = float(q_post.mean())

    # Store the values
    q_pre_values.append(q_pre_value)
    q_post_values.append(q_post_value)

    # Calculate and store the ratio (with protection against division by zero)
    if abs(q_pre_value) > 1e-8:
        ratio = q_post_value / q_pre_value
    else:
        ratio = 1.0  # Default if pre value is near zero

    ratios.append(ratio)

# Convert to numpy arrays for analysis
q_pre_values = np.array(q_pre_values)
q_post_values = np.array(q_post_values)
ratios = np.array(ratios)

# Print some basic statistics
print(f"Number of interventions analyzed: {len(ratios)}")
print(f"Q-pre values - min: {q_pre_values.min():.4f}, max: {q_pre_values.max():.4f}, mean: {q_pre_values.mean():.4f}")
print(f"Q-post values - min: {q_post_values.min():.4f}, max: {q_post_values.max():.4f}, mean: {q_post_values.mean():.4f}")
print(f"Ratios (Q-post/Q-pre) - min: {ratios.min():.4f}, max: {ratios.max():.4f}, mean: {ratios.mean():.4f}")
print(f"Percentage of interventions with improved Q-values: {(ratios > 1.0).mean() * 100:.1f}%")

# Create a figure for the histogram
plt.figure(figsize=(12, 8))

# Clip extreme ratio values for better visualization (adjust if needed)
ratios_clipped = np.clip(ratios, 0.5, 2.0)

# Create the histogram with more bins for better detail
plt.hist(ratios_clipped, bins=50, alpha=0.8, color='royalblue',
         edgecolor='black', linewidth=0.5)

# Add vertical line at 1.0 (no change in Q-value)
plt.axvline(x=1.0, color='r', linestyle='--', linewidth=2,
            label='No change (ratio = 1.0)')

# Add labels and title
plt.xlabel('Q-value Ratio (Q_post / Q_pre)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Q-value Ratios After Intervention\n(All Interventions)', fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

# Add annotation showing improvement percentage
improved_pct = (ratios > 1.0).mean() * 100
plt.annotate(f"Interventions with improved Q-value: {improved_pct:.1f}%",
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=14, ha='left',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Add annotation for mean ratio
plt.annotate(f"Mean ratio: {ratios.mean():.3f}",
            xy=(0.05, 0.88), xycoords='axes fraction',
            fontsize=14, ha='left',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Add note about clipping (if applied)
if (ratios.min() < 0.5) or (ratios.max() > 2.0):
    plt.annotate(f"Note: Histogram shows ratios clipped to [0.5, 2.0]\nActual range: [{ratios.min():.2f}, {ratios.max():.2f}]",
                xy=(0.05, 0.81), xycoords='axes fraction',
                fontsize=12, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))

plt.tight_layout()
plt.show()

# Also create a scatter plot of pre vs post Q-values
plt.figure(figsize=(10, 8))
plt.scatter(q_pre_values, q_post_values, alpha=0.6, s=50)
plt.plot([min(q_pre_values), max(q_pre_values)], [min(q_pre_values), max(q_pre_values)],
         'r--', linewidth=2, label='Equal Q-values (no change)')

plt.xlabel('Q-value (Pre-intervention)', fontsize=14)
plt.ylabel('Q-value (Post-intervention)', fontsize=14)
plt.title('Pre vs Post Intervention Q-values', fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

# Add annotation showing improvement percentage
plt.annotate(f"Points above line: {improved_pct:.1f}% (improved Q-value)",
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=14, ha='left',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# %%
# Visualize (log_0.99(q_pre/10) - log_0.99(q_post/10)) / k for all interventions
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math

# Function to calculate logarithm with base 0.99
def log_base_099(x):
    # Ensure the input is positive for logarithm
    if x <= 0:
        return 0  # Handle non-positive values
    return math.log(x) / math.log(0.99)

# Create arrays to store the values
q_pre_values = []
q_post_values = []
intervention_lengths = []
optimality_metrics = []

# Process all interventions
for i in tqdm(range(len(preference_buffer)), desc="Calculating optimality metrics"):
    intervention = preference_buffer[i]

    # Get intervention length (k)
    k = len(intervention['observations'])
    intervention_lengths.append(k)

    # Get the pre-intervention state and action (first state in sequence)
    pre_obs = {
        'front': np.expand_dims(intervention['observations'][0]['front'], axis=0),
        'state': np.expand_dims(intervention['observations'][0]['state'], axis=0),
        'wrist': np.expand_dims(intervention['observations'][0]['wrist'], axis=0),
    }
    pre_action = np.expand_dims(intervention['actions'][0][:6], axis=0)

    # Get the post-intervention state and action (last state in sequence)
    post_obs = {
        'front': np.expand_dims(intervention['observations'][-1]['front'], axis=0),
        'state': np.expand_dims(intervention['observations'][-1]['state'], axis=0),
        'wrist': np.expand_dims(intervention['observations'][-1]['wrist'], axis=0),
    }
    post_action = np.expand_dims(intervention['actions'][-1][:6], axis=0)

    # Calculate Q-values
    q_pre, rng = expert_critic(pre_obs, pre_action, rng)
    q_post, rng = expert_critic(post_obs, post_action, rng)

    # Extract the values
    q_pre_value = float(q_pre.mean())
    q_post_value = float(q_post.mean())

    q_pre_values.append(q_pre_value)
    q_post_values.append(q_post_value)

    # Calculate the optimality metric
    # Add a safety offset to ensure values are positive for logarithm
    offset = 10  # This matches the "/10" in the formula

    # Calculate logarithms with appropriate handling of potential negative values
    try:
        log_pre = log_base_099((q_pre_value + offset) / offset)
        log_post = log_base_099((q_post_value + offset) / offset)
        # Calculate the metric
        optimality = (log_pre - log_post) / k if k > 0 else 0
    except (ValueError, ZeroDivisionError):
        # Handle any errors in calculation
        optimality = 0

    optimality_metrics.append(optimality)

# Convert to numpy arrays
optimality_metrics = np.array(optimality_metrics)
intervention_lengths = np.array(intervention_lengths)

# Print statistics
print(f"Number of interventions analyzed: {len(optimality_metrics)}")
print(f"Optimality metrics - min: {optimality_metrics.min():.6f}, max: {optimality_metrics.max():.6f}, mean: {optimality_metrics.mean():.6f}")
print(f"Intervention lengths - min: {intervention_lengths.min()}, max: {intervention_lengths.max()}, mean: {intervention_lengths.mean():.2f}")

# Create a histogram of the optimality metrics
plt.figure(figsize=(12, 8))

# Determine reasonable clip bounds based on data
lower_clip = np.percentile(optimality_metrics, 1)
upper_clip = np.percentile(optimality_metrics, 99)
optimality_metrics_clipped = np.clip(optimality_metrics, lower_clip, upper_clip)

plt.hist(optimality_metrics_clipped, bins=50, alpha=0.8, color='teal',
         edgecolor='black', linewidth=0.5)

# Add vertical line at 0 (neutral optimality)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Neutral (0)')

# Add labels and title
plt.xlabel('Optimality Metric: (log₀.₉₉(q_pre/10) - log₀.₉₉(q_post/10)) / k', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Intervention Optimality Metrics', fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

# Add annotations
plt.annotate(f"Mean optimality: {optimality_metrics.mean():.6f}",
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=14, ha='left',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Add note about clipping if applied
if np.any(optimality_metrics < lower_clip) or np.any(optimality_metrics > upper_clip):
    plt.annotate(f"Note: Histogram shows values clipped to [{lower_clip:.6f}, {upper_clip:.6f}]\nActual range: [{optimality_metrics.min():.6f}, {optimality_metrics.max():.6f}]",
                xy=(0.05, 0.88), xycoords='axes fraction',
                fontsize=12, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))

plt.tight_layout()
plt.show()

# Also create a scatter plot of optimality vs intervention length
plt.figure(figsize=(12, 8))
plt.scatter(intervention_lengths, optimality_metrics, alpha=0.6, s=50, c='teal')

plt.xlabel('Intervention Length (k)', fontsize=14)
plt.ylabel('Optimality Metric', fontsize=14)
plt.title('Intervention Optimality vs. Length', fontsize=16)
plt.grid(alpha=0.3)

# Add horizontal line at y=0
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Neutral (0)')
plt.legend(fontsize=12)

# Add trend line
if len(intervention_lengths) > 1:
    z = np.polyfit(intervention_lengths, optimality_metrics, 1)
    p = np.poly1d(z)
    plt.plot(sorted(intervention_lengths), p(sorted(intervention_lengths)),
             "r--", linewidth=2, alpha=0.7, label=f"Trend: y = {z[0]:.6f}x + {z[1]:.6f}")
    plt.legend(fontsize=12)

plt.tight_layout()
plt.show()

# Visualize a joint plot of q_pre vs q_post colored by optimality
plt.figure(figsize=(12, 10))
scatter = plt.scatter(q_pre_values, q_post_values,
                     c=optimality_metrics, cmap='coolwarm', alpha=0.7, s=60)

# Add a line for q_pre = q_post
min_val = min(min(q_pre_values), min(q_post_values))
max_val = max(max(q_pre_values), max(q_post_values))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='q_pre = q_post')

plt.colorbar(scatter, label='Optimality Metric')
plt.xlabel('Pre-intervention Q-value', fontsize=14)
plt.ylabel('Post-intervention Q-value', fontsize=14)
plt.title('Pre vs Post Q-values Colored by Optimality Metric', fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Plot histogram of intervention lengths
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Extract intervention lengths
intervention_lengths = []

for intervention in tqdm(preference_buffer, desc="Extracting intervention lengths"):
    # Get the number of steps in this intervention
    length = len(intervention['observations'])
    intervention_lengths.append(length)

# Convert to numpy array for analysis
intervention_lengths = np.array(intervention_lengths)

# Print statistics about intervention lengths
print(f"Number of interventions: {len(intervention_lengths)}")
print(f"Intervention length statistics:")
print(f"  Min length: {intervention_lengths.min()}")
print(f"  Max length: {intervention_lengths.max()}")
print(f"  Mean length: {intervention_lengths.mean():.2f}")
print(f"  Median length: {np.median(intervention_lengths):.2f}")
print(f"  Standard deviation: {np.std(intervention_lengths):.2f}")

# Create histogram
plt.figure(figsize=(12, 8))
n, bins, patches = plt.hist(intervention_lengths, bins=30, color='royalblue',
                           alpha=0.7, edgecolor='black', linewidth=0.5)

# Add vertical line at mean length
plt.axvline(x=intervention_lengths.mean(), color='r', linestyle='--',
           linewidth=2, label=f'Mean length: {intervention_lengths.mean():.2f}')

# Add vertical line at median length
plt.axvline(x=np.median(intervention_lengths), color='g', linestyle='--',
           linewidth=2, label=f'Median length: {np.median(intervention_lengths):.2f}')

# Add labels and title
plt.xlabel('Intervention Length (steps)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Intervention Lengths', fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

# Add annotations with additional statistics
stats_text = (f"Total interventions: {len(intervention_lengths)}\n"
              f"Min length: {intervention_lengths.min()}\n"
              f"Max length: {intervention_lengths.max()}\n"
              f"Std deviation: {np.std(intervention_lengths):.2f}")

plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
            fontsize=12, ha='right', va='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Optional: Add percentile lines
percentiles = [25, 75]
for p in percentiles:
    value = np.percentile(intervention_lengths, p)
    plt.axvline(x=value, color='orange', alpha=0.5, linestyle=':',
              linewidth=1.5, label=f'{p}th percentile: {value:.2f}')

# Update legend after adding percentile lines
plt.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.show()

# Optional: Create a cumulative distribution plot
plt.figure(figsize=(12, 8))
counts, bin_edges = np.histogram(intervention_lengths, bins=30, density=True)
cdf = np.cumsum(counts) * (bin_edges[1] - bin_edges[0])

plt.plot(bin_edges[1:], cdf, 'b-', linewidth=2,
        label='Cumulative Distribution')

# Add horizontal lines at key percentages
for p in [0.25, 0.5, 0.75, 0.9]:
    plt.axhline(y=p, color='gray', alpha=0.5, linestyle='--')
    # Find the corresponding x value (intervention length) at this percentile
    p_value = np.percentile(intervention_lengths, p*100)
    plt.annotate(f'{int(p*100)}%: {p_value:.1f} steps',
                xy=(p_value, p), xytext=(p_value+1, p+0.03),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

plt.xlabel('Intervention Length (steps)', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.title('Cumulative Distribution of Intervention Lengths', fontsize=16)
plt.grid(alpha=0.3)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# %%
# Plot how intervention lengths change over time/episodes
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# First, determine chronological order of interventions
# We'll use the timestamp of the first observation in each intervention if available,
# or just the order in the preference buffer as a proxy for time

# Check if interventions have timestamps
has_timestamps = False
if len(preference_buffer) > 0 and 'observations' in preference_buffer[0]:
    if len(preference_buffer[0]['observations']) > 0:
        sample_obs = preference_buffer[0]['observations'][0]
        has_timestamps = hasattr(sample_obs, 'timestamp') or 'timestamp' in sample_obs

if has_timestamps:
    # Sort interventions by timestamp if available
    print("Using timestamps to order interventions...")
    # Extract timestamps (implementation depends on data structure)
    # This is a placeholder - adjust based on actual data structure
    timestamps = [intervention['observations'][0].get('timestamp', i)
                  for i, intervention in enumerate(preference_buffer)]

    # Create indices sorted by timestamp
    sorted_indices = np.argsort(timestamps)
else:
    print("No timestamps found, using intervention order in buffer as proxy for time...")
    # Use original order as proxy for time
    sorted_indices = np.arange(len(preference_buffer))

# Now we'll group interventions into chronological bins/episodes
# We can use fixed-size bins or identify episode boundaries if that information is available

# Approach 1: Fixed-size bins (if episode info is not available)
bin_size = 10  # Number of interventions per bin
num_bins = len(preference_buffer) // bin_size + (1 if len(preference_buffer) % bin_size > 0 else 0)

# For storing average lengths per bin
bin_avg_lengths = []
bin_centers = []

for bin_idx in range(num_bins):
    start_idx = bin_idx * bin_size
    end_idx = min(start_idx + bin_size, len(sorted_indices))

    # Get interventions in this bin
    bin_indices = sorted_indices[start_idx:end_idx]
    bin_lengths = [len(preference_buffer[i]['observations']) for i in bin_indices]

    # Calculate average length for this bin
    if bin_lengths:
        bin_avg_lengths.append(np.mean(bin_lengths))
        bin_centers.append(bin_idx * bin_size + bin_size/2)  # Center of bin

# Plot average intervention length over time (fixed bins)
plt.figure(figsize=(14, 8))
plt.plot(bin_centers, bin_avg_lengths, 'o-', linewidth=2, markersize=8)
plt.xlabel('Intervention Index (grouped by bins of size {})'.format(bin_size), fontsize=14)
plt.ylabel('Average Intervention Length (steps)', fontsize=14)
plt.title('Trend of Intervention Lengths Over Time', fontsize=16)
plt.grid(alpha=0.3)

# Add smoothed trend line
if len(bin_centers) > 5:
    try:
        from scipy.signal import savgol_filter
        window_length = min(len(bin_avg_lengths) - (len(bin_avg_lengths) % 2 - 1), 11)
        if window_length > 3:
            smoothed = savgol_filter(bin_avg_lengths, window_length, 3)
            plt.plot(bin_centers, smoothed, 'r-', linewidth=3, alpha=0.7,
                    label='Smoothed trend')
            plt.legend(fontsize=12)
    except:
        # Fall back to simple polynomial fit if savgol_filter fails
        z = np.polyfit(bin_centers, bin_avg_lengths, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(bin_centers), max(bin_centers), 100)
        plt.plot(x_smooth, p(x_smooth), 'r-', linewidth=3, alpha=0.7,
                label='Trend line (polynomial fit)')
        plt.legend(fontsize=12)

# Calculate overall trend (percentage increase or decrease)
if len(bin_avg_lengths) > 1:
    first_avg = bin_avg_lengths[0]
    last_avg = bin_avg_lengths[-1]
    change_pct = ((last_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0

    change_text = f"Overall change: {change_pct:.1f}% ({first_avg:.1f} to {last_avg:.1f} steps)"
    plt.annotate(change_text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# Approach 2: If we have episode information from replay buffer
# Let's see if we can map interventions to specific episodes

# We'll try to extract episode information from replay_buffer
# Episode starts are already collected in episode_indices
print(f"Found {len(episode_indices)} episode boundaries in replay buffer")

# Function to determine which episode an intervention belongs to
def find_episode_for_intervention(intervention, replay_buffer, episode_indices):
    """Find which episode an intervention belongs to based on matching observations"""
    # Get first observation from intervention
    first_obs = intervention['observations'][0]

    # Try to find matching observation in replay buffer
    for i, transition in enumerate(replay_buffer):
        # Simplified matching - this might need to be adapted based on actual data structure
        if np.array_equal(transition['observations']['front'], first_obs['front']) and \
           np.array_equal(transition['observations']['state'], first_obs['state']):
            # Found a match, now determine which episode this is
            for ep_idx, start_idx in enumerate(episode_indices[:-1]):
                end_idx = episode_indices[ep_idx+1]
                if start_idx <= i < end_idx:
                    return ep_idx
    return None

# Try to map interventions to episodes
intervention_to_episode = {}
episodes_with_interventions = defaultdict(list)

# Use a sample of interventions to check if we can map them to episodes
# (checking all might be too slow)
sample_size = min(100, len(preference_buffer))
sample_indices = np.random.choice(len(preference_buffer), sample_size, replace=False)

print(f"Trying to map {sample_size} interventions to episodes...")
mappable_count = 0

for idx in tqdm(sample_indices):
    intervention = preference_buffer[idx]
    ep_idx = find_episode_for_intervention(intervention, replay_buffer, episode_indices)
    if ep_idx is not None:
        intervention_to_episode[idx] = ep_idx
        episodes_with_interventions[ep_idx].append(idx)
        mappable_count += 1

print(f"Successfully mapped {mappable_count}/{sample_size} interventions to episodes")

# If we have enough mappings, plot episode-based trends
if mappable_count > sample_size * 0.5:  # If we were able to map more than half
    print("Plotting episode-based intervention length trends...")

    # Calculate average intervention length per episode
    episode_avg_lengths = []
    episode_nums = []

    for ep_idx, intervention_indices in sorted(episodes_with_interventions.items()):
        if intervention_indices:  # If this episode has interventions
            lengths = [len(preference_buffer[i]['observations']) for i in intervention_indices]
            episode_avg_lengths.append(np.mean(lengths))
            episode_nums.append(ep_idx)

    # Plot episode-based average intervention lengths
    plt.figure(figsize=(14, 8))
    plt.plot(episode_nums, episode_avg_lengths, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Episode Number', fontsize=14)
    plt.ylabel('Average Intervention Length (steps)', fontsize=14)
    plt.title('Average Intervention Length by Episode', fontsize=16)
    plt.grid(alpha=0.3)

    # Add trend line if we have enough episodes
    if len(episode_nums) > 5:
        z = np.polyfit(episode_nums, episode_avg_lengths, 1)
        p = np.poly1d(z)
        plt.plot(episode_nums, p(episode_nums), 'r--', linewidth=2,
                label=f'Trend: {z[0]:.4f}x + {z[1]:.2f}')
        plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

    # Create a bar plot showing number of interventions per episode
    interventions_per_episode = [len(intervention_indices) for _, intervention_indices
                               in sorted(episodes_with_interventions.items())]
    episode_indices = sorted(episodes_with_interventions.keys())

    plt.figure(figsize=(14, 8))
    plt.bar(episode_indices, interventions_per_episode, alpha=0.7)
    plt.xlabel('Episode Number', fontsize=14)
    plt.ylabel('Number of Interventions', fontsize=14)
    plt.title('Interventions per Episode', fontsize=16)
    plt.grid(alpha=0.3, axis='y')

    # Add moving average
    window_size = 3
    if len(episode_indices) >= window_size:
        moving_avg = pd.Series(interventions_per_episode).rolling(window=window_size).mean().values
        valid_indices = ~np.isnan(moving_avg)
        plt.plot(np.array(episode_indices)[valid_indices],
                moving_avg[valid_indices], 'r-', linewidth=3,
                label=f'{window_size}-episode moving average')
        plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

else:
    print("Not enough mappings to create episode-based plots")

# Alternative: Plot a rolling average of intervention lengths over the sequence
print("Plotting rolling average of intervention lengths...")

# Calculate intervention lengths
intervention_lengths = [len(preference_buffer[i]['observations']) for i in sorted_indices]

# Create a rolling average
window_size = 20  # Adjust based on number of interventions
rolling_avg = []
indices = []

for i in range(len(intervention_lengths) - window_size + 1):
    rolling_avg.append(np.mean(intervention_lengths[i:i+window_size]))
    indices.append(i + window_size//2)  # Center point of window

plt.figure(figsize=(14, 8))
plt.plot(range(len(intervention_lengths)), intervention_lengths, 'o',
         alpha=0.3, label='Individual interventions')
plt.plot(indices, rolling_avg, 'r-', linewidth=3,
         label=f'{window_size}-intervention rolling average')

plt.xlabel('Intervention Index (chronological)', fontsize=14)
plt.ylabel('Intervention Length (steps)', fontsize=14)
plt.title('Intervention Lengths Over Time with Rolling Average', fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

# Add annotations for trend
if rolling_avg:
    first_avg = rolling_avg[0]
    last_avg = rolling_avg[-1]
    change_pct = ((last_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0

    trend_text = f"Overall trend: {change_pct:.1f}% change in average length"
    plt.annotate(trend_text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# %%
# %%
# Visualize Q values across trajectories with interventions highlighted
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import os

# Create output directory for plots if it doesn't exist
plots_dir = "q_value_plots"
os.makedirs(plots_dir, exist_ok=True)

# Identify all episodes in the replay buffer
episode_indices = [0]  # Start of first episode
for i in range(len(replay_buffer)):
    if replay_buffer[i]['dones']:
        episode_indices.append(i+1)  # Start of next episode

# Remove the last entry if it's beyond the buffer length
if episode_indices[-1] >= len(replay_buffer):
    episode_indices.pop()

# Calculate Q-values and identify interventions for each step in each episode
print(f"Calculating Q-values for {len(episode_indices)-1} episodes...")

episodes_data = []
for ep_idx in tqdm(range(len(episode_indices)-1), desc="Processing episodes"):
    start_idx = episode_indices[ep_idx]
    end_idx = episode_indices[ep_idx+1]

    episode_steps = []
    episode_q_values = []
    episode_is_intervention = []

    # Process each step in this episode
    for step_idx, i in enumerate(range(start_idx, end_idx)):
        transition = replay_buffer[i]
        obs = transition['observations']
        action = transition['actions']

        # Determine if this is an intervention
        is_intervention = ('info' in transition and
                          isinstance(transition['info'], dict) and
                          'intervene_action' in transition['info'])

        # Prepare observation for Q-value calculation
        prepared_obs = {
            'front': np.expand_dims(obs['front'], axis=0),
            'state': np.expand_dims(obs['state'], axis=0),
            'wrist': np.expand_dims(obs['wrist'], axis=0),
        }

        # Prepare action (first 6 dimensions)
        prepared_action = np.expand_dims(action[:6], axis=0)

        # Calculate Q-value
        key, rng = jax.random.split(rng)
        q_value = float(agent.forward_critic(prepared_obs, prepared_action, key).mean())

        # Store data
        episode_steps.append(step_idx)
        episode_q_values.append(q_value)
        episode_is_intervention.append(is_intervention)

    # Store episode data
    episodes_data.append({
        'steps': episode_steps,
        'q_values': episode_q_values,
        'is_intervention': episode_is_intervention,
        'episode_idx': ep_idx
    })

print(f"Generating plots with 5 episodes per plot...")

# Generate plots with 5 episodes per plot
num_episodes = len(episodes_data)
plots_needed = (num_episodes + 4) // 5  # Ceiling division

# Create a custom colormap for a smooth transition
intervention_cmap = LinearSegmentedColormap.from_list(
    "intervention_cmap", ["blue", "red"])

for plot_idx in tqdm(range(plots_needed), desc="Generating group plots"):
    start_ep = plot_idx * 5
    end_ep = min(start_ep + 5, num_episodes)

    plt.figure(figsize=(15, 10))

    # Plot each episode in this group
    for i, ep_idx in enumerate(range(start_ep, end_ep)):
        episode = episodes_data[ep_idx]
        steps = episode['steps']
        q_values = episode['q_values']
        is_intervention = episode['is_intervention']

        # Create segments for continuous coloring
        points = np.array([steps, q_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create line collection with colors based on intervention status
        from matplotlib.collections import LineCollection
        colors = np.array(['blue' if not interv else 'red' for interv in is_intervention[:-1]])
        lc = LineCollection(segments, colors=colors, linewidth=2, label=f'Episode {ep_idx+1}')

        # Add line collection to plot
        plt.gca().add_collection(lc)

        # Also plot points to show exact values
        intervention_points = [steps[j] for j in range(len(steps)) if is_intervention[j]]
        intervention_q_values = [q_values[j] for j in range(len(q_values)) if is_intervention[j]]
        normal_points = [steps[j] for j in range(len(steps)) if not is_intervention[j]]
        normal_q_values = [q_values[j] for j in range(len(q_values)) if not is_intervention[j]]

        plt.scatter(intervention_points, intervention_q_values, color='red', s=30, alpha=0.6)
        plt.scatter(normal_points, normal_q_values, color='blue', s=20, alpha=0.6)

        # Add episode label at the end of the line
        if steps:
            plt.text(steps[-1], q_values[-1], f" Ep {ep_idx+1}", fontsize=10)

    # Set plot limits and labels
    plt.xlim(0, max([max(ep['steps']) for ep in episodes_data[start_ep:end_ep]]) + 5)
    y_min = min([min(ep['q_values']) for ep in episodes_data[start_ep:end_ep]])
    y_max = max([max(ep['q_values']) for ep in episodes_data[start_ep:end_ep]])
    margin = (y_max - y_min) * 0.1
    plt.ylim(y_min - margin, y_max + margin)

    plt.xlabel('Steps in Episode', fontsize=14)
    plt.ylabel('Q-value', fontsize=14)
    plt.title(f'Q-values Across Episodes {start_ep+1}-{end_ep}\nRed = Intervention, Blue = Regular Action', fontsize=16)
    plt.grid(alpha=0.3)

    # Add legend for intervention vs normal
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Intervention'),
        Line2D([0], [0], color='blue', lw=3, label='Regular Action')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/q_values_episodes_{start_ep+1}_to_{end_ep}.png", dpi=150)
    plt.close()

# Also generate individual plots for each episode
print(f"Generating individual plots for each episode...")

for ep_idx, episode in tqdm(enumerate(episodes_data), desc="Generating individual plots", total=len(episodes_data)):
    steps = episode['steps']
    q_values = episode['q_values']
    is_intervention = episode['is_intervention']

    if not steps:  # Skip if episode is empty
        continue

    plt.figure(figsize=(12, 8))

    # Plot step-by-step colored line
    for i in range(len(steps)-1):
        color = 'red' if is_intervention[i] else 'blue'
        plt.plot([steps[i], steps[i+1]], [q_values[i], q_values[i+1]], color=color, linewidth=2.5)

    # Add markers
    intervention_points = [steps[j] for j in range(len(steps)) if is_intervention[j]]
    intervention_q_values = [q_values[j] for j in range(len(q_values)) if is_intervention[j]]
    normal_points = [steps[j] for j in range(len(steps)) if not is_intervention[j]]
    normal_q_values = [q_values[j] for j in range(len(q_values)) if not is_intervention[j]]

    # Calculate statistics
    avg_q = np.mean(q_values)
    avg_intervention_q = np.mean(intervention_q_values) if intervention_q_values else None
    avg_normal_q = np.mean(normal_q_values) if normal_q_values else None

    # Calculate percentage of steps that were interventions
    intervention_pct = (sum(is_intervention) / len(is_intervention)) * 100 if is_intervention else 0

    # Plot scatter points
    if intervention_points:
        plt.scatter(intervention_points, intervention_q_values, color='red', s=50, alpha=0.7,
                    label=f'Intervention (avg Q: {avg_intervention_q:.3f})' if avg_intervention_q is not None else 'Intervention')
    if normal_points:
        plt.scatter(normal_points, normal_q_values, color='blue', s=40, alpha=0.7,
                   label=f'Regular Action (avg Q: {avg_normal_q:.3f})' if avg_normal_q is not None else 'Regular Action')

    plt.xlabel('Steps in Episode', fontsize=14)
    plt.ylabel('Q-value', fontsize=14)
    plt.title(f'Episode {ep_idx+1} Q-values\n{intervention_pct:.1f}% Intervention Steps', fontsize=16)
    plt.grid(alpha=0.3)

    # Add horizontal line at average Q-value
    plt.axhline(y=avg_q, color='green', linestyle='--', alpha=0.7, label=f'Avg Q-value: {avg_q:.3f}')

    # Add text annotation with episode statistics
    stats_text = (
        f"Episode length: {len(steps)} steps\n"
        f"Intervention steps: {sum(is_intervention)} ({intervention_pct:.1f}%)\n"
        f"Max Q-value: {max(q_values):.3f}\n"
        f"Min Q-value: {min(q_values):.3f}"
    )
    plt.annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction',
                fontsize=11, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/q_values_episode_{ep_idx+1}.png", dpi=150)
    plt.close()

print(f"All plots saved to {plots_dir}/")

# Display summary of all episodes
print("\nSummary of all episodes:")
total_steps = sum(len(ep['steps']) for ep in episodes_data)
total_interventions = sum(sum(ep['is_intervention']) for ep in episodes_data)
print(f"Total episodes: {len(episodes_data)}")
print(f"Total steps: {total_steps}")
print(f"Total intervention steps: {total_interventions} ({(total_interventions/total_steps)*100:.2f}%)")
print(f"Average episode length: {total_steps/len(episodes_data):.2f} steps")
print(f"Average interventions per episode: {total_interventions/len(episodes_data):.2f} steps")
# %%
# Visualize Q values across trajectories with interventions highlighted (with batching)
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import os
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# Create output directory for plots if it doesn't exist
plots_dir = "q_value_plots"
os.makedirs(plots_dir, exist_ok=True)

# Set batch size for more efficient processing
BATCH_SIZE = 1024

# Identify all episodes in the replay buffer
episode_indices = [0]  # Start of first episode
for i in range(len(replay_buffer)):
    if replay_buffer[i]['dones']:
        episode_indices.append(i+1)  # Start of next episode

# Remove the last entry if it's beyond the buffer length
if episode_indices[-1] >= len(replay_buffer):
    episode_indices.pop()

print(f"Calculating Q-values for {len(episode_indices)-1} episodes...")

# First, prepare data for each episode
episodes_data = []
all_obs_batches = []
all_action_batches = []
all_batch_indices = []  # To track which episode and step each batch element belongs to

for ep_idx in tqdm(range(len(episode_indices)-1), desc="Preparing episode data"):
    start_idx = episode_indices[ep_idx]
    end_idx = episode_indices[ep_idx+1]

    episode_steps = list(range(end_idx - start_idx))
    episode_is_intervention = []

    # Process each step in this episode
    for step_idx, i in enumerate(range(start_idx, end_idx)):
        transition = replay_buffer[i]

        # Determine if this is an intervention
        is_intervention = ('info' in transition and
                          isinstance(transition['info'], dict) and
                          'intervene_action' in transition['info'])
        episode_is_intervention.append(is_intervention)

        # Prepare data for batching
        obs = transition['observations']
        action = transition['actions']

        # Add to global batching lists
        all_obs_batches.append(obs)
        all_action_batches.append(action[:6])  # First 6 dimensions of action
        all_batch_indices.append((ep_idx, step_idx))  # Track which episode and step

    # Store episode data (without Q-values for now)
    episodes_data.append({
        'steps': episode_steps,
        'q_values': [0] * len(episode_steps),  # Placeholder, will fill later
        'is_intervention': episode_is_intervention,
        'episode_idx': ep_idx
    })

# Now batch process all the data to get Q-values
print(f"Processing {len(all_obs_batches)} total steps in batches of {BATCH_SIZE}...")

# Process in batches
q_values = []
try:
    for batch_idx in tqdm(range(0, len(all_obs_batches), BATCH_SIZE), desc="Computing Q-values in batches"):
        # Get current batch
        batch_end = min(batch_idx + BATCH_SIZE, len(all_obs_batches))
        batch_size = batch_end - batch_idx

        # Prepare batch
        batch_obs = {
            'front': np.stack([all_obs_batches[i]['front'] for i in range(batch_idx, batch_end)]),
            'state': np.stack([all_obs_batches[i]['state'] for i in range(batch_idx, batch_end)]),
            'wrist': np.stack([all_obs_batches[i]['wrist'] for i in range(batch_idx, batch_end)]),
        }
        batch_actions = np.stack([all_action_batches[i] for i in range(batch_idx, batch_end)])

        # Calculate Q-values for the batch
        key, rng = jax.random.split(rng)
        batch_q_values = agent.forward_critic(batch_obs, batch_actions, key)
        batch_q_values = np.asarray(batch_q_values)

        # Print batch shape information for debugging
        if batch_idx == 0:  # Only print for first batch
            print(f"Batch Q-values shape: {batch_q_values.shape}")

        # Take minimum across critic ensemble (first dimension)
        # This is the standard approach in SAC where we use the minimum of the critic values
        batch_q_values_min = batch_q_values.min(axis=0).tolist()

        # Add to overall list - make sure we're getting one value per input
        if len(batch_q_values_min) != batch_size:
            print(f"Warning: Expected {batch_size} Q-values but got {len(batch_q_values_min)}")

        q_values.extend(batch_q_values_min)

    print(f"Computed {len(q_values)} Q-values for {len(all_batch_indices)} steps")

    # Check for mismatch
    if len(q_values) != len(all_batch_indices):
        print(f"Warning: Mismatch between Q-values ({len(q_values)}) and batch indices ({len(all_batch_indices)})")
        # Truncate to the shorter length
        min_len = min(len(q_values), len(all_batch_indices))
        q_values = q_values[:min_len]
        all_batch_indices = all_batch_indices[:min_len]

    # Assign Q-values back to the episodes safely
    for i, (ep_idx, step_idx) in enumerate(all_batch_indices):
        if i < len(q_values):  # Safety check
            if ep_idx < len(episodes_data) and step_idx < len(episodes_data[ep_idx]['q_values']):
                episodes_data[ep_idx]['q_values'][step_idx] = q_values[i]
            else:
                print(f"Warning: Invalid episode index ({ep_idx},{step_idx})")
except Exception as e:
    print(f"Error during batch processing: {e}")
    # Fall back to non-batched processing if batching fails
    print("Falling back to step-by-step processing...")

    # Reset episodes_data q_values
    for ep_data in episodes_data:
        ep_data['q_values'] = [0] * len(ep_data['steps'])

    # Process each episode step by step
    for ep_idx in tqdm(range(len(episode_indices)-1), desc="Processing episodes (fallback)"):
        start_idx = episode_indices[ep_idx]
        end_idx = episode_indices[ep_idx+1]

        # Process each step in this episode
        for step_idx, i in enumerate(range(start_idx, end_idx)):
            transition = replay_buffer[i]
            obs = transition['observations']
            action = transition['actions']

            # Prepare observation for Q-value calculation
            prepared_obs = {
                'front': np.expand_dims(obs['front'], axis=0),
                'state': np.expand_dims(obs['state'], axis=0),
                'wrist': np.expand_dims(obs['wrist'], axis=0),
            }

            # Prepare action (first 6 dimensions)
            prepared_action = np.expand_dims(action[:6], axis=0)

            # Calculate Q-value
            key, rng = jax.random.split(rng)
            q_value = float(agent.forward_critic(prepared_obs, prepared_action, key).min())  # Take minimum not mean

            # Store Q-value
            episodes_data[ep_idx]['q_values'][step_idx] = q_value

print(f"Generating plots with 5 episodes per plot...")

# Generate plots with 5 episodes per plot
num_episodes = len(episodes_data)
plots_needed = (num_episodes + 4) // 5  # Ceiling division

for plot_idx in tqdm(range(plots_needed), desc="Generating group plots"):
    start_ep = plot_idx * 5
    end_ep = min(start_ep + 5, num_episodes)

    plt.figure(figsize=(15, 10))

    # Plot each episode in this group
    for i, ep_idx in enumerate(range(start_ep, end_ep)):
        episode = episodes_data[ep_idx]
        steps = episode['steps']
        q_values = episode['q_values']
        is_intervention = episode['is_intervention']

        if not steps:  # Skip if episode is empty
            continue

        # Create segments for continuous coloring
        points = np.array([steps, q_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create line collection with colors based on intervention status
        colors = np.array(['blue' if not interv else 'red' for interv in is_intervention[:-1]])
        lc = LineCollection(segments, colors=colors, linewidth=2, label=f'Episode {ep_idx+1}')

        # Add line collection to plot
        plt.gca().add_collection(lc)

        # Also plot points to show exact values
        intervention_points = [steps[j] for j in range(len(steps)) if is_intervention[j]]
        intervention_q_values = [q_values[j] for j in range(len(q_values)) if is_intervention[j]]
        normal_points = [steps[j] for j in range(len(steps)) if not is_intervention[j]]
        normal_q_values = [q_values[j] for j in range(len(q_values)) if not is_intervention[j]]

        plt.scatter(intervention_points, intervention_q_values, color='red', s=30, alpha=0.6)
        plt.scatter(normal_points, normal_q_values, color='blue', s=20, alpha=0.6)

        # Add episode label at the end of the line
        if steps:
            plt.text(steps[-1], q_values[-1], f" Ep {ep_idx+1}", fontsize=10)

    # Set plot limits and labels
    plt.xlim(0, max([max(ep['steps']) for ep in episodes_data[start_ep:end_ep] if ep['steps']]) + 5)
    y_values = [q for ep in episodes_data[start_ep:end_ep] if ep['steps'] for q in ep['q_values']]
    if y_values:  # Only set y limits if we have valid data
        y_min = min(y_values)
        y_max = max(y_values)
        margin = (y_max - y_min) * 0.1
        plt.ylim(y_min - margin, y_max + margin)

    plt.xlabel('Steps in Episode', fontsize=14)
    plt.ylabel('Q-value', fontsize=14)
    plt.title(f'Q-values Across Episodes {start_ep+1}-{end_ep}\nRed = Intervention, Blue = Regular Action', fontsize=16)
    plt.grid(alpha=0.3)

    # Add legend for intervention vs normal
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Intervention'),
        Line2D([0], [0], color='blue', lw=3, label='Regular Action')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/q_values_episodes_{start_ep+1}_to_{end_ep}.png", dpi=150)
    plt.close()

# Also generate individual plots for each episode
print(f"Generating individual plots for each episode...")

for ep_idx, episode in tqdm(enumerate(episodes_data), desc="Generating individual plots", total=len(episodes_data)):
    steps = episode['steps']
    q_values = episode['q_values']
    is_intervention = episode['is_intervention']

    if not steps:  # Skip if episode is empty
        continue

    plt.figure(figsize=(12, 8))

    # Plot step-by-step colored line
    for i in range(len(steps)-1):
        color = 'red' if is_intervention[i] else 'blue'
        plt.plot([steps[i], steps[i+1]], [q_values[i], q_values[i+1]], color=color, linewidth=2.5)

    # Add markers
    intervention_points = [steps[j] for j in range(len(steps)) if is_intervention[j]]
    intervention_q_values = [q_values[j] for j in range(len(q_values)) if is_intervention[j]]
    normal_points = [steps[j] for j in range(len(steps)) if not is_intervention[j]]
    normal_q_values = [q_values[j] for j in range(len(q_values)) if not is_intervention[j]]

    # Calculate statistics
    avg_q = np.mean(q_values)
    avg_intervention_q = np.mean(intervention_q_values) if intervention_q_values else None
    avg_normal_q = np.mean(normal_q_values) if normal_q_values else None

    # Calculate percentage of steps that were interventions
    intervention_pct = (sum(is_intervention) / len(is_intervention)) * 100 if is_intervention else 0

    # Plot scatter points
    if intervention_points:
        plt.scatter(intervention_points, intervention_q_values, color='red', s=50, alpha=0.7,
                    label=f'Intervention (avg Q: {avg_intervention_q:.3f})' if avg_intervention_q is not None else 'Intervention')
    if normal_points:
        plt.scatter(normal_points, normal_q_values, color='blue', s=40, alpha=0.7,
                   label=f'Regular Action (avg Q: {avg_normal_q:.3f})' if avg_normal_q is not None else 'Regular Action')

    plt.xlabel('Steps in Episode', fontsize=14)
    plt.ylabel('Q-value', fontsize=14)
    plt.title(f'Episode {ep_idx+1} Q-values\n{intervention_pct:.1f}% Intervention Steps', fontsize=16)
    plt.grid(alpha=0.3)

    # Add horizontal line at average Q-value
    plt.axhline(y=avg_q, color='green', linestyle='--', alpha=0.7, label=f'Avg Q-value: {avg_q:.3f}')

    # Add text annotation with episode statistics
    stats_text = (
        f"Episode length: {len(steps)} steps\n"
        f"Intervention steps: {sum(is_intervention)} ({intervention_pct:.1f}%)\n"
        f"Max Q-value: {max(q_values):.3f}\n"
        f"Min Q-value: {min(q_values):.3f}"
    )
    plt.annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction',
                fontsize=11, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/q_values_episode_{ep_idx+1}.png", dpi=150)
    plt.close()

print(f"All plots saved to {plots_dir}/")

# Display summary of all episodes
print("\nSummary of all episodes:")
total_steps = sum(len(ep['steps']) for ep in episodes_data)
total_interventions = sum(sum(ep['is_intervention']) for ep in episodes_data)
print(f"Total episodes: {len(episodes_data)}")
print(f"Total steps: {total_steps}")
print(f"Total intervention steps: {total_interventions} ({(total_interventions/total_steps)*100:.2f}%)")
print(f"Average episode length: {total_steps/len(episodes_data):.2f} steps")
print(f"Average interventions per episode: {total_interventions/len(episodes_data):.2f} steps")
# %%
