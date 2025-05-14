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
from tqdm import tqdm
import math
from collections import defaultdict
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

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

# %% Utility Functions

def prepare_observation(obs):
    """Prepare a single observation for model input."""
    return {
        'front': np.expand_dims(obs['front'], axis=0),
        'state': np.expand_dims(obs['state'], axis=0),
        'wrist': np.expand_dims(obs['wrist'], axis=0),
    }

def prepare_observations_batch(obses):
    """Prepare a batch of observations for model input."""
    return {
        'front': np.array([obs['front'] for obs in obses]),
        'state': np.array([obs['state'] for obs in obses]),
        'wrist': np.array([obs['wrist'] for obs in obses]),
    }

def get_action(agent, obs, rng, argmax=True):
    """Get action from agent for given observation."""
    rng, key = jax.random.split(rng)
    actions = agent.sample_actions(
        observations=jax.device_put(obs),
        argmax=argmax,
        seed=key
    )
    return actions, rng

def compute_q_value(agent, obs, action, rng):
    """Compute Q-value for a given observation-action pair."""
    key, rng = jax.random.split(rng)
    q = agent.forward_critic(obs, action[:, :6], key)
    return q, rng

def is_intervention(transition):
    """Check if a transition is an intervention."""
    return ('info' in transition and
           isinstance(transition['info'], dict) and
           'intervene_action' in transition['info'])

def safe_ratio(a, b, default=1.0):
    """Calculate ratio while avoiding division by zero."""
    mask = (np.abs(b) > 1e-8)
    ratio = np.full_like(a, default, dtype=float)
    ratio[mask] = a[mask] / b[mask]
    return ratio

def log_base_099(x):
    """Calculate logarithm with base 0.99, handling non-positive values."""
    if hasattr(x, "__iter__"):
        x_safe = np.maximum(x, 1e-10)
        return np.log(x_safe) / np.log(0.99)
    elif x <= 0:
        return 0  # Handle non-positive values
    else:
        return math.log(x) / math.log(0.99)

def find_episode_boundaries(replay_buffer):
    """Find the starting indices of each episode in the replay buffer."""
    episode_indices = [0]  # Start of first episode
    for i in range(len(replay_buffer)):
        if replay_buffer[i]['dones']:
            episode_indices.append(i+1)  # Start of next episode

    # Remove the last entry if it's beyond the buffer length
    if episode_indices[-1] >= len(replay_buffer):
        episode_indices.pop()

    return episode_indices

def process_q_values_batch(agent, replay_buffer, episode_indices, rng, batch_size=1024):
    """Process Q-values for all episodes in batches."""
    # Prepare data for each episode
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
            episode_is_intervention.append(is_intervention(transition))

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

    print(f"Processing {len(all_obs_batches)} total steps in batches of {batch_size}...")

    # Process in batches
    q_values = []
    try:
        for batch_idx in tqdm(range(0, len(all_obs_batches), batch_size), desc="Computing Q-values in batches"):
            # Get current batch
            batch_end = min(batch_idx + batch_size, len(all_obs_batches))
            batch_size_actual = batch_end - batch_idx

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

            # Take minimum across critic ensemble (first dimension)
            batch_q_values_min = batch_q_values.min(axis=0).tolist()

            # Ensure we're getting one value per input
            if len(batch_q_values_min) != batch_size_actual:
                print(f"Warning: Expected {batch_size_actual} Q-values but got {len(batch_q_values_min)}")

            q_values.extend(batch_q_values_min)

        # Assign Q-values back to the episodes
        for i, (ep_idx, step_idx) in enumerate(all_batch_indices):
            if i < len(q_values):  # Safety check
                if ep_idx < len(episodes_data) and step_idx < len(episodes_data[ep_idx]['q_values']):
                    episodes_data[ep_idx]['q_values'][step_idx] = q_values[i]

    except Exception as e:
        print(f"Error during batch processing: {e}")
        # Fall back to non-batched processing if batching fails
        print("Falling back to step-by-step processing...")

        # Process each episode step by step
        for ep_idx in tqdm(range(len(episode_indices)-1), desc="Processing episodes (fallback)"):
            start_idx = episode_indices[ep_idx]
            end_idx = episode_indices[ep_idx+1]

            # Process each step in this episode
            for step_idx, i in enumerate(range(start_idx, end_idx)):
                transition = replay_buffer[i]
                obs = transition['observations']
                action = transition['actions']

                # Prepare observation and action
                prepared_obs = prepare_observation(obs)
                prepared_action = np.expand_dims(action[:6], axis=0)

                # Calculate Q-value
                key, rng = jax.random.split(rng)
                q_value = float(agent.forward_critic(prepared_obs, prepared_action, key).min())

                # Store Q-value
                episodes_data[ep_idx]['q_values'][step_idx] = q_value

    return episodes_data, rng

def plot_q_values_multiple_episodes(episodes_data, start_ep, end_ep, filename=None):
    """Plot Q-values for multiple episodes in a single figure."""
    plt.figure(figsize=(15, 10))

    # Plot each episode in this group
    for ep_idx in range(start_ep, end_ep):
        if ep_idx >= len(episodes_data):
            break

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
    valid_episodes = [ep for ep in episodes_data[start_ep:end_ep] if ep['steps']]
    if valid_episodes:
        plt.xlim(0, max([max(ep['steps']) for ep in valid_episodes]) + 5)
        y_values = [q for ep in valid_episodes for q in ep['q_values']]
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

    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_q_values_single_episode(episode, ep_idx, filename=None):
    """Plot Q-values for a single episode with detailed statistics."""
    steps = episode['steps']
    q_values = episode['q_values']
    is_intervention = episode['is_intervention']

    if not steps:  # Skip if episode is empty
        return

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

    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()

def analyze_intervention_lengths(preference_buffer):
    """Analyze and plot the distribution of intervention lengths."""
    # Extract intervention lengths
    intervention_lengths = [len(intervention['observations']) for intervention in preference_buffer]
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
    plt.hist(intervention_lengths, bins=30, color='royalblue',
             alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add vertical lines for statistics
    plt.axvline(x=intervention_lengths.mean(), color='r', linestyle='--',
               linewidth=2, label=f'Mean length: {intervention_lengths.mean():.2f}')
    plt.axvline(x=np.median(intervention_lengths), color='g', linestyle='--',
               linewidth=2, label=f'Median length: {np.median(intervention_lengths):.2f}')

    # Add percentile lines
    percentiles = [25, 75]
    for p in percentiles:
        value = np.percentile(intervention_lengths, p)
        plt.axvline(x=value, color='orange', alpha=0.5, linestyle=':',
                  linewidth=1.5, label=f'{p}th percentile: {value:.2f}')

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

    plt.tight_layout()
    plt.show()

    # Create a cumulative distribution plot
    plt.figure(figsize=(12, 8))
    counts, bin_edges = np.histogram(intervention_lengths, bins=30, density=True)
    cdf = np.cumsum(counts) * (bin_edges[1] - bin_edges[0])

    plt.plot(bin_edges[1:], cdf, 'b-', linewidth=2, label='Cumulative Distribution')

    # Add horizontal lines at key percentages
    for p in [0.25, 0.5, 0.75, 0.9]:
        plt.axhline(y=p, color='gray', alpha=0.5, linestyle='--')
        # Find the corresponding x value at this percentile
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

    return intervention_lengths

def analyze_intervention_effectiveness(agent, preference_buffer, rng):
    """Analyze the effectiveness of interventions based on Q-values."""
    # Create arrays to store the pre and post Q-values
    q_pre_values = []
    q_post_values = []
    ratios = []

    # Process all intervention pairs in the preference buffer
    for intervention in tqdm(preference_buffer, desc="Processing interventions"):
        # Get the pre-intervention state and action (first state in sequence)
        pre_obs = prepare_observation(intervention['observations'][0])
        pre_action = np.expand_dims(intervention['actions'][0][:6], axis=0)

        # Get the post-intervention state and action (last state in sequence)
        post_obs = prepare_observation(intervention['observations'][-1])
        post_action = np.expand_dims(intervention['actions'][-1][:6], axis=0)

        # Calculate Q-values
        q_pre, rng = compute_q_value(agent, pre_obs, pre_action, rng)
        q_post, rng = compute_q_value(agent, post_obs, post_action, rng)

        # Extract the values
        q_pre_value = float(q_pre.mean())
        q_post_value = float(q_post.mean())

        # Store the values
        q_pre_values.append(q_pre_value)
        q_post_values.append(q_post_value)

        # Calculate and store the ratio (with protection against division by zero)
        ratio = safe_ratio(np.array([q_post_value]), np.array([q_pre_value]))[0]
        ratios.append(ratio)

    # Convert to numpy arrays for analysis
    q_pre_values = np.array(q_pre_values)
    q_post_values = np.array(q_post_values)
    ratios = np.array(ratios)

    # Print statistics
    print(f"Number of interventions analyzed: {len(ratios)}")
    print(f"Q-pre values - min: {q_pre_values.min():.4f}, max: {q_pre_values.max():.4f}, mean: {q_pre_values.mean():.4f}")
    print(f"Q-post values - min: {q_post_values.min():.4f}, max: {q_post_values.max():.4f}, mean: {q_post_values.mean():.4f}")
    print(f"Ratios (Q-post/Q-pre) - min: {ratios.min():.4f}, max: {ratios.max():.4f}, mean: {ratios.mean():.4f}")
    print(f"Percentage of interventions with improved Q-values: {(ratios > 1.0).mean() * 100:.1f}%")

    # Plot histogram of ratios
    plt.figure(figsize=(12, 8))
    ratios_clipped = np.clip(ratios, 0.5, 2.0)
    plt.hist(ratios_clipped, bins=50, alpha=0.8, color='royalblue',
             edgecolor='black', linewidth=0.5)
    plt.axvline(x=1.0, color='r', linestyle='--', linewidth=2,
                label='No change (ratio = 1.0)')
    plt.xlabel('Q-value Ratio (Q_post / Q_pre)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Q-value Ratios After Intervention', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    # Add annotation showing improvement percentage
    improved_pct = (ratios > 1.0).mean() * 100
    plt.annotate(f"Interventions with improved Q-value: {improved_pct:.1f}%",
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=14, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
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
    plt.annotate(f"Points above line: {improved_pct:.1f}% (improved Q-value)",
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=14, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    plt.tight_layout()
    plt.show()

    return q_pre_values, q_post_values, ratios

def analyze_optimality_metrics(agent, preference_buffer, rng, offset=10):
    """Calculate and analyze optimality metrics for interventions."""
    # Create arrays to store the values
    q_pre_values = []
    q_post_values = []
    intervention_lengths = []
    optimality_metrics = []

    # Process all interventions
    for intervention in tqdm(preference_buffer, desc="Calculating optimality metrics"):
        # Get intervention length (k)
        k = len(intervention['observations'])
        intervention_lengths.append(k)

        # Get the pre-intervention state and action (first state in sequence)
        pre_obs = prepare_observation(intervention['observations'][0])
        pre_action = np.expand_dims(intervention['actions'][0][:6], axis=0)

        # Get the post-intervention state and action (last state in sequence)
        post_obs = prepare_observation(intervention['observations'][-1])
        post_action = np.expand_dims(intervention['actions'][-1][:6], axis=0)

        # Calculate Q-values
        q_pre, rng = compute_q_value(agent, pre_obs, pre_action, rng)
        q_post, rng = compute_q_value(agent, post_obs, post_action, rng)

        # Extract the values
        q_pre_value = float(q_pre.mean())
        q_post_value = float(q_post.mean())

        q_pre_values.append(q_pre_value)
        q_post_values.append(q_post_value)

        # Calculate the optimality metric
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
    q_pre_values = np.array(q_pre_values)
    q_post_values = np.array(q_post_values)

    # Print statistics
    print(f"Number of interventions analyzed: {len(optimality_metrics)}")
    print(f"Optimality metrics - min: {optimality_metrics.min():.6f}, max: {optimality_metrics.max():.6f}, mean: {optimality_metrics.mean():.6f}")
    print(f"Intervention lengths - min: {intervention_lengths.min()}, max: {intervention_lengths.max()}, mean: {intervention_lengths.mean():.2f}")

    # Create a histogram of the optimality metrics
    plt.figure(figsize=(12, 8))
    lower_clip = np.percentile(optimality_metrics, 1)
    upper_clip = np.percentile(optimality_metrics, 99)
    optimality_metrics_clipped = np.clip(optimality_metrics, lower_clip, upper_clip)

    plt.hist(optimality_metrics_clipped, bins=50, alpha=0.8, color='teal',
             edgecolor='black', linewidth=0.5)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Neutral (0)')
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

    return optimality_metrics, intervention_lengths, q_pre_values, q_post_values

def analyze_when2intervene(agent, replay_buffer, rng, threshold=0.98, max_samples=1000):
    """Analyze when2intervene method accuracy using pre-intervention and non-intervention states."""
    # Find pre-intervention states (states right before an intervention)
    pre_intervention_states = []
    non_intervention_states = []

    # Identify pre-intervention states (the state right before an intervention starts)
    for i in range(len(replay_buffer)-1):
        cur_transition = replay_buffer[i]
        next_transition = replay_buffer[i+1]

        # Check if the current state is not an intervention but the next one is
        cur_has_intervention = is_intervention(cur_transition)
        next_has_intervention = is_intervention(next_transition)

        if not cur_has_intervention and next_has_intervention:
            # This is a pre-intervention state
            pre_intervention_states.append(cur_transition['observations'])
        elif not cur_has_intervention:
            # This is a regular non-intervention state
            non_intervention_states.append(cur_transition['observations'])

    print(f"Found {len(pre_intervention_states)} pre-intervention states and {len(non_intervention_states)} regular non-intervention states")

    if len(pre_intervention_states) == 0 or len(non_intervention_states) == 0:
        print("Not enough pre-intervention or non-intervention states for evaluation")
        return None

    # Sample a balanced subset if needed
    pre_intervention_sample_indices = np.random.choice(len(pre_intervention_states),
                                                     min(max_samples, len(pre_intervention_states)),
                                                     replace=False)
    non_intervention_sample_indices = np.random.choice(len(non_intervention_states),
                                                     min(max_samples, len(non_intervention_states)),
                                                     replace=False)

    pre_intervention_states_sample = [pre_intervention_states[i] for i in pre_intervention_sample_indices]
    non_intervention_states_sample = [non_intervention_states[i] for i in non_intervention_sample_indices]

    # Prepare observations
    pre_intervention_obs = prepare_observations_batch(pre_intervention_states_sample)
    non_intervention_obs = prepare_observations_batch(non_intervention_states_sample)

    # Get expert actions (argmax=True)
    pre_intervention_expert_action, rng = get_action(agent, pre_intervention_obs, rng, argmax=True)
    non_intervention_expert_action, rng = get_action(agent, non_intervention_obs, rng, argmax=True)

    # Get policy actions (argmax=False)
    rng, policy_key1 = jax.random.split(rng)
    pre_intervention_policy_action = agent.sample_actions(
        observations=jax.device_put(pre_intervention_obs),
        argmax=False,  # Sample from policy distribution
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
    q_pre_intervention_expert, rng = compute_q_value(agent, pre_intervention_obs, pre_intervention_expert_action, rng)
    q_pre_intervention_policy, rng = compute_q_value(agent, pre_intervention_obs, pre_intervention_policy_action, rng)
    q_non_intervention_expert, rng = compute_q_value(agent, non_intervention_obs, non_intervention_expert_action, rng)
    q_non_intervention_policy, rng = compute_q_value(agent, non_intervention_obs, non_intervention_policy_action, rng)

    # Debug: Print Q-value statistics
    print(f"\nQ-Value Statistics:")
    print(f"Pre-Intervention Expert Q-values: min={q_pre_intervention_expert.min():.4f}, max={q_pre_intervention_expert.max():.4f}, mean={q_pre_intervention_expert.mean():.4f}")
    print(f"Pre-Intervention Policy Q-values: min={q_pre_intervention_policy.min():.4f}, max={q_pre_intervention_policy.max():.4f}, mean={q_pre_intervention_policy.mean():.4f}")
    print(f"Non-Intervention Expert Q-values: min={q_non_intervention_expert.min():.4f}, max={q_non_intervention_expert.max():.4f}, mean={q_non_intervention_expert.mean():.4f}")
    print(f"Non-Intervention Policy Q-values: min={q_non_intervention_policy.min():.4f}, max={q_non_intervention_policy.max():.4f}, mean={q_non_intervention_policy.mean():.4f}")

    # Calculate ratios (avoid division by zero)
    pre_intervention_ratios = safe_ratio(q_pre_intervention_policy, q_pre_intervention_expert)
    non_intervention_ratios = safe_ratio(q_non_intervention_policy, q_non_intervention_expert)

    # Calculate average ratios
    avg_pre_intervention_ratio = pre_intervention_ratios.mean()
    avg_non_intervention_ratio = non_intervention_ratios.mean()

    print(f"\nRatio Analysis (Q(s, policy action) / Q(s, expert action)):")
    print(f"Average ratio at pre-intervention states: {avg_pre_intervention_ratio:.4f}")
    print(f"Average ratio at non-intervention states: {avg_non_intervention_ratio:.4f}")

    # Calculate when2intervene decision using the threshold
    when2intervene_pre = (pre_intervention_ratios < threshold).mean()
    when2intervene_non = (non_intervention_ratios < threshold).mean()

    print(f"\nWhen2Intervene Method (ratio < {threshold}):")
    print(f"Percentage recommending intervention at pre-intervention states: {when2intervene_pre:.4f}")
    print(f"Percentage recommending intervention at non-intervention states: {when2intervene_non:.4f}")

    # Ensure ratios are flattened to 1D arrays
    pre_intervention_ratios_flat = pre_intervention_ratios.flatten()
    non_intervention_ratios_flat = non_intervention_ratios.flatten()

    # Calculate stats with flattened arrays
    avg_pre_intervention_ratio = pre_intervention_ratios_flat.mean()
    avg_non_intervention_ratio = non_intervention_ratios_flat.mean()
    when2intervene_pre = (pre_intervention_ratios_flat < threshold).mean()
    when2intervene_non = (non_intervention_ratios_flat < threshold).mean()

    # Clip extreme values for better visualization (focused on the range of interest)
    pre_intervention_ratios_clip = np.clip(pre_intervention_ratios_flat, 0.95, 1.05)
    non_intervention_ratios_clip = np.clip(non_intervention_ratios_flat, 0.95, 1.05)

    # Set up the figure and axes for histogram
    plt.figure(figsize=(12, 6))
    bins = np.linspace(0.95, 1.05, 40)  # Focused bins between 0.95 and 1.05

    plt.hist(pre_intervention_ratios_clip, bins=bins, alpha=0.7,
             label=f'Pre-intervention states (mean={avg_pre_intervention_ratio:.3f})')
    plt.hist(non_intervention_ratios_clip, bins=bins, alpha=0.7,
             label=f'Non-intervention states (mean={avg_non_intervention_ratio:.3f})')

    # Add vertical line at threshold
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'When2Intervene threshold ({threshold})')

    # Add labels and title
    plt.xlabel('Q-value Ratio (Q(s, policy action) / Q(s, expert action))', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Q-value Ratios (Clipped to 0.95-1.05 Range)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlim(0.95, 1.05)

    # Add text annotation showing percentages below threshold
    plt.annotate(f"Pre-intervention states below threshold: {when2intervene_pre:.1%}",
                xy=(threshold, plt.gca().get_ylim()[1]*0.95),
                xytext=(threshold+0.01, plt.gca().get_ylim()[1]*0.95),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    plt.annotate(f"Non-intervention states below threshold: {when2intervene_non:.1%}",
                xy=(threshold, plt.gca().get_ylim()[1]*0.85),
                xytext=(threshold+0.01, plt.gca().get_ylim()[1]*0.85),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    plt.tight_layout()
    plt.show()

    # Optional: Create a second figure with separate subplots for clearer comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pre-intervention ratios histogram
    ax1.hist(pre_intervention_ratios_clip, bins=bins, alpha=0.7)
    ax1.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    ax1.set_title(f'Pre-intervention States\nMean: {avg_pre_intervention_ratio:.3f}, Below threshold: {when2intervene_pre:.1%}')
    ax1.set_xlabel('Q-value Ratio (clipped)')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(0.95, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Non-intervention ratios histogram
    ax2.hist(non_intervention_ratios_clip, bins=bins, alpha=0.7)
    ax2.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    ax2.set_title(f'Non-intervention States\nMean: {avg_non_intervention_ratio:.3f}, Below threshold: {when2intervene_non:.1%}')
    ax2.set_xlabel('Q-value Ratio (clipped)')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim(0.95, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'pre_intervention_ratios': pre_intervention_ratios_flat,
        'non_intervention_ratios': non_intervention_ratios_flat,
        'when2intervene_pre': when2intervene_pre,
        'when2intervene_non': when2intervene_non
    }

def analyze_intervention_trend(preference_buffer, bin_size=10):
    """Analyze how intervention lengths change over time."""
    # Use original order as proxy for time if no timestamps are available
    sorted_indices = np.arange(len(preference_buffer))

    # Fixed-size bins for grouping interventions
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
    plt.xlabel(f'Intervention Index (grouped by bins of size {bin_size})', fontsize=14)
    plt.ylabel('Average Intervention Length (steps)', fontsize=14)
    plt.title('Trend of Intervention Lengths Over Time', fontsize=16)
    plt.grid(alpha=0.3)

    # Add smoothed trend line
    if len(bin_centers) > 5:
        try:
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

    # Plot a rolling average of intervention lengths
    intervention_lengths = [len(preference_buffer[i]['observations']) for i in sorted_indices]

    # Create a rolling average
    window_size = min(20, len(intervention_lengths) // 4)  # Adjust based on number of interventions
    if window_size < 2:
        window_size = 2

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

    return bin_centers, bin_avg_lengths, intervention_lengths

def analyze_constraints(agent, preference_buffer, rng):
    """Analyze constraint satisfaction for interventions."""
    pre_intervention_obs = [p['observations'][0] for p in preference_buffer]
    intervene_actions = [p['actions'][0] for p in preference_buffer]
    policy_actions = [p['policy_actions'][0] for p in preference_buffer]
    post_intervention_obs = [p['observations'][-1] for p in preference_buffer]

    pre_intervention_obs = prepare_observations_batch(pre_intervention_obs)
    post_intervention_obs = prepare_observations_batch(post_intervention_obs)

    # Get expert actions for pre and post observations
    pre_intervention_expert_action, rng = get_action(agent, pre_intervention_obs, rng)
    post_intervention_expert_action, rng = get_action(agent, post_intervention_obs, rng)

    policy_actions = np.array(policy_actions)
    intervene_actions = np.array(intervene_actions)

    # Compute Q-values for analysis
    q_pre_expert, rng = compute_q_value(agent, pre_intervention_obs, pre_intervention_expert_action, rng)
    q_post_expert, rng = compute_q_value(agent, post_intervention_obs, post_intervention_expert_action, rng)
    q_pre_policy, rng = compute_q_value(agent, pre_intervention_obs, policy_actions[:, :6], rng)
    q_pre_intervene, rng = compute_q_value(agent, pre_intervention_obs, intervene_actions[:, :6], rng)

    # Calculate constraint satisfaction metrics
    constraint1_acc = ((q_pre_expert - q_post_expert) < 0).mean()
    qvalue_based_learning_intervene = ((q_pre_policy - q_pre_intervene) < 0).mean()
    qvalue_based_learning_expert = ((q_pre_policy / q_pre_expert) < 0.98).mean()
    constraint2_acc = ((q_pre_intervene - q_post_expert) < 0).mean()

    print(f"Constraint 1 accuracy: {constraint1_acc}")
    print(f"Q-value based learning intervene accuracy: {qvalue_based_learning_intervene}")
    print(f"Q-value based when2intervene accuracy (pre-intervention states): {qvalue_based_learning_expert}")
    print(f"Constraint 2 accuracy: {constraint2_acc}")

    return {
        'constraint1_acc': constraint1_acc,
        'qvalue_based_learning_intervene': qvalue_based_learning_intervene,
        'qvalue_based_learning_expert': qvalue_based_learning_expert,
        'constraint2_acc': constraint2_acc
    }

# %%
# Setup paths and configuration
checkpoint_path = "/Users/pranavnt/jax-hitl-hil-serl/examples/experiments/franka_sim/sriyash_runs/hil-bc-coeff0.1-decay"
plots_dir = "q_value_plots"
os.makedirs(plots_dir, exist_ok=True)

# Define training configuration
from experiments.config import DefaultTrainingConfig
class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front", "wrist"]
    classifier_keys = ["front", "wrist"]
    proprio_keys = ['panda/tcp_pos', 'panda/tcp_vel', 'panda/gripper_pos']
    pretraining_steps = 0
    reward_scale = 1
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
# Initialize agent and load checkpoint
intervene_steps = 0  # Default number of steps between pre and post intervention states
constraint_eps = 0.1  # Default constraint epsilon

# Sample observation and action for model initialization
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

# Initialize agent
agent = make_sac_pixel_agent_hybrid_single_arm(
    seed=0,
    sample_obs=obs_sample,
    sample_action=action_sample,
    image_keys=config.image_keys,
    encoder_type=config.encoder_type,
    discount=config.discount,
    enable_cl=False,
    cl_config=cl_config,
)

# Load the latest checkpoint
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
# Load replay buffer
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
# Load preference buffer (interventions)
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

# Initialize random key
rng = jax.random.PRNGKey(0)

# %%
# Analyze constraint satisfaction
print("Analyzing constraint satisfaction...")
constraint_metrics = analyze_constraints(agent, preference_buffer, rng)

# %%
# Analyze when2intervene method
print("\nAnalyzing when2intervene accuracy...")
when2intervene_results = analyze_when2intervene(agent, replay_buffer, rng)

# %%
# Analyze intervention lengths
print("\nAnalyzing intervention lengths...")
intervention_lengths = analyze_intervention_lengths(preference_buffer)

# %%
# Analyze intervention effectiveness based on Q-values
print("\nAnalyzing intervention effectiveness...")
q_pre_values, q_post_values, ratios = analyze_intervention_effectiveness(agent, preference_buffer, rng)

# %%
# Analyze optimality metrics
print("\nAnalyzing optimality metrics...")
optimality_metrics, intervention_lens, q_pre_vals, q_post_vals = analyze_optimality_metrics(agent, preference_buffer, rng)

# %%
# Analyze intervention trends over time
print("\nAnalyzing intervention trends over time...")
bin_centers, bin_avg_lengths, intervention_lengths_raw = analyze_intervention_trend(preference_buffer)

# %%
# Visualize Q-values across trajectories
print("\nProcessing Q-values for all episodes...")
episode_indices = find_episode_boundaries(replay_buffer)
episodes_data, rng = process_q_values_batch(agent, replay_buffer, episode_indices, rng)

# Generate plots with episodes grouped
print(f"\nGenerating Q-value plots with episodes grouped...")
num_episodes = len(episodes_data)
plots_needed = (num_episodes + 4) // 5  # Ceiling division

for plot_idx in tqdm(range(plots_needed), desc="Generating group plots"):
    start_ep = plot_idx * 5
    end_ep = min(start_ep + 5, num_episodes)
    filename = f"{plots_dir}/q_values_episodes_{start_ep+1}_to_{end_ep}.png"
    plot_q_values_multiple_episodes(episodes_data, start_ep, end_ep, filename)

# %%
