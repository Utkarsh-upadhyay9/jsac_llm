import numpy as np
import os
import matplotlib.pyplot as plt


def moving_avg(x, k=400):
    return np.convolve(x, np.ones(k)/k, mode='valid')

def plot_comparison(save_path='plots/actor_comparison.png'):
    plt.figure(figsize=(12, 7))
    agent_names = ["MLP", "LLM", "Hybrid"]
    colors = ['#1f77b4', '#2ca02c', '#d62728'] # Blue, Green, Red

    for name, color in zip(agent_names, colors):
        try:
            rewards = np.load(f'plots/{name}_rewards.npy')
            print(f"Loaded {name} rewards: {len(rewards)} episodes")
            
            # Use smaller window for moving average if data is short
            window_size = 3000
            if len(rewards) > window_size:
                smoothed_rewards = moving_avg(rewards, k=window_size)
                plt.plot(smoothed_rewards, label=f'DDPG-{name}', linewidth=2.5, color=color)
            else:
                # Plot raw data if too short for moving average
                plt.plot(rewards, label=f'DDPG-{name}', linewidth=2.5, color=color)
                
        except FileNotFoundError:
            print(f"Warning: 'plots/{name}_rewards.npy' not found. Skipping.")

    plt.xlabel('Episode', fontsize=14)
    # Include unit in brackets after the label
    plt.ylabel('Secracy Rate (bps/Hz)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to '{save_path}'")
    # Removed plt.show() for headless execution

# Generate the final comparison plot
plot_comparison()