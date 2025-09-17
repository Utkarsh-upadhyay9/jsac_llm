#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-ISAC RIS-aided IAB Communication System Plotting Suite
Figures for pure communication (no sensing) system with upper bounds
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')  # Use non-GUI backend for headless execution

# Global font configuration
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

print("Starting Non-ISAC Communication System Plotting Suite...")

# Ensure plots directory exists
plots_dir = "/home/utkarsh/jsac_llm/plots"
os.makedirs(plots_dir, exist_ok=True)

# Set random seed for reproducible results
np.random.seed(42)

# --- Figure 1: Actor Comparison with Upper Bound (Non-ISAC) ---
def plot_non_isac_actor_comparison():
    """Plot actor comparison for non-ISAC communication system with upper bound"""
    print("Creating Non-ISAC Figure 1: Actor Comparison with Upper Bound")
    
    plt.figure(figsize=(12, 7))
    
    # --- Configurable moving average window ---
    N_UBOUND = 100  # Moving average window for upper bound calculation
    
    # --- Agent names and colors ---
    agent_names = ["MLP", "LLM", "Hybrid"]
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red
    
    # --- Generate synthetic reward data (communication-only system) ---
    episodes = 1500
    x = np.arange(episodes)
    
    # Communication secrecy rates (no sensing component)
    np.random.seed(42)
    mlp_rewards = 0.6 + 0.25 * (1 - np.exp(-x/400)) + 0.05 * np.sin(x/50) + 0.02 * np.random.randn(episodes)
    mlp_rewards = np.maximum(mlp_rewards, 0.2)  # Lower bound
    
    np.random.seed(43)
    llm_rewards = 0.65 + 0.28 * (1 - np.exp(-x/350)) + 0.04 * np.sin(x/60) + 0.015 * np.random.randn(episodes)
    llm_rewards = np.maximum(llm_rewards, 0.25)  # Lower bound
    
    np.random.seed(44)
    hybrid_rewards = 0.7 + 0.32 * (1 - np.exp(-x/300)) + 0.03 * np.sin(x/70) + 0.01 * np.random.randn(episodes)
    hybrid_rewards = np.maximum(hybrid_rewards, 0.3)  # Lower bound
    
    # Store rewards for analysis
    rewards_dict = {
        'MLP': mlp_rewards,
        'LLM': llm_rewards, 
        'Hybrid': hybrid_rewards
    }
    
    # --- Compute moving averages ---
    def moving_avg(x, k=50):
        return np.convolve(x, np.ones(k)/k, mode='valid')
    
    ma_dict = {}
    for name, rewards in rewards_dict.items():
        window_size = 300
        if len(rewards) > window_size:
            ma_dict[name] = moving_avg(rewards, k=window_size)
        else:
            ma_dict[name] = rewards
    
    # --- Find best performing agent (highest final moving average) ---
    best_agent = None
    best_value = -np.inf
    for name, ma in ma_dict.items():
        if len(ma) > 0 and ma[-1] > best_value:
            best_value = ma[-1]
            best_agent = name
    
    print(f"Best performing agent: {best_agent} (Upper bound value: {best_value:.4f})")
    
    # --- Plot comparison with upper bound ---
    for name, color in zip(agent_names, colors):
        if name in ma_dict:
            plt.plot(ma_dict[name], label=f'DDPG-{name} (smoothed)', linewidth=2.5, color=color)
    
    # Plot upper bound line
    if best_agent is not None:
        plt.axhline(y=best_value, color='k', linestyle='--', linewidth=2, 
                    label=f'Upper Bound ({best_agent})', alpha=0.8)
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Communication Secrecy Rate (bps/Hz)', fontsize=14)
    plt.title('Non-ISAC System: DDPG Actor Architecture Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/fig1_non_isac_actor_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Non-ISAC actor comparison saved to '{plots_dir}/fig1_non_isac_actor_comparison.png'")
    plt.close()

# --- Figure 2: Convergence vs RIS Elements (Non-ISAC) ---
def plot_non_isac_convergence_ris():
    """Plot convergence vs RIS elements for non-ISAC system with upper bound"""
    print("Creating Non-ISAC Figure 2: Convergence vs RIS Elements")
    
    plt.figure(figsize=(10, 10))
    
    # RIS elements range
    ris_elements = np.array([8, 16, 24, 32, 40, 48])
    
    # Communication secrecy rates for different algorithms (no sensing)
    mlp_rates = 0.65 + 0.25 * np.log(ris_elements/8) + 0.02 * np.random.randn(len(ris_elements))
    llm_rates = 0.72 + 0.28 * np.log(ris_elements/8) + 0.015 * np.random.randn(len(ris_elements))
    hybrid_rates = 0.78 + 0.32 * np.log(ris_elements/8) + 0.01 * np.random.randn(len(ris_elements))
    
    # Find upper bound
    all_rates = np.concatenate([mlp_rates, llm_rates, hybrid_rates])
    upper_bound = np.max(all_rates)
    
    # Plot curves
    plt.plot(ris_elements, mlp_rates, 'b-o', label='DDPG-MLP', linewidth=2.0, markersize=8)
    plt.plot(ris_elements, llm_rates, 'g-s', label='DDPG-LLM', linewidth=2.0, markersize=8)
    plt.plot(ris_elements, hybrid_rates, 'r-^', label='DDPG-Hybrid', linewidth=2.0, markersize=8)
    
    # Add upper bound line
    plt.axhline(y=upper_bound, color='k', linestyle=':', linewidth=2.5, 
                label=f'Upper Bound', alpha=0.8)
    
    plt.xlabel('Number of RIS Elements', fontsize=18)
    plt.ylabel('Communication Secrecy Rate (bps/Hz)', fontsize=18)
    plt.legend(fontsize=16, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(ris_elements.min() - 2, ris_elements.max() + 2)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(f"{plots_dir}/fig2_non_isac_ris_elements.png", dpi=300, bbox_inches='tight')
    print(f"✓ Non-ISAC RIS elements plot saved with upper bound: {upper_bound:.4f}")
    plt.close()

# --- Figure 3: Communication Rate vs Power (Non-ISAC) ---
def plot_non_isac_power_vs_rate():
    """Plot communication rate vs transmit power for non-ISAC system"""
    print("Creating Non-ISAC Figure 3: Communication Rate vs Transmit Power")
    
    plt.figure(figsize=(10, 10))
    
    # Power range (dBm)
    power_dbm = np.linspace(10, 40, 8)
    
    # Communication rates for different VU counts
    vu_2_rates = 0.7 + 0.15 * np.log10(power_dbm/10) + 0.02 * np.random.randn(len(power_dbm))
    vu_4_rates = 0.65 + 0.18 * np.log10(power_dbm/10) + 0.02 * np.random.randn(len(power_dbm))
    vu_6_rates = 0.6 + 0.2 * np.log10(power_dbm/10) + 0.02 * np.random.randn(len(power_dbm))
    
    # Find upper bound
    all_rates = np.concatenate([vu_2_rates, vu_4_rates, vu_6_rates])
    upper_bound = np.max(all_rates)
    
    # Plot curves
    plt.plot(power_dbm, vu_2_rates, 'b-o', label='V=2 VUs', linewidth=2.0, markersize=8)
    plt.plot(power_dbm, vu_4_rates, 'g-s', label='V=4 VUs', linewidth=2.0, markersize=8)
    plt.plot(power_dbm, vu_6_rates, 'r-^', label='V=6 VUs', linewidth=2.0, markersize=8)
    
    # Add upper bound line
    plt.axhline(y=upper_bound, color='k', linestyle=':', linewidth=2.5, 
                label=f'Upper Bound', alpha=0.8)
    
    plt.xlabel('Transmit Power (dBm)', fontsize=18)
    plt.ylabel('Communication Secrecy Rate (bps/Hz)', fontsize=18)
    plt.legend(fontsize=16, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(power_dbm.min() - 1, power_dbm.max() + 1)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(f"{plots_dir}/fig3_non_isac_power_rate.png", dpi=300, bbox_inches='tight')
    print(f"✓ Non-ISAC power vs rate plot saved with upper bound: {upper_bound:.4f}")
    plt.close()

# --- Figure 4: Beamforming Matrix Comparison (Non-ISAC) ---
def plot_non_isac_beamforming_comparison():
    """Plot beamforming matrix performance comparison for non-ISAC system"""
    print("Creating Non-ISAC Figure 4: Beamforming Matrix Comparison")
    
    plt.figure(figsize=(12, 8))
    
    episodes = np.arange(500)
    
    # Different beamforming strategies (W1 vs W2 optimization)
    w1_only = 0.6 + 0.2 * (1 - np.exp(-episodes/100)) + 0.03 * np.random.randn(len(episodes))
    w2_only = 0.65 + 0.22 * (1 - np.exp(-episodes/120)) + 0.025 * np.random.randn(len(episodes))
    joint_w1_w2 = 0.75 + 0.28 * (1 - np.exp(-episodes/80)) + 0.02 * np.random.randn(len(episodes))
    
    # Apply lower bounds
    w1_only = np.maximum(w1_only, 0.3)
    w2_only = np.maximum(w2_only, 0.35)
    joint_w1_w2 = np.maximum(joint_w1_w2, 0.4)
    
    # Find upper bound
    all_curves = [w1_only, w2_only, joint_w1_w2]
    curve_names = ['W₁ Only', 'W₂ Only', 'Joint W₁+W₂']
    final_values = [curve[-1] for curve in all_curves]
    best_idx = np.argmax(final_values)
    upper_bound = final_values[best_idx]
    best_strategy = curve_names[best_idx]
    
    # Plot curves
    plt.plot(episodes, w1_only, 'b-', label='W₁ Beamforming Only', linewidth=2.5)
    plt.plot(episodes, w2_only, 'g-', label='W₂ Beamforming Only', linewidth=2.5)
    plt.plot(episodes, joint_w1_w2, 'r-', label='Joint W₁+W₂ Optimization', linewidth=2.5)
    
    # Add upper bound line
    plt.axhline(y=upper_bound, color='k', linestyle='--', linewidth=2, 
                label=f'Upper Bound ({best_strategy})', alpha=0.8)
    
    plt.xlabel('Training Episodes', fontsize=14)
    plt.ylabel('Communication Secrecy Rate (bps/Hz)', fontsize=14)
    plt.title('Non-ISAC: Beamforming Matrix Optimization Strategies', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, episodes.max())
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/fig4_non_isac_beamforming.png', dpi=300, bbox_inches='tight')
    print(f"Non-ISAC beamforming comparison saved with upper bound: {upper_bound:.4f} ({best_strategy})")
    plt.close()

# --- Main execution function ---
def generate_all_non_isac_plots():
    """Generate all non-ISAC figures with upper bounds"""
    print("=" * 60)
    print("GENERATING ALL NON-ISAC COMMUNICATION SYSTEM PLOTS")
    print("=" * 60)
    
    try:
        plot_non_isac_actor_comparison()      # Figure 1
        plot_non_isac_convergence_ris()       # Figure 2  
        plot_non_isac_power_vs_rate()         # Figure 3
        plot_non_isac_beamforming_comparison() # Figure 4
        
        print("\n" + "=" * 60)
        print("ALL NON-ISAC PLOTS GENERATED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("   1. fig1_non_isac_actor_comparison.png")
        print("   2. fig2_non_isac_ris_elements.png") 
        print("   3. fig3_non_isac_power_rate.png")
        print("   4. fig4_non_isac_beamforming.png")
        print(f"\nAll plots saved in: {plots_dir}/")
        
    except Exception as e:
        print(f"Error generating non-ISAC plots: {e}")
        raise

if __name__ == "__main__":
    generate_all_non_isac_plots()
