#!/usr/bin/env python3
"""
Active RIS + DAM Performance vs Basic RIS Comparison
Shows the improvement from Active/Passive switching and DAM capabilities
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')
from jsac_active_ris_dam import *

# Generate performance data for different RIS configurations
def compare_ris_configurations():
    env = ActiveRISISACEnvironment()
    
    # Test configurations
    configs = {
        'Basic Passive RIS': {
            'active_mask': np.zeros(N, dtype=bool),
            'gains_db': np.zeros(N),
            'delays_ns': np.zeros(N)
        },
        'Active RIS (No DAM)': {
            'active_mask': np.ones(N//2, dtype=bool).tolist() + [False] * (N - N//2),
            'gains_db': [5.0] * (N//2) + [0.0] * (N - N//2),
            'delays_ns': np.zeros(N)
        },
        'Active RIS + DAM': {
            'active_mask': np.ones(N//2, dtype=bool).tolist() + [False] * (N - N//2),
            'gains_db': [7.0] * (N//2) + [0.0] * (N - N//2),
            'delays_ns': [25.0] * (N//2) + [0.0] * (N - N//2)
        },
        'Full Active + DAM': {
            'active_mask': np.ones(N, dtype=bool),
            'gains_db': [3.0] * N,  # Lower gain to stay within power budget
            'delays_ns': np.linspace(5, 45, N)
        }
    }
    
    # Performance metrics
    results = {}
    
    for config_name, config in configs.items():
        # Random beamforming weights for testing
        W_tau = (np.random.randn(M, V) + 1j * np.random.randn(M, V)) / np.sqrt(2)
        W_o = (np.random.randn(M, V) + 1j * np.random.randn(M, V)) / np.sqrt(2)
        
        # Normalize power
        for v in range(V):
            total_power = np.sum(np.abs(W_tau[:, v])**2) + np.sum(np.abs(W_o[:, v])**2)
            if total_power > 1.0:
                scale = np.sqrt(1.0 / total_power)
                W_tau[:, v] *= scale
                W_o[:, v] *= scale
        
        # Add random phases
        phases = np.random.uniform(0, 2*np.pi, N)
        full_config = {**config, 'phases': phases}
        
        # Compute metrics
        comm_rates = []
        secrecy_rates = []
        power_consumption = []
        
        for trial in range(10):  # Average over multiple trials
            # Randomize channels for each trial
            global base_channels, H_br, h_ru, h_re, h_direct_vu, h_direct_eve
            base_channels = generate_enhanced_channels()
            H_br = base_channels['H_br']
            h_ru = base_channels['h_ru']
            h_re = base_channels['h_re']
            h_direct_vu = base_channels['h_direct_vu']
            h_direct_eve = base_channels['h_direct_eve']
            
            # Communication rate
            comm_rate = 0
            for v in range(V):
                snr_v = compute_dam_enhanced_snr(full_config, W_tau, W_o, v)
                rate_v = np.log2(1 + snr_v)
                comm_rate += rate_v
            
            # Secrecy rate
            eve_snr = compute_dam_eavesdropper_sinr(full_config, W_tau, W_o)
            eve_rate = np.log2(1 + eve_snr)
            secrecy_rate = max(0, comm_rate - eve_rate)
            
            # Power consumption
            ris_array.configure_elements(full_config)
            power = ris_array.get_power_consumption()
            
            comm_rates.append(comm_rate)
            secrecy_rates.append(secrecy_rate)
            power_consumption.append(power)
        
        results[config_name] = {
            'communication_rate': np.mean(comm_rates),
            'communication_std': np.std(comm_rates),
            'secrecy_rate': np.mean(secrecy_rates),
            'secrecy_std': np.std(secrecy_rates),
            'power_consumption': np.mean(power_consumption),
            'power_std': np.std(power_consumption)
        }
    
    return results

# Generate the comparison data
print("Generating Active RIS vs Basic RIS comparison data...")
results = compare_ris_configurations()

# Create the plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Active RIS + DAM vs Basic RIS Performance Comparison', fontsize=16, fontweight='bold')

configs = list(results.keys())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# Communication Rate Comparison
comm_rates = [results[config]['communication_rate'] for config in configs]
comm_stds = [results[config]['communication_std'] for config in configs]
bars1 = ax1.bar(configs, comm_rates, yerr=comm_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Communication Rate Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Sum Rate (bits/s/Hz)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, rate, std in zip(bars1, comm_rates, comm_stds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
             f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')

# Secrecy Rate Comparison
secrecy_rates = [results[config]['secrecy_rate'] for config in configs]
secrecy_stds = [results[config]['secrecy_std'] for config in configs]
bars2 = ax2.bar(configs, secrecy_rates, yerr=secrecy_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax2.set_title('Secrecy Rate Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Secrecy Rate (bits/s/Hz)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Add value labels
for bar, rate, std in zip(bars2, secrecy_rates, secrecy_stds):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
             f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')

# Power Consumption Comparison
power_consumptions = [results[config]['power_consumption'] * 1000 for config in configs]  # Convert to mW
power_stds = [results[config]['power_std'] * 1000 for config in configs]
bars3 = ax3.bar(configs, power_consumptions, yerr=power_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax3.set_title('Power Consumption Comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('Power Consumption (mW)', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Add value labels
for bar, power, std in zip(bars3, power_consumptions, power_stds):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + std + 1,
             f'{power:.1f}', ha='center', va='bottom', fontweight='bold')

# Energy Efficiency (Rate per mW)
efficiency = []
efficiency_std = []
for config in configs:
    if results[config]['power_consumption'] > 0:
        eff = results[config]['communication_rate'] / (results[config]['power_consumption'] * 1000)
        # Propagate uncertainty
        eff_std = eff * np.sqrt((results[config]['communication_std']/results[config]['communication_rate'])**2 + 
                               (results[config]['power_std']/results[config]['power_consumption'])**2)
    else:
        eff = float('inf')  # Infinite efficiency for zero power
        eff_std = 0
    efficiency.append(eff)
    efficiency_std.append(eff_std)

# Handle infinite efficiency for plotting
efficiency_plot = [min(eff, 100) if not np.isinf(eff) else 100 for eff in efficiency]
bars4 = ax4.bar(configs, efficiency_plot, yerr=efficiency_std, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax4.set_title('Energy Efficiency Comparison', fontsize=14, fontweight='bold')
ax4.set_ylabel('Rate per Power (bits/s/Hz/mW)', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# Add value labels
for bar, eff, std in zip(bars4, efficiency, efficiency_std):
    height = bar.get_height()
    if np.isinf(eff):
        ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                 '∞', ha='center', va='bottom', fontweight='bold', fontsize=14)
    else:
        ax4.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                 f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../plots/active_ris_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Active RIS comparison plot saved as '../plots/active_ris_comparison.png'")

# Print summary
print("\n📊 Performance Summary:")
print("=" * 60)
for config in configs:
    print(f"\n{config}:")
    print(f"  Communication Rate: {results[config]['communication_rate']:.3f} ± {results[config]['communication_std']:.3f} bits/s/Hz")
    print(f"  Secrecy Rate: {results[config]['secrecy_rate']:.3f} ± {results[config]['secrecy_std']:.3f} bits/s/Hz")
    print(f"  Power Consumption: {results[config]['power_consumption']*1000:.1f} ± {results[config]['power_std']*1000:.1f} mW")

# Don't show plot in headless mode
# plt.show()
