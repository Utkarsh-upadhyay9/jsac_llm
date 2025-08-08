#!/usr/bin/env python3
"""
Active Element Power Budget Optimization
Shows the trade-off between number of active elements, power consumption, and performance
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')
from jsac_active_ris_dam import *

def analyze_power_budget_optimization():
    """Analyze optimal allocation of power budget across active elements"""
    
    # Test different numbers of active elements
    active_counts = range(0, N+1)
    
    # Different power allocation strategies
    strategies = {
        'Equal Power': 'equal',
        'Random Allocation': 'random',
        'Gradient Allocation': 'gradient',
        'Binary High-Low': 'binary'
    }
    
    results = {}
    
    # Generate fixed beamforming weights for consistent comparison
    np.random.seed(42)  # For reproducibility
    W_tau = (np.random.randn(M, V) + 1j * np.random.randn(M, V)) / np.sqrt(2)
    W_o = (np.random.randn(M, V) + 1j * np.random.randn(M, V)) / np.sqrt(2)
    
    # Normalize power
    for v in range(V):
        total_power = np.sum(np.abs(W_tau[:, v])**2) + np.sum(np.abs(W_o[:, v])**2)
        if total_power > 1.0:
            scale = np.sqrt(1.0 / total_power)
            W_tau[:, v] *= scale
            W_o[:, v] *= scale
    
    print("Analyzing power budget optimization...")
    
    for strategy_name, strategy_type in strategies.items():
        results[strategy_name] = {
            'active_counts': [],
            'secrecy_rates': [],
            'communication_rates': [],
            'power_consumptions': [],
            'energy_efficiencies': [],
            'active_elements': []
        }
        
        for num_active in active_counts:
            if num_active == 0:
                # All passive
                config = {
                    'phases': np.random.uniform(0, 2*np.pi, N),
                    'active_mask': np.zeros(N, dtype=bool),
                    'gains_db': np.zeros(N),
                    'delays_ns': np.zeros(N)
                }
            else:
                # Create active mask
                active_mask = np.zeros(N, dtype=bool)
                active_indices = np.random.choice(N, num_active, replace=False)
                active_mask[active_indices] = True
                
                # Allocate gains based on strategy
                gains_db = np.zeros(N)
                if strategy_type == 'equal':
                    # Equal power across active elements
                    total_power_budget = N * P_active_max * 0.5
                    power_per_element = total_power_budget / num_active
                    gain_per_element = np.sqrt(power_per_element / P_active_max * G_active_lin) - 1
                    gain_db = 10 * np.log10(1 + gain_per_element)
                    gains_db[active_indices] = min(gain_db, G_active_max)
                    
                elif strategy_type == 'random':
                    # Random allocation
                    random_gains = np.random.exponential(3, num_active)
                    gains_db[active_indices] = np.clip(random_gains, 0, G_active_max)
                    
                elif strategy_type == 'gradient':
                    # Linear gradient allocation
                    gradient_gains = np.linspace(1, G_active_max, num_active)
                    gains_db[active_indices] = gradient_gains
                    
                elif strategy_type == 'binary':
                    # Half high, half low
                    high_count = num_active // 2
                    gains_db[active_indices[:high_count]] = G_active_max * 0.8
                    gains_db[active_indices[high_count:]] = G_active_max * 0.3
                
                # Add some DAM delays for active elements
                delays_ns = np.zeros(N)
                delays_ns[active_indices] = np.random.uniform(10, 40, num_active)
                
                config = {
                    'phases': np.random.uniform(0, 2*np.pi, N),
                    'active_mask': active_mask,
                    'gains_db': gains_db,
                    'delays_ns': delays_ns
                }
            
            # Average over multiple trials
            secrecy_rates = []
            comm_rates = []
            power_consumptions = []
            
            for trial in range(8):  # Multiple trials for averaging
                # Generate new channels
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
                    snr_v = compute_dam_enhanced_snr(config, W_tau, W_o, v)
                    rate_v = np.log2(1 + snr_v)
                    comm_rate += rate_v
                
                # Eavesdropper rate
                eve_snr = compute_dam_eavesdropper_sinr(config, W_tau, W_o)
                eve_rate = np.log2(1 + eve_snr)
                
                # Secrecy rate
                secrecy_rate = max(0, comm_rate - eve_rate)
                
                # Power consumption
                ris_array.configure_elements(config)
                power = ris_array.get_power_consumption()
                
                secrecy_rates.append(secrecy_rate)
                comm_rates.append(comm_rate)
                power_consumptions.append(power)
            
            # Store results
            avg_secrecy = np.mean(secrecy_rates)
            avg_comm = np.mean(comm_rates)
            avg_power = np.mean(power_consumptions)
            
            results[strategy_name]['active_counts'].append(num_active)
            results[strategy_name]['secrecy_rates'].append(avg_secrecy)
            results[strategy_name]['communication_rates'].append(avg_comm)
            results[strategy_name]['power_consumptions'].append(avg_power)
            
            # Energy efficiency (rate per mW)
            if avg_power > 0:
                efficiency = avg_comm / (avg_power * 1000)
            else:
                efficiency = float('inf')
            results[strategy_name]['energy_efficiencies'].append(efficiency)
            results[strategy_name]['active_elements'].append(num_active)
    
    return results

# Generate optimization analysis
power_results = analyze_power_budget_optimization()

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Active RIS Power Budget Optimization Analysis', fontsize=16, fontweight='bold')

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
markers = ['o', 's', '^', 'D']

# Secrecy Rate vs Number of Active Elements
for i, (strategy, data) in enumerate(power_results.items()):
    ax1.plot(data['active_counts'], data['secrecy_rates'], 
             color=colors[i], marker=markers[i], linewidth=2, markersize=6,
             label=strategy, alpha=0.8)

ax1.set_title('Secrecy Rate vs Active Elements', fontsize=14, fontweight='bold')
ax1.set_xlabel('Number of Active Elements')
ax1.set_ylabel('Secrecy Rate (bits/s/Hz)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Power Consumption vs Performance
for i, (strategy, data) in enumerate(power_results.items()):
    power_mw = [p * 1000 for p in data['power_consumptions']]
    ax2.scatter(power_mw, data['secrecy_rates'], 
               color=colors[i], s=60, alpha=0.7, label=strategy, marker=markers[i])

ax2.set_title('Power vs Secrecy Rate Trade-off', fontsize=14, fontweight='bold')
ax2.set_xlabel('Power Consumption (mW)')
ax2.set_ylabel('Secrecy Rate (bits/s/Hz)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Energy Efficiency Analysis
for i, (strategy, data) in enumerate(power_results.items()):
    # Handle infinite efficiency (zero power case)
    efficiencies = [min(eff, 50) if not np.isinf(eff) else 50 for eff in data['energy_efficiencies']]
    ax3.plot(data['active_counts'], efficiencies,
             color=colors[i], marker=markers[i], linewidth=2, markersize=6,
             label=strategy, alpha=0.8)

ax3.set_title('Energy Efficiency vs Active Elements', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Active Elements')
ax3.set_ylabel('Energy Efficiency (bits/s/Hz/mW)')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Optimal Operating Points (Pareto Front)
all_secrecy = []
all_power = []
all_strategies = []
all_active_counts = []

for strategy, data in power_results.items():
    all_secrecy.extend(data['secrecy_rates'])
    all_power.extend([p * 1000 for p in data['power_consumptions']])
    all_strategies.extend([strategy] * len(data['secrecy_rates']))
    all_active_counts.extend(data['active_counts'])

# Find Pareto optimal points
pareto_indices = []
for i in range(len(all_secrecy)):
    is_pareto = True
    for j in range(len(all_secrecy)):
        if (all_secrecy[j] >= all_secrecy[i] and all_power[j] <= all_power[i] and 
            (all_secrecy[j] > all_secrecy[i] or all_power[j] < all_power[i])):
            is_pareto = False
            break
    if is_pareto:
        pareto_indices.append(i)

# Plot all points
strategy_colors = {strategy: colors[i] for i, strategy in enumerate(power_results.keys())}
for strategy in power_results.keys():
    strategy_mask = [all_strategies[i] == strategy for i in range(len(all_strategies))]
    ax4.scatter([all_power[i] for i in range(len(all_power)) if strategy_mask[i]],
               [all_secrecy[i] for i in range(len(all_secrecy)) if strategy_mask[i]],
               color=strategy_colors[strategy], alpha=0.6, s=40, label=strategy)

# Highlight Pareto optimal points
pareto_power = [all_power[i] for i in pareto_indices]
pareto_secrecy = [all_secrecy[i] for i in pareto_indices]
ax4.scatter(pareto_power, pareto_secrecy, color='red', s=100, marker='*', 
           alpha=0.8, label='Pareto Optimal', edgecolors='black', linewidth=1)

ax4.set_title('Pareto Optimal Operating Points', fontsize=14, fontweight='bold')
ax4.set_xlabel('Power Consumption (mW)')
ax4.set_ylabel('Secrecy Rate (bits/s/Hz)')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('../plots/power_budget_optimization.png', dpi=300, bbox_inches='tight')
print("✓ Power budget optimization plot saved as '../plots/power_budget_optimization.png'")

# Analysis summary
print("\n⚡ Power Budget Optimization Analysis:")
print("=" * 60)

# Find best configurations for each strategy
for strategy, data in power_results.items():
    best_idx = np.argmax(data['secrecy_rates'])
    best_efficiency_idx = np.argmax([eff if not np.isinf(eff) else 0 for eff in data['energy_efficiencies']])
    
    print(f"\n{strategy}:")
    print(f"  Best Secrecy Rate: {data['secrecy_rates'][best_idx]:.3f} bits/s/Hz")
    print(f"    @ {data['active_counts'][best_idx]} active elements")
    print(f"    @ {data['power_consumptions'][best_idx]*1000:.1f} mW")
    
    print(f"  Best Efficiency: {data['energy_efficiencies'][best_efficiency_idx]:.2f} bits/s/Hz/mW")
    print(f"    @ {data['active_counts'][best_efficiency_idx]} active elements")

# Pareto optimal summary
print(f"\n🎯 Found {len(pareto_indices)} Pareto optimal operating points:")
for i, idx in enumerate(pareto_indices[:5]):  # Show top 5
    print(f"  {i+1}. {all_strategies[idx]}: {all_secrecy[idx]:.3f} bits/s/Hz @ {all_power[idx]:.1f} mW")
    print(f"     ({all_active_counts[idx]} active elements)")

plt.show()
