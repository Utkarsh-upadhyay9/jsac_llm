#!/usr/bin/env python3
"""
DAM Delay Optimization Analysis
Shows how different delay patterns affect secrecy performance
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')
from jsac_active_ris_dam import *

def analyze_dam_delays():
    """Analyze the effect of different DAM delay patterns on secrecy rate"""
    
    # Fixed active configuration for fair comparison
    base_config = {
        'phases': np.random.uniform(0, 2*np.pi, N),
        'active_mask': np.ones(N, dtype=bool),
        'gains_db': [3.0] * N  # Moderate gain to stay within power budget
    }
    
    # Different delay strategies
    delay_strategies = {
        'No DAM': np.zeros(N),
        'Uniform 25ns': np.full(N, 25.0),
        'Linear Gradient': np.linspace(5, 45, N),
        'Random Pattern': np.random.uniform(5, 45, N),
        'Alternating': [10, 40] * (N//2) + [25] * (N % 2),
        'Exponential': 5 * np.exp(np.linspace(0, 2, N)),
        'Sinusoidal': 25 + 15 * np.sin(np.linspace(0, 2*np.pi, N)),
        'Optimal Pattern': [5, 15, 25, 35, 45, 35, 25, 15][:N]  # Hand-crafted pattern
    }
    
    results = {}
    
    # Generate random beamforming weights
    W_tau = (np.random.randn(M, V) + 1j * np.random.randn(M, V)) / np.sqrt(2)
    W_o = (np.random.randn(M, V) + 1j * np.random.randn(M, V)) / np.sqrt(2)
    
    # Normalize power
    for v in range(V):
        total_power = np.sum(np.abs(W_tau[:, v])**2) + np.sum(np.abs(W_o[:, v])**2)
        if total_power > 1.0:
            scale = np.sqrt(1.0 / total_power)
            W_tau[:, v] *= scale
            W_o[:, v] *= scale
    
    print("Analyzing DAM delay patterns...")
    
    for strategy_name, delays in delay_strategies.items():
        config = {**base_config, 'delays_ns': delays}
        
        # Metrics over multiple channel realizations
        secrecy_rates = []
        communication_rates = []
        eve_rates = []
        
        for trial in range(15):  # More trials for statistical significance
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
            
            secrecy_rates.append(secrecy_rate)
            communication_rates.append(comm_rate)
            eve_rates.append(eve_rate)
        
        results[strategy_name] = {
            'secrecy_rate': np.mean(secrecy_rates),
            'secrecy_std': np.std(secrecy_rates),
            'communication_rate': np.mean(communication_rates),
            'eve_rate': np.mean(eve_rates),
            'delays': delays,
            'avg_delay': np.mean(delays),
            'delay_variance': np.var(delays)
        }
    
    return results

# Generate analysis
dam_results = analyze_dam_delays()

# Create comprehensive plots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main secrecy rate comparison
ax1 = fig.add_subplot(gs[0, :])
strategies = list(dam_results.keys())
secrecy_rates = [dam_results[s]['secrecy_rate'] for s in strategies]
secrecy_stds = [dam_results[s]['secrecy_std'] for s in strategies]

colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
bars = ax1.bar(strategies, secrecy_rates, yerr=secrecy_stds, capsize=5, 
               color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('DAM Delay Pattern Performance Comparison', fontsize=16, fontweight='bold')
ax1.set_ylabel('Secrecy Rate (bits/s/Hz)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Add value labels
for bar, rate, std in zip(bars, secrecy_rates, secrecy_stds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
             f'{rate:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Delay pattern visualization
ax2 = fig.add_subplot(gs[1, 0])
for i, (strategy, data) in enumerate(dam_results.items()):
    if strategy == 'No DAM':
        continue
    delays = data['delays']
    ax2.plot(range(N), delays, 'o-', label=strategy, linewidth=2, markersize=6)

ax2.set_title('Delay Patterns Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('RIS Element Index')
ax2.set_ylabel('Delay (ns)')
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Secrecy vs Communication Rate Trade-off
ax3 = fig.add_subplot(gs[1, 1])
comm_rates = [dam_results[s]['communication_rate'] for s in strategies]
for i, strategy in enumerate(strategies):
    ax3.scatter(comm_rates[i], secrecy_rates[i], s=100, c=[colors[i]], 
               label=strategy, alpha=0.8, edgecolors='black')

ax3.set_title('Secrecy vs Communication Trade-off', fontsize=14, fontweight='bold')
ax3.set_xlabel('Communication Rate (bits/s/Hz)')
ax3.set_ylabel('Secrecy Rate (bits/s/Hz)')
ax3.grid(True, alpha=0.3)

# Delay statistics effect
ax4 = fig.add_subplot(gs[1, 2])
avg_delays = [dam_results[s]['avg_delay'] for s in strategies]
delay_vars = [dam_results[s]['delay_variance'] for s in strategies]

scatter = ax4.scatter(avg_delays, delay_vars, c=secrecy_rates, s=100, 
                     cmap='viridis', alpha=0.8, edgecolors='black')
ax4.set_title('Delay Statistics vs Performance', fontsize=14, fontweight='bold')
ax4.set_xlabel('Average Delay (ns)')
ax4.set_ylabel('Delay Variance')
ax4.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Secrecy Rate', rotation=270, labelpad=15)

# Detailed comparison table
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('tight')
ax5.axis('off')

# Create table data
table_data = []
headers = ['Strategy', 'Secrecy Rate', 'Comm Rate', 'Eve Rate', 'Avg Delay', 'Delay Var']

for strategy in strategies:
    data = dam_results[strategy]
    row = [
        strategy,
        f"{data['secrecy_rate']:.3f} ± {data['secrecy_std']:.3f}",
        f"{data['communication_rate']:.3f}",
        f"{data['eve_rate']:.3f}",
        f"{data['avg_delay']:.1f} ns",
        f"{data['delay_variance']:.1f}"
    ]
    table_data.append(row)

table = ax5.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Style the table
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold')

plt.suptitle('Active RIS Delay Alignment Modulation (DAM) Analysis', fontsize=18, fontweight='bold', y=0.98)
plt.savefig('../plots/dam_analysis.png', dpi=300, bbox_inches='tight')
print("✓ DAM analysis plot saved as '../plots/dam_analysis.png'")

# Print detailed results
print("\n🔒 DAM Delay Pattern Analysis Results:")
print("=" * 70)

# Sort by secrecy rate for ranking
sorted_strategies = sorted(strategies, key=lambda s: dam_results[s]['secrecy_rate'], reverse=True)

for i, strategy in enumerate(sorted_strategies):
    data = dam_results[strategy]
    print(f"\n{i+1}. {strategy}:")
    print(f"   Secrecy Rate: {data['secrecy_rate']:.3f} ± {data['secrecy_std']:.3f} bits/s/Hz")
    print(f"   Communication: {data['communication_rate']:.3f} bits/s/Hz")
    print(f"   Eavesdropper: {data['eve_rate']:.3f} bits/s/Hz")
    print(f"   Avg Delay: {data['avg_delay']:.1f} ns")
    print(f"   Delay Pattern: {np.array(data['delays'])[:4]}... (showing first 4 elements)")

# Find best performing pattern
best_strategy = sorted_strategies[0]
improvement = ((dam_results[best_strategy]['secrecy_rate'] - dam_results['No DAM']['secrecy_rate']) / 
               dam_results['No DAM']['secrecy_rate'] * 100)

print(f"\n🏆 Best Strategy: {best_strategy}")
print(f"📈 Improvement over No DAM: {improvement:.1f}%")

plt.show()
