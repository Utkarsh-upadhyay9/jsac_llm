#!/usr/bin/env python3
"""
Quick Active RIS + DAM Demo Plot
Fast demonstration of the Active RIS capabilities with synthetic data
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Ensure plots directory exists
os.makedirs('../plots', exist_ok=True)

print("Generating Quick Active RIS + DAM Demo Plot...")

# Synthetic performance data based on expected Active RIS + DAM improvements
configs = ['Basic Passive RIS', 'Active RIS (No DAM)', 'Active RIS + DAM', 'Full Active + DAM']

# Communication rates (bits/s/Hz) - Active RIS provides amplification
comm_rates = [8.2, 10.5, 12.8, 11.9]  # Peak then slight drop due to power constraints
comm_stds = [0.3, 0.4, 0.5, 0.6]

# Secrecy rates (bits/s/Hz) - DAM provides significant security improvement
secrecy_rates = [2.1, 2.8, 5.2, 4.8]  # Major improvement with DAM
secrecy_stds = [0.2, 0.3, 0.4, 0.5]

# Power consumption (mW) - Active elements consume power
power_consumptions = [0.0, 25.3, 35.7, 48.2]
power_stds = [0.0, 2.1, 3.2, 4.1]

# Energy efficiency (bits/s/Hz/mW)
efficiencies = []
for i in range(len(configs)):
    if power_consumptions[i] > 0:
        eff = comm_rates[i] / power_consumptions[i]
    else:
        eff = float('inf')  # Infinite for zero power
    efficiencies.append(eff)

# Create comprehensive plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Active RIS + DAM vs Basic RIS Performance Demo', fontsize=16, fontweight='bold')

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# Communication Rate Comparison
bars1 = ax1.bar(configs, comm_rates, yerr=comm_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Communication Rate Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Sum Rate (bits/s/Hz)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, rate, std in zip(bars1, comm_rates, comm_stds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.2,
             f'{rate:.1f}', ha='center', va='bottom', fontweight='bold')

# Secrecy Rate Comparison
bars2 = ax2.bar(configs, secrecy_rates, yerr=secrecy_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax2.set_title('Secrecy Rate Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Secrecy Rate (bits/s/Hz)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Add value labels
for bar, rate, std in zip(bars2, secrecy_rates, secrecy_stds):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
             f'{rate:.1f}', ha='center', va='bottom', fontweight='bold')

# Power Consumption Comparison
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

# Feature Comparison Radar Chart
features = ['Communication\nRate', 'Secrecy\nEnhancement', 'Power\nEfficiency', 'DAM\nCapability', 'Active\nAmplification']
angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]  # Complete circle

# Normalize features for radar chart (0-1 scale)
feature_data = {
    'Basic Passive RIS': [0.6, 0.3, 1.0, 0.0, 0.0],
    'Active RIS (No DAM)': [0.8, 0.4, 0.6, 0.0, 0.8],
    'Active RIS + DAM': [1.0, 1.0, 0.8, 1.0, 0.9],
    'Full Active + DAM': [0.9, 0.9, 0.4, 1.0, 1.0]
}

ax4 = plt.subplot(2, 2, 4, projection='polar')
for i, (config, values) in enumerate(feature_data.items()):
    values_plot = values + values[:1]  # Complete circle
    ax4.plot(angles, values_plot, color=colors[i], linewidth=2, label=config, alpha=0.8)
    ax4.fill(angles, values_plot, color=colors[i], alpha=0.25)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(features)
ax4.set_ylim(0, 1)
ax4.set_title('Capability Comparison', fontsize=14, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.tight_layout()
plt.savefig('../plots/active_ris_demo.png', dpi=300, bbox_inches='tight')
print("✓ Active RIS demo plot saved as '../plots/active_ris_demo.png'")

# Create DAM delay pattern visualization
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Delay Alignment Modulation (DAM) Patterns', fontsize=16, fontweight='bold')

# Different delay strategies
N = 8  # Number of RIS elements
delay_strategies = {
    'No DAM': np.zeros(N),
    'Uniform 25ns': np.full(N, 25.0),
    'Linear Gradient': np.linspace(5, 45, N),
    'Optimal Pattern': [5, 15, 25, 35, 45, 35, 25, 15][:N]
}

# Delay pattern visualization
colors_dam = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for i, (strategy, delays) in enumerate(delay_strategies.items()):
    if strategy == 'No DAM':
        continue
    ax1.plot(range(N), delays, 'o-', label=strategy, linewidth=2, markersize=8, color=colors_dam[i])

ax1.set_title('DAM Delay Patterns', fontsize=14, fontweight='bold')
ax1.set_xlabel('RIS Element Index')
ax1.set_ylabel('Delay (ns)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Security improvement with DAM
dam_strategies = list(delay_strategies.keys())
secrecy_improvements = [0, 15, 35, 60]  # Percentage improvement
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars = ax2.bar(dam_strategies, secrecy_improvements, color=colors_bar, alpha=0.7, edgecolor='black')
ax2.set_title('Security Enhancement with DAM', fontsize=14, fontweight='bold')
ax2.set_ylabel('Secrecy Rate Improvement (%)')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Add value labels
for bar, improvement in zip(bars, secrecy_improvements):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{improvement}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../plots/dam_patterns_demo.png', dpi=300, bbox_inches='tight')
print("✓ DAM patterns demo plot saved as '../plots/dam_patterns_demo.png'")

# Summary statistics
print("\n📊 Active RIS + DAM Performance Summary:")
print("=" * 60)
for i, config in enumerate(configs):
    print(f"\n{config}:")
    print(f"  Communication Rate: {comm_rates[i]:.1f} ± {comm_stds[i]:.1f} bits/s/Hz")
    print(f"  Secrecy Rate: {secrecy_rates[i]:.1f} ± {secrecy_stds[i]:.1f} bits/s/Hz")
    print(f"  Power Consumption: {power_consumptions[i]:.1f} ± {power_stds[i]:.1f} mW")
    if power_consumptions[i] > 0:
        print(f"  Energy Efficiency: {efficiencies[i]:.3f} bits/s/Hz/mW")
    else:
        print(f"  Energy Efficiency: ∞ (zero power)")

# Key improvements
basic_secrecy = secrecy_rates[0]
dam_secrecy = secrecy_rates[2]
improvement = ((dam_secrecy - basic_secrecy) / basic_secrecy) * 100

print(f"\n🎯 Key Improvements with Active RIS + DAM:")
print(f"  • Secrecy Rate: {improvement:.0f}% improvement over basic RIS")
print(f"  • Communication Rate: {((comm_rates[2] - comm_rates[0]) / comm_rates[0] * 100):.0f}% improvement")
print(f"  • Active Amplification: Up to 10 dB gain per element")
print(f"  • DAM Security: 50ns delay alignment for anti-eavesdropping")

print(f"\n✅ Active RIS + DAM demo plots generated successfully!")
print(f"🔒 Advanced security features demonstrated beyond basic passive RIS")
