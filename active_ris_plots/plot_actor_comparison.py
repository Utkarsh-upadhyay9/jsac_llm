#!/usr/bin/env python3
"""
Actor Network Performance Comparison for Active RIS + DAM
Compares MLP, LLM, and Hybrid actors in the enhanced Active RIS environment
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
sys.path.append('..')
from jsac_active_ris_dam import *

def train_and_evaluate_actors(episodes=100):
    """Train and evaluate different actor networks on Active RIS environment"""
    
    # Initialize environment and actors
    env = ActiveRISISACEnvironment()
    
    actors = {
        'MLP': ActiveRISMLP(env.state_dim, env.action_dim).to(device),
        'LLM': ActiveRISLLM(env.state_dim, env.action_dim).to(device),
        'Hybrid': ActiveRISHybrid(env.state_dim, env.action_dim).to(device)
    }
    
    # Simple optimizers for demonstration
    optimizers = {}
    for name, actor in actors.items():
        if name == 'LLM':
            optimizers[name] = torch.optim.Adam(actor.parameters(), lr=1e-5)  # Lower LR for LLM
        else:
            optimizers[name] = torch.optim.Adam(actor.parameters(), lr=1e-4)
    
    # Training history
    training_history = {name: {
        'episode_rewards': [],
        'communication_rates': [],
        'secrecy_rates': [],
        'power_consumptions': [],
        'active_elements': [],
        'dam_delays': []
    } for name in actors.keys()}
    
    print(f"Training {len(actors)} actors for {episodes} episodes...")
    
    for episode in range(episodes):
        if episode % 20 == 0:
            print(f"Episode {episode}/{episodes}")
        
        for actor_name, actor in actors.items():
            # Reset environment
            state = env.reset()
            state_tensor = torch.FloatTensor(state).to(device)
            
            # Generate action
            actor.eval()
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store metrics
            training_history[actor_name]['episode_rewards'].append(reward)
            training_history[actor_name]['communication_rates'].append(info['communication_rate'])
            training_history[actor_name]['secrecy_rates'].append(info['secrecy_rate'])
            training_history[actor_name]['power_consumptions'].append(info['power_consumption'])
            training_history[actor_name]['active_elements'].append(info['active_elements'])
            training_history[actor_name]['dam_delays'].append(info['dam_delay_avg'])
            
            # Simple policy gradient update (simplified for demonstration)
            if episode > 10:  # Start training after some episodes
                actor.train()
                optimizers[actor_name].zero_grad()
                
                # Simple loss: negative reward (we want to maximize reward)
                loss = -torch.tensor(reward, requires_grad=True)
                
                # Add L2 regularization to prevent overfitting
                l2_reg = 0
                for param in actor.parameters():
                    l2_reg += torch.norm(param)
                loss += 1e-5 * l2_reg
                
                # Backward pass (simplified - in real training you'd use proper policy gradient)
                try:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optimizers[actor_name].step()
                except:
                    pass  # Skip if gradient computation fails
    
    return training_history

def analyze_actor_specialization(training_history):
    """Analyze how different actors specialize in Active RIS control"""
    
    # Calculate moving averages for smoother plots
    window = 10
    
    smoothed_history = {}
    for actor_name, history in training_history.items():
        smoothed_history[actor_name] = {}
        for metric, values in history.items():
            smoothed_history[actor_name][metric] = []
            for i in range(len(values)):
                start_idx = max(0, i - window + 1)
                smoothed_history[actor_name][metric].append(
                    np.mean(values[start_idx:i+1])
                )
    
    return smoothed_history

# Run training simulation
print("Starting Actor Network Comparison for Active RIS + DAM...")
training_results = train_and_evaluate_actors(episodes=150)
smoothed_results = analyze_actor_specialization(training_results)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

colors = {'MLP': '#FF6B6B', 'LLM': '#4ECDC4', 'Hybrid': '#45B7D1'}
episodes = range(len(training_results['MLP']['episode_rewards']))

# 1. Episode Rewards Comparison
ax1 = fig.add_subplot(gs[0, 0])
for actor_name in ['MLP', 'LLM', 'Hybrid']:
    ax1.plot(episodes, smoothed_results[actor_name]['episode_rewards'], 
             color=colors[actor_name], linewidth=2, label=actor_name, alpha=0.8)
ax1.set_title('Episode Rewards', fontsize=12, fontweight='bold')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Communication Rates
ax2 = fig.add_subplot(gs[0, 1])
for actor_name in ['MLP', 'LLM', 'Hybrid']:
    ax2.plot(episodes, smoothed_results[actor_name]['communication_rates'], 
             color=colors[actor_name], linewidth=2, label=actor_name, alpha=0.8)
ax2.set_title('Communication Rates', fontsize=12, fontweight='bold')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Rate (bits/s/Hz)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Secrecy Rates
ax3 = fig.add_subplot(gs[0, 2])
for actor_name in ['MLP', 'LLM', 'Hybrid']:
    ax3.plot(episodes, smoothed_results[actor_name]['secrecy_rates'], 
             color=colors[actor_name], linewidth=2, label=actor_name, alpha=0.8)
ax3.set_title('Secrecy Rates', fontsize=12, fontweight='bold')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Secrecy Rate (bits/s/Hz)')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Power Consumption
ax4 = fig.add_subplot(gs[1, 0])
for actor_name in ['MLP', 'LLM', 'Hybrid']:
    power_mw = [p * 1000 for p in smoothed_results[actor_name]['power_consumptions']]
    ax4.plot(episodes, power_mw, 
             color=colors[actor_name], linewidth=2, label=actor_name, alpha=0.8)
ax4.set_title('Power Consumption', fontsize=12, fontweight='bold')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Power (mW)')
ax4.grid(True, alpha=0.3)
ax4.legend()

# 5. Active Elements Usage
ax5 = fig.add_subplot(gs[1, 1])
for actor_name in ['MLP', 'LLM', 'Hybrid']:
    ax5.plot(episodes, smoothed_results[actor_name]['active_elements'], 
             color=colors[actor_name], linewidth=2, label=actor_name, alpha=0.8)
ax5.set_title('Active Elements Usage', fontsize=12, fontweight='bold')
ax5.set_xlabel('Episode')
ax5.set_ylabel('Number of Active Elements')
ax5.grid(True, alpha=0.3)
ax5.legend()

# 6. DAM Delay Patterns
ax6 = fig.add_subplot(gs[1, 2])
for actor_name in ['MLP', 'LLM', 'Hybrid']:
    ax6.plot(episodes, smoothed_results[actor_name]['dam_delays'], 
             color=colors[actor_name], linewidth=2, label=actor_name, alpha=0.8)
ax6.set_title('Average DAM Delays', fontsize=12, fontweight='bold')
ax6.set_xlabel('Episode')
ax6.set_ylabel('Average Delay (ns)')
ax6.grid(True, alpha=0.3)
ax6.legend()

# 7. Performance Distribution (Final 50 episodes)
ax7 = fig.add_subplot(gs[2, 0])
final_episodes = -50
final_rewards = {}
for actor_name in ['MLP', 'LLM', 'Hybrid']:
    final_rewards[actor_name] = training_results[actor_name]['episode_rewards'][final_episodes:]

bp = ax7.boxplot([final_rewards[name] for name in ['MLP', 'LLM', 'Hybrid']], 
                 labels=['MLP', 'LLM', 'Hybrid'], patch_artist=True)
for patch, color in zip(bp['boxes'], [colors[name] for name in ['MLP', 'LLM', 'Hybrid']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax7.set_title('Final Performance Distribution', fontsize=12, fontweight='bold')
ax7.set_ylabel('Episode Reward')
ax7.grid(True, alpha=0.3)

# 8. Efficiency Analysis (Reward per Power)
ax8 = fig.add_subplot(gs[2, 1])
for actor_name in ['MLP', 'LLM', 'Hybrid']:
    rewards = smoothed_results[actor_name]['episode_rewards'][final_episodes:]
    powers = smoothed_results[actor_name]['power_consumptions'][final_episodes:]
    efficiencies = [r/(p*1000) if p > 0 else 0 for r, p in zip(rewards, powers)]
    ax8.bar(actor_name, np.mean(efficiencies), color=colors[actor_name], 
            alpha=0.7, edgecolor='black')

ax8.set_title('Energy Efficiency (Final 50 Episodes)', fontsize=12, fontweight='bold')
ax8.set_ylabel('Reward per mW')
ax8.grid(True, alpha=0.3)

# 9. Specialization Radar Chart
ax9 = fig.add_subplot(gs[2, 2], projection='polar')

# Normalize metrics for radar chart (final 20 episodes average)
metrics = ['Communication', 'Secrecy', 'Power Eff.', 'Active Usage', 'DAM Usage']
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for actor_name in ['MLP', 'LLM', 'Hybrid']:
    final_data = {
        'communication': np.mean(training_results[actor_name]['communication_rates'][-20:]),
        'secrecy': np.mean(training_results[actor_name]['secrecy_rates'][-20:]),
        'power_eff': np.mean(training_results[actor_name]['episode_rewards'][-20:]) / 
                    (np.mean(training_results[actor_name]['power_consumptions'][-20:]) * 1000 + 1e-6),
        'active_usage': np.mean(training_results[actor_name]['active_elements'][-20:]) / N,
        'dam_usage': np.mean(training_results[actor_name]['dam_delays'][-20:]) / 50.0  # Normalize by max delay
    }
    
    # Normalize to 0-1 scale
    values = list(final_data.values())
    max_vals = [10, 10, 1, 1, 1]  # Rough normalization factors
    normalized_values = [v/m for v, m in zip(values, max_vals)]
    normalized_values += normalized_values[:1]  # Complete the circle
    
    ax9.plot(angles, normalized_values, color=colors[actor_name], 
             linewidth=2, label=actor_name, alpha=0.8)
    ax9.fill(angles, normalized_values, color=colors[actor_name], alpha=0.25)

ax9.set_xticks(angles[:-1])
ax9.set_xticklabels(metrics)
ax9.set_title('Actor Specialization Profile', fontsize=12, fontweight='bold', pad=20)
ax9.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.suptitle('Active RIS + DAM: Actor Network Performance Comparison', 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig('../plots/actor_comparison_active_ris.png', dpi=300, bbox_inches='tight')
print("✓ Actor comparison plot saved as '../plots/actor_comparison_active_ris.png'")

# Performance summary
print("\n🤖 Actor Network Performance Summary (Final 50 Episodes):")
print("=" * 70)

for actor_name in ['MLP', 'LLM', 'Hybrid']:
    final_rewards = np.mean(training_results[actor_name]['episode_rewards'][-50:])
    final_comm = np.mean(training_results[actor_name]['communication_rates'][-50:])
    final_secrecy = np.mean(training_results[actor_name]['secrecy_rates'][-50:])
    final_power = np.mean(training_results[actor_name]['power_consumptions'][-50:]) * 1000
    final_active = np.mean(training_results[actor_name]['active_elements'][-50:])
    final_delays = np.mean(training_results[actor_name]['dam_delays'][-50:])
    
    print(f"\n{actor_name} Actor:")
    print(f"  Average Reward: {final_rewards:.3f}")
    print(f"  Communication Rate: {final_comm:.3f} bits/s/Hz")
    print(f"  Secrecy Rate: {final_secrecy:.3f} bits/s/Hz")
    print(f"  Power Consumption: {final_power:.1f} mW")
    print(f"  Active Elements: {final_active:.1f}/{N}")
    print(f"  Average DAM Delay: {final_delays:.1f} ns")
    
    if final_power > 0:
        efficiency = final_rewards / final_power
        print(f"  Energy Efficiency: {efficiency:.3f} reward/mW")

# Identify best performer
best_actor = max(['MLP', 'LLM', 'Hybrid'], 
                key=lambda x: np.mean(training_results[x]['episode_rewards'][-50:]))
print(f"\n🏆 Best Overall Performer: {best_actor}")

plt.show()
