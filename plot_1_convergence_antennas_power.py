#!/usr/bin/env python3
"""
Plot 1: Convergence versus episodes for changing number of antennas at BS and transmitting power
This plot shows how different configurations of antennas and power levels affect convergence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import os
from transformers import DistilBertTokenizer, DistilBertModel
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Create plots directory
if not os.path.exists('plots'):
    os.makedirs('plots')

# Import necessary classes and functions from jsac.py
exec(open('/home/utkarsh/jsac_llm/jsac.py').read())

def run_training_experiment(N_antennas, M_bs_antennas, P_max_val, episodes=1000):
    """
    Run training experiment with specific antenna and power configurations
    """
    global N, M, P_max
    N = N_antennas  # RIS elements
    M = M_bs_antennas  # BS antennas
    P_max = P_max_val  # Maximum power
    
    # Update system parameters
    V_users = 3
    state_dim = N + (2 * M * V_users)
    action_dim = N + (2 * M * V_users)
    
    # Recreate channels with new dimensions
    pl_br = compute_pathloss(abs(iab_pos - ris_pos))
    pl_ru = compute_pathloss(abs(ris_pos - vu_pos))
    pl_be = compute_pathloss(abs(iab_pos - eve_pos))
    pl_e = compute_pathloss(abs(ris_pos - eve_pos))
    
    # Update channel matrices
    global H_be, h_e, h_ru, H_br
    H_be = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) / np.sqrt(2) * np.sqrt(1 / pl_be)
    h_e = (np.random.randn(1, N) + 1j*np.random.randn(1, N)) / np.sqrt(2) * np.sqrt(1 / pl_e)
    
    # RIS-to-VU Channel
    phi_sv = 1
    d_sv = abs(ris_pos - vu_pos)
    pl_sv = compute_pathloss(d_sv)
    h_ru = np.array([np.exp(-1j * 2 * np.pi / lambda_c * n * d_sv * phi_sv) for n in range(N)])
    h_ru = (1 / np.sqrt(pl_sv)) * h_ru.reshape(1, -1)
    
    # IAB-to-RIS Channel
    kappa = 10
    phi_is = 1
    d_is = abs(iab_pos - ris_pos)
    pl_is = compute_pathloss(d_is)
    H_br = np.zeros((N, M), dtype=complex)
    for m in range(M):
        h_los = np.array([np.exp(-1j * 2 * np.pi / lambda_c * n * d_is * phi_is) for n in range(N)])
        h_nlos = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        H_br[:, m] = np.sqrt(kappa / (1 + kappa)) * h_los + np.sqrt(1 / (1 + kappa)) * h_nlos
    H_br *= (1 / np.sqrt(pl_is))
    
    # Initialize tokenizer and agents
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    agents = {
        "MLP": DDPGAgent("MLP", ActorMLP, Critic, state_dim, action_dim),
        "LLM": DDPGAgent("LLM", ActorLLM, Critic, state_dim, action_dim, is_text_based=True),
        "Hybrid": DDPGAgent("Hybrid", ActorHybrid, Critic, state_dim, action_dim, is_hybrid=True)
    }
    
    # Training hyperparameters
    batch_size = 16
    gamma = 0.99
    tau = 0.0005
    noise_std = 0.3
    noise_decay = 0.998
    min_noise_std = 0.02
    
    print(f"Training with N={N}, M={M}, P_max={P_max}")
    
    # Training loop (simplified)
    for ep in range(episodes):
        # Initialize state
        ris_phases = np.random.uniform(0, 2*np.pi, N)
        bs_w_real = np.random.randn(M * V_users)
        bs_w_imag = np.random.randn(M * V_users)
        bs_w_complex = bs_w_real + 1j * bs_w_imag
        bs_w_complex = bs_w_complex / np.linalg.norm(bs_w_complex) * np.sqrt(P_max)
        bs_w_real = bs_w_complex.real.flatten()
        bs_w_imag = bs_w_complex.imag.flatten()
        
        state_np = np.concatenate([ris_phases, bs_w_real, bs_w_imag])
        state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        
        # Default KPIs for prompt generation
        default_kpis = {
            'secrecy_rate': 0.5,
            'sensing_secrecy_rate': 1.0,
            'min_user_rate': 1.0,
            'rate_eve': 0.8,
            'rate_sense_iab': 2.0,
            'rate_sense_eve': 1.0,
            'P_tx_total': 0.5,
            'P_max': P_max
        }
        
        for name, agent in agents.items():
            # Generate action
            if agent.is_text_based or agent.is_hybrid:
                prompt = create_descriptive_prompt(**default_kpis)
                inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
            
            with torch.no_grad():
                if agent.is_text_based:
                    action = agent.actor(inputs['input_ids'], inputs['attention_mask']).cpu().numpy()[0]
                elif agent.is_hybrid:
                    action = agent.actor(state_tensor, inputs['input_ids'], inputs['attention_mask']).cpu().numpy()[0]
                else:
                    action = agent.actor(state_tensor).cpu().numpy()[0]
            
            # Add noise
            noisy_action = action + np.random.normal(0, noise_std, action_dim)
            
            # Extract RIS phases and beamforming
            ris_action = np.mod((noisy_action[:N] + 1) / 2 * 2 * np.pi, 2 * np.pi)
            w_flat = noisy_action[N:]
            w_real = w_flat[:M*V_users].reshape(M, V_users)
            w_imag = w_flat[M*V_users:].reshape(M, V_users)
            W_tau = w_real + 1j * w_imag
            W_o = W_tau.copy()
            
            # Power constraint
            P_tx_total = np.trace(W_tau @ W_tau.conj().T + W_o @ W_o.conj().T).real
            if P_tx_total > P_max:
                scale = np.sqrt(P_max / P_tx_total)
                W_tau *= scale
                W_o *= scale
            
            # Compute SNRs and rates
            snr_eve = compute_eve_sinr_maxcase(ris_action, W_tau, W_o)
            snr_comm = compute_snr_delayed(ris_action, W_tau, W_o, v_idx=0)
            snr_sense = compute_sensing_snr(W_tau, W_o, ris_action)
            
            # Calculate rates
            gamma_req = 0.001
            gamma_s_min = 2
            
            if snr_comm >= gamma_req:
                R_v = beta * B * np.log2(1 + max(snr_comm, snr_min))
                R_e = beta * B * np.log2(1 + max(snr_eve, snr_min))
                C_D_i = beta * B * np.log2(1 + max(np.abs(h_backhaul)**2 / sigma2, snr_min))
                total_Rv = R_v * V_users
                R_D_v = R_v / total_Rv * C_D_i
                R_E2E_v = min(R_v, R_D_v)
                secrecy_rate = max(R_E2E_v - R_e, 0)
            else:
                secrecy_rate = 0
                
            if snr_sense >= gamma_s_min:
                sensing_rate = beta * B * np.log2(1 + max(snr_sense, snr_min))
            else:
                sensing_rate = 0
            
            # Compute sensing at eavesdropper (simplified)
            R_sense_eve = 0.5  # Simplified
            sensing_secrecy_rate = max(sensing_rate - R_sense_eve, 0)
            
            # Calculate reward
            reward = omega * secrecy_rate + (1 - omega) * sensing_secrecy_rate
            reward = np.clip(reward, -10.0, 10.0)
            
            # Store reward
            agent.reward_history.append(reward)
            
            # Update agent (simplified - no replay buffer for speed)
            if len(agent.reward_history) > 10:
                agent.update(batch_size, gamma, tau, tokenizer)
        
        # Update noise
        noise_std = max(noise_std * noise_decay, min_noise_std)
        
        if (ep + 1) % 200 == 0:
            print(f"Episode {ep + 1}/{episodes} completed")
    
    return agents

# Configuration parameters
antenna_configs = [
    (16, 8),   # N=16 RIS elements, M=8 BS antennas
    (32, 16),  # N=32 RIS elements, M=16 BS antennas
    (64, 32),  # N=64 RIS elements, M=32 BS antennas
]

power_configs = [0.5, 1.0, 2.0]  # Different power levels

# Run experiments
print("Running convergence experiments for different antenna and power configurations...")

results = {}
episodes = 800  # Reduced for faster execution

for N_ris, M_bs in antenna_configs:
    for P_max_val in power_configs:
        config_name = f"N{N_ris}_M{M_bs}_P{P_max_val}"
        print(f"\nRunning configuration: {config_name}")
        
        agents = run_training_experiment(N_ris, M_bs, P_max_val, episodes)
        
        # Store results
        results[config_name] = {}
        for name, agent in agents.items():
            results[config_name][name] = agent.reward_history.copy()

# Plotting
def moving_avg(x, window=50):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.figure(figsize=(15, 10))

# Plot 1: Effect of number of antennas (fixed power P=1.0)
plt.subplot(2, 2, 1)
colors = ['blue', 'green', 'red']
for i, (N_ris, M_bs) in enumerate(antenna_configs):
    config_name = f"N{N_ris}_M{M_bs}_P1.0"
    if config_name in results:
        for j, (algo_name, color) in enumerate(zip(['MLP', 'LLM', 'Hybrid'], colors)):
            if algo_name in results[config_name]:
                rewards = results[config_name][algo_name]
                smoothed = moving_avg(rewards, 30)
                plt.plot(smoothed, color=color, linestyle=['-', '--', '-.'][i], 
                        label=f'{algo_name} (N={N_ris}, M={M_bs})', alpha=0.8)

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Convergence vs Episodes (Different Antenna Configurations)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Effect of transmitting power (fixed antennas N=32, M=16)
plt.subplot(2, 2, 2)
for i, P_max_val in enumerate(power_configs):
    config_name = f"N32_M16_P{P_max_val}"
    if config_name in results:
        for j, (algo_name, color) in enumerate(zip(['MLP', 'LLM', 'Hybrid'], colors)):
            if algo_name in results[config_name]:
                rewards = results[config_name][algo_name]
                smoothed = moving_avg(rewards, 30)
                plt.plot(smoothed, color=color, linestyle=['-', '--', '-.'][i], 
                        label=f'{algo_name} (P={P_max_val})', alpha=0.8)

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Convergence vs Episodes (Different Power Levels)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Hybrid algorithm performance across all configurations
plt.subplot(2, 2, 3)
config_idx = 0
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
for N_ris, M_bs in antenna_configs:
    for P_max_val in power_configs:
        config_name = f"N{N_ris}_M{M_bs}_P{P_max_val}"
        if config_name in results and 'Hybrid' in results[config_name]:
            rewards = results[config_name]['Hybrid']
            smoothed = moving_avg(rewards, 30)
            plt.plot(smoothed, linestyle=linestyles[config_idx], 
                    label=f'N={N_ris}, M={M_bs}, P={P_max_val}', alpha=0.8)
            config_idx += 1

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Hybrid Algorithm - All Configurations')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Final convergence comparison
plt.subplot(2, 2, 4)
final_rewards = {}
for config_name, config_results in results.items():
    for algo_name, rewards in config_results.items():
        if algo_name not in final_rewards:
            final_rewards[algo_name] = []
        # Take average of last 100 episodes
        final_rewards[algo_name].append(np.mean(rewards[-100:]))

x_pos = np.arange(len(antenna_configs) * len(power_configs))
width = 0.25

for i, algo_name in enumerate(['MLP', 'LLM', 'Hybrid']):
    if algo_name in final_rewards:
        plt.bar(x_pos + i*width, final_rewards[algo_name], width, 
                label=algo_name, alpha=0.8)

config_labels = []
for N_ris, M_bs in antenna_configs:
    for P_max_val in power_configs:
        config_labels.append(f'N{N_ris}\nM{M_bs}\nP{P_max_val}')

plt.xlabel('Configuration')
plt.ylabel('Final Average Reward')
plt.title('Final Convergence Performance')
plt.xticks(x_pos + width, config_labels, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/plot_1_convergence_antennas_power.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlot 1 completed and saved to 'plots/plot_1_convergence_antennas_power.png'")
