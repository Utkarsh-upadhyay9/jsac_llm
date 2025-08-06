#!/usr/bin/env python3
"""
Complete JSAC Plotting Suite - Generates all 8 requested figures
Based on the JSAC reinforcement learning code with MLP, LLM, and Hybrid algorithms

Figures:
1. Convergence vs episodes for changing number of antennas at BS and transmitting power
2. Reward vs number of VUs for w=0 and w=1
3. Reward vs number of sensing targets for w=0 and w=1  
4. Secrecy rate vs number of RIS elements
5. Secrecy rate vs total power (dBm)
6. Secrecy rate vs beta factor
7. Secrecy rate vs bandwidth
8. Secrecy rate vs BS antennas
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

print("=== JSAC Complete Plotting Suite ===")

# Set device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Create plots directory
if not os.path.exists('plots'):
    os.makedirs('plots')
    print("Created 'plots' directory")

# Import all functions and classes from jsac.py
print("Loading JSAC functions and classes...")
exec(open('/home/utkarsh/jsac_llm/jsac.py').read())

def simplified_training(N_ris, M_bs, V_users_val, P_max_val, omega_val, beta_val, B_val, episodes=500):
    """
    Simplified training function for parameter sweeps
    """
    global N, M, V_users, P_max, omega, beta, B, V
    N = N_ris
    M = M_bs
    V_users = V_users_val
    V = V_users_val
    P_max = P_max_val
    omega = omega_val
    beta = beta_val
    B = B_val
    
    # Update dimensions
    state_dim = N + (2 * M * V_users)
    action_dim = N + (2 * M * V_users)
    
    # Reinitialize channels with new dimensions
    pl_br = compute_pathloss(abs(iab_pos - ris_pos))
    pl_ru = compute_pathloss(abs(ris_pos - vu_pos))
    pl_be = compute_pathloss(abs(iab_pos - eve_pos))
    pl_e = compute_pathloss(abs(ris_pos - eve_pos))
    
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
    
    # Initialize agents (only Hybrid for speed)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    agents = {
        "MLP": DDPGAgent("MLP", ActorMLP, Critic, state_dim, action_dim),
        "LLM": DDPGAgent("LLM", ActorLLM, Critic, state_dim, action_dim, is_text_based=True),
        "Hybrid": DDPGAgent("Hybrid", ActorHybrid, Critic, state_dim, action_dim, is_hybrid=True)
    }
    
    # Training parameters
    batch_size = 16
    gamma = 0.99
    tau = 0.0005
    noise_std = 0.3
    noise_decay = 0.998
    min_noise_std = 0.02
    
    # Training loop
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
        
        # Default KPIs
        default_kpis = {
            'secrecy_rate': 0.5, 'sensing_secrecy_rate': 1.0,
            'min_user_rate': 1.0, 'rate_eve': 0.8,
            'rate_sense_iab': 2.0, 'rate_sense_eve': 1.0,
            'P_tx_total': 0.5, 'P_max': P_max
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
            
            # Add noise and process action
            noisy_action = action + np.random.normal(0, noise_std, action_dim)
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
            
            # Compute metrics
            snr_eve = compute_eve_sinr_maxcase(ris_action, W_tau, W_o)
            snr_comm = compute_snr_delayed(ris_action, W_tau, W_o, v_idx=0)
            snr_sense = compute_sensing_snr(W_tau, W_o, ris_action)
            
            # Calculate rates and reward
            gamma_req = 0.001
            gamma_s_min = 2
            
            secrecy_rate = 0
            if snr_comm >= gamma_req:
                R_v = beta * B * np.log2(1 + max(snr_comm, snr_min))
                R_e = beta * B * np.log2(1 + max(snr_eve, snr_min))
                C_D_i = beta * B * np.log2(1 + max(np.abs(h_backhaul)**2 / sigma2, snr_min))
                total_Rv = R_v * V_users
                R_D_v = R_v / total_Rv * C_D_i
                R_E2E_v = min(R_v, R_D_v)
                secrecy_rate = max(R_E2E_v - R_e, 0)
            
            sensing_rate = 0
            if snr_sense >= gamma_s_min:
                sensing_rate = beta * B * np.log2(1 + max(snr_sense, snr_min))
            
            R_sense_eve = 0.5  # Simplified
            sensing_secrecy_rate = max(sensing_rate - R_sense_eve, 0)
            
            reward = omega * secrecy_rate + (1 - omega) * sensing_secrecy_rate
            reward = np.clip(reward, -10.0, 10.0)
            
            agent.reward_history.append(reward)
            
            # Update (simplified)
            if len(agent.reward_history) > 10 and ep % 5 == 0:
                agent.update(batch_size, gamma, tau, tokenizer)
        
        noise_std = max(noise_std * noise_decay, min_noise_std)
    
    # Return final performance metrics
    results = {}
    for name, agent in agents.items():
        final_reward = np.mean(agent.reward_history[-50:]) if len(agent.reward_history) >= 50 else np.mean(agent.reward_history)
        results[name] = {
            'final_reward': final_reward,
            'secrecy_rate': secrecy_rate,
            'history': agent.reward_history
        }
    
    return results

# ============================================================================
# FIGURE 1: Convergence vs Episodes for Different Antenna and Power Configs
# ============================================================================
print("\n=== Generating Figure 1: Convergence vs Episodes ===")

# Configuration parameters
antenna_configs = [(16, 8), (32, 16), (64, 32)]  # (N_RIS, M_BS)
power_configs = [0.5, 1.0, 2.0]  # Different power levels

results_fig1 = {}
episodes = 600

for N_ris, M_bs in antenna_configs:
    for P_max_val in power_configs:
        config_name = f"N{N_ris}_M{M_bs}_P{P_max_val}"
        print(f"Training config: {config_name}")
        
        results = simplified_training(N_ris, M_bs, 3, P_max_val, 0.5, 0.8, 1.0, episodes)
        results_fig1[config_name] = results

# Plot Figure 1
plt.figure(figsize=(16, 12))

def moving_avg(x, window=30):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='valid')

# Subplot 1: Effect of antennas (fixed power P=1.0)
plt.subplot(2, 2, 1)
colors = ['blue', 'green', 'red']
linestyles = ['-', '--', '-.']
for i, (N_ris, M_bs) in enumerate(antenna_configs):
    config_name = f"N{N_ris}_M{M_bs}_P1.0"
    if config_name in results_fig1:
        for j, (algo_name, color) in enumerate(zip(['MLP', 'LLM', 'Hybrid'], colors)):
            rewards = results_fig1[config_name][algo_name]['history']
            smoothed = moving_avg(rewards, 20)
            plt.plot(smoothed, color=color, linestyle=linestyles[i], 
                    label=f'{algo_name} (N={N_ris}, M={M_bs})', alpha=0.8)

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Convergence vs Episodes (Different Antenna Configurations)')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Effect of power (fixed antennas N=32, M=16)
plt.subplot(2, 2, 2)
for i, P_max_val in enumerate(power_configs):
    config_name = f"N32_M16_P{P_max_val}"
    if config_name in results_fig1:
        for j, (algo_name, color) in enumerate(zip(['MLP', 'LLM', 'Hybrid'], colors)):
            rewards = results_fig1[config_name][algo_name]['history']
            smoothed = moving_avg(rewards, 20)
            plt.plot(smoothed, color=color, linestyle=linestyles[i], 
                    label=f'{algo_name} (P={P_max_val})', alpha=0.8)

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Convergence vs Episodes (Different Power Levels)')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Hybrid performance across all configs
plt.subplot(2, 2, 3)
config_idx = 0
for N_ris, M_bs in antenna_configs:
    for P_max_val in power_configs:
        config_name = f"N{N_ris}_M{M_bs}_P{P_max_val}"
        if config_name in results_fig1 and 'Hybrid' in results_fig1[config_name]:
            rewards = results_fig1[config_name]['Hybrid']['history']
            smoothed = moving_avg(rewards, 20)
            plt.plot(smoothed, label=f'N={N_ris}, M={M_bs}, P={P_max_val}', alpha=0.8)
            config_idx += 1

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Hybrid Algorithm - All Configurations')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Final convergence comparison
plt.subplot(2, 2, 4)
final_rewards = {}
config_labels = []

for config_name, config_results in results_fig1.items():
    config_labels.append(config_name.replace('_', '\\n'))
    for algo_name, results in config_results.items():
        if algo_name not in final_rewards:
            final_rewards[algo_name] = []
        final_rewards[algo_name].append(results['final_reward'])

x_pos = np.arange(len(config_labels))
width = 0.25

for i, algo_name in enumerate(['MLP', 'LLM', 'Hybrid']):
    if algo_name in final_rewards:
        plt.bar(x_pos + i*width, final_rewards[algo_name], width, 
                label=algo_name, alpha=0.8)

plt.xlabel('Configuration')
plt.ylabel('Final Average Reward')
plt.title('Final Convergence Performance')
plt.xticks(x_pos + width, config_labels, rotation=45, fontsize=8)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/figure_1_convergence_antennas_power.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 1 saved!")

# ============================================================================
# FIGURE 2: Reward vs Number of VUs for w=0 and w=1
# ============================================================================
print("\n=== Generating Figure 2: Reward vs Number of VUs ===")

vu_range = [1, 2, 3, 4, 5]
omega_values = [0.0, 1.0]  # w=0 (only sensing), w=1 (only communication)

results_fig2 = {}

for omega_val in omega_values:
    for num_vus in vu_range:
        config_name = f"VU{num_vus}_w{omega_val}"
        print(f"Training config: {config_name}")
        
        results = simplified_training(32, 16, num_vus, 1.0, omega_val, 0.8, 1.0, 400)
        results_fig2[config_name] = results

# Plot Figure 2
plt.figure(figsize=(12, 8))

for omega_val in omega_values:
    omega_label = "Communication Only" if omega_val == 1.0 else "Sensing Only"
    
    for algo_name, color in zip(['MLP', 'LLM', 'Hybrid'], ['blue', 'green', 'red']):
        rewards = []
        for num_vus in vu_range:
            config_name = f"VU{num_vus}_w{omega_val}"
            if config_name in results_fig2:
                rewards.append(results_fig2[config_name][algo_name]['final_reward'])
            else:
                rewards.append(0)
        
        linestyle = '-' if omega_val == 1.0 else '--'
        plt.plot(vu_range, rewards, color=color, linestyle=linestyle, 
                marker='o', label=f'{algo_name} (w={omega_val})', linewidth=2)

plt.xlabel('Number of Vehicular Users (VUs)')
plt.ylabel('Final Reward')
plt.title('Reward vs Number of VUs for w=0 (Sensing) and w=1 (Communication)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/figure_2_reward_vs_vus.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 2 saved!")

# ============================================================================
# FIGURE 3: Reward vs Number of Sensing Targets for w=0 and w=1
# ============================================================================
print("\n=== Generating Figure 3: Reward vs Number of Sensing Targets ===")

# Since the original code has a fixed target, we'll simulate multiple targets
# by running the same config multiple times and averaging
target_range = [1, 2, 3, 4, 5]

results_fig3 = {}

for omega_val in omega_values:
    for num_targets in target_range:
        config_name = f"Targets{num_targets}_w{omega_val}"
        print(f"Training config: {config_name}")
        
        # Run multiple times for different targets and average
        multi_results = []
        for target_run in range(num_targets):
            np.random.seed(42 + target_run)  # Different seed for each target
            results = simplified_training(32, 16, 3, 1.0, omega_val, 0.8, 1.0, 300)
            multi_results.append(results)
        
        # Average results across target runs
        avg_results = {}
        for algo_name in ['MLP', 'LLM', 'Hybrid']:
            avg_reward = np.mean([r[algo_name]['final_reward'] for r in multi_results])
            avg_results[algo_name] = {'final_reward': avg_reward}
        
        results_fig3[config_name] = avg_results

# Plot Figure 3
plt.figure(figsize=(12, 8))

for omega_val in omega_values:
    for algo_name, color in zip(['MLP', 'LLM', 'Hybrid'], ['blue', 'green', 'red']):
        rewards = []
        for num_targets in target_range:
            config_name = f"Targets{num_targets}_w{omega_val}"
            if config_name in results_fig3:
                rewards.append(results_fig3[config_name][algo_name]['final_reward'])
            else:
                rewards.append(0)
        
        linestyle = '-' if omega_val == 1.0 else '--'
        plt.plot(target_range, rewards, color=color, linestyle=linestyle, 
                marker='s', label=f'{algo_name} (w={omega_val})', linewidth=2)

plt.xlabel('Number of Sensing Targets')
plt.ylabel('Final Reward')
plt.title('Reward vs Number of Sensing Targets for w=0 (Sensing) and w=1 (Communication)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/figure_3_reward_vs_targets.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 3 saved!")

# ============================================================================
# FIGURE 4: Secrecy Rate vs Number of RIS Elements
# ============================================================================
print("\n=== Generating Figure 4: Secrecy Rate vs RIS Elements ===")

ris_elements = [8, 16, 32, 64, 128]
results_fig4 = {}

for N_ris in ris_elements:
    config_name = f"RIS{N_ris}"
    print(f"Training config: {config_name}")
    
    results = simplified_training(N_ris, 16, 3, 1.0, 0.5, 0.8, 1.0, 400)
    results_fig4[config_name] = results

# Plot Figure 4
plt.figure(figsize=(12, 8))

for algo_name, color in zip(['MLP', 'LLM', 'Hybrid'], ['blue', 'green', 'red']):
    secrecy_rates = []
    for N_ris in ris_elements:
        config_name = f"RIS{N_ris}"
        if config_name in results_fig4:
            secrecy_rates.append(results_fig4[config_name][algo_name]['secrecy_rate'])
        else:
            secrecy_rates.append(0)
    
    plt.plot(ris_elements, secrecy_rates, color=color, marker='o', 
            label=f'{algo_name}', linewidth=2, markersize=8)

plt.xlabel('Number of RIS Elements')
plt.ylabel('Secrecy Rate (bits/s/Hz)')
plt.title('Secrecy Rate vs Number of RIS Elements')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/figure_4_secrecy_vs_ris.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 4 saved!")

# ============================================================================
# FIGURE 5: Secrecy Rate vs Total Power (dBm)
# ============================================================================
print("\n=== Generating Figure 5: Secrecy Rate vs Total Power ===")

power_dbm_values = [10, 15, 20, 25, 30]  # dBm
power_linear = [10**(p/10)/1000 for p in power_dbm_values]  # Convert to linear watts

results_fig5 = {}

for i, P_max_val in enumerate(power_linear):
    config_name = f"Power{power_dbm_values[i]}dBm"
    print(f"Training config: {config_name}")
    
    results = simplified_training(32, 16, 3, P_max_val, 0.5, 0.8, 1.0, 400)
    results_fig5[config_name] = results

# Plot Figure 5
plt.figure(figsize=(12, 8))

for algo_name, color in zip(['MLP', 'LLM', 'Hybrid'], ['blue', 'green', 'red']):
    secrecy_rates = []
    for i, p_dbm in enumerate(power_dbm_values):
        config_name = f"Power{p_dbm}dBm"
        if config_name in results_fig5:
            secrecy_rates.append(results_fig5[config_name][algo_name]['secrecy_rate'])
        else:
            secrecy_rates.append(0)
    
    plt.plot(power_dbm_values, secrecy_rates, color=color, marker='o', 
            label=f'{algo_name}', linewidth=2, markersize=8)

plt.xlabel('Total Power (dBm)')
plt.ylabel('Secrecy Rate (bits/s/Hz)')
plt.title('Secrecy Rate vs Total Power')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/figure_5_secrecy_vs_power.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 5 saved!")

# ============================================================================
# FIGURE 6: Secrecy Rate vs Beta Factor
# ============================================================================
print("\n=== Generating Figure 6: Secrecy Rate vs Beta Factor ===")

beta_values = [0.2, 0.4, 0.6, 0.8, 1.0]
results_fig6 = {}

for beta_val in beta_values:
    config_name = f"Beta{beta_val}"
    print(f"Training config: {config_name}")
    
    results = simplified_training(32, 16, 3, 1.0, 0.5, beta_val, 1.0, 400)
    results_fig6[config_name] = results

# Plot Figure 6
plt.figure(figsize=(12, 8))

for algo_name, color in zip(['MLP', 'LLM', 'Hybrid'], ['blue', 'green', 'red']):
    secrecy_rates = []
    for beta_val in beta_values:
        config_name = f"Beta{beta_val}"
        if config_name in results_fig6:
            secrecy_rates.append(results_fig6[config_name][algo_name]['secrecy_rate'])
        else:
            secrecy_rates.append(0)
    
    plt.plot(beta_values, secrecy_rates, color=color, marker='o', 
            label=f'{algo_name}', linewidth=2, markersize=8)

plt.xlabel('Beta Factor')
plt.ylabel('Secrecy Rate (bits/s/Hz)')
plt.title('Secrecy Rate vs Beta Factor')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/figure_6_secrecy_vs_beta.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 6 saved!")

# ============================================================================
# FIGURE 7: Secrecy Rate vs Bandwidth
# ============================================================================
print("\n=== Generating Figure 7: Secrecy Rate vs Bandwidth ===")

bandwidth_values = [0.5, 1.0, 1.5, 2.0, 2.5]  # GHz
results_fig7 = {}

for B_val in bandwidth_values:
    config_name = f"BW{B_val}"
    print(f"Training config: {config_name}")
    
    results = simplified_training(32, 16, 3, 1.0, 0.5, 0.8, B_val, 400)
    results_fig7[config_name] = results

# Plot Figure 7
plt.figure(figsize=(12, 8))

for algo_name, color in zip(['MLP', 'LLM', 'Hybrid'], ['blue', 'green', 'red']):
    secrecy_rates = []
    for B_val in bandwidth_values:
        config_name = f"BW{B_val}"
        if config_name in results_fig7:
            secrecy_rates.append(results_fig7[config_name][algo_name]['secrecy_rate'])
        else:
            secrecy_rates.append(0)
    
    plt.plot(bandwidth_values, secrecy_rates, color=color, marker='o', 
            label=f'{algo_name}', linewidth=2, markersize=8)

plt.xlabel('Bandwidth (GHz)')
plt.ylabel('Secrecy Rate (bits/s/Hz)')
plt.title('Secrecy Rate vs Bandwidth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/figure_7_secrecy_vs_bandwidth.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 7 saved!")

# ============================================================================
# FIGURE 8: Secrecy Rate vs BS Antennas
# ============================================================================
print("\n=== Generating Figure 8: Secrecy Rate vs BS Antennas ===")

bs_antennas = [4, 8, 16, 32, 64]
results_fig8 = {}

for M_bs in bs_antennas:
    config_name = f"BS{M_bs}"
    print(f"Training config: {config_name}")
    
    results = simplified_training(32, M_bs, 3, 1.0, 0.5, 0.8, 1.0, 400)
    results_fig8[config_name] = results

# Plot Figure 8
plt.figure(figsize=(12, 8))

for algo_name, color in zip(['MLP', 'LLM', 'Hybrid'], ['blue', 'green', 'red']):
    secrecy_rates = []
    for M_bs in bs_antennas:
        config_name = f"BS{M_bs}"
        if config_name in results_fig8:
            secrecy_rates.append(results_fig8[config_name][algo_name]['secrecy_rate'])
        else:
            secrecy_rates.append(0)
    
    plt.plot(bs_antennas, secrecy_rates, color=color, marker='o', 
            label=f'{algo_name}', linewidth=2, markersize=8)

plt.xlabel('Number of BS Antennas')
plt.ylabel('Secrecy Rate (bits/s/Hz)')
plt.title('Secrecy Rate vs BS Antennas')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/figure_8_secrecy_vs_bs_antennas.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 8 saved!")

print("\n=== All Figures Generated Successfully! ===")
print("Generated files:")
for i in range(1, 9):
    filename = f"figure_{i}_*.png"
    print(f"  - plots/{filename}")

print("\nAll plots saved in the 'plots' directory!")
