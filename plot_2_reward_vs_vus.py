#!/usr/bin/env python3
"""
JSAC Figure 2: Reward vs Number of VUs for w=0 and w=1
Individual plotting script for Figure 2 from the JSAC research suite
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

print("=== JSAC Figure 2: Reward vs Number of VUs ===")

# Set device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Create plots directory
if not os.path.exists('plots'):
    os.makedirs('plots')

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
    
    # Initialize agents
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
# FIGURE 2: Reward vs Number of VUs for w=0 and w=1
# ============================================================================

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

print("Figure 2 saved as 'plots/figure_2_reward_vs_vus.png'!")
