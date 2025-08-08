# -*- coding: utf-8 -*-
"""
JSAC Active RIS with Delay Alignment Modulation (DAM) Implementation

This file implements the advanced features from:
"Secure Transmission for Active RIS-Assisted THz ISAC Systems With Delay Alignment Modulation"
Extension of basic jsac.py implementation with advanced RIS capabilities.
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

print("--- JSAC Active RIS with DAM Implementation ---")

# --- GPU Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Enhanced System Parameters for Active RIS + DAM ---
M = 16          # BS antennas
N = 8           # RIS elements  
V = 4           # Number of VUs
K = 3           # Sensing targets
f_c = 30e9      # THz carrier frequency (30 GHz for initial testing)
c = 3e8         # Speed of light
wavelength = c / f_c
d_spacing = wavelength / 2  # Half-wavelength spacing

# Active RIS Parameters
P_active_max = 0.1      # Maximum active element power (W)
P_passive = 0.0         # Passive element power consumption  
eta_active = 0.8        # Active element efficiency
G_active_max = 10       # Maximum active amplification gain (dB)
G_active_lin = 10**(G_active_max/10)  # Linear gain

# DAM Parameters
tau_max = 50e-9         # Maximum delay (50 ns)
tau_resolution = 1e-9   # Delay resolution (1 ns)
N_delay_levels = int(tau_max / tau_resolution) + 1
B_signal = 1e9          # Signal bandwidth (1 GHz)

# Noise and interference
sigma2 = 1e-12          # Noise power
beta = 0.3              # Secrecy weight factor
omega = 0.7             # Communication vs sensing trade-off

print(f"Active RIS Configuration: {N} elements, {G_active_max} dB max gain")
print(f"DAM Configuration: {tau_max*1e9:.0f} ns max delay, {tau_resolution*1e9:.0f} ns resolution")

# --- Position Setup ---
iab_pos = 0
ris_pos = 30
vu_positions = np.array([50, 60, 70, 80])
eve_pos = 45  # Eavesdropper position
target_positions = np.array([40, 55, 75])

def compute_pathloss(distance, freq=f_c):
    """Enhanced path loss for THz frequencies"""
    if distance == 0:
        return 1.0
    # THz path loss with atmospheric absorption
    alpha_atm = 0.01  # Atmospheric absorption coefficient (dB/km per GHz)
    alpha_total = alpha_atm * (freq / 1e9) * (distance / 1000)  # Total atmospheric loss
    
    # Free space path loss + atmospheric loss
    fspl_db = 20 * np.log10(distance) + 20 * np.log10(freq) + 20 * np.log10(4 * np.pi / c)
    total_loss_db = fspl_db + alpha_total
    return 10**(-total_loss_db / 10)

# --- Enhanced Channel Modeling for Active RIS ---

# Generate enhanced channel matrices
def generate_enhanced_channels():
    """Generate channel matrices for Active RIS system"""
    channels = {}
    
    # BS to RIS channel (H_br) - Enhanced with THz characteristics
    H_br = np.zeros((N, M), dtype=complex)
    for n in range(N):
        for m in range(M):
            # LoS component with spatial correlation
            d_nm = np.sqrt((n * d_spacing)**2 + (m * d_spacing)**2)
            phase_nm = 2 * np.pi * d_nm / wavelength
            amplitude = np.sqrt(compute_pathloss(abs(iab_pos - ris_pos)))
            
            # Add NLoS scattering for realism
            los_component = amplitude * np.exp(1j * phase_nm)
            nlos_component = 0.1 * amplitude * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            
            H_br[n, m] = los_component + nlos_component
    
    channels['H_br'] = H_br
    
    # RIS to users channels (h_ru for each user)
    h_ru_all = np.zeros((V, N), dtype=complex)
    for v in range(V):
        for n in range(N):
            distance = abs(ris_pos - vu_positions[v])
            amplitude = np.sqrt(compute_pathloss(distance))
            phase = 2 * np.pi * distance / wavelength + 2 * np.pi * n * d_spacing * np.sin(np.pi/6) / wavelength
            h_ru_all[v, n] = amplitude * np.exp(1j * phase)
    
    channels['h_ru'] = h_ru_all
    
    # RIS to eavesdropper channel
    h_re = np.zeros(N, dtype=complex)
    for n in range(N):
        distance = abs(ris_pos - eve_pos)
        amplitude = np.sqrt(compute_pathloss(distance))
        phase = 2 * np.pi * distance / wavelength + 2 * np.pi * n * d_spacing * np.sin(np.pi/4) / wavelength
        h_re[n] = amplitude * np.exp(1j * phase)
    
    channels['h_re'] = h_re
    
    # Direct channels (BS to users, BS to eavesdropper)
    h_direct_vu = np.zeros((V, M), dtype=complex)
    for v in range(V):
        for m in range(M):
            distance = abs(iab_pos - vu_positions[v])
            amplitude = np.sqrt(compute_pathloss(distance))
            phase = 2 * np.pi * distance / wavelength + 2 * np.pi * m * d_spacing * np.sin(np.pi/8) / wavelength
            h_direct_vu[v, m] = amplitude * np.exp(1j * phase)
    
    channels['h_direct_vu'] = h_direct_vu
    
    h_direct_eve = np.zeros(M, dtype=complex)
    for m in range(M):
        distance = abs(iab_pos - eve_pos)
        amplitude = np.sqrt(compute_pathloss(distance))
        phase = 2 * np.pi * distance / wavelength + 2 * np.pi * m * d_spacing * np.sin(np.pi/5) / wavelength
        h_direct_eve[m] = amplitude * np.exp(1j * phase)
    
    channels['h_direct_eve'] = h_direct_eve
    
    return channels

# Generate base channels
base_channels = generate_enhanced_channels()
H_br = base_channels['H_br']
h_ru = base_channels['h_ru'] 
h_re = base_channels['h_re']
h_direct_vu = base_channels['h_direct_vu']
h_direct_eve = base_channels['h_direct_eve']

# --- Active RIS Element Model ---

class ActiveRISElement:
    """Model for individual active RIS element with amplification and DAM"""
    
    def __init__(self, element_id):
        self.element_id = element_id
        self.is_active = False  # Start in passive mode
        self.phase_shift = 0.0  # Phase shift (radians)
        self.amplification_gain = 1.0  # Linear amplification gain
        self.delay = 0.0  # DAM delay (seconds)
        self.power_consumption = 0.0
        
    def set_active_mode(self, gain_db=0.0, delay_ns=0.0):
        """Switch to active mode with specified gain and delay"""
        self.is_active = True
        self.amplification_gain = 10**(min(gain_db, G_active_max) / 10)
        self.delay = min(delay_ns * 1e-9, tau_max)
        self.power_consumption = P_active_max * (self.amplification_gain - 1) / G_active_lin
        
    def set_passive_mode(self):
        """Switch to passive reflection mode"""
        self.is_active = False
        self.amplification_gain = 1.0
        self.delay = 0.0
        self.power_consumption = P_passive
        
    def set_phase(self, phase_rad):
        """Set phase shift for the element"""
        self.phase_shift = phase_rad % (2 * np.pi)
        
    def apply_transformation(self, incident_signal):
        """Apply RIS transformation to incident signal"""
        # Phase shift
        phase_factor = np.exp(1j * self.phase_shift)
        
        # Amplification (active mode only)
        if self.is_active:
            # Add noise for active amplification
            noise_power = self.power_consumption * 0.1  # 10% noise figure
            noise = np.sqrt(noise_power) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            amplified_signal = self.amplification_gain * incident_signal + noise
        else:
            amplified_signal = incident_signal
            
        # Apply DAM delay (simplified as phase shift for narrowband)
        if self.delay > 0:
            delay_phase = 2 * np.pi * f_c * self.delay
            delay_factor = np.exp(-1j * delay_phase)
        else:
            delay_factor = 1.0
            
        return phase_factor * delay_factor * amplified_signal

# --- Active RIS Array Management ---

class ActiveRISArray:
    """Manage array of active RIS elements with DAM"""
    
    def __init__(self, num_elements=N):
        self.num_elements = num_elements
        self.elements = [ActiveRISElement(i) for i in range(num_elements)]
        self.total_power_budget = N * P_active_max * 0.5  # 50% power budget
        
    def configure_elements(self, config):
        """Configure RIS elements based on optimization result
        
        config: dict with 'phases', 'active_mask', 'gains_db', 'delays_ns'
        """
        phases = config.get('phases', np.zeros(self.num_elements))
        active_mask = config.get('active_mask', np.zeros(self.num_elements, dtype=bool))
        gains_db = config.get('gains_db', np.zeros(self.num_elements))
        delays_ns = config.get('delays_ns', np.zeros(self.num_elements))
        
        total_power = 0.0
        
        for i, element in enumerate(self.elements):
            element.set_phase(phases[i])
            
            if active_mask[i] and total_power < self.total_power_budget:
                element.set_active_mode(gains_db[i], delays_ns[i])
                total_power += element.power_consumption
                
                # If power budget exceeded, switch to passive
                if total_power > self.total_power_budget:
                    element.set_passive_mode()
                    total_power -= element.power_consumption
            else:
                element.set_passive_mode()
    
    def get_reflection_matrix(self):
        """Get the combined reflection matrix for all elements"""
        reflection_coeffs = np.zeros(self.num_elements, dtype=complex)
        
        for i, element in enumerate(self.elements):
            # Reflection coefficient includes phase, gain, and delay
            coeff = element.amplification_gain * np.exp(1j * element.phase_shift)
            if element.delay > 0:
                delay_phase = 2 * np.pi * f_c * element.delay
                coeff *= np.exp(-1j * delay_phase)
            reflection_coeffs[i] = coeff
            
        return np.diag(reflection_coeffs)
    
    def get_power_consumption(self):
        """Get total power consumption of the array"""
        return sum(element.power_consumption for element in self.elements)
    
    def get_active_count(self):
        """Get number of active elements"""
        return sum(1 for element in self.elements if element.is_active)

# Initialize RIS array
ris_array = ActiveRISArray(N)

# --- Enhanced Signal Processing with DAM ---

def compute_dam_enhanced_snr(ris_config, W_tau, W_o, v_idx=0):
    """Compute SNR with DAM-enhanced Active RIS"""
    
    # Configure RIS array
    ris_array.configure_elements(ris_config)
    
    # Get reflection matrix
    Theta = ris_array.get_reflection_matrix()
    
    # Enhanced channel with active amplification and DAM
    h_ris_enhanced = h_ru[v_idx, :] @ Theta @ H_br
    h_direct = h_direct_vu[v_idx, :]
    
    # Beamforming vectors for target user
    W_tau_v = W_tau[:, v_idx].reshape(-1, 1)
    W_o_v = W_o[:, v_idx].reshape(-1, 1)
    
    # Signal power with DAM enhancement
    signal_direct = h_direct @ W_tau_v
    signal_ris = h_ris_enhanced @ W_o_v
    signal_total = signal_direct + signal_ris
    signal_power = np.abs(signal_total) ** 2
    
    # Interference from other users
    interference = 0
    for j in range(V):
        if j != v_idx:
            W_tau_j = W_tau[:, j].reshape(-1, 1)
            W_o_j = W_o[:, j].reshape(-1, 1)
            
            interference_j = h_direct @ W_tau_j + h_ris_enhanced @ W_o_j
            interference += np.abs(interference_j) ** 2
    
    # Add thermal noise and active element noise
    noise_power = sigma2 + ris_array.get_power_consumption() * 0.05  # 5% noise figure
    
    sinr = signal_power / (interference + noise_power)
    return float(np.real(sinr).ravel()[0])

def compute_dam_eavesdropper_sinr(ris_config, W_tau, W_o):
    """Compute worst-case eavesdropper SINR with DAM countermeasures"""
    
    # Configure RIS array
    ris_array.configure_elements(ris_config)
    Theta = ris_array.get_reflection_matrix()
    
    # Eavesdropper channels
    h_direct_e = h_direct_eve
    h_ris_e = h_re @ Theta @ H_br
    
    # Eavesdropper tries to decode each user's signal
    max_sinr_e = 0
    
    for v in range(V):
        W_tau_v = W_tau[:, v].reshape(-1, 1)
        W_o_v = W_o[:, v].reshape(-1, 1)
        
        # Signal intended for user v
        signal_e = h_direct_e @ W_tau_v + h_ris_e @ W_o_v
        signal_power_e = np.abs(signal_e) ** 2
        
        # Interference from other users
        interference_e = 0
        for j in range(V):
            if j != v:
                W_tau_j = W_tau[:, j].reshape(-1, 1)
                W_o_j = W_o[:, j].reshape(-1, 1)
                interference_j = h_direct_e @ W_tau_j + h_ris_e @ W_o_j
                interference_e += np.abs(interference_j) ** 2
        
        # DAM creates additional interference for eavesdropper
        dam_interference = 0
        for element in ris_array.elements:
            if element.delay > 0:
                # Delayed signals create constructive interference for legitimate users
                # but destructive interference for eavesdropper due to different geometry
                dam_factor = np.abs(np.sin(2 * np.pi * f_c * element.delay)) * 0.1
                dam_interference += signal_power_e * dam_factor
        
        sinr_e = signal_power_e / (interference_e + dam_interference + sigma2)
        max_sinr_e = max(max_sinr_e, sinr_e)
    
    return float(np.real(max_sinr_e).ravel()[0])

# --- Enhanced DDPG Environment for Active RIS + DAM ---

class ActiveRISISACEnvironment:
    """Enhanced ISAC Environment with Active RIS and DAM capabilities"""
    
    def __init__(self):
        # Action space: [phases(N), active_mask(N), gains(N), delays(N), beamforming_weights(2*M*V)]
        self.action_dim = N + N + N + N + 2 * M * V  # Extended for active RIS + DAM
        
        # State space: [channel_info, power_levels, previous_rewards, RIS_status]
        self.state_dim = 2 * M * V + V + 3 + N  # Extended for RIS status
        
        self.max_power = 1.0
        self.current_channels = None
        self.episode_step = 0
        
    def reset(self):
        """Reset environment and generate new channel realization"""
        self.current_channels = generate_enhanced_channels()
        self.episode_step = 0
        
        # Initial state with channel information
        state = np.zeros(self.state_dim)
        
        # Channel state information (simplified)
        H_flat = np.concatenate([self.current_channels['H_br'].real.flatten(),
                                self.current_channels['H_br'].imag.flatten()])
        channel_dim = self.state_dim - V - 3 - N
        state[:channel_dim] = H_flat[:channel_dim]
        
        # Power levels
        state[-V-3-N:-3-N] = np.random.uniform(0.5, 1.0, V)
        
        # Previous rewards
        state[-3-N:-N] = [0, 0, 0]  # [communication, sensing, secrecy]
        
        # RIS element status (all passive initially)
        state[-N:] = np.zeros(N)
        
        return state
    
    def step(self, action):
        """Execute action and return new state, reward, done flag"""
        self.episode_step += 1
        
        # Parse action
        phases = action[:N] * 2 * np.pi  # Scale to [0, 2π]
        active_probs = torch.sigmoid(torch.tensor(action[N:2*N]))  # Convert to probabilities
        active_mask = active_probs > 0.5  # Threshold for activation
        gains_db = np.clip(action[2*N:3*N], 0, G_active_max)  # Clip gains
        delays_ns = np.clip(action[3*N:4*N] * tau_max * 1e9, 0, tau_max * 1e9)  # Scale delays
        
        # Beamforming weights (complex)
        bf_weights = action[4*N:]
        W_tau = (bf_weights[:M*V] + 1j * bf_weights[M*V:]).reshape(M, V)
        W_o = W_tau.copy()  # Simplified: same weights for both
        
        # Normalize beamforming weights
        for v in range(V):
            power_tau = np.sum(np.abs(W_tau[:, v])**2)
            power_o = np.sum(np.abs(W_o[:, v])**2)
            total_power = power_tau + power_o
            
            if total_power > self.max_power:
                scale = np.sqrt(self.max_power / total_power)
                W_tau[:, v] *= scale
                W_o[:, v] *= scale
        
        # RIS configuration
        ris_config = {
            'phases': phases,
            'active_mask': active_mask.numpy(),
            'gains_db': gains_db,
            'delays_ns': delays_ns
        }
        
        # Compute performance metrics
        communication_reward = 0
        secrecy_reward = 0
        sensing_reward = 0
        
        # Communication performance
        for v in range(V):
            snr_v = compute_dam_enhanced_snr(ris_config, W_tau, W_o, v)
            rate_v = np.log2(1 + snr_v)
            communication_reward += rate_v
        
        # Secrecy performance
        eve_snr = compute_dam_eavesdropper_sinr(ris_config, W_tau, W_o)
        eve_rate = np.log2(1 + eve_snr)
        secrecy_reward = max(0, communication_reward - eve_rate)
        
        # Sensing performance (simplified)
        sensing_power = np.sum(np.abs(W_tau)**2) + np.sum(np.abs(W_o)**2)
        sensing_reward = sensing_power * K  # Proportional to number of targets
        
        # Power penalty for active elements
        power_penalty = ris_array.get_power_consumption() / P_active_max * 0.1
        
        # Combined reward with DAM benefits
        dam_bonus = 0
        active_count = ris_array.get_active_count()
        if active_count > 0:
            # DAM provides security bonus
            dam_bonus = 0.1 * active_count / N
        
        reward = (omega * communication_reward + 
                 (1 - omega) * sensing_reward + 
                 beta * secrecy_reward + 
                 dam_bonus - power_penalty)
        
        # Next state
        next_state = self.reset()  # Simplified: new channel each step
        next_state[-3-N:-N] = [communication_reward, sensing_reward, secrecy_reward]
        next_state[-N:] = active_mask.float().numpy()
        
        done = self.episode_step >= 200
        
        info = {
            'communication_rate': communication_reward,
            'sensing_performance': sensing_reward,
            'secrecy_rate': secrecy_reward,
            'active_elements': active_count,
            'power_consumption': ris_array.get_power_consumption(),
            'dam_delay_avg': np.mean(delays_ns)
        }
        
        return next_state, reward, done, info

# --- Enhanced Actor Networks for Active RIS + DAM ---

class ActiveRISMLP(nn.Module):
    """Enhanced MLP Actor for Active RIS with DAM"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            
            # Specialized heads for different action types
            nn.Linear(hidden_dim//2, action_dim)
        )
        
        # Separate heads for better control
        self.phase_head = nn.Linear(hidden_dim//2, N)
        self.active_head = nn.Linear(hidden_dim//2, N)
        self.gain_head = nn.Linear(hidden_dim//2, N)
        self.delay_head = nn.Linear(hidden_dim//2, N)
        self.bf_head = nn.Linear(hidden_dim//2, 2*M*V)
        
    def forward(self, state):
        # Shared features
        features = self.network[:-1](state)  # All layers except final
        
        # Specialized outputs
        phases = torch.tanh(self.phase_head(features))  # [-1, 1] for phase scaling
        active_logits = self.active_head(features)  # Logits for active/passive decision
        gains = torch.sigmoid(self.gain_head(features))  # [0, 1] for gain scaling
        delays = torch.sigmoid(self.delay_head(features))  # [0, 1] for delay scaling
        bf_weights = torch.tanh(self.bf_head(features))  # [-1, 1] for beamforming
        
        # Concatenate all outputs
        action = torch.cat([phases, active_logits, gains, delays, bf_weights], dim=-1)
        return action

class ActiveRISLLM(nn.Module):
    """Enhanced LLM Actor for Active RIS with DAM using semantic understanding"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        # LLM components
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.llm = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze LLM parameters initially
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # State encoding
        self.state_encoder = nn.Linear(state_dim, 768)  # Match LLM hidden size
        
        # Context templates for RIS decisions
        self.templates = [
            "optimize active ris elements for secure communication with delay alignment",
            "configure passive ris reflection for energy efficient beamforming",
            "balance active and passive elements for maximum secrecy rate enhancement",
            "apply delay alignment modulation to counter eavesdropping attacks",
            "minimize power consumption while maintaining communication quality"
        ]
        
        # Action decoders
        self.phase_decoder = nn.Linear(768, N)
        self.active_decoder = nn.Linear(768, N)
        self.gain_decoder = nn.Linear(768, N)
        self.delay_decoder = nn.Linear(768, N)
        self.bf_decoder = nn.Linear(768, 2*M*V)
        
    def get_semantic_context(self, state):
        """Generate semantic context based on current state"""
        # Analyze state to determine context
        power_levels = state[-V-3-N:-3-N]
        prev_rewards = state[-3-N:-N]
        ris_status = state[-N:]
        
        avg_power = torch.mean(power_levels)
        secrecy_reward = prev_rewards[2]
        active_ratio = torch.mean(ris_status)
        
        # Choose appropriate template based on state
        if secrecy_reward < 0.1:
            context = self.templates[3]  # Focus on security
        elif avg_power < 0.3:
            context = self.templates[1]  # Focus on efficiency
        elif active_ratio > 0.7:
            context = self.templates[4]  # Reduce power consumption
        else:
            context = self.templates[0]  # General optimization
            
        return context
    
    def forward(self, state):
        batch_size = state.shape[0] if len(state.shape) > 1 else 1
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Get semantic context
        context = self.get_semantic_context(state[0])
        
        # Tokenize context
        tokens = self.tokenizer(context, return_tensors='pt', padding=True, truncation=True)
        tokens = {k: v.to(state.device) for k, v in tokens.items()}
        
        # Get LLM embeddings
        with torch.no_grad():
            llm_output = self.llm(**tokens)
            semantic_features = llm_output.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Encode state
        state_features = self.state_encoder(state)
        
        # Combine semantic and state features
        combined_features = semantic_features + state_features
        
        # Generate actions with semantic guidance
        phases = torch.tanh(self.phase_decoder(combined_features))
        active_logits = self.active_decoder(combined_features)
        gains = torch.sigmoid(self.gain_decoder(combined_features))
        delays = torch.sigmoid(self.delay_decoder(combined_features))
        bf_weights = torch.tanh(self.bf_decoder(combined_features))
        
        action = torch.cat([phases, active_logits, gains, delays, bf_weights], dim=-1)
        
        if batch_size == 1:
            action = action.squeeze(0)
            
        return action

class ActiveRISHybrid(nn.Module):
    """Hybrid Actor combining MLP and LLM for Active RIS with DAM"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        self.mlp_actor = ActiveRISMLP(state_dim, action_dim, hidden_dim)
        self.llm_actor = ActiveRISLLM(state_dim, action_dim, hidden_dim)
        
        # Gating network to combine outputs
        self.gate = nn.Sequential(
            nn.Linear(state_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        # Get actions from both networks
        mlp_action = self.mlp_actor(state)
        llm_action = self.llm_actor(state)
        
        # Compute gating weights
        gate_weights = self.gate(state)
        
        # Combine actions
        if len(state.shape) > 1:
            gate_weights = gate_weights.unsqueeze(-1)
        
        hybrid_action = (gate_weights[..., 0:1] * mlp_action + 
                        gate_weights[..., 1:2] * llm_action)
        
        return hybrid_action

# --- Training Setup ---

print("\n--- Active RIS + DAM Training Configuration ---")
print(f"Action Dimension: {4*N + 2*M*V} (phases, active, gains, delays, beamforming)")
print(f"State Dimension: {2*M*V + V + 3 + N} (channels, power, rewards, RIS status)")
print(f"Maximum Active Elements: {N}")
print(f"DAM Delay Range: 0-{tau_max*1e9:.0f} ns")

# Initialize environment
env = ActiveRISISACEnvironment()

# Initialize networks
state_dim = env.state_dim
action_dim = env.action_dim

actors = {
    'MLP': ActiveRISMLP(state_dim, action_dim),
    'LLM': ActiveRISLLM(state_dim, action_dim), 
    'Hybrid': ActiveRISHybrid(state_dim, action_dim)
}

# Move to GPU if available
for name, actor in actors.items():
    actor.to(device)
    print(f"✓ {name} Actor initialized on {device}")

print(f"\n🚀 Active RIS with DAM implementation ready!")
print(f"📊 Features: Active/Passive switching, {tau_max*1e9:.0f}ns DAM, {G_active_max}dB amplification")
print(f"🔒 Security: Enhanced anti-eavesdropping with delay alignment modulation")
print(f"⚡ Power: Dynamic active element management with {P_active_max*1000:.0f}mW budget")

if __name__ == "__main__":
    print("\n--- Quick Test of Active RIS + DAM System ---")
    
    # Test environment
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Test actors
    for name, actor in actors.items():
        actor.eval()
        with torch.no_grad():
            test_state = torch.FloatTensor(state).to(device)
            test_action = actor(test_state)
            print(f"{name} action shape: {test_action.shape}")
    
    # Test RIS configuration
    test_config = {
        'phases': np.random.uniform(0, 2*np.pi, N),
        'active_mask': np.random.choice([True, False], N),
        'gains_db': np.random.uniform(0, G_active_max, N),
        'delays_ns': np.random.uniform(0, tau_max*1e9, N)
    }
    
    ris_array.configure_elements(test_config)
    print(f"RIS Configuration: {ris_array.get_active_count()}/{N} active elements")
    print(f"Power Consumption: {ris_array.get_power_consumption()*1000:.1f} mW")
    
    print("\n Active RIS + DAM system test completed successfully!")
