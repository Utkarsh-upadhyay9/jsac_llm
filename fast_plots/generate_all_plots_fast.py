#!/usr/bin/env python3

import os
import subprocess
import time

print("=== JSAC Fast Plotting Suite (HONEST VERSION) ===")
print("Generating all 8 figures using pre-saved data (no retraining)")
print("Showing actual implementation: basic RIS phase optimization only")

# Check if base data files exist
base_files = ['plots/MLP_rewards.npy', 'plots/LLM_rewards.npy', 'plots/Hybrid_rewards.npy']
missing_files = []

for file in base_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print("❌ Error: Missing required data files:")
    for file in missing_files:
        print(f"   - {file}")
    print("\nPlease ensure you have the reward data files from previous training runs.")
    exit(1)

print("✓ All base reward data files found")

# List of fast plotting scripts
fast_scripts = [
    'fast_plots/plot_1_convergence_antennas_power.py',
    'fast_plots/plot_2_reward_vs_vus.py',
    'fast_plots/plot_3_reward_vs_targets.py',
    'fast_plots/plot_4_secrecy_vs_ris.py',
    'fast_plots/plot_5_secrecy_vs_power.py',
    'fast_plots/plot_6_secrecy_vs_beta.py',
    'fast_plots/plot_7_secrecy_vs_bandwidth.py',
    'fast_plots/plot_8_secrecy_vs_bs_antennas.py'
]

figure_names = [
    "Convergence vs Episodes",
    "Reward vs Number of VUs", 
    "Reward vs Sensing Targets",
    "Secrecy Rate vs RIS Elements",
    "Secrecy Rate vs Power",
    "Secrecy Rate vs Beta Factor",
    "Secrecy Rate vs Bandwidth", 
    "Secrecy Rate vs BS Antennas"
]

print(f"\nGenerating {len(fast_scripts)} figures...")
start_time = time.time()

for i, (script, name) in enumerate(zip(fast_scripts, figure_names), 1):
    print(f"\n[{i}/{len(fast_scripts)}] {name}")
    
    if os.path.exists(script):
        try:
            result = subprocess.run(['python', script], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                print(f"✓ Generated successfully")
            else:
                print(f"❌ Error: {result.stderr}")
        except Exception as e:
            print(f"❌ Failed to run {script}: {e}")
    else:
        print(f"❌ Script not found: {script}")

end_time = time.time()
total_time = end_time - start_time

print(f"\n=== Fast Plotting Complete ===")
print(f"Total time: {total_time:.1f} seconds")
print(f"Average time per figure: {total_time/len(fast_scripts):.1f} seconds")

print("\n📊 Generated figures (with '_fast' suffix):")
expected_outputs = [
    'plots/figure_1_convergence_antennas_power_fast.png',
    'plots/figure_2_reward_vs_vus_fast.png', 
    'plots/figure_3_reward_vs_targets_fast.png',
    'plots/figure_4_secrecy_vs_ris_fast.png',
    'plots/figure_5_secrecy_vs_power_fast.png',
    'plots/figure_6_secrecy_vs_beta_fast.png',
    'plots/figure_7_secrecy_vs_bandwidth_fast.png',
    'plots/figure_8_secrecy_vs_bs_antennas_fast.png'
]

for output in expected_outputs:
    if os.path.exists(output):
        print(f"✓ {output}")
    else:
        print(f"❌ {output} (not generated)")

print("\n🚀 Fast plotting completed! All figures use existing training data with parameter modeling.")
print("💡 These plots are generated instantly without requiring model retraining.")
