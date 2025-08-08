#!/usr/bin/env python3
"""
Generate All Active RIS + DAM Plots
Master script to generate all visualization plots for the Active RIS implementation
"""

import os
import subprocess
import time
import sys

print("=== Active RIS + DAM Plotting Suite ===")
print("Generating comprehensive analysis plots for Active RIS with DAM implementation")
print("Showing advanced features: Active/Passive switching, Delay Alignment Modulation")

# Ensure plots directory exists
if not os.path.exists('../plots'):
    os.makedirs('../plots')
    print("✓ Created plots directory")

# List of plotting scripts
plot_scripts = [
    'plot_active_vs_basic_ris.py',
    'plot_dam_analysis.py', 
    'plot_power_optimization.py',
    'plot_actor_comparison.py'
]

plot_descriptions = [
    "Active vs Basic RIS Performance Comparison",
    "DAM Delay Pattern Analysis",
    "Power Budget Optimization",
    "Actor Network Performance Comparison"
]

print(f"\nGenerating {len(plot_scripts)} comprehensive analysis plots...")
start_time = time.time()

success_count = 0
failed_plots = []

for i, (script, description) in enumerate(zip(plot_scripts, plot_descriptions), 1):
    print(f"\n[{i}/{len(plot_scripts)}] {description}")
    print(f"Running: {script}")
    
    if os.path.exists(script):
        try:
            # Run the plotting script
            result = subprocess.run(['python', script], 
                                  capture_output=True, text=True, 
                                  cwd='.', timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print(f"✓ Generated successfully")
                success_count += 1
            else:
                print(f"❌ Error: {result.stderr}")
                failed_plots.append(script)
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout: {script} took too long to execute")
            failed_plots.append(script)
        except Exception as e:
            print(f"❌ Failed to run {script}: {e}")
            failed_plots.append(script)
    else:
        print(f"❌ Script not found: {script}")
        failed_plots.append(script)

end_time = time.time()
total_time = end_time - start_time

print(f"\n=== Active RIS + DAM Plotting Complete ===")
print(f"Total time: {total_time:.1f} seconds")
print(f"Successfully generated: {success_count}/{len(plot_scripts)} plots")

if failed_plots:
    print(f"Failed plots: {', '.join(failed_plots)}")

print(f"\n📊 Generated Active RIS + DAM Analysis Plots:")
expected_outputs = [
    '../plots/active_ris_comparison.png',
    '../plots/dam_analysis.png',
    '../plots/power_budget_optimization.png', 
    '../plots/actor_comparison_active_ris.png'
]

for output in expected_outputs:
    if os.path.exists(output):
        print(f"✓ {output}")
    else:
        print(f"❌ {output} (not generated)")

print(f"\n🚀 Active RIS + DAM analysis completed!")
print(f"📈 Key Features Analyzed:")
print(f"   • Active vs Passive RIS performance comparison")
print(f"   • DAM delay pattern optimization for security")
print(f"   • Power budget allocation strategies")
print(f"   • MLP/LLM/Hybrid actor network specialization")
print(f"   • Energy efficiency and Pareto optimal operation")

print(f"\n🔬 Advanced Capabilities Demonstrated:")
print(f"   • {50}ns DAM delay control for anti-eavesdropping")
print(f"   • Up to {10}dB active element amplification") 
print(f"   • Dynamic active/passive element switching")
print(f"   • Multi-objective optimization (secrecy, power, efficiency)")
print(f"   • Enhanced THz channel modeling with atmospheric effects")

if success_count == len(plot_scripts):
    print(f"\n✅ All Active RIS + DAM analysis plots generated successfully!")
else:
    print(f"\n⚠️  {len(plot_scripts) - success_count} plots failed to generate. Check individual script outputs.")

print(f"\n💡 These plots demonstrate the advanced Active RIS + DAM capabilities")
print(f"   beyond the basic passive RIS implementation in jsac.py")
