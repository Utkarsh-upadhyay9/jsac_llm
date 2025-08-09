#!/usr/bin/env python
"""
Quick script to update all font sizes to 14 and fix axis limits in comprehensive_plots.py
"""

def update_styling():
    with open('comprehensive_plots.py', 'r') as f:
        content = f.read()
    
    # Update font sizes from 12 to 14
    content = content.replace("plt.rcParams.update({'font.size': 12})", "plt.rcParams.update({'font.size': 14})")
    content = content.replace("fontsize=12", "fontsize=14")
    
    # Add axis limits to remove padding - update set_xlim calls to ensure exact boundaries
    content = content.replace("ax_left.set_xlim(vu_range.min(), vu_range.max())", 
                             "ax_left.set_xlim(vu_range[0], vu_range[-1])")
    content = content.replace("ax_left.set_xlim(targets.min(), targets.max())", 
                             "ax_left.set_xlim(targets[0], targets[-1])")
    
    # Add explicit axis limits for single-axis plots
    content = content.replace("plt.tight_layout()", """# Force exact x-axis limits - no padding
    ax = plt.gca()
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
    
    plt.tight_layout()""")
    
    with open('comprehensive_plots.py', 'w') as f:
        f.write(content)
    
    print("✓ Updated styling in comprehensive_plots.py")

if __name__ == "__main__":
    update_styling()
