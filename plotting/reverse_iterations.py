import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Data extracted from the GTest logs
raw_data = {
    "int": {
        "Sequential": [
            (10, 0), (100, 2), (1000, 20), (10000, 137),
            (100000, 1349), (1000000, 13303), (2000000, 28143)
        ],
        "Reverse Sequential": [
            (10, 0), (100, 1), (1000, 14), (10000, 131),
            (100000, 1333), (1000000, 14957), (2000000, 30518)
        ],
        "Random": [  # Jump Iterate
            (10, 2), (100, 5), (1000, 51), (10000, 951),
            (100000, 11031), (1000000, 161868), (2000000, 326825)
        ],
        "Reverse Random": [  # Reverse Jump Iterate
            (10, 0), (100, 5), (1000, 56), (10000, 969),
            (100000, 10907), (1000000, 160237), (2000000, 330656)
        ],
    },
    "long": {
        "Sequential": [
            (10, 0), (100, 2), (1000, 25), (10000, 248),
            (100000, 2637), (1000000, 26727), (2000000, 114343)
        ],
        "Reverse Sequential": [
            (10, 0), (100, 2), (1000, 24), (10000, 248),
            (100000, 2606), (1000000, 26366), (2000000, 119823)
        ],
        "Random": [
            (10, 1), (100, 5), (1000, 49), (10000, 1000),
            (100000, 15644), (1000000, 161880), (2000000, 750104)
        ],
        "Reverse Random": [
            (10, 0), (100, 5), (1000, 67), (10000, 1004),
            (100000, 15184), (1000000, 164766), (2000000, 808657)
        ],
    },
    "double": {
        "Sequential": [
            (10, 0), (100, 2), (1000, 25), (10000, 253),
            (100000, 2674), (1000000, 28038), (2000000, 117098)
        ],
        "Reverse Sequential": [
            (10, 0), (100, 2), (1000, 25), (10000, 250),
            (100000, 2611), (1000000, 26645), (2000000, 121011)
        ],
        "Random": [
            (10, 1), (100, 5), (1000, 49), (10000, 1004),
            (100000, 14041), (1000000, 152211), (2000000, 894905)
        ],
        "Reverse Random": [
            (10, 0), (100, 6), (1000, 72), (10000, 1059),
            (100000, 15548), (1000000, 197694), (2000000, 1926702)
        ],
    }
}

def smart_format_time(x, pos):
    """Format time values with appropriate units and precision"""
    if x == 0:
        return '0'
    elif x < 1:
        return f'{x*1000:.0f}μs'
    elif x < 1000:
        return f'{x:.0f}ms'
    elif x < 60000:
        return f'{x/1000:.1f}s'
    else:
        return f'{x/60000:.1f}min'

def smart_format_size(x, pos):
    """Format array size with appropriate units"""
    if x < 1000:
        return f'{x:.0f}'
    elif x < 1000000:
        return f'{x/1000:.0f}K'
    else:
        return f'{x/1000000:.1f}M'

def calculate_complexity_line(sizes, base_time, complexity='linear'):
    """Calculate theoretical complexity lines for comparison"""
    if complexity == 'linear':
        return [base_time * (size / sizes[0]) for size in sizes]
    elif complexity == 'quadratic':
        return [base_time * (size / sizes[0])**2 for size in sizes]
    elif complexity == 'nlogn':
        return [base_time * (size / sizes[0]) * np.log2(size / sizes[0]) for size in sizes]

def add_performance_annotations(ax, sizes, seq_times, rand_times):
    """Add performance ratio annotations and highlight significant differences"""
    
    # Calculate performance ratios
    ratios = []
    for i, (seq, rand) in enumerate(zip(seq_times, rand_times)):
        if seq > 0:
            ratio = rand / seq
            ratios.append((sizes[i], ratio))
    
    # Add ratio annotations for largest sizes
    if len(ratios) >= 2:
        size, ratio = ratios[-1]  # Last (largest) size
        if ratio > 2:  # Only annotate significant differences
            ax.annotate(f'{ratio:.1f}× slower',
                       xy=(size, rand_times[-1]),
                       xytext=(20, 20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='orange', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', lw=1.2),
                       fontsize=9, fontweight='bold')

def plot_individual_benchmarks(data):
    """Plot individual benchmark comparisons for each data type with CS standards"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Memory Access Pattern Performance Analysis\nAll Iteration Patterns', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    # Updated colors for 4 patterns
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # Blue, Green, Orange, Red
    markers = ['o', '^', 's', 'D']  # Circle, Triangle, Square, Diamond
    
    for idx, (dtype, modes) in enumerate(data.items()):
        ax = axes[idx]
        
        all_times = []
        
        # Plot all four patterns
        for i, (mode, values) in enumerate(modes.items()):
            sizes = [x[0] for x in values]
            times = [max(x[1], 0.1) for x in values]  # Avoid log(0)
            all_times.extend(times)
            
            # Plot lines with different styles for each pattern
            ax.plot(sizes, times, marker=markers[i], color=colors[i], 
                   label=mode, linewidth=2.5, markersize=6,
                   markerfacecolor='white', markeredgewidth=2)
        
        # Add theoretical complexity reference (linear for sequential)
        seq_times = [max(x[1], 0.1) for x in modes['Sequential']]
        if seq_times[1] > 0:
            theoretical = calculate_complexity_line(sizes, seq_times[1])
            ax.plot(sizes, theoretical, '--', color='gray', alpha=0.6, 
                   label='O(n) Reference', linewidth=1.5)
        
        # Formatting
        ax.set_title(f'{dtype.upper()} Data Type\n({8 if dtype in ["double", "long"] else 4} bytes per element)', 
                    fontweight='bold', pad=15)
        ax.set_xlabel('Array Size (elements)')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Custom formatters
        ax.xaxis.set_major_formatter(FuncFormatter(smart_format_size))
        ax.yaxis.set_major_formatter(FuncFormatter(smart_format_time))
        
        # Enhanced grid
        ax.grid(True, which="major", linestyle='-', alpha=0.4)
        ax.grid(True, which="minor", linestyle=':', alpha=0.2)
        
        # Legend with better positioning
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('memory_access_patterns.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def plot_combined_analysis(data):
    """Create a comprehensive analysis with multiple subplots for all patterns"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Comprehensive Memory Access Performance Analysis - All Patterns', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))
    
    # Create subplots for each access pattern
    patterns = ['Sequential', 'Reverse Sequential', 'Random', 'Reverse Random']
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for pattern_idx, pattern in enumerate(patterns):
        ax = fig.add_subplot(gs[subplot_positions[pattern_idx][0], 
                                subplot_positions[pattern_idx][1]])
        
        for i, (dtype, modes) in enumerate(data.items()):
            if pattern in modes:
                sizes = [x[0] for x in modes[pattern]]
                times = [max(x[1], 0.1) for x in modes[pattern]]
                
                ax.plot(sizes, times, 'o-', color=colors[i], label=f'{dtype}',
                        linewidth=2, markersize=6)
        
        ax.set_title(f'{pattern} Access', fontweight='bold')
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Time (ms)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(smart_format_size))
        ax.yaxis.set_major_formatter(FuncFormatter(smart_format_time))
    
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

def create_summary_table(data):
    """Create a summary table of key performance metrics"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Data Type', 'Size (bytes)', 'Seq. Time (ms)\n@ 2M elements', 
               'Rand. Time (ms)\n@ 2M elements', 'Performance Ratio', 
               'Cache Efficiency']
    
    for dtype, modes in data.items():
        size_bytes = 8 if dtype in ['double', 'long'] else 4
        seq_time = modes['Sequential'][-1][1]
        rand_time = modes['Random'][-1][1]
        ratio = rand_time / seq_time if seq_time > 0 else float('inf')
        
        # Calculate cache efficiency (lower ratio = better cache usage)
        efficiency = "Excellent" if ratio < 5 else "Good" if ratio < 15 else "Poor"
        
        table_data.append([
            dtype.upper(),
            str(size_bytes),
            f"{seq_time:,}",
            f"{rand_time:,}",
            f"{ratio:.1f}×",
            efficiency
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if j == 5:  # Cache efficiency column
                cell_text = table_data[i-1][j]
                if cell_text == "Excellent":
                    table[(i, j)].set_facecolor('#E8F5E8')
                elif cell_text == "Good":
                    table[(i, j)].set_facecolor('#FFF3CD')
                else:
                    table[(i, j)].set_facecolor('#F8D7DA')
    
    plt.title('Performance Summary - Memory Access Patterns\n2M Element Arrays',
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

# Execute the analysis
if __name__ == "__main__":
    print("Generating comprehensive benchmark visualizations...")
    print("=" * 60)
    
    # Generate all visualizations
    plot_individual_benchmarks(raw_data)
    plot_combined_analysis(raw_data)
    
    print("\nVisualization complete!")
    print("Generated files:")
    print("• memory_access_patterns.png - Individual data type comparisons")
    print("• comprehensive_analysis.png - Combined analysis with ratios")
