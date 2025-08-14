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
            (10, 0), (100, 2), (1000, 16), (10000, 131),
            (100000, 1382), (1000000, 13782), (2000000, 31189)
        ],
        "Random": [  # Renamed from "Jump" for clarity
            (10, 2), (100, 6), (1000, 51), (10000, 945),
            (100000, 11832), (1000000, 158561), (2000000, 327873)
        ],
    },
    "long": {
        "Sequential": [
            (10, 0), (100, 2), (1000, 28), (10000, 266),
            (100000, 2815), (1000000, 28604), (2000000, 146210)
        ],
        "Random": [
            (10, 1), (100, 5), (1000, 51), (10000, 1038),
            (100000, 15165), (1000000, 166890), (2000000, 1076490)
        ],
    },
    "float": {
        "Sequential": [
            (10, 0), (100, 1), (1000, 12), (10000, 127),
            (100000, 1302), (1000000, 13352), (2000000, 27942)
        ],
        "Random": [
            (10, 2), (100, 6), (1000, 53), (10000, 999),
            (100000, 11159), (1000000, 148445), (2000000, 329537)
        ],
    },
    "double": {
        "Sequential": [
            (10, 0), (100, 2), (1000, 27), (10000, 257),
            (100000, 2656), (1000000, 29799), (2000000, 165417)
        ],
        "Random": [
            (10, 1), (100, 5), (1000, 53), (10000, 1047),
            (100000, 15545), (1000000, 187580), (2000000, 2253729)
        ],
    }
}


def smart_format_time(x, pos):
    """Format time values with appropriate units and precision"""
    if x == 0:
        return '0'
    elif x < 1:
        return f'{x * 1000:.0f}μs'
    elif x < 1000:
        return f'{x:.0f}ms'
    elif x < 60000:
        return f'{x / 1000:.1f}s'
    else:
        return f'{x / 60000:.1f}min'


def smart_format_size(x, pos):
    """Format array size with appropriate units"""
    if x < 1000:
        return f'{x:.0f}'
    elif x < 1000000:
        return f'{x / 1000:.0f}K'
    else:
        return f'{x / 1000000:.1f}M'


def calculate_complexity_line(sizes, base_time, complexity='linear'):
    """Calculate theoretical complexity lines for comparison"""
    if complexity == 'linear':
        return [base_time * (size / sizes[0]) for size in sizes]
    elif complexity == 'quadratic':
        return [base_time * (size / sizes[0]) ** 2 for size in sizes]
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

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Memory Access Pattern Performance Analysis\nSequential vs Random Access',
                 fontsize=16, fontweight='bold', y=0.98)

    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e']  # Blue for sequential, orange for random

    for idx, (dtype, modes) in enumerate(data.items()):
        ax = axes[idx]

        # Extract data
        seq_sizes = [x[0] for x in modes['Sequential']]
        seq_times = [max(x[1], 0.1) for x in modes['Sequential']]  # Avoid log(0)
        rand_sizes = [x[0] for x in modes['Random']]
        rand_times = [max(x[1], 0.1) for x in modes['Random']]

        # Plot lines with better styling
        ax.plot(seq_sizes, seq_times, 'o-', color=colors[0],
                label='Sequential Access', linewidth=2.5, markersize=7,
                markerfacecolor='white', markeredgewidth=2)
        ax.plot(rand_sizes, rand_times, 's-', color=colors[1],
                label='Random Access', linewidth=2.5, markersize=7,
                markerfacecolor='white', markeredgewidth=2)

        # Add theoretical complexity reference (linear for sequential)
        if seq_times[1] > 0:
            theoretical = calculate_complexity_line(seq_sizes, seq_times[1])
            ax.plot(seq_sizes, theoretical, '--', color='gray', alpha=0.6,
                    label='O(n) Reference', linewidth=1.5)

        # Formatting
        ax.set_title(
            f'{dtype.upper()} Data Type\n({8 if dtype == "double" else 4 if dtype == "float" else 8 if dtype == "long" else 4} bytes per element)',
            fontweight='bold', pad=15)
        ax.set_xlabel('Array Size (elements)')
        ax.set_ylabel('Execution Time (ms)')
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Custom formatters
        ax.xaxis.set_major_formatter(FuncFormatter(smart_format_size))
        ax.yaxis.set_major_formatter(FuncFormatter(smart_format_time))

        # Add performance annotations
        add_performance_annotations(ax, seq_sizes, seq_times, rand_times)

        # Enhanced grid
        ax.grid(True, which="major", linestyle='-', alpha=0.4)
        ax.grid(True, which="minor", linestyle=':', alpha=0.2)

        # Legend with better positioning
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('memory_access_patterns.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()


def plot_combined_analysis(data):
    """Create a comprehensive analysis with multiple subplots"""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Main comparison plot
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[1, :2])
    ax3 = fig.add_subplot(gs[:, 2])

    fig.suptitle('Comprehensive Memory Access Performance Analysis',
                 fontsize=16, fontweight='bold')

    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))

    # Sequential access patterns
    for i, (dtype, modes) in enumerate(data.items()):
        sizes = [x[0] for x in modes['Sequential']]
        times = [max(x[1], 0.1) for x in modes['Sequential']]

        ax1.plot(sizes, times, 'o-', color=colors[i], label=f'{dtype}',
                 linewidth=2, markersize=6)

    ax1.set_title('Sequential Access Performance', fontweight='bold')
    ax1.set_xlabel('Array Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Random access patterns
    for i, (dtype, modes) in enumerate(data.items()):
        sizes = [x[0] for x in modes['Random']]
        times = [max(x[1], 0.1) for x in modes['Random']]

        ax2.plot(sizes, times, 's-', color=colors[i], label=f'{dtype}',
                 linewidth=2, markersize=6)

    ax2.set_title('Random Access Performance', fontweight='bold')
    ax2.set_xlabel('Array Size')
    ax2.set_ylabel('Time (ms)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Performance ratio analysis
    sizes_for_ratio = [x[0] for x in data['int']['Sequential']]

    for i, (dtype, modes) in enumerate(data.items()):
        seq_times = [max(x[1], 0.1) for x in modes['Sequential']]
        rand_times = [max(x[1], 0.1) for x in modes['Random']]
        ratios = [r / s if s > 0 else 1 for s, r in zip(seq_times, rand_times)]

        ax3.plot(sizes_for_ratio, ratios, 'o-', color=colors[i],
                 label=f'{dtype}', linewidth=2, markersize=6)

    ax3.set_title('Performance Ratio\n(Random/Sequential)', fontweight='bold')
    ax3.set_xlabel('Array Size')
    ax3.set_ylabel('Performance Ratio')
    ax3.set_xscale('log')
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight',
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