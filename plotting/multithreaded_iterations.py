import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 16,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


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


def plot_cache_locality_impact(data):
    """Analyze the impact of cache locality across different access patterns"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cache Locality Impact: Sequential vs Jump Access Patterns', fontsize=16, fontweight='bold')

    data_types = ['Int', 'Long', 'Double']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for i, dtype in enumerate(data_types):
        ax = axes[i]
        thread_counts = sorted([int(k) for k in data[dtype].keys()])

        # Calculate ratio of Jump vs Sequential for each thread count
        for j, threads in enumerate(thread_counts):
            thread_data = data[dtype][threads]

            # Get 100K element performance for comparison
            seq_time = None
            jump_time = None

            for size, time in thread_data['SequentialIterate']:
                if size == 100000:
                    seq_time = time
                    break

            for size, time in thread_data['JumpIterate']:
                if size == 100000:
                    jump_time = time
                    break

            if seq_time and jump_time and seq_time > 0:
                penalty_ratio = jump_time / seq_time
                ax.bar(threads, penalty_ratio, color=colors[j], alpha=0.7,
                       label=f'{threads} threads')

        ax.set_title(f'{dtype} Arrays (100K elements)', fontweight='bold')
        ax.set_xlabel('Thread Count')
        ax.set_ylabel('Cache Miss Penalty\n(Jump/Sequential Ratio)')
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No penalty')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cache_locality_impact.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_scaling_efficiency_heatmap(data):
    """Create a heatmap showing scaling efficiency across patterns and thread counts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Thread Scaling Efficiency Heatmap\n(Efficiency = Speedup / Thread Count)',
                 fontsize=16, fontweight='bold')

    patterns = ['SequentialIterate', 'ReverseSequentialIterate', 'JumpIterate', 'ReverseJumpIterate']
    data_types = ['Int', 'Long', 'Double']
    array_size = 100000

    for idx, pattern in enumerate(patterns):
        ax = axes[idx // 2, idx % 2]

        # Prepare efficiency matrix
        thread_counts = [2, 4, 8, 14]
        efficiency_matrix = np.zeros((len(data_types), len(thread_counts)))

        for i, dtype in enumerate(data_types):
            baseline_time = None
            # Get single-thread baseline (use 2-thread as approximation)
            if '2' in data[dtype]:
                for size, time in data[dtype]['2'][pattern]:
                    if size == array_size:
                        baseline_time = time * 2  # Approximate single-thread time
                        break

            if baseline_time:
                for j, threads in enumerate(thread_counts):
                    if threads in data[dtype]:
                        for size, time in data[dtype][threads][pattern]:
                            if size == array_size and time > 0:
                                speedup = baseline_time / time
                                efficiency = speedup / threads
                                efficiency_matrix[i, j] = efficiency
                                break

        # Create heatmap
        im = ax.imshow(efficiency_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Add text annotations
        for i in range(len(data_types)):
            for j in range(len(thread_counts)):
                text = ax.text(j, i, f'{efficiency_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')

        ax.set_title(pattern.replace('Iterate', ''), fontweight='bold')
        ax.set_xticks(range(len(thread_counts)))
        ax.set_xticklabels([f'{t}T' for t in thread_counts])
        ax.set_yticks(range(len(data_types)))
        ax.set_yticklabels(data_types)
        ax.set_xlabel('Thread Count')
        if idx % 2 == 0:
            ax.set_ylabel('Data Type')

    # Add colorbar
    plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('scaling_efficiency_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_memory_bandwidth_analysis(data):
    """Analyze memory bandwidth utilization across different configurations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Memory Bandwidth Analysis: Throughput vs Array Size', fontsize=16, fontweight='bold')

    data_types = ['Int', 'Long', 'Double']
    type_sizes = {'Int': 4, 'Long': 8, 'Double': 8}  # bytes
    patterns = ['SequentialIterate', 'JumpIterate']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, dtype in enumerate(data_types):
        for j, pattern in enumerate(patterns):
            ax = axes[j, i]

            thread_counts = sorted([int(k) for k in data[dtype].keys()])

            for k, threads in enumerate(thread_counts):
                thread_data = data[dtype][threads][pattern]
                sizes = [s for s, _ in thread_data if s >= 1000]  # Focus on larger arrays
                throughputs = []

                for size, time_ms in thread_data:
                    if size >= 1000 and time_ms > 0:
                        time_s = time_ms / 1000.0
                        bytes_processed = size * type_sizes[dtype]
                        throughput_mbs = (bytes_processed / (1024 * 1024)) / time_s
                        throughputs.append(throughput_mbs)

                if sizes and throughputs:
                    ax.loglog(sizes, throughputs, 'o-', color=colors[k],
                              label=f'{threads} threads', linewidth=2, markersize=4)

            ax.set_title(f'{dtype} - {pattern.replace("Iterate", "")}', fontweight='bold')
            ax.set_xlabel('Array Size (elements)')
            ax.set_ylabel('Throughput (MB/s)')
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            ax.xaxis.set_major_formatter(FuncFormatter(smart_format_size))

    plt.tight_layout()
    plt.savefig('memory_bandwidth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_data_type_comparison(data):
    """Compare performance characteristics across data types"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Data Type Performance Comparison (8 Threads, 100K Elements)',
                 fontsize=16, fontweight='bold')

    patterns = ['SequentialIterate', 'ReverseSequentialIterate', 'JumpIterate', 'ReverseJumpIterate']
    data_types = ['Int', 'Long', 'Double']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    array_size = 100000
    threads = '8'

    for idx, pattern in enumerate(patterns):
        ax = axes[idx // 2, idx % 2]

        times = []
        for dtype in data_types:
            if threads in data[dtype] and pattern in data[dtype][threads]:
                for size, time in data[dtype][threads][pattern]:
                    if size == array_size:
                        times.append(time)
                        break
                else:
                    times.append(0)
            else:
                times.append(0)

        bars = ax.bar(data_types, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bar, time in zip(bars, times):
            if time > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                        f'{time:.0f}ms', ha='center', va='bottom', fontweight='bold')

        ax.set_title(pattern.replace('Iterate', ''), fontweight='bold')
        ax.set_ylabel('Execution Time (ms)')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('data_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_comprehensive_performance_profile(data):
    """Create a comprehensive performance profile showing key insights"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('Comprehensive Cache Performance Analysis', fontsize=18, fontweight='bold')

    # 1. Cache penalty comparison (top-left)
    ax1 = fig.add_subplot(gs[0, :2])
    data_types = ['Int', 'Long', 'Double']
    thread_counts = [2, 4, 8, 14]
    array_size = 100000

    x = np.arange(len(data_types))
    width = 0.2

    for i, threads in enumerate(thread_counts):
        penalties = []
        for dtype in data_types:
            if threads in data[dtype]:
                seq_time = jump_time = None
                for size, time in data[dtype][threads]['SequentialIterate']:
                    if size == array_size:
                        seq_time = time
                        break
                for size, time in data[dtype][threads]['JumpIterate']:
                    if size == array_size:
                        jump_time = time
                        break
                if seq_time and jump_time and seq_time > 0:
                    penalties.append(jump_time / seq_time)
                else:
                    penalties.append(0)
            else:
                penalties.append(0)

        ax1.bar(x + i * width, penalties, width, label=f'{threads}T', alpha=0.8)

    ax1.set_title('Cache Miss Penalty by Data Type', fontweight='bold')
    ax1.set_xlabel('Data Type')
    ax1.set_ylabel('Jump/Sequential Ratio')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(data_types)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Best scaling pattern (top-right)
    ax2 = fig.add_subplot(gs[0, 2:])
    patterns = ['SequentialIterate', 'ReverseSequentialIterate', 'JumpIterate', 'ReverseJumpIterate']
    best_efficiencies = []

    for pattern in patterns:
        max_eff = 0
        for dtype in data_types:
            baseline = None
            if '2' in data[dtype]:
                for size, time in data[dtype]['2'][pattern]:
                    if size == array_size:
                        baseline = time * 2
                        break

            if baseline:
                for threads in [8, 14]:
                    if threads in data[dtype]:
                        for size, time in data[dtype][threads][pattern]:
                            if size == array_size and time > 0:
                                speedup = baseline / time
                                efficiency = speedup / threads
                                max_eff = max(max_eff, efficiency)
                                break
        best_efficiencies.append(max_eff)

    bars = ax2.bar(range(len(patterns)), best_efficiencies,
                   color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'], alpha=0.7)
    ax2.set_title('Peak Threading Efficiency by Pattern', fontweight='bold')
    ax2.set_ylabel('Best Efficiency Achieved')
    ax2.set_xticks(range(len(patterns)))
    ax2.set_xticklabels([p.replace('Iterate', '') for p in patterns], rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add efficiency values on bars
    for bar, eff in zip(bars, best_efficiencies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. Memory hierarchy impact (bottom)
    ax3 = fig.add_subplot(gs[1:, :])

    # Show performance degradation with array size for different patterns
    thread_count = 8
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, pattern in enumerate(patterns):
        for dtype in ['Double']:  # Focus on one data type for clarity
            if str(thread_count) in data[dtype] and pattern in data[dtype][str(thread_count)]:
                thread_data = data[dtype][str(thread_count)][pattern]
                sizes = [s for s, _ in thread_data]
                times = [t for _, t in thread_data]

                # Calculate normalized performance (elements per ms)
                normalized_perf = [s / t if t > 0 else 0 for s, t in zip(sizes, times)]

                ax3.semilogx(sizes, normalized_perf, 'o-', color=colors[i],
                             label=pattern.replace('Iterate', ''), linewidth=2, markersize=6)

    ax3.set_title(f'Memory Hierarchy Impact (Double, {thread_count} Threads)', fontweight='bold')
    ax3.set_xlabel('Array Size (elements)')
    ax3.set_ylabel('Performance (elements/ms)')
    ax3.legend()
    ax3.grid(True, which="both", alpha=0.3)
    ax3.xaxis.set_major_formatter(FuncFormatter(smart_format_size))

    # Add annotations for cache levels
    ax3.axvline(x=1000, color='red', linestyle='--', alpha=0.5, label='L1 Cache (~4KB)')
    ax3.axvline(x=100000, color='orange', linestyle='--', alpha=0.5, label='L2 Cache (~256KB)')
    ax3.axvline(x=1000000, color='purple', linestyle='--', alpha=0.5, label='L3 Cache (~8MB)')

    plt.savefig('comprehensive_performance_profile.png', dpi=300, bbox_inches='tight')
    plt.show()


# Enhanced function for access pattern comparison with better insights
def plot_access_pattern_comparison(data):
    """Enhanced access pattern comparison with scientific insights"""
    data_types = ['Int', 'Long', 'Double']
    colors = {'2': '#1f77b4', '4': '#ff7f0e', '8': '#2ca02c', '14': '#d62728'}

    for dtype in data_types:
        if dtype not in data:
            continue

        thread_blocks = data[dtype]
        patterns = list(next(iter(thread_blocks.values())).keys())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        axes = axes.flatten()

        fig.suptitle(f'{dtype} Array Performance: Cache Behavior Analysis',
                     fontsize=16, fontweight='bold', y=0.96)

        for idx, pattern in enumerate(patterns):
            ax = axes[idx]

            for threads, pat_dict in sorted(thread_blocks.items(), key=lambda x: int(x[0])):
                if pattern not in pat_dict:
                    continue

                entries = pat_dict[pattern]
                if not entries:
                    continue

                sizes = [s for s, _ in entries]
                times = [max(t, 0.1) for _, t in entries]

                # Calculate throughput for better comparison
                type_size = {'Int': 4, 'Long': 8, 'Double': 8}[dtype]
                throughputs = [(s * type_size) / (t * 1024 * 1024) for s, t in zip(sizes, times)]

                ax.loglog(sizes, throughputs, marker='o', color=colors.get(threads, '#333333'),
                          label=f'{threads} Threads', linewidth=2, markersize=5, alpha=0.8)

            pattern_clean = pattern.replace('Iterate', '')
            ax.set_title(f'{pattern_clean} Access Pattern', fontweight='bold', pad=15)
            ax.grid(True, which="major", linestyle='-', alpha=0.4)
            ax.grid(True, which="minor", linestyle=':', alpha=0.2)
            ax.xaxis.set_major_formatter(FuncFormatter(smart_format_size))
            ax.set_ylabel('Throughput (MB/s)', fontsize=11)

            # Add cache level indicators
            ax.axvline(x=1000, color='red', linestyle='--', alpha=0.3)
            ax.axvline(x=100000, color='orange', linestyle='--', alpha=0.3)
            ax.axvline(x=1000000, color='purple', linestyle='--', alpha=0.3)

            if idx == 0:  # Only show legend on first subplot
                ax.legend(fontsize=9, loc='upper right')

        # Add cache level labels
        fig.text(0.5, 0.02, 'Array Size (elements) | Vertical lines: L1 (~1K), L2 (~100K), L3 (~1M) cache boundaries',
                 ha='center', fontsize=10, style='italic')
        fig.text(0.02, 0.5, 'Memory Throughput (MB/s)', va='center', rotation='vertical', fontsize=12)

        plt.tight_layout(rect=[0.03, 0.05, 1, 0.94])
        plt.savefig(f'enhanced_access_pattern_{dtype.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()

raw_data = {
    "Int": {
        2: {
            "SequentialIterate": [(10, 2), (100, 2), (1000, 10), (10000, 76), (100000, 668), (1000000, 8086),
                                  (2000000, 16894)],
            "ReverseSequentialIterate": [(10, 2), (100, 3), (1000, 11), (10000, 75), (100000, 693), (1000000, 7394),
                                         (2000000, 14645)],
            "JumpIterate": [(10, 3), (100, 3), (1000, 35), (10000, 307), (100000, 5115), (1000000, 82812),
                            (2000000, 164000)],
            "ReverseJumpIterate": [(10, 5), (100, 10), (1000, 50), (10000, 557), (100000, 5246), (1000000, 94367),
                                   (2000000, 190048)],
        },
        4: {
            "SequentialIterate": [(10, 4), (100, 5), (1000, 8), (10000, 40), (100000, 356), (1000000, 4238),
                                  (2000000, 9299)],
            "ReverseSequentialIterate": [(10, 3), (100, 5), (1000, 9), (10000, 44), (100000, 378), (1000000, 5040),
                                         (2000000, 9667)],
            "JumpIterate": [(10, 4), (100, 2), (1000, 16), (10000, 169), (100000, 2568), (1000000, 39785),
                            (2000000, 81235)],
            "ReverseJumpIterate": [(10, 7), (100, 10), (1000, 36), (10000, 312), (100000, 2767), (1000000, 47648),
                                   (2000000, 102508)],
        },
        8: {
            "SequentialIterate": [(10, 6), (100, 5), (1000, 10), (10000, 35), (100000, 293), (1000000, 3345),
                                  (2000000, 6914)],
            "ReverseSequentialIterate": [(10, 9), (100, 5), (1000, 12), (10000, 43), (100000, 292), (1000000, 3548),
                                         (2000000, 7609)],
            "JumpIterate": [(10, 6), (100, 3), (1000, 7), (10000, 84), (100000, 1271), (1000000, 16864),
                            (2000000, 38775)],
            "ReverseJumpIterate": [(10, 17), (100, 11), (1000, 28), (10000, 162), (100000, 2111), (1000000, 32269),
                                   (2000000, 78964)],
        },
        14: {
            "SequentialIterate": [(10, 5), (100, 6), (1000, 9), (10000, 35), (100000, 328), (1000000, 3445),
                                  (2000000, 7815)],
            "ReverseSequentialIterate": [(10, 11), (100, 6), (1000, 13), (10000, 42), (100000, 367), (1000000, 3776),
                                         (2000000, 8360)],
            "JumpIterate": [(10, 7), (100, 3), (1000, 5), (10000, 46), (100000, 424), (1000000, 7347),
                            (2000000, 18962)],
            "ReverseJumpIterate": [(10, 24), (100, 17), (1000, 26), (10000, 162), (100000, 1976), (1000000, 28396),
                                   (2000000, 77311)],
        }
    },
    "Long": {
        2: {
            "SequentialIterate": [(10, 2), (100, 4), (1000, 20), (10000, 143), (100000, 1364), (1000000, 17166), (2000000, 56983)],
            "ReverseSequentialIterate": [(10, 1), (100, 3), (1000, 19), (10000, 137), (100000, 1352), (1000000, 16073), (2000000, 64069)],
            "JumpIterate": [(10, 2), (100, 4), (1000, 29), (10000, 495), (100000, 5622), (1000000, 78362), (2000000, 166771)],
            "ReverseJumpIterate": [(10, 2), (100, 8), (1000, 55), (10000, 564), (100000, 6460), (1000000, 97170), (2000000, 433030)],
        },
        4: {
            "SequentialIterate": [(10, 2), (100, 5), (1000, 10), (10000, 75), (100000, 716), (1000000, 7763), (2000000, 30964)],
            "ReverseSequentialIterate": [(10, 2), (100, 5), (1000, 12), (10000, 77), (100000, 692), (1000000, 7885), (2000000, 31198)],
            "JumpIterate": [(10, 2), (100, 1), (1000, 14), (10000, 150), (100000, 2541), (1000000, 39469), (2000000, 78389)],
            "ReverseJumpIterate": [(10, 2), (100, 10), (1000, 32), (10000, 295), (100000, 3046), (1000000, 55501), (2000000, 188263)],
        },
        8: {
            "SequentialIterate": [(10, 4), (100, 5), (1000, 10), (10000, 59), (100000, 585), (1000000, 6732), (2000000, 16369)],
            "ReverseSequentialIterate": [(10, 3), (100, 5), (1000, 13), (10000, 60), (100000, 528), (1000000, 6330), (2000000, 16459)],
            "JumpIterate": [(10, 3), (100, 1), (1000, 7), (10000, 82), (100000, 1267), (1000000, 19596), (2000000, 38688)],
            "ReverseJumpIterate": [(10, 4), (100, 7), (1000, 23), (10000, 178), (100000, 2207), (1000000, 42940), (2000000, 127463)],
        },
        14: {
            "SequentialIterate": [(10, 5), (100, 5), (1000, 11), (10000, 53), (100000, 604), (1000000, 7176), (2000000, 13563)],
            "ReverseSequentialIterate": [(10, 4), (100, 6), (1000, 13), (10000, 59), (100000, 591), (1000000, 6616), (2000000, 13578)],
            "JumpIterate": [(10, 5), (100, 0), (1000, 4), (10000, 54), (100000, 723), (1000000, 9217), (2000000, 23833)],
            "ReverseJumpIterate": [(10, 6), (100, 4), (1000, 21), (10000, 167), (100000, 2175), (1000000, 38281), (2000000, 77438)],
        },
    },
    "Double": {
        2: {
            "SequentialIterate": [(10, 5), (100, 4), (1000, 21), (10000, 143), (100000, 1364), (1000000, 17046), (2000000, 65312)],
            "ReverseSequentialIterate": [(10, 5), (100, 4), (1000, 17), (10000, 143), (100000, 1490), (1000000, 17573), (2000000, 85087)],
            "JumpIterate": [(10, 5), (100, 3), (1000, 35), (10000, 519), (100000, 5772), (1000000, 78278), (2000000, 174461)],
            "ReverseJumpIterate": [(10, 4), (100, 9), (1000, 54), (10000, 594), (100000, 6212), (1000000, 96361), (2000000, 385206)],
        },
        4: {
            "SequentialIterate": [(10, 5), (100, 6), (1000, 11), (10000, 75), (100000, 702), (1000000, 9523), (2000000, 35189)],
            "ReverseSequentialIterate": [(10, 5), (100, 7), (1000, 12), (10000, 80), (100000, 716), (1000000, 8322), (2000000, 36905)],
            "JumpIterate": [(10, 4), (100, 2), (1000, 18), (10000, 168), (100000, 2592), (1000000, 40559), (2000000, 80124)],
            "ReverseJumpIterate": [(10, 4), (100, 8), (1000, 32), (10000, 312), (100000, 2981), (1000000, 54456), (2000000, 195087)],
        },
        8: {
            "SequentialIterate": [(10, 14), (100, 7), (1000, 12), (10000, 61), (100000, 529), (1000000, 7193), (2000000, 16919)],
            "ReverseSequentialIterate": [(10, 13), (100, 7), (1000, 13), (10000, 62), (100000, 571), (1000000, 6784), (2000000, 19508)],
            "JumpIterate": [(10, 13), (100, 1), (1000, 8), (10000, 91), (100000, 1310), (1000000, 19446), (2000000, 38881)],
            "ReverseJumpIterate": [(10, 14), (100, 7), (1000, 25), (10000, 222), (100000, 2188), (1000000, 46230), (2000000, 122498)],
        },
        14: {
            "SequentialIterate": [(10, 21), (100, 7), (1000, 12), (10000, 55), (100000, 569), (1000000, 6085), (2000000, 12983)],
            "ReverseSequentialIterate": [(10, 17), (100, 7), (1000, 14), (10000, 63), (100000, 617), (1000000, 8341), (2000000, 15163)],
            "JumpIterate": [(10, 19), (100, 1), (1000, 5), (10000, 57), (100000, 752), (1000000, 9861), (2000000, 21424)],
            "ReverseJumpIterate": [(10, 18), (100, 6), (1000, 26), (10000, 200), (100000, 2136), (1000000, 33066), (2000000, 80716)],
        },
    },
}


if __name__ == '__main__':
    plot_cache_locality_impact(raw_data)
    plot_scaling_efficiency_heatmap(raw_data)
    plot_memory_bandwidth_analysis(raw_data)
    plot_data_type_comparison(raw_data)
    plot_comprehensive_performance_profile(raw_data)
    plot_access_pattern_comparison(raw_data)
