import matplotlib.pyplot as plt
import numpy as np

# Cleaned data focusing on key patterns
raw_data = {
    "Int": {
        2: {
            "SequentialIterate": [(10, 2), (100, 2), (1000, 10), (10000, 76), (100000, 668), (1000000, 8086),
                                  (2000000, 16894)],
            "JumpIterate": [(10, 3), (100, 3), (1000, 35), (10000, 307), (100000, 5115), (1000000, 82812),
                            (2000000, 164000)],
            "NeighbourSequentialIterate": [(10, 2), (100, 11), (1000, 115), (10000, 396), (100000, 3896),
                                           (1000000, 30314), (2000000, 83899)],
        },
        4: {
            "SequentialIterate": [(10, 4), (100, 5), (1000, 8), (10000, 40), (100000, 356), (1000000, 4238),
                                  (2000000, 9299)],
            "JumpIterate": [(10, 4), (100, 2), (1000, 16), (10000, 169), (100000, 2568), (1000000, 39785),
                            (2000000, 81235)],
            "NeighbourSequentialIterate": [(10, 4), (100, 174), (1000, 160), (10000, 1014), (100000, 8144),
                                           (1000000, 17355), (2000000, 38653)],
        },
        8: {
            "SequentialIterate": [(10, 6), (100, 5), (1000, 10), (10000, 35), (100000, 293), (1000000, 3345),
                                  (2000000, 6914)],
            "JumpIterate": [(10, 6), (100, 3), (1000, 7), (10000, 84), (100000, 1271), (1000000, 16864),
                            (2000000, 38775)],
            "NeighbourSequentialIterate": [(10, 3), (100, 167), (1000, 1363), (10000, 13454), (100000, 37918),
                                           (1000000, 38065), (2000000, 69009)],
        },
        14: {
            "SequentialIterate": [(10, 5), (100, 6), (1000, 9), (10000, 35), (100000, 328), (1000000, 3445),
                                  (2000000, 7815)],
            "JumpIterate": [(10, 7), (100, 3), (1000, 5), (10000, 46), (100000, 424), (1000000, 7347),
                            (2000000, 18962)],
            "NeighbourSequentialIterate": [(10, 2), (100, 132), (1000, 1333), (10000, 12486), (100000, 47098),
                                           (1000000, 60437), (2000000, 97240)],
        }
    },
}


def create_focused_analysis():
    """Create focused visualizations highlighting cache behavior impacts"""

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multithreaded Cache Performance Analysis', fontsize=16, fontweight='bold')

    # Define colors and styles
    colors = {
        'SequentialIterate': '#2E8B57',  # Sea Green
        'JumpIterate': '#FF6B35',  # Orange Red
        'NeighbourSequentialIterate': '#8E44AD'  # Purple
    }

    thread_counts = [2, 4, 8, 14]
    patterns = ['SequentialIterate', 'JumpIterate', 'NeighbourSequentialIterate']

    # 1. Scaling Performance by Array Size
    # 1. Scaling Performance by Array Size (Updated)
    ax1 = axes[0, 0]
    ax1.set_title('Performance Scaling by Array Size', fontweight='bold')
    ax1.set_xlabel('Array Size', fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    markers = {2: 'o', 4: 's', 8: '^', 14: 'D'}

    for pattern in patterns:
        for threads in thread_counts:
            data = raw_data["Int"][threads][pattern]
            sizes = [x[0] for x in data]
            times = [x[1] for x in data]

            ax1.plot(sizes, times,
                     color=colors[pattern],
                     marker=markers[threads],
                     markersize=6,
                     linewidth=2,
                     linestyle='-' if pattern != 'NeighbourSequentialIterate' else '--',
                     label=f'{pattern.replace("Iterate", "")} ({threads}T)')

    ax1.legend(fontsize=8, loc='upper left')

    # 2. Cache Invalidation Impact (Key Insight)
    ax2 = axes[0, 1]

    array_sizes = [100, 1000, 10000, 100000, 1000000, 2000000]

    for threads in thread_counts:
        cache_impact = []
        for size in array_sizes:
            # Compare NeighbourSequentialIterate vs SequentialIterate
            seq_data = raw_data["Int"][threads]["SequentialIterate"]
            neighbor_data = raw_data["Int"][threads]["NeighbourSequentialIterate"]

            seq_time = next(x[1] for x in seq_data if x[0] == size)
            neighbor_time = next(x[1] for x in neighbor_data if x[0] == size)

            impact = neighbor_time / seq_time
            cache_impact.append(impact)

        ax2.semilogx(array_sizes, cache_impact,
                     marker='o',
                     linewidth=2.5,
                     markersize=6,
                     label=f'{threads} threads')

    ax2.set_xlabel('Array Size', fontweight='bold')
    ax2.set_ylabel('Performance Degradation Factor', fontweight='bold')
    ax2.set_title('Cache Invalidation Impact\n(Neighbour vs Sequential)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No degradation')

    # 3. Threading Efficiency Analysis
    ax3 = axes[1, 0]

    # Focus on large array (2M elements) where threading effects are clearest
    large_array_size = 2000000

    for pattern in patterns:
        times_by_threads = []
        for threads in thread_counts:
            data = raw_data["Int"][threads][pattern]
            time = next(x[1] for x in data if x[0] == large_array_size)
            times_by_threads.append(time)

        # Calculate speedup relative to 2 threads
        baseline_time = times_by_threads[0]  # 2 threads
        speedups = [baseline_time / time for time in times_by_threads]

        ax3.plot(thread_counts, speedups,
                 color=colors[pattern],
                 marker='s',
                 linewidth=3,
                 markersize=8,
                 label=pattern)

    # Add ideal speedup line
    ideal_speedup = [threads / 2 for threads in thread_counts]
    ax3.plot(thread_counts, ideal_speedup, 'k--',
             linewidth=2, alpha=0.7, label='Ideal Linear Speedup')

    ax3.set_xlabel('Number of Threads', fontweight='bold')
    ax3.set_ylabel('Speedup (vs 2 threads)', fontweight='bold')
    ax3.set_title('Threading Efficiency (2M Elements)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(thread_counts)

    # 4. Memory Access Pattern Comparison
    ax4 = axes[1, 1]

    # Show the performance penalty for each pattern across thread counts
    x_pos = np.arange(len(thread_counts))
    width = 0.25

    # Get times for 2M elements for each pattern
    seq_times = []
    jump_times = []
    neighbor_times = []

    for threads in thread_counts:
        seq_data = raw_data["Int"][threads]["SequentialIterate"]
        jump_data = raw_data["Int"][threads]["JumpIterate"]
        neighbor_data = raw_data["Int"][threads]["NeighbourSequentialIterate"]

        seq_time = next(x[1] for x in seq_data if x[0] == 2000000)
        jump_time = next(x[1] for x in jump_data if x[0] == 2000000)
        neighbor_time = next(x[1] for x in neighbor_data if x[0] == 2000000)

        seq_times.append(seq_time)
        jump_times.append(jump_time)
        neighbor_times.append(neighbor_time)

    bars1 = ax4.bar(x_pos - width, seq_times, width,
                    color=colors['SequentialIterate'], alpha=0.8,
                    label='Sequential (Cache Friendly)')
    bars2 = ax4.bar(x_pos, jump_times, width,
                    color=colors['JumpIterate'], alpha=0.8,
                    label='Jump (Cache Miss)')
    bars3 = ax4.bar(x_pos + width, neighbor_times, width,
                    color=colors['NeighbourSequentialIterate'], alpha=0.8,
                    label='Neighbour (Cache Invalidation)')

    ax4.set_xlabel('Number of Threads', fontweight='bold')
    ax4.set_ylabel('Execution Time (ms)', fontweight='bold')
    ax4.set_title('Memory Access Pattern Impact (2M Elements)', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(thread_counts)
    ax4.legend()
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars for clarity
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{int(height):,}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=8, rotation=90)

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    plt.tight_layout()
    plt.show()


def print_key_insights():
    """Print the most important benchmark insights"""
    print("=== KEY CACHE PERFORMANCE INSIGHTS ===\n")

    # Cache invalidation impact analysis
    print("1. CACHE INVALIDATION IMPACT (2M elements):")
    print("   Pattern                    | 2T     | 4T     | 8T     | 14T")
    print("   " + "-" * 55)

    for pattern in ['SequentialIterate', 'JumpIterate', 'NeighbourSequentialIterate']:
        times_str = []
        for threads in [2, 4, 8, 14]:
            data = raw_data["Int"][threads][pattern]
            time = next(x[1] for x in data if x[0] == 2000000)
            times_str.append(f"{time / 1000:5.1f}s")

        pattern_name = pattern.replace('Iterate', '').ljust(25)
        print(f"   {pattern_name} | {' | '.join(times_str)}")

    print("\n2. CACHE PENALTY FACTORS (vs Sequential):")
    for threads in [2, 4, 8, 14]:
        seq_data = raw_data["Int"][threads]["SequentialIterate"]
        jump_data = raw_data["Int"][threads]["JumpIterate"]
        neighbor_data = raw_data["Int"][threads]["NeighbourSequentialIterate"]

        seq_time = next(x[1] for x in seq_data if x[0] == 2000000)
        jump_time = next(x[1] for x in jump_data if x[0] == 2000000)
        neighbor_time = next(x[1] for x in neighbor_data if x[0] == 2000000)

        jump_penalty = jump_time / seq_time
        neighbor_penalty = neighbor_time / seq_time

        print(f"   {threads:2d} threads: Jump={jump_penalty:4.1f}x, Neighbour={neighbor_penalty:4.1f}x")

    print("\n3. SCALING EFFICIENCY (2M elements, vs 2 threads):")
    for pattern in ['SequentialIterate', 'JumpIterate', 'NeighbourSequentialIterate']:
        print(f"   {pattern.replace('Iterate', '')}:")

        baseline_data = raw_data["Int"][2][pattern]
        baseline_time = next(x[1] for x in baseline_data if x[0] == 2000000)

        for threads in [4, 8, 14]:
            data = raw_data["Int"][threads][pattern]
            time = next(x[1] for x in data if x[0] == 2000000)
            speedup = baseline_time / time
            efficiency = speedup / (threads / 2)
            print(f"     {threads:2d} threads: {speedup:4.1f}x speedup ({efficiency:4.1f} efficiency)")
        print()


# Run the focused analysis
create_focused_analysis()
print_key_insights()