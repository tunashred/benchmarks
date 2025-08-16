import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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


def calculate_speedup(base_times, comparison_times):
    """Calculate speedup ratio between two timing datasets"""
    speedups = []
    for (size_b, time_b), (size_c, time_c) in zip(base_times, comparison_times):
        if size_b == size_c and time_c > 0:
            speedups.append((size_b, time_b / time_c))
    return speedups


def calculate_efficiency(speedup_data, num_threads):
    """Calculate parallel efficiency (speedup / num_threads)"""
    return [(size, speedup / num_threads) for size, speedup in speedup_data]


def calculate_throughput(timing_data):
    """Calculate throughput (elements processed per time unit)"""
    return [(size, size / time if time > 0 else 0) for size, time in timing_data]


# 1. Scalability Analysis - Performance vs Array Size
def plot_scalability_analysis():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scalability Analysis: Performance vs Array Size', fontsize=16, fontweight='bold')

    data_types = ['Int', 'Long', 'Double']
    algorithms = ['SequentialIterate', 'JumpIterate']

    for i, algorithm in enumerate(algorithms):
        for j, data_type in enumerate(['Int', 'Double']):
            ax = axes[i][j]

            for threads in [2, 4, 8, 14]:
                if data_type in raw_data and threads in raw_data[data_type]:
                    data = raw_data[data_type][threads][algorithm]
                    sizes, times = zip(*data)
                    ax.loglog(sizes, times, marker='o', linewidth=2,
                              label=f'{threads} threads', markersize=6)

            ax.set_xlabel('Array Size (elements)', fontweight='bold')
            ax.set_ylabel('Execution Time (time units)', fontweight='bold')
            ax.set_title(f'{data_type} - {algorithm}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 2. Speedup Analysis
def plot_speedup_analysis():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Speedup Analysis: Parallel Performance Gains', fontsize=16, fontweight='bold')

    algorithms = ['SequentialIterate', 'ReverseSequentialIterate', 'JumpIterate', 'ReverseJumpIterate']

    for i, data_type in enumerate(['Int', 'Double']):
        for j, algorithm in enumerate(['SequentialIterate', 'JumpIterate']):
            ax = axes[i][j]

            # Use 2-thread performance as baseline
            if data_type in raw_data and 2 in raw_data[data_type]:
                baseline_data = raw_data[data_type][2][algorithm]

                for threads in [4, 8, 14]:
                    if threads in raw_data[data_type]:
                        comparison_data = raw_data[data_type][threads][algorithm]
                        speedup_data = calculate_speedup(baseline_data, comparison_data)

                        if speedup_data:
                            sizes, speedups = zip(*speedup_data)
                            ax.semilogx(sizes, speedups, marker='o', linewidth=2,
                                        label=f'{threads} threads', markersize=6)

            # Add ideal speedup lines
            sizes = [10, 100, 1000, 10000, 100000, 1000000, 2000000]
            for threads in [4, 8, 14]:
                ideal_speedup = [threads / 2] * len(sizes)  # threads/2 because baseline is 2 threads
                ax.semilogx(sizes, ideal_speedup, '--', alpha=0.5,
                            label=f'Ideal {threads}t' if j == 0 else "")

            ax.set_xlabel('Array Size (elements)', fontweight='bold')
            ax.set_ylabel('Speedup Factor', fontweight='bold')
            ax.set_title(f'{data_type} - {algorithm}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 3. Parallel Efficiency Analysis
def plot_efficiency_analysis():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Parallel Efficiency Analysis (Sequential Iteration)', fontsize=16, fontweight='bold')

    for i, data_type in enumerate(['Int', 'Long', 'Double']):
        ax = axes[i]

        algorithm = 'SequentialIterate'  # Focus on sequential for efficiency
        if data_type in raw_data and 2 in raw_data[data_type]:
            baseline_data = raw_data[data_type][2][algorithm]

            for threads in [4, 8, 14]:
                if threads in raw_data[data_type]:
                    comparison_data = raw_data[data_type][threads][algorithm]
                    speedup_data = calculate_speedup(baseline_data, comparison_data)
                    efficiency_data = calculate_efficiency(speedup_data, threads / 2)

                    if efficiency_data:
                        sizes, efficiencies = zip(*efficiency_data)
                        ax.semilogx(sizes, [e * 100 for e in efficiencies],
                                    marker='o', linewidth=2, label=f'{threads} threads', markersize=6)

        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency')
        ax.set_xlabel('Array Size (elements)', fontweight='bold')
        ax.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
        ax.set_title(f'{data_type} Data Type', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 150)

    plt.tight_layout()
    plt.show()


# 4. Memory Access Pattern Comparison
def plot_memory_access_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Memory Access Pattern Performance Comparison (8 Threads)', fontsize=16, fontweight='bold')

    patterns = [
        ('SequentialIterate', 'ReverseSequentialIterate'),
        ('JumpIterate', 'ReverseJumpIterate')
    ]

    for i, (pattern1, pattern2) in enumerate(patterns):
        for j, data_type in enumerate(['Int', 'Double']):
            ax = axes[i][j]

            threads = 8  # Focus on 8 threads for this comparison
            if data_type in raw_data and threads in raw_data[data_type]:
                # Forward pattern
                data1 = raw_data[data_type][threads][pattern1]
                sizes1, times1 = zip(*data1)
                ax.loglog(sizes1, times1, marker='o', linewidth=2,
                          label=pattern1, markersize=6)

                # Reverse pattern
                data2 = raw_data[data_type][threads][pattern2]
                sizes2, times2 = zip(*data2)
                ax.loglog(sizes2, times2, marker='s', linewidth=2,
                          label=pattern2, markersize=6)

            ax.set_xlabel('Array Size (elements)', fontweight='bold')
            ax.set_ylabel('Execution Time (time units)', fontweight='bold')
            ax.set_title(f'{data_type} - {pattern1} vs {pattern2}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 5. Data Type Performance Comparison
def plot_data_type_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Data Type Performance Comparison (8 Threads)', fontsize=16, fontweight='bold')

    algorithms = ['SequentialIterate', 'ReverseSequentialIterate', 'JumpIterate', 'ReverseJumpIterate']

    for i, algorithm in enumerate(algorithms):
        ax = axes[i // 2][i % 2]

        threads = 8
        for data_type in ['Int', 'Long', 'Double']:
            if data_type in raw_data and threads in raw_data[data_type]:
                data = raw_data[data_type][threads][algorithm]
                sizes, times = zip(*data)
                ax.loglog(sizes, times, marker='o', linewidth=2,
                          label=data_type, markersize=6)

        ax.set_xlabel('Array Size (elements)', fontweight='bold')
        ax.set_ylabel('Execution Time (time units)', fontweight='bold')
        ax.set_title(algorithm, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 6. Throughput Analysis
def plot_throughput_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Throughput Analysis: Elements Processed per Time Unit', fontsize=16, fontweight='bold')

    for i, algorithm in enumerate(['SequentialIterate', 'JumpIterate']):
        ax = axes[i]

        data_type = 'Int'  # Focus on Int for throughput
        for threads in [2, 4, 8, 14]:
            if data_type in raw_data and threads in raw_data[data_type]:
                data = raw_data[data_type][threads][algorithm]
                throughput_data = calculate_throughput(data)
                sizes, throughputs = zip(*throughput_data)
                ax.loglog(sizes, throughputs, marker='o', linewidth=2,
                          label=f'{threads} threads', markersize=6)

        ax.set_xlabel('Array Size (elements)', fontweight='bold')
        ax.set_ylabel('Throughput (elements/time)', fontweight='bold')
        ax.set_title(f'{algorithm}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 7. Performance Heatmap
def plot_performance_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Performance Heatmap: Execution Time by Thread Count and Array Size', fontsize=16, fontweight='bold')

    algorithm = 'SequentialIterate'
    array_sizes = [10, 100, 1000, 10000, 100000, 1000000, 2000000]
    thread_counts = [2, 4, 8, 14]

    for i, data_type in enumerate(['Int', 'Long', 'Double']):
        # Create matrix for heatmap
        performance_matrix = np.zeros((len(thread_counts), len(array_sizes)))

        for j, threads in enumerate(thread_counts):
            if data_type in raw_data and threads in raw_data[data_type]:
                data = raw_data[data_type][threads][algorithm]
                for size, time in data:
                    if size in array_sizes:
                        k = array_sizes.index(size)
                        performance_matrix[j][k] = time

        # Create heatmap
        im = axes[i].imshow(performance_matrix, cmap='YlOrRd', aspect='auto')

        # Set ticks and labels
        axes[i].set_xticks(range(len(array_sizes)))
        axes[i].set_xticklabels([f'{size:,}' for size in array_sizes], rotation=45)
        axes[i].set_yticks(range(len(thread_counts)))
        axes[i].set_yticklabels(thread_counts)

        axes[i].set_xlabel('Array Size (elements)', fontweight='bold')
        axes[i].set_ylabel('Thread Count', fontweight='bold')
        axes[i].set_title(f'{data_type} Data Type', fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Execution Time')

    plt.tight_layout()
    plt.show()


# Generate all plots
if __name__ == "__main__":
    print("Generating Multithreading Performance Analysis Plots...")
    print("=" * 60)

    print("\n1. Generating Scalability Analysis...")
    plot_scalability_analysis()

    print("2. Generating Speedup Analysis...")
    plot_speedup_analysis()

    print("3. Generating Parallel Efficiency Analysis...")
    plot_efficiency_analysis()

    print("4. Generating Memory Access Pattern Comparison...")
    plot_memory_access_comparison()

    print("5. Generating Data Type Performance Comparison...")
    plot_data_type_comparison()

    print("6. Generating Throughput Analysis...")
    plot_throughput_analysis()

    print("7. Generating Performance Heatmap...")
    plot_performance_heatmap()

    print("\nAll plots generated successfully!")
    print("=" * 60)
