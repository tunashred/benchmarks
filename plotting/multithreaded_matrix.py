import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

matrix_data = {
    "Int": {
        2: {
            "NaiveMul": [(512, 56), (1024, 472), (2048, 7945), (4096, 75897), (8192, 886293)],
            "OptimizedMul": [(512, 13), (1024, 100), (2048, 1173), (4096, 10767), (8192, 87659)],
        },
        4: {
            "NaiveMul": [(512, 29), (1024, 244), (2048, 4261), (4096, 40060), (8192, 494412)],
            "OptimizedMul": [(512, 9), (1024, 67), (2048, 1096), (4096, 8442), (8192, 61871)],
        },
        8: {
            "NaiveMul": [(512, 20), (1024, 192), (2048, 3044), (4096, 28783), (8192, 375189)],
            "OptimizedMul": [(512, 6), (1024, 43), (2048, 402), (4096, 3973), (8192, 43578)],
        },
        12: {
            "NaiveMul": [(512, 23), (1024, 171), (2048, 2592), (4096, 26049), (8192, 353620)],
            "OptimizedMul": [(512, 5), (1024, 42), (2048, 380), (4096, 4087), (8192, 59454)],
        },
    },
    "Long": {
        2: {
            "NaiveMul": [(512, 68), (1024, 592), (2048, 10452), (4096, 114501), (8192, 1095735)],
            "OptimizedMul": [(512, 32), (1024, 229), (2048, 2950), (4096, 23577), (8192, 170894)],
        },
        4: {
            "NaiveMul": [(512, 36), (1024, 313), (2048, 6026), (4096, 63443), (8192, 642312)],
            "OptimizedMul": [(512, 16), (1024, 143), (2048, 2360), (4096, 15779), (8192, 118379)],
        },
        8: {
            "NaiveMul": [(512, 28), (1024, 243), (2048, 3921), (4096, 43056), (8192, 581111)],
            "OptimizedMul": [(512, 12), (1024, 93), (2048, 1047), (4096, 9250), (8192, 103975)],
        },
        12: {
            "NaiveMul": [(512, 28), (1024, 255), (2048, 3393), (4096, 43500), (8192, 564294)],
            "OptimizedMul": [(512, 11), (1024, 112), (2048, 1190), (4096, 10384), (8192, 148880)],
        },
    },
    "Double": {
        2: {
            "NaiveMul": [(512, 111), (1024, 869), (2048, 13989), (4096, 139861), (8192, 1366289)],
            "OptimizedMul": [(512, 42), (1024, 384), (2048, 3251), (4096, 27700), (8192, 201510)],
        },
        4: {
            "NaiveMul": [(512, 53), (1024, 434), (2048, 6796), (4096, 79819), (8192, 785673)],
            "OptimizedMul": [(512, 24), (1024, 225), (2048, 2243), (4096, 17541), (8192, 122442)],
        },
        8: {
            "NaiveMul": [(512, 35), (1024, 328), (2048, 3810), (4096, 52901), (8192, 566516)],
            "OptimizedMul": [(512, 14), (1024, 134), (2048, 1071), (4096, 11515), (8192, 70909)],
        },
        12: {
            "NaiveMul": [(512, 35), (1024, 296), (2048, 5074), (4096, 52397), (8192, 598647)],
            "OptimizedMul": [(512, 15), (1024, 120), (2048, 1048), (4096, 14664), (8192, 95510)],
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


def calculate_gflops(matrix_size, time):
    """Calculate GFLOPS for matrix multiplication (2*n^3 operations)"""
    operations = 2 * (matrix_size ** 3)
    if time > 0:
        return operations / (time * 1e9)  # Convert to GFLOPS
    return 0


def calculate_optimization_factor(naive_data, optimized_data):
    """Calculate how much faster optimized version is vs naive"""
    factors = []
    for (size_n, time_n), (size_o, time_o) in zip(naive_data, optimized_data):
        if size_n == size_o and time_o > 0:
            factors.append((size_n, time_n / time_o))
    return factors


# 1. Algorithm Comparison: Naive vs Optimized
def plot_algorithm_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Matrix Multiplication: Algorithm Performance Comparison', fontsize=16, fontweight='bold')

    for i, data_type in enumerate(['Int', 'Long', 'Double']):
        if i >= 2:  # We have 3 data types but only 4 subplots, so we'll use 2x2 and skip one
            ax = axes[1][1] if i == 2 else None
        else:
            ax = axes[i // 2][i % 2] if i < 2 else axes[1][0]

        if ax is None:
            continue

        threads = 8  # Focus on 8 threads for comparison
        if data_type in matrix_data and threads in matrix_data[data_type]:
            # Naive algorithm
            naive_data = matrix_data[data_type][threads]["NaiveMul"]
            sizes_n, times_n = zip(*naive_data)
            ax.loglog(sizes_n, times_n, 'o-', linewidth=3, markersize=8,
                      label='Naive Algorithm', color='red', alpha=0.8)

            # Optimized algorithm
            opt_data = matrix_data[data_type][threads]["OptimizedMul"]
            sizes_o, times_o = zip(*opt_data)
            ax.loglog(sizes_o, times_o, 's-', linewidth=3, markersize=8,
                      label='Optimized Algorithm', color='blue', alpha=0.8)

            # Theoretical O(n^3) line for reference
            theoretical_sizes = np.array(sizes_n)
            theoretical_times = times_n[0] * (theoretical_sizes / sizes_n[0]) ** 3
            ax.loglog(theoretical_sizes, theoretical_times, '--', alpha=0.5,
                      color='gray', label='O(n³) Reference')

        ax.set_xlabel('Matrix Size (N×N)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Execution Time (time units)', fontweight='bold', fontsize=12)
        ax.set_title(f'{data_type} Data Type (8 Threads)', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([512, 1024, 2048, 4096, 8192])
        ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Remove the unused subplot
    axes[1][0].remove()

    plt.tight_layout()
    plt.show()


# 2. Scalability Analysis - Performance vs Matrix Size
def plot_scalability_analysis():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Scalability Analysis: Thread Count vs Matrix Size Performance', fontsize=16, fontweight='bold')

    algorithms = ['NaiveMul', 'OptimizedMul']
    data_types = ['Int', 'Long', 'Double']

    for i, algorithm in enumerate(algorithms):
        for j, data_type in enumerate(data_types):
            ax = axes[i][j]

            for threads in [2, 4, 8, 12]:
                if data_type in matrix_data and threads in matrix_data[data_type]:
                    data = matrix_data[data_type][threads][algorithm]
                    sizes, times = zip(*data)
                    ax.loglog(sizes, times, 'o-', linewidth=2, markersize=6,
                              label=f'{threads} threads', alpha=0.8)

            ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
            ax.set_ylabel('Execution Time (time units)', fontweight='bold')
            ax.set_title(f'{data_type} - {algorithm}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks([512, 1024, 2048, 4096, 8192])
            ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    plt.tight_layout()
    plt.show()


# 3. Parallel Speedup Analysis
def plot_speedup_analysis():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parallel Speedup Analysis (Relative to 2 Threads)', fontsize=16, fontweight='bold')

    algorithms = ['NaiveMul', 'OptimizedMul']
    data_types = ['Int', 'Long', 'Double']

    for i, algorithm in enumerate(algorithms):
        for j, data_type in enumerate(data_types):
            ax = axes[i][j]

            if data_type in matrix_data and 2 in matrix_data[data_type]:
                baseline_data = matrix_data[data_type][2][algorithm]

                for threads in [4, 8, 12]:
                    if threads in matrix_data[data_type]:
                        comparison_data = matrix_data[data_type][threads][algorithm]
                        speedup_data = calculate_speedup(baseline_data, comparison_data)

                        if speedup_data:
                            sizes, speedups = zip(*speedup_data)
                            ax.semilogx(sizes, speedups, 'o-', linewidth=2, markersize=6,
                                        label=f'{threads} threads', alpha=0.8)

                # Add ideal speedup lines
                sizes = [512, 1024, 2048, 4096, 8192]
                for threads in [4, 8, 12]:
                    ideal_speedup = [threads / 2] * len(sizes)
                    ax.semilogx(sizes, ideal_speedup, '--', alpha=0.4,
                                label=f'Ideal {threads}t' if j == 0 else "")

            ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
            ax.set_ylabel('Speedup Factor', fontweight='bold')
            ax.set_title(f'{data_type} - {algorithm}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks([512, 1024, 2048, 4096, 8192])
            ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    plt.tight_layout()
    plt.show()


# 4. GFLOPS Performance Analysis
def plot_gflops_analysis():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GFLOPS Performance Analysis', fontsize=16, fontweight='bold')

    # Plot 1: GFLOPS vs Matrix Size for different algorithms
    ax = axes[0][0]
    data_type = 'Int'  # Focus on Int for this comparison
    threads = 8

    if data_type in matrix_data and threads in matrix_data[data_type]:
        for algorithm in ['NaiveMul', 'OptimizedMul']:
            data = matrix_data[data_type][threads][algorithm]
            gflops_data = [(size, calculate_gflops(size, time)) for size, time in data]
            sizes, gflops = zip(*gflops_data)
            ax.semilogx(sizes, gflops, 'o-', linewidth=2, markersize=6,
                        label=algorithm, alpha=0.8)

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('GFLOPS', fontweight='bold')
    ax.set_title(f'GFLOPS Comparison ({data_type}, 8 threads)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Plot 2: GFLOPS vs Thread Count
    ax = axes[0][1]
    algorithm = 'OptimizedMul'
    matrix_size = 4096  # Focus on large matrix

    for data_type in ['Int', 'Long', 'Double']:
        thread_gflops = []
        thread_counts = []

        for threads in [2, 4, 8, 12]:
            if data_type in matrix_data and threads in matrix_data[data_type]:
                data = matrix_data[data_type][threads][algorithm]
                for size, time in data:
                    if size == matrix_size:
                        gflops = calculate_gflops(size, time)
                        thread_gflops.append(gflops)
                        thread_counts.append(threads)
                        break

        if thread_gflops:
            ax.plot(thread_counts, thread_gflops, 'o-', linewidth=2, markersize=6,
                    label=data_type, alpha=0.8)

    ax.set_xlabel('Thread Count', fontweight='bold')
    ax.set_ylabel('GFLOPS', fontweight='bold')
    ax.set_title(f'GFLOPS vs Threads ({algorithm}, {matrix_size}×{matrix_size})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Peak GFLOPS by Data Type
    ax = axes[1][0]
    data_types = ['Int', 'Long', 'Double']
    algorithms = ['NaiveMul', 'OptimizedMul']

    x_pos = np.arange(len(data_types))
    width = 0.35

    for i, algorithm in enumerate(algorithms):
        peak_gflops = []
        for data_type in data_types:
            max_gflops = 0
            for threads in [2, 4, 8, 12]:
                if data_type in matrix_data and threads in matrix_data[data_type]:
                    data = matrix_data[data_type][threads][algorithm]
                    for size, time in data:
                        gflops = calculate_gflops(size, time)
                        max_gflops = max(max_gflops, gflops)
            peak_gflops.append(max_gflops)

        ax.bar(x_pos + i * width, peak_gflops, width, label=algorithm, alpha=0.8)

    ax.set_xlabel('Data Type', fontweight='bold')
    ax.set_ylabel('Peak GFLOPS', fontweight='bold')
    ax.set_title('Peak GFLOPS by Data Type', fontweight='bold')
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(data_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Optimization Factor
    ax = axes[1][1]
    data_type = 'Int'
    threads = 8

    if data_type in matrix_data and threads in matrix_data[data_type]:
        naive_data = matrix_data[data_type][threads]['NaiveMul']
        opt_data = matrix_data[data_type][threads]['OptimizedMul']
        opt_factors = calculate_optimization_factor(naive_data, opt_data)

        sizes, factors = zip(*opt_factors)
        ax.semilogx(sizes, factors, 'o-', linewidth=3, markersize=8,
                    color='green', alpha=0.8)

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Optimization Factor (Naive/Optimized)', fontweight='bold')
    ax.set_title(f'Optimization Effectiveness ({data_type}, 8 threads)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    plt.tight_layout()
    plt.show()


# 5. Memory Efficiency Analysis
def plot_memory_efficiency():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Memory Efficiency: Data Type Performance Comparison', fontsize=16, fontweight='bold')

    algorithms = ['NaiveMul', 'OptimizedMul']
    threads = 8

    for i, algorithm in enumerate(algorithms[:2]):  # Only plot first 2 algorithms
        ax = axes[i]

        for data_type in ['Int', 'Long', 'Double']:
            if data_type in matrix_data and threads in matrix_data[data_type]:
                data = matrix_data[data_type][threads][algorithm]
                sizes, times = zip(*data)

                # Calculate memory throughput (bytes processed per time)
                type_sizes = {'Int': 4, 'Long': 8, 'Double': 8}  # bytes per element
                memory_throughput = []
                for size, time in data:
                    total_bytes = 3 * (size ** 2) * type_sizes[data_type]  # 2 input + 1 output matrix
                    if time > 0:
                        throughput = total_bytes / time / 1e6  # MB/s
                        memory_throughput.append(throughput)
                    else:
                        memory_throughput.append(0)

                ax.semilogx(sizes, memory_throughput, 'o-', linewidth=2, markersize=6,
                            label=data_type, alpha=0.8)

        ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
        ax.set_ylabel('Memory Throughput (MB/s)', fontweight='bold')
        ax.set_title(f'{algorithm} - Memory Efficiency', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([512, 1024, 2048, 4096, 8192])
        ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Plot 3: Parallel Efficiency
    ax = axes[2]
    algorithm = 'OptimizedMul'
    data_type = 'Int'

    if data_type in matrix_data and 2 in matrix_data[data_type]:
        baseline_data = matrix_data[data_type][2][algorithm]

        for threads in [4, 8, 12]:
            if threads in matrix_data[data_type]:
                comparison_data = matrix_data[data_type][threads][algorithm]
                speedup_data = calculate_speedup(baseline_data, comparison_data)
                efficiency_data = calculate_efficiency(speedup_data, threads / 2)

                if efficiency_data:
                    sizes, efficiencies = zip(*efficiency_data)
                    ax.semilogx(sizes, [e * 100 for e in efficiencies], 'o-',
                                linewidth=2, markersize=6, label=f'{threads} threads', alpha=0.8)

    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency')
    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
    ax.set_title(f'Parallel Efficiency ({data_type}, {algorithm})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 150)

    plt.tight_layout()
    plt.show()


# 6. Performance Heatmaps
def plot_performance_heatmaps():
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Performance Heatmaps: Execution Time by Thread Count and Matrix Size', fontsize=16, fontweight='bold')

    algorithms = ['NaiveMul', 'OptimizedMul']
    data_types = ['Int', 'Long', 'Double']
    matrix_sizes = [512, 1024, 2048, 4096, 8192]
    thread_counts = [2, 4, 8, 12]

    for i, algorithm in enumerate(algorithms):
        for j, data_type in enumerate(data_types):
            ax = axes[i][j]

            # Create matrix for heatmap
            performance_matrix = np.zeros((len(thread_counts), len(matrix_sizes)))

            for t_idx, threads in enumerate(thread_counts):
                if data_type in matrix_data and threads in matrix_data[data_type]:
                    data = matrix_data[data_type][threads][algorithm]
                    for size, time in data:
                        if size in matrix_sizes:
                            s_idx = matrix_sizes.index(size)
                            performance_matrix[t_idx][s_idx] = time

            # Create heatmap with log scale
            log_matrix = np.log10(performance_matrix + 1)  # +1 to avoid log(0)
            im = ax.imshow(log_matrix, cmap='YlOrRd', aspect='auto')

            # Set ticks and labels
            ax.set_xticks(range(len(matrix_sizes)))
            ax.set_xticklabels([f'{size}' for size in matrix_sizes])
            ax.set_yticks(range(len(thread_counts)))
            ax.set_yticklabels(thread_counts)

            ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
            ax.set_ylabel('Thread Count', fontweight='bold')
            ax.set_title(f'{data_type} - {algorithm}', fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('log₁₀(Execution Time)')

    plt.tight_layout()
    plt.show()


# 8. Cache Performance Analysis
def plot_cache_analysis():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle('Cache Performance Analysis: Memory Access Patterns', fontsize=16, fontweight='bold')

    # Plot 1: Performance per element vs matrix size (cache efficiency indicator)
    ax = axes[0][0]
    algorithm = 'OptimizedMul'
    threads = 8

    for data_type in ['Int', 'Long', 'Double']:
        if data_type in matrix_data and threads in matrix_data[data_type]:
            data = matrix_data[data_type][threads][algorithm]
            perf_per_element = [(size, time / (size * size)) for size, time in data if time > 0]
            sizes, perf = zip(*perf_per_element)
            ax.loglog(sizes, perf, 'o-', linewidth=2, markersize=6,
                      label=data_type, alpha=0.8)

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Time per Element', fontweight='bold')
    ax.set_title('Cache Efficiency Indicator', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Plot 2: Algorithm efficiency comparison across sizes
    ax = axes[0][1]
    data_type = 'Int'
    threads = 8

    if data_type in matrix_data and threads in matrix_data[data_type]:
        naive_data = matrix_data[data_type][threads]['NaiveMul']
        opt_data = matrix_data[data_type][threads]['OptimizedMul']

        # Calculate operations per second
        naive_ops = [(size, 2 * (size ** 3) / time) for size, time in naive_data if time > 0]
        opt_ops = [(size, 2 * (size ** 3) / time) for size, time in opt_data if time > 0]

        sizes_n, ops_n = zip(*naive_ops)
        sizes_o, ops_o = zip(*opt_ops)

        ax.loglog(sizes_n, ops_n, 'o-', linewidth=2, markersize=6,
                  label='Naive', alpha=0.8, color='red')
        ax.loglog(sizes_o, ops_o, 's-', linewidth=2, markersize=6,
                  label='Optimized', alpha=0.8, color='blue')

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Operations per Time Unit', fontweight='bold')
    ax.set_title(f'Computational Efficiency ({data_type})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Plot 3: Memory bandwidth utilization
    ax = axes[1][0]
    algorithm = 'OptimizedMul'
    threads = 8

    for data_type in ['Int', 'Long', 'Double']:
        type_sizes = {'Int': 4, 'Long': 8, 'Double': 8}
        if data_type in matrix_data and threads in matrix_data[data_type]:
            data = matrix_data[data_type][threads][algorithm]
            bandwidth = []
            sizes = []

            for size, time in data:
                if time > 0:
                    # Estimate memory accesses: 2*N^2 reads + N^2 writes per iteration, N iterations
                    total_bytes = size * (2 * size * size + size * size) * type_sizes[data_type]
                    bw = total_bytes / time / 1e6  # MB/s
                    bandwidth.append(bw)
                    sizes.append(size)

            if sizes:
                ax.semilogx(sizes, bandwidth, 'o-', linewidth=2, markersize=6,
                            label=data_type, alpha=0.8)

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Memory Bandwidth (MB/s)', fontweight='bold')
    ax.set_title('Memory Bandwidth Utilization', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Plot 4: Thread scaling efficiency
    ax = axes[1][1]
    algorithm = 'OptimizedMul'
    data_type = 'Int'

    sizes_to_plot = [2048, 4096, 8192]
    colors = ['blue', 'green', 'red']

    for i, matrix_size in enumerate(sizes_to_plot):
        thread_counts = []
        efficiencies = []
        baseline_time = None

        for threads in [2, 4, 8, 12]:
            if data_type in matrix_data and threads in matrix_data[data_type]:
                data = matrix_data[data_type][threads][algorithm]
                for size, time in data:
                    if size == matrix_size:
                        if baseline_time is None:
                            baseline_time = time
                            efficiency = 100.0
                        else:
                            speedup = baseline_time / time
                            efficiency = (speedup / (threads / 2)) * 100

                        thread_counts.append(threads)
                        efficiencies.append(efficiency)
                        break

        if thread_counts:
            ax.plot(thread_counts, efficiencies, 'o-', linewidth=2, markersize=6,
                    label=f'{matrix_size}×{matrix_size}', color=colors[i], alpha=0.8)

    ax.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax.set_xlabel('Thread Count', fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
    ax.set_title(f'Thread Scaling Efficiency ({algorithm})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 150)

    plt.tight_layout()
    plt.show()


# Generate all plots
if __name__ == "__main__":
    print("Generating Matrix Multiplication Performance Analysis Plots...")
    print("=" * 70)

    print("\n1. Generating Algorithm Comparison...")
    plot_algorithm_comparison()

    print("2. Generating Scalability Analysis...")
    plot_scalability_analysis()

    print("3. Generating Parallel Speedup Analysis...")
    plot_speedup_analysis()

    print("4. Generating GFLOPS Performance Analysis...")
    plot_gflops_analysis()

    print("5. Generating Memory Efficiency Analysis...")
    plot_memory_efficiency()

    print("6. Generating Performance Heatmaps...")
    plot_performance_heatmaps()

    print("7. Generating Cache Performance Analysis...")
    plot_cache_analysis()

    print("\nAll matrix multiplication analysis plots generated successfully!")
    print("=" * 70)

    # Print some key insights
    print("\nKey Research Insights:")
    print("-" * 30)

    # Calculate and display optimization factors
    data_type = 'Int'
    threads = 8
    if data_type in matrix_data and threads in matrix_data[data_type]:
        naive_data = matrix_data[data_type][threads]['NaiveMul']
        opt_data = matrix_data[data_type][threads]['OptimizedMul']

        print(f"\nOptimization Factors ({data_type}, {threads} threads):")
        for (size_n, time_n), (size_o, time_o) in zip(naive_data, opt_data):
            if size_n == size_o and time_o > 0:
                factor = time_n / time_o
                print(f"  {size_n}×{size_n}: {factor:.1f}x faster")

    # Calculate peak GFLOPS
    print(f"\nPeak GFLOPS Performance:")
    for data_type in ['Int', 'Long', 'Double']:
        max_gflops = 0
        best_config = ""
        for threads in [2, 4, 8, 12]:
            if data_type in matrix_data and threads in matrix_data[data_type]:
                for algorithm in ['NaiveMul', 'OptimizedMul']:
                    data = matrix_data[data_type][threads][algorithm]
                    for size, time in data:
                        gflops = calculate_gflops(size, time)
                        if gflops > max_gflops:
                            max_gflops = gflops
                            best_config = f"{threads}t, {algorithm}, {size}×{size}"

        print(f"  {data_type}: {max_gflops:.2f} GFLOPS ({best_config})")
