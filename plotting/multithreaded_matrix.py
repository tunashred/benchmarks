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

matrix_as_array_data = {
    "Int": {
        2: {
            "NaiveMul": [(512, 232), (1024, 1580), (2048, 13308), (4096, 111369), (8192, 882399)],
            "OptimizedMul": [(512, 4), (1024, 35), (2048, 269), (4096, 2462), (8192, 21827)],
        },
        4: {
            "NaiveMul": [(512, 117), (1024, 820), (2048, 7554), (4096, 58074), (8192, 464269)],
            "OptimizedMul": [(512, 4), (1024, 21), (2048, 167), (4096, 1408), (8192, 10173)],
        },
        8: {
            "NaiveMul": [(512, 68), (1024, 629), (2048, 5062), (4096, 39025), (8192, 319374)],
            "OptimizedMul": [(512, 2), (1024, 14), (2048, 116), (4096, 1008), (8192, 9196)],
        },
        12: {
            "NaiveMul": [(512, 69), (1024, 583), (2048, 4534), (4096, 34221), (8192, 283027)],
            "OptimizedMul": [(512, 2), (1024, 15), (2048, 105), (4096, 1047), (8192, 9520)],
        },
    },
    "Long": {
        2: {
            "NaiveMul": [(512, 212), (1024, 1624), (2048, 14027), (4096, 108050), (8192, 1031067)],
            "OptimizedMul": [(512, 12), (1024, 92), (2048, 830), (4096, 7282), (8192, 65527)],
        },
        4: {
            "NaiveMul": [(512, 105), (1024, 824), (2048, 7179), (4096, 54775), (8192, 533921)],
            "OptimizedMul": [(512, 8), (1024, 52), (2048, 485), (4096, 3506), (8192, 30237)],
        },
        8: {
            "NaiveMul": [(512, 76), (1024, 624), (2048, 5351), (4096, 40373), (8192, 401814)],
            "OptimizedMul": [(512, 6), (1024, 37), (2048, 337), (4096, 2926), (8192, 24866)],
        },
        12: {
            "NaiveMul": [(512, 70), (1024, 510), (2048, 3874), (4096, 34442), (8192, 337268)],
            "OptimizedMul": [(512, 5), (1024, 38), (2048, 317), (4096, 2795), (8192, 22122)],
        },
    },
    "Double": {
        2: {
            "NaiveMul": [(512, 214), (1024, 1652), (2048, 15785), (4096, 110847), (8192, 1065871)],
            "OptimizedMul": [(512, 7), (1024, 62), (2048, 601), (4096, 5487), (8192, 53597)],
        },
        4: {
            "NaiveMul": [(512, 104), (1024, 837), (2048, 8042), (4096, 56453), (8192, 554700)],
            "OptimizedMul": [(512, 6), (1024, 35), (2048, 361), (4096, 2643), (8192, 23109)],
        },
        8: {
            "NaiveMul": [(512, 75), (1024, 601), (2048, 5371), (4096, 41607), (8192, 394149)],
            "OptimizedMul": [(512, 4), (1024, 28), (2048, 263), (4096, 2337), (8192, 21221)],
        },
        12: {
            "NaiveMul": [(512, 60), (1024, 484), (2048, 3984), (4096, 34503), (8192, 324134)],
            "OptimizedMul": [(512, 4), (1024, 25), (2048, 264), (4096, 2370), (8192, 19050)],
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

def get_data_from_source(source, data_type, threads, algorithm):
    """Helper function to get data from either matrix_data or matrix_as_array_data"""
    datasets = {
        'matrix': matrix_data,
        'array': matrix_as_array_data
    }
    
    if source in datasets and data_type in datasets[source] and threads in datasets[source][data_type]:
        return datasets[source][data_type][threads].get(algorithm, [])
    return []

# 1. Implementation Comparison: Matrix vs Array vs Algorithms
def plot_implementation_comparison():
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Implementation Comparison: Matrix Objects vs Arrays vs Algorithms', fontsize=16, fontweight='bold')

    data_types = ['Int', 'Long', 'Double']
    threads = 8

    for i, data_type in enumerate(data_types):
        # Top row: Algorithm comparison within matrix implementation
        ax = axes[0][i]
        
        matrix_naive = get_data_from_source('matrix', data_type, threads, 'NaiveMul')
        matrix_opt = get_data_from_source('matrix', data_type, threads, 'OptimizedMul')
        
        if matrix_naive:
            sizes_n, times_n = zip(*matrix_naive)
            ax.loglog(sizes_n, times_n, 'o-', linewidth=3, markersize=8,
                      label='Matrix Naive', color='red', alpha=0.8)
        
        if matrix_opt:
            sizes_o, times_o = zip(*matrix_opt)
            ax.loglog(sizes_o, times_o, 's-', linewidth=3, markersize=8,
                      label='Matrix Optimized', color='blue', alpha=0.8)

        ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
        ax.set_ylabel('Execution Time (time units)', fontweight='bold')
        ax.set_title(f'{data_type} - Matrix Implementation', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([512, 1024, 2048, 4096, 8192])
        ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

        # Bottom row: Implementation comparison (Matrix vs Array)
        ax = axes[1][i]
        
        array_naive = get_data_from_source('array', data_type, threads, 'NaiveMul')
        array_opt = get_data_from_source('array', data_type, threads, 'OptimizedMul')
        
        if matrix_opt:
            ax.loglog(sizes_o, times_o, 's-', linewidth=2, markersize=6,
                      label='Matrix Optimized', color='blue', alpha=0.8)
        
        if array_naive:
            sizes_an, times_an = zip(*array_naive)
            ax.loglog(sizes_an, times_an, '^-', linewidth=2, markersize=6,
                      label='Array Naive', color='orange', alpha=0.8)
        
        if array_opt:
            sizes_ao, times_ao = zip(*array_opt)
            ax.loglog(sizes_ao, times_ao, 'v-', linewidth=2, markersize=6,
                      label='Array Optimized', color='green', alpha=0.8)

        ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
        ax.set_ylabel('Execution Time (time units)', fontweight='bold')
        ax.set_title(f'{data_type} - Implementation Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([512, 1024, 2048, 4096, 8192])
        ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    plt.tight_layout()
    plt.show()

# 2. Cross-Implementation Performance Analysis
def plot_cross_implementation_analysis():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Implementation Performance Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Best of each implementation type
    ax = axes[0][0]
    data_type = 'Int'
    threads = 8
    
    implementations = {
        'Matrix Naive': get_data_from_source('matrix', data_type, threads, 'NaiveMul'),
        'Matrix Optimized': get_data_from_source('matrix', data_type, threads, 'OptimizedMul'),
        'Array Naive': get_data_from_source('array', data_type, threads, 'NaiveMul'),
        'Array Optimized': get_data_from_source('array', data_type, threads, 'OptimizedMul')
    }
    
    colors = {'Matrix Naive': 'red', 'Matrix Optimized': 'blue', 
              'Array Naive': 'orange', 'Array Optimized': 'green'}
    markers = {'Matrix Naive': 'o', 'Matrix Optimized': 's', 
               'Array Naive': '^', 'Array Optimized': 'v'}
    
    for impl_name, data in implementations.items():
        if data:
            sizes, times = zip(*data)
            ax.loglog(sizes, times, f'{markers[impl_name]}-', linewidth=2, markersize=6,
                      label=impl_name, color=colors[impl_name], alpha=0.8)
    
    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Execution Time (time units)', fontweight='bold')
    ax.set_title(f'All Implementations ({data_type}, {threads} threads)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Plot 2: Implementation speedup factors
    ax = axes[0][1]
    
    matrix_naive = get_data_from_source('matrix', data_type, threads, 'NaiveMul')
    matrix_opt = get_data_from_source('matrix', data_type, threads, 'OptimizedMul')
    array_naive = get_data_from_source('array', data_type, threads, 'NaiveMul')
    array_opt = get_data_from_source('array', data_type, threads, 'OptimizedMul')
    
    if matrix_naive and matrix_opt:
        matrix_factors = calculate_optimization_factor(matrix_naive, matrix_opt)
        sizes, factors = zip(*matrix_factors)
        ax.semilogx(sizes, factors, 's-', linewidth=2, markersize=6,
                    label='Matrix: Opt vs Naive', color='blue', alpha=0.8)
    
    if array_naive and array_opt:
        array_factors = calculate_optimization_factor(array_naive, array_opt)
        sizes, factors = zip(*array_factors)
        ax.semilogx(sizes, factors, 'v-', linewidth=2, markersize=6,
                    label='Array: Opt vs Naive', color='green', alpha=0.8)
    
    if matrix_opt and array_opt:
        impl_factors = calculate_optimization_factor(matrix_opt, array_opt)
        sizes, factors = zip(*impl_factors)
        ax.semilogx(sizes, factors, 'o-', linewidth=2, markersize=6,
                    label='Array vs Matrix (Opt)', color='purple', alpha=0.8)

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontweight='bold')
    ax.set_title('Implementation Speedup Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Plot 3: GFLOPS comparison across implementations
    ax = axes[1][0]
    
    for impl_name, data in implementations.items():
        if data:
            gflops_data = [(size, calculate_gflops(size, time)) for size, time in data]
            sizes, gflops = zip(*gflops_data)
            ax.semilogx(sizes, gflops, f'{markers[impl_name]}-', linewidth=2, markersize=6,
                        label=impl_name, color=colors[impl_name], alpha=0.8)

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('GFLOPS', fontweight='bold')
    ax.set_title('GFLOPS Comparison Across Implementations', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Plot 4: Memory efficiency comparison
    ax = axes[1][1]
    
    type_sizes = {'Int': 4, 'Long': 8, 'Double': 8}
    
    for impl_name, data in implementations.items():
        if data and 'Optimized' in impl_name:  # Focus on optimized versions
            memory_throughput = []
            sizes_list = []
            for size, time in data:
                total_bytes = 3 * (size ** 2) * type_sizes[data_type]
                if time > 0:
                    throughput = total_bytes / time / 1e6  # MB/s
                    memory_throughput.append(throughput)
                    sizes_list.append(size)
            
            if memory_throughput:
                ax.semilogx(sizes_list, memory_throughput, f'{markers[impl_name]}-',
                            linewidth=2, markersize=6, label=impl_name,
                            color=colors[impl_name], alpha=0.8)

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Memory Throughput (MB/s)', fontweight='bold')
    ax.set_title('Memory Efficiency: Optimized Implementations', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    plt.tight_layout()
    plt.show()

# 3. Comprehensive Scalability Analysis
def plot_comprehensive_scalability():
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Comprehensive Scalability Analysis: All Implementations', fontsize=16, fontweight='bold')

    data_sources = [('matrix', matrix_data), ('array', matrix_as_array_data)]
    
    for col, (source_name, source_data) in enumerate(data_sources):
        # Plot 1: Thread scaling for naive algorithms
        ax = axes[0][col]
        algorithm = 'NaiveMul'
        data_type = 'Int'
        
        for threads in [2, 4, 8, 12]:
            data = get_data_from_source(source_name, data_type, threads, algorithm)
            if data:
                sizes, times = zip(*data)
                ax.loglog(sizes, times, 'o-', linewidth=2, markersize=6,
                          label=f'{threads} threads', alpha=0.8)

        ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
        ax.set_ylabel('Execution Time (time units)', fontweight='bold')
        ax.set_title(f'{source_name.title()} - {algorithm} Scaling', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([512, 1024, 2048, 4096, 8192])
        ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

        # Plot 2: Thread scaling for optimized algorithms
        ax = axes[1][col]
        algorithm = 'OptimizedMul'
        
        for threads in [2, 4, 8, 12]:
            data = get_data_from_source(source_name, data_type, threads, algorithm)
            if data:
                sizes, times = zip(*data)
                ax.loglog(sizes, times, 's-', linewidth=2, markersize=6,
                          label=f'{threads} threads', alpha=0.8)

        ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
        ax.set_ylabel('Execution Time (time units)', fontweight='bold')
        ax.set_title(f'{source_name.title()} - {algorithm} Scaling', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([512, 1024, 2048, 4096, 8192])
        ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

        # Plot 3: Parallel efficiency
        ax = axes[2][col]
        algorithm = 'OptimizedMul'
        matrix_size = 4096
        
        thread_counts = []
        efficiencies = []
        baseline_time = None
        
        for threads in [2, 4, 8, 12]:
            data = get_data_from_source(source_name, data_type, threads, algorithm)
            if data:
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
            ax.plot(thread_counts, efficiencies, 'o-', linewidth=3, markersize=8, alpha=0.8)
            ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency')

        ax.set_xlabel('Thread Count', fontweight='bold')
        ax.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
        ax.set_title(f'{source_name.title()} - Parallel Efficiency ({matrix_size}×{matrix_size})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 150)

    plt.tight_layout()
    plt.show()

# 4. Implementation Efficiency Heatmaps
def plot_implementation_heatmaps():
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Implementation Performance Heatmaps', fontsize=16, fontweight='bold')

    matrix_sizes = [512, 1024, 2048, 4096, 8192]
    thread_counts = [2, 4, 8, 12]
    
    data_sources = [('Matrix', matrix_data), ('Array', matrix_as_array_data)]
    algorithms = ['NaiveMul', 'OptimizedMul']
    
    for row, algorithm in enumerate(algorithms):
        for col, (source_name, source_data) in enumerate(data_sources):
            for data_type_idx, data_type in enumerate(['Int', 'Double']):
                ax_col = col * 2 + data_type_idx
                if ax_col >= 4:
                    continue
                    
                ax = axes[row][ax_col]
                
                # Create performance matrix
                performance_matrix = np.zeros((len(thread_counts), len(matrix_sizes)))
                
                for t_idx, threads in enumerate(thread_counts):
                    data = get_data_from_source(source_name.lower(), data_type, threads, algorithm)
                    for size, time in data:
                        if size in matrix_sizes:
                            s_idx = matrix_sizes.index(size)
                            performance_matrix[t_idx][s_idx] = time

                # Create heatmap with log scale
                log_matrix = np.log10(performance_matrix + 1)
                im = ax.imshow(log_matrix, cmap='YlOrRd', aspect='auto')

                # Set ticks and labels
                ax.set_xticks(range(len(matrix_sizes)))
                ax.set_xticklabels([f'{size}' for size in matrix_sizes])
                ax.set_yticks(range(len(thread_counts)))
                ax.set_yticklabels(thread_counts)

                ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
                ax.set_ylabel('Thread Count', fontweight='bold')
                ax.set_title(f'{source_name} {data_type} - {algorithm}', fontweight='bold')

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('log₁₀(Time)', fontsize=10)

    plt.tight_layout()
    plt.show()

# 5. Peak Performance Comparison
def plot_peak_performance():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Peak Performance Comparison Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Peak GFLOPS by implementation and data type
    ax = axes[0][0]
    data_types = ['Int', 'Long', 'Double']
    implementations = ['Matrix Naive', 'Matrix Opt', 'Array Naive', 'Array Opt']
    
    x_pos = np.arange(len(data_types))
    width = 0.2
    colors = ['red', 'blue', 'orange', 'green']
    
    for i, (impl_name, color) in enumerate(zip(implementations, colors)):
        peak_gflops = []
        source_name = 'matrix' if 'Matrix' in impl_name else 'array'
        algorithm = 'OptimizedMul' if 'Opt' in impl_name else 'NaiveMul'
        
        for data_type in data_types:
            max_gflops = 0
            for threads in [2, 4, 8, 12]:
                data = get_data_from_source(source_name, data_type, threads, algorithm)
                for size, time in data:
                    gflops = calculate_gflops(size, time)
                    max_gflops = max(max_gflops, gflops)
            peak_gflops.append(max_gflops)
        
        ax.bar(x_pos + i * width, peak_gflops, width, label=impl_name, 
               color=color, alpha=0.8)

    ax.set_xlabel('Data Type', fontweight='bold')
    ax.set_ylabel('Peak GFLOPS', fontweight='bold')
    ax.set_title('Peak GFLOPS by Implementation', fontweight='bold')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(data_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Implementation advantage analysis
    ax = axes[0][1]
    data_type = 'Int'
    threads = 8
    
    matrix_naive = get_data_from_source('matrix', data_type, threads, 'NaiveMul')
    array_opt = get_data_from_source('array', data_type, threads, 'OptimizedMul')
    
    if matrix_naive and array_opt:
        advantage_factors = calculate_optimization_factor(matrix_naive, array_opt)
        sizes, factors = zip(*advantage_factors)
        ax.semilogx(sizes, factors, 'o-', linewidth=3, markersize=8,
                    color='purple', label='Array Opt vs Matrix Naive')

    matrix_opt = get_data_from_source('matrix', data_type, threads, 'OptimizedMul')
    if matrix_opt and array_opt:
        impl_comparison = calculate_optimization_factor(matrix_opt, array_opt)
        sizes, factors = zip(*impl_comparison)
        ax.semilogx(sizes, factors, 's-', linewidth=3, markersize=8,
                    color='green', label='Array Opt vs Matrix Opt')

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Performance Advantage Factor', fontweight='bold')
    ax.set_title('Implementation Advantage Analysis', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Plot 3: Memory throughput comparison
    ax = axes[1][0]
    type_sizes = {'Int': 4, 'Long': 8, 'Double': 8}
    data_type = 'Int'
    threads = 8
    
    implementations = {
        'Matrix Opt': get_data_from_source('matrix', data_type, threads, 'OptimizedMul'),
        'Array Opt': get_data_from_source('array', data_type, threads, 'OptimizedMul')
    }
    
    for impl_name, data in implementations.items():
        if data:
            memory_throughput = []
            sizes_list = []
            for size, time in data:
                total_bytes = 3 * (size ** 2) * type_sizes[data_type]
                if time > 0:
                    throughput = total_bytes / time / 1e6
                    memory_throughput.append(throughput)
                    sizes_list.append(size)
            
            if memory_throughput:
                marker = 's' if 'Matrix' in impl_name else 'o'
                color = 'blue' if 'Matrix' in impl_name else 'green'
                ax.semilogx(sizes_list, memory_throughput, f'{marker}-',
                            linewidth=3, markersize=8, label=impl_name,
                            color=color, alpha=0.8)

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Memory Throughput (MB/s)', fontweight='bold')
    ax.set_title('Memory Throughput: Optimized Implementations', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels([512, 1024, 2048, 4096, 8192])

    # Plot 4: Best configuration summary
    ax = axes[1][1]
    matrix_sizes = [1024, 2048, 4096, 8192]
    
    best_times_matrix = []
    best_times_array = []
    
    for size in matrix_sizes:
        # Find best time for matrix implementation
        best_matrix = float('inf')
        best_array = float('inf')
        
        for threads in [2, 4, 8, 12]:
            for algorithm in ['NaiveMul', 'OptimizedMul']:
                # Matrix
                matrix_data_point = get_data_from_source('matrix', 'Int', threads, algorithm)
                for s, t in matrix_data_point:
                    if s == size and t < best_matrix:
                        best_matrix = t
                
                # Array
                array_data_point = get_data_from_source('array', 'Int', threads, algorithm)
                for s, t in array_data_point:
                    if s == size and t < best_array:
                        best_array = t
        
        best_times_matrix.append(best_matrix if best_matrix != float('inf') else 0)
        best_times_array.append(best_array if best_array != float('inf') else 0)

    x_pos = np.arange(len(matrix_sizes))
    width = 0.35
    
    ax.bar(x_pos - width/2, best_times_matrix, width, label='Matrix (Best)',
           color='blue', alpha=0.8)
    ax.bar(x_pos + width/2, best_times_array, width, label='Array (Best)',
           color='green', alpha=0.8)

    ax.set_xlabel('Matrix Size (N×N)', fontweight='bold')
    ax.set_ylabel('Best Execution Time', fontweight='bold')
    ax.set_title('Best Performance by Implementation', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{size}×{size}' for size in matrix_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

# Updated main execution with new analysis
if __name__ == "__main__":
    print("Generating Extended Matrix Multiplication Performance Analysis...")
    print("=" * 80)

    print("\n1. Generating Implementation Comparison (Matrix vs Array)...")
    plot_implementation_comparison()

    print("2. Generating Cross-Implementation Analysis...")
    plot_cross_implementation_analysis()

    print("3. Generating Comprehensive Scalability Analysis...")
    plot_comprehensive_scalability()

    print("4. Generating Implementation Performance Heatmaps...")
    plot_implementation_heatmaps()

    print("5. Generating Peak Performance Comparison...")
    plot_peak_performance()

    print("\nAll extended analysis plots generated successfully!")
    print("=" * 80)

    # Print comprehensive insights
    print("\nComprehensive Performance Insights:")
    print("-" * 40)

    # Compare implementations
    data_type = 'Int'
    threads = 8
    matrix_opt = get_data_from_source('matrix', data_type, threads, 'OptimizedMul')
    array_opt = get_data_from_source('array', data_type, threads, 'OptimizedMul')

    if matrix_opt and array_opt:
        print(f"\nImplementation Comparison ({data_type}, {threads} threads):")
        for (size_m, time_m), (size_a, time_a) in zip(matrix_opt, array_opt):
            if size_m == size_a and time_a > 0:
                advantage = time_m / time_a
                winner = "Array" if advantage > 1 else "Matrix"
                factor = max(advantage, 1/advantage)
                print(f"  {size_m}×{size_m}: {winner} is {factor:.1f}x faster")

    # Peak GFLOPS comparison
    print(f"\nPeak GFLOPS by Implementation:")
    implementations = [
        ('Matrix Naive', 'matrix', 'NaiveMul'),
        ('Matrix Optimized', 'matrix', 'OptimizedMul'),
        ('Array Naive', 'array', 'NaiveMul'),
        ('Array Optimized', 'array', 'OptimizedMul')
    ]
    
    for impl_name, source, algorithm in implementations:
        max_gflops = 0
        best_config = ""
        
        for data_type in ['Int', 'Long', 'Double']:
            for threads in [2, 4, 8, 12]:
                data = get_data_from_source(source, data_type, threads, algorithm)
                for size, time in data:
                    gflops = calculate_gflops(size, time)
                    if gflops > max_gflops:
                        max_gflops = gflops
                        best_config = f"{data_type}, {threads}t, {size}×{size}"
        
        print(f"  {impl_name}: {max_gflops:.2f} GFLOPS ({best_config})")
