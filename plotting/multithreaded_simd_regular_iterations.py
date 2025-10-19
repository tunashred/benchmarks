import matplotlib.pyplot as plt
import numpy as np

simd_data = {
    "int": {
        192: {2: 1, 4: 0, 8: 0, 12: 0},
        960: {2: 7, 4: 2, 8: 1, 12: 1},
        9984: {2: 58, 4: 29, 8: 22, 12: 21},
        99840: {2: 643, 4: 332, 8: 239, 12: 209},
        1000128: {2: 6809, 4: 4246, 8: 2857, 12: 2552},
        2000640: {2: 13585, 4: 8931, 8: 6494, 12: 5417},
    },
    "long": {
        192: {2: 2, 4: 1, 8: 0, 12: 1},
        960: {2: 11, 4: 5, 8: 4, 12: 3},
        9984: {2: 124, 4: 56, 8: 40, 12: 37},
        99840: {2: 1289, 4: 648, 8: 484, 12: 416},
        1000128: {2: 17957, 4: 8428, 8: 6412, 12: 5346},
        2000640: {2: 55580, 4: 28438, 8: 17552, 12: 18819},
    },
    "double": {
        192: {2: 2, 4: 1, 8: 1, 12: 1},
        960: {2: 10, 4: 5, 8: 3, 12: 3},
        9984: {2: 118, 4: 54, 8: 39, 12: 37},
        99840: {2: 1293, 4: 642, 8: 480, 12: 418},
        1000128: {2: 13526, 4: 8459, 8: 6331, 12: 5371},
        2000640: {2: 55226, 4: 34214, 8: 18738, 12: 19535},
    }
}

regular_data = {
    "int": {
        192: {2: 7, 4: 5, 8: 4, 12: 7},
        960: {2: 18, 4: 12, 8: 11, 12: 12},
        9984: {2: 146, 4: 76, 8: 62, 12: 60},
        99840: {2: 1576, 4: 820, 8: 613, 12: 542},
        1000128: {2: 16001, 4: 8747, 8: 6404, 12: 5637},
        2000640: {2: 33060, 4: 16994, 8: 13450, 12: 12310},
    },
    "long": {
        192: {2: 7, 4: 6, 8: 6, 12: 6},
        960: {2: 16, 4: 8, 8: 8, 12: 12},
        9984: {2: 170, 4: 80, 8: 64, 12: 68},
        99840: {2: 1720, 4: 877, 8: 651, 12: 570},
        1000128: {2: 17309, 4: 9443, 8: 7190, 12: 7000},
        2000640: {2: 66306, 4: 54458, 8: 32764, 12: 18896},
    },
    "double": {
        192: {2: 12, 4: 8, 8: 9, 12: 8},
        960: {2: 40, 4: 21, 8: 15, 12: 18},
        9984: {2: 405, 4: 209, 8: 162, 12: 152},
        99840: {2: 3931, 4: 2058, 8: 1562, 12: 1372},
        1000128: {2: 39722, 4: 20672, 8: 15518, 12: 14208},
        2000640: {2: 97446, 4: 54458, 8: 32764, 12: 33600},
    }
}

# Extract array sizes and thread counts
sizes = sorted(list(simd_data["int"].keys()))
threads = sorted(list(simd_data["int"][sizes[0]].keys()))
data_types = ["int", "long", "double"]


# Helper function to format array sizes
def format_size(size):
    if size >= 1000000:
        return f"{size / 1000000:.1f}M"
    elif size >= 1000:
        return f"{size / 1000:.1f}K"
    return str(size)


# Plot 1: Performance Comparison by Data Type
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle('SIMD vs Regular Performance Across Array Sizes', fontsize=16, fontweight='bold')

for idx, dtype in enumerate(data_types):
    ax = axes1[idx]

    for thread in threads:
        simd_times = [simd_data[dtype][size][thread] for size in sizes]
        regular_times = [regular_data[dtype][size][thread] for size in sizes]

        ax.plot(range(len(sizes)), simd_times, marker='o', linewidth=2.5,
                markersize=8, label=f'{thread}T SIMD', linestyle='-')
        ax.plot(range(len(sizes)), regular_times, marker='s', linewidth=2,
                markersize=7, label=f'{thread}T Regular', linestyle='--', alpha=0.7)

    ax.set_xlabel('Array Size', fontweight='bold', fontsize=11)
    ax.set_ylabel('Time (ms)', fontweight='bold', fontsize=11)
    ax.set_title(f'{dtype.upper()} Data Type', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([format_size(s) for s in sizes], rotation=45)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('array_performance_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: array_performance_comparison.png")

# Plot 2: Speedup Factor Analysis
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle('SIMD Speedup Factor by Thread Count', fontsize=16, fontweight='bold')

for idx, thread in enumerate(threads):
    ax = axes2[idx // 2, idx % 2]
    x = np.arange(len(sizes))
    width = 0.25

    for i, dtype in enumerate(data_types):
        speedups = []
        for size in sizes:
            regular_time = regular_data[dtype][size][thread]
            simd_time = simd_data[dtype][size][thread]
            # Handle zero values
            if simd_time == 0:
                speedups.append(0)
            else:
                speedups.append(regular_time / simd_time)

        ax.bar(x + i * width, speedups, width, label=dtype.upper(), alpha=0.8)

    ax.set_xlabel('Array Size', fontweight='bold')
    ax.set_ylabel('Speedup Factor (Regular/SIMD)', fontweight='bold')
    ax.set_title(f'{thread} Threads', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([format_size(s) for s in sizes], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, linewidth=2)

plt.tight_layout()
plt.savefig('speedup_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: speedup_analysis.png")

# Plot 3: Thread Scaling Efficiency
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle('Thread Scaling: SIMD vs Regular', fontsize=16, fontweight='bold')

for idx, dtype in enumerate(data_types):
    # SIMD scaling
    ax_simd = axes3[0, idx]
    for size in sizes:
        times = [simd_data[dtype][size][t] for t in threads]
        ax_simd.plot(threads, times, marker='o', linewidth=2.5,
                     markersize=8, label=format_size(size))

    ax_simd.set_xlabel('Threads', fontweight='bold')
    ax_simd.set_ylabel('Time (ms)', fontweight='bold')
    ax_simd.set_title(f'{dtype.upper()} - SIMD', fontweight='bold')
    ax_simd.legend(fontsize=8, title='Array Size')
    ax_simd.grid(True, alpha=0.3)
    ax_simd.set_yscale('log')

    # Regular scaling
    ax_regular = axes3[1, idx]
    for size in sizes:
        times = [regular_data[dtype][size][t] for t in threads]
        ax_regular.plot(threads, times, marker='s', linewidth=2.5,
                        markersize=8, label=format_size(size))

    ax_regular.set_xlabel('Threads', fontweight='bold')
    ax_regular.set_ylabel('Time (ms)', fontweight='bold')
    ax_regular.set_title(f'{dtype.upper()} - Regular', fontweight='bold')
    ax_regular.legend(fontsize=8, title='Array Size')
    ax_regular.grid(True, alpha=0.3)
    ax_regular.set_yscale('log')

plt.tight_layout()
plt.savefig('thread_scaling.png', dpi=300, bbox_inches='tight')
print("Saved: thread_scaling.png")

# Plot 4: Throughput (Elements per ms)
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
fig4.suptitle('Throughput: Elements Processed per Millisecond', fontsize=16, fontweight='bold')

for idx, dtype in enumerate(data_types):
    ax = axes4[idx]

    for thread in threads:
        simd_throughput = []
        regular_throughput = []

        for size in sizes:
            simd_time = simd_data[dtype][size][thread]
            regular_time = regular_data[dtype][size][thread]

            # Calculate throughput (elements per ms)
            simd_tp = size / simd_time if simd_time > 0 else 0
            regular_tp = size / regular_time if regular_time > 0 else 0

            simd_throughput.append(simd_tp)
            regular_throughput.append(regular_tp)

        ax.plot(range(len(sizes)), simd_throughput, marker='o', linewidth=2.5,
                markersize=8, label=f'{thread}T SIMD')
        ax.plot(range(len(sizes)), regular_throughput, marker='s', linewidth=2,
                markersize=7, label=f'{thread}T Regular', linestyle='--', alpha=0.7)

    ax.set_xlabel('Array Size', fontweight='bold')
    ax.set_ylabel('Elements/ms', fontweight='bold')
    ax.set_title(f'{dtype.upper()} Data Type', fontweight='bold')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([format_size(s) for s in sizes], rotation=45)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('throughput_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: throughput_comparison.png")

# Plot 5: Speedup Heatmaps
fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
fig5.suptitle('SIMD Speedup Heatmaps', fontsize=16, fontweight='bold')

for idx, dtype in enumerate(data_types):
    # Speedup heatmap
    ax_speedup = axes5[0, idx]
    speedup_matrix = np.zeros((len(sizes), len(threads)))

    for i, size in enumerate(sizes):
        for j, thread in enumerate(threads):
            regular_time = regular_data[dtype][size][thread]
            simd_time = simd_data[dtype][size][thread]
            if simd_time > 0:
                speedup_matrix[i, j] = regular_time / simd_time

    im1 = ax_speedup.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)
    ax_speedup.set_xticks(np.arange(len(threads)))
    ax_speedup.set_yticks(np.arange(len(sizes)))
    ax_speedup.set_xticklabels(threads)
    ax_speedup.set_yticklabels([format_size(s) for s in sizes])
    ax_speedup.set_xlabel('Threads', fontweight='bold')
    ax_speedup.set_ylabel('Array Size', fontweight='bold')
    ax_speedup.set_title(f'{dtype.upper()} - Speedup Factor', fontweight='bold')

    for i in range(len(sizes)):
        for j in range(len(threads)):
            val = speedup_matrix[i, j]
            text_color = 'white' if val > 5 else 'black'
            if val > 0:
                ax_speedup.text(j, i, f'{val:.1f}x', ha="center", va="center",
                                color=text_color, fontsize=9, fontweight='bold')

    plt.colorbar(im1, ax=ax_speedup, label='Speedup')

    # Absolute time comparison
    ax_time = axes5[1, idx]
    time_data = []
    labels = []

    for thread in threads:
        simd_times = [simd_data[dtype][size][thread] for size in sizes]
        regular_times = [regular_data[dtype][size][thread] for size in sizes]
        time_data.append(simd_times)
        time_data.append(regular_times)
        labels.append(f'{thread}T SIMD')
        labels.append(f'{thread}T Reg')

    time_matrix = np.array(time_data).T
    im2 = ax_time.imshow(time_matrix, cmap='YlOrRd', aspect='auto', norm=plt.matplotlib.colors.LogNorm())
    ax_time.set_xticks(np.arange(len(labels)))
    ax_time.set_yticks(np.arange(len(sizes)))
    ax_time.set_xticklabels(labels, rotation=45, ha='right')
    ax_time.set_yticklabels([format_size(s) for s in sizes])
    ax_time.set_xlabel('Implementation', fontweight='bold')
    ax_time.set_ylabel('Array Size', fontweight='bold')
    ax_time.set_title(f'{dtype.upper()} - Execution Time (ms)', fontweight='bold')

    plt.colorbar(im2, ax=ax_time, label='Time (ms)')

plt.tight_layout()
plt.savefig('heatmap_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: heatmap_analysis.png")

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

for dtype in data_types:
    print(f"\n{dtype.upper()} Data Type:")
    print("-" * 70)

    for thread in threads:
        speedups = []
        for size in sizes:
            regular_time = regular_data[dtype][size][thread]
            simd_time = simd_data[dtype][size][thread]
            if simd_time > 0:
                speedups.append(regular_time / simd_time)

        valid_speedups = [s for s in speedups if s > 0]
        if valid_speedups:
            print(f"  {thread:2d} threads - Avg: {np.mean(valid_speedups):5.2f}x, "
                  f"Max: {np.max(valid_speedups):5.2f}x, Min: {np.min(valid_speedups):5.2f}x")

print("\n" + "=" * 70)
print("All plots generated successfully!")
print("=" * 70)

plt.show()
