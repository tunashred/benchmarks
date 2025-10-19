import matplotlib.pyplot as plt
import numpy as np

simd_data = {
    "int": {
        (512, 512): {2: 8, 4: 5, 8: 3, 12: 3},
        (1024, 1024): {2: 64, 4: 33, 8: 27, 12: 43},
        (2048, 2048): {2: 682, 4: 384, 8: 228, 12: 227},
        (4096, 4096): {2: 7350, 4: 4013, 8: 2241, 12: 2008},
        (8192, 8192): {2: 58189, 4: 31106, 8: 17037, 12: 15874},
    },
    "long": {
        (512, 512): {2: 15, 4: 8, 8: 6, 12: 6},
        (1024, 1024): {2: 111, 4: 54, 8: 45, 12: 49},
        (2048, 2048): {2: 1871, 4: 1034, 8: 570, 12: 536},
        (4096, 4096): {2: 14664, 4: 7903, 8: 4285, 12: 4250},
        (8192, 8192): {2: 112993, 4: 59987, 8: 32324, 12: 34689},
    },
    "double": {
        (512, 512): {2: 13, 4: 11, 8: 5, 12: 7},
        (1024, 1024): {2: 109, 4: 57, 8: 53, 12: 49},
        (2048, 2048): {2: 1937, 4: 1044, 8: 601, 12: 537},
        (4096, 4096): {2: 14789, 4: 7988, 8: 4377, 12: 4242},
        (8192, 8192): {2: 113771, 4: 60483, 8: 34129, 12: 34118},
    }
}

scalar_data = {
    "int": {
        (512, 512): {2: 23, 4: 13, 8: 9, 12: 8},
        (1024, 1024): {2: 151, 4: 93, 8: 59, 12: 59},
        (2048, 2048): {2: 1133, 4: 676, 8: 461, 12: 434},
        (4096, 4096): {2: 8949, 4: 4800, 8: 3598, 12: 3290},
        (8192, 8192): {2: 71254, 4: 37675, 8: 28952, 12: 27295},
    },
    "long": {
        (512, 512): {2: 26, 4: 17, 8: 11, 12: 9},
        (1024, 1024): {2: 169, 4: 95, 8: 74, 12: 73},
        (2048, 2048): {2: 1171, 4: 663, 8: 490, 12: 485},
        (4096, 4096): {2: 10229, 4: 5077, 8: 4102, 12: 3804},
        (8192, 8192): {2: 84131, 4: 42705, 8: 33650, 12: 31194},
    },
    "double": {
        (512, 512): {2: 46, 4: 26, 8: 19, 12: 17},
        (1024, 1024): {2: 311, 4: 167, 8: 122, 12: 117},
        (2048, 2048): {2: 2410, 4: 1275, 8: 962, 12: 876},
        (4096, 4096): {2: 19046, 4: 9971, 8: 7583, 12: 6983},
        (8192, 8192): {2: 151572, 4: 79150, 8: 61201, 12: 57067},
    }
}

# Extract matrix sizes and thread counts
sizes = sorted(list(simd_data["int"].keys()))
threads = sorted(list(simd_data["int"][sizes[0]].keys()))
data_types = ["int", "long", "double"]

# Plot 1: Performance comparison across data types (one plot per thread count)
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('SIMD vs Scalar Performance by Thread Count', fontsize=16, fontweight='bold')

for idx, thread in enumerate(threads):
    ax = axes1[idx // 2, idx % 2]
    x = np.arange(len(sizes))
    width = 0.12

    for i, dtype in enumerate(data_types):
        simd_times = [simd_data[dtype][size][thread] for size in sizes]
        scalar_times = [scalar_data[dtype][size][thread] for size in sizes]

        ax.bar(x + (i * 2 - 2) * width, simd_times, width, label=f'{dtype} SIMD', alpha=0.8)
        ax.bar(x + (i * 2 - 1) * width, scalar_times, width, label=f'{dtype} Scalar', alpha=0.8)

    ax.set_xlabel('Matrix Size', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title(f'{thread} Threads', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s[0]}x{s[1]}' for s in sizes], rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('thread_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: thread_comparison.png")

# Plot 2: Speedup comparison (Scalar/SIMD ratio)
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('SIMD Speedup Factor (Scalar Time / SIMD Time)', fontsize=16, fontweight='bold')

for idx, dtype in enumerate(data_types):
    ax = axes2[idx]

    for thread in threads:
        speedups = []
        for size in sizes:
            scalar_time = scalar_data[dtype][size][thread]
            simd_time = simd_data[dtype][size][thread]
            speedups.append(scalar_time / simd_time)

        ax.plot([f'{s[0]}' for s in sizes], speedups, marker='o', linewidth=2,
                markersize=8, label=f'{thread} threads')

    ax.set_xlabel('Matrix Size', fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontweight='bold')
    ax.set_title(f'{dtype.upper()} Data Type', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')

plt.tight_layout()
plt.savefig('speedup_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: speedup_comparison.png")

# Plot 3: Scaling efficiency (one plot per data type)
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle('Thread Scaling Efficiency', fontsize=16, fontweight='bold')

for idx, dtype in enumerate(data_types):
    ax = axes3[idx]

    for size in sizes:
        simd_times = [simd_data[dtype][size][t] for t in threads]
        scalar_times = [scalar_data[dtype][size][t] for t in threads]

        ax.plot(threads, simd_times, marker='o', linewidth=2, markersize=8,
                label=f'{size[0]} SIMD', linestyle='-')
        ax.plot(threads, scalar_times, marker='s', linewidth=2, markersize=8,
                label=f'{size[0]} Scalar', linestyle='--', alpha=0.7)

    ax.set_xlabel('Number of Threads', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title(f'{dtype.upper()} Data Type', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('scaling_efficiency.png', dpi=300, bbox_inches='tight')
print("Saved: scaling_efficiency.png")

# Plot 4: Heatmap of speedup factors
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
fig4.suptitle('SIMD Speedup Heatmap (Darker = Better)', fontsize=16, fontweight='bold')

for idx, dtype in enumerate(data_types):
    ax = axes4[idx]
    speedup_matrix = np.zeros((len(sizes), len(threads)))

    for i, size in enumerate(sizes):
        for j, thread in enumerate(threads):
            scalar_time = scalar_data[dtype][size][thread]
            simd_time = simd_data[dtype][size][thread]
            speedup_matrix[i, j] = scalar_time / simd_time

    im = ax.imshow(speedup_matrix, cmap='YlGn', aspect='auto')
    ax.set_xticks(np.arange(len(threads)))
    ax.set_yticks(np.arange(len(sizes)))
    ax.set_xticklabels(threads)
    ax.set_yticklabels([f'{s[0]}x{s[1]}' for s in sizes])
    ax.set_xlabel('Threads', fontweight='bold')
    ax.set_ylabel('Matrix Size', fontweight='bold')
    ax.set_title(f'{dtype.upper()}', fontweight='bold')

    # Add text annotations
    for i in range(len(sizes)):
        for j in range(len(threads)):
            text = ax.text(j, i, f'{speedup_matrix[i, j]:.2f}x',
                           ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax, label='Speedup Factor')

plt.tight_layout()
plt.savefig('speedup_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: speedup_heatmap.png")

# Plot 5: Summary statistics
fig5, ax5 = plt.subplots(figsize=(12, 6))
fig5.suptitle('Average Speedup by Data Type and Thread Count', fontsize=16, fontweight='bold')

x = np.arange(len(threads))
width = 0.25

for i, dtype in enumerate(data_types):
    avg_speedups = []
    for thread in threads:
        speedups = [scalar_data[dtype][size][thread] / simd_data[dtype][size][thread]
                    for size in sizes]
        avg_speedups.append(np.mean(speedups))

    ax5.bar(x + i * width, avg_speedups, width, label=dtype.upper(), alpha=0.8)

ax5.set_xlabel('Number of Threads', fontweight='bold')
ax5.set_ylabel('Average Speedup Factor', fontweight='bold')
ax5.set_title('Average SIMD Performance Gain', fontweight='bold')
ax5.set_xticks(x + width)
ax5.set_xticklabels(threads)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
ax5.axhline(y=1, color='r', linestyle='--', alpha=0.5, linewidth=2)

plt.tight_layout()
plt.savefig('average_speedup.png', dpi=300, bbox_inches='tight')
print("Saved: average_speedup.png")

print("\nAll plots generated successfully!")
print("\nSummary Statistics:")
print("=" * 60)
for dtype in data_types:
    print(f"\n{dtype.upper()} Data Type:")
    for thread in threads:
        speedups = [scalar_data[dtype][size][thread] / simd_data[dtype][size][thread]
                    for size in sizes]
        print(f"  {thread:2d} threads - Avg Speedup: {np.mean(speedups):.2f}x, "
              f"Max: {np.max(speedups):.2f}x, Min: {np.min(speedups):.2f}x")

plt.show()
