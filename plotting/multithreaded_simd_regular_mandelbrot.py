import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

simd_data = {
    "1K": {  # 1920x1080
        "mandelbrot": {2: 286, 4: 214, 8: 130, 12: 85},
        "shells": {2: 485, 4: 256, 8: 153, 12: 110},
        "seastar": {2: 130, 4: 70, 8: 39, 12: 31},
        "stuff": {2: 650, 4: 337, 8: 187, 12: 133},
        "galaxy": {2: 696, 4: 363, 8: 207, 12: 142},
    },
    "4K": {  # 3840x2160
        "mandelbrot": {2: 1126, 4: 849, 8: 497, 12: 346},
        "shells": {2: 1926, 4: 1029, 8: 605, 12: 439},
        "seastar": {2: 509, 4: 277, 8: 149, 12: 121},
        "stuff": {2: 2573, 4: 1328, 8: 750, 12: 537},
        "galaxy": {2: 2778, 4: 1444, 8: 830, 12: 564},
    },
    "6K": {  # 5760x3240
        "mandelbrot": {2: 2537, 4: 1932, 8: 1142, 12: 803},
        "shells": {2: 4340, 4: 2325, 8: 1339, 12: 996},
        "seastar": {2: 1177, 4: 650, 8: 367, 12: 281},
        "stuff": {2: 5742, 4: 3052, 8: 1730, 12: 1246},
        "galaxy": {2: 6216, 4: 3270, 8: 1885, 12: 1304},
    },
    "8K": {  # 7680x4320
        "mandelbrot": {2: 4455, 4: 3420, 8: 2031, 12: 1424},
        "shells": {2: 7693, 4: 4116, 8: 2474, 12: 1750},
        "seastar": {2: 2058, 4: 1152, 8: 650, 12: 493},
        "stuff": {2: 10322, 4: 5365, 8: 2978, 12: 2231},
        "galaxy": {2: 11112, 4: 5789, 8: 3342, 12: 2313},
    },
}

regular_data = {
    "1K": {  # 1920x1080
        "mandelbrot": {2: 2317, 4: 1790, 8: 1040, 12: 714},
        "shells": {2: 3957, 4: 2091, 8: 1254, 12: 855},
        "seastar": {2: 956, 4: 519, 8: 273, 12: 208},
        "stuff": {2: 5191, 4: 2701, 8: 1516, 12: 1109},
        "galaxy": {2: 5475, 4: 2854, 8: 1602, 12: 1085},
    },
    "4K": {  # 3840x2160
        "mandelbrot": {2: 9273, 4: 7096, 8: 4143, 12: 2883},
        "shells": {2: 15709, 4: 8450, 8: 4902, 12: 3376},
        "seastar": {2: 3816, 4: 2061, 8: 1139, 12: 792},
        "stuff": {2: 20564, 4: 10801, 8: 6065, 12: 4317},
        "galaxy": {2: 21732, 4: 11282, 8: 6389, 12: 4346},
    },
    "6K": {  # 5760x3240
        "mandelbrot": {2: 20742, 4: 16104, 8: 9431, 12: 6486},
        "shells": {2: 35369, 4: 18804, 8: 11188, 12: 7647},
        "seastar": {2: 8667, 4: 4667, 8: 2496, 12: 1823},
        "stuff": {2: 45944, 4: 24304, 8: 13648, 12: 9743},
        "galaxy": {2: 48350, 4: 25732, 8: 14413, 12: 9825},
    },
    "8K": {  # 7680x4320
        "mandelbrot": {2: 36675, 4: 28240, 8: 16576, 12: 11606},
        "shells": {2: 62771, 4: 33768, 8: 19723, 12: 13701},
        "seastar": {2: 15348, 4: 8286, 8: 4436, 12: 3230},
        "stuff": {2: 82827, 4: 43164, 8: 24195, 12: 17323},
        "galaxy": {2: 85917, 4: 45649, 8: 25598, 12: 17467},
    },
}

resolutions = ["1K", "4K", "6K", "8K"]
fractals = ["mandelbrot", "shells", "seastar", "stuff", "galaxy"]
threads = [2, 4, 8, 12]

# Resolution pixel counts for reference
res_pixels = {
    "1K": 1920 * 1080,
    "4K": 3840 * 2160,
    "6K": 5760 * 3240,
    "8K": 7680 * 4320
}

# Plot 1: Fractal Complexity Comparison (Performance Profile)
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('Rendering Performance by Fractal Type and Resolution', fontsize=16, fontweight='bold')

for idx, thread in enumerate(threads):
    ax = axes1[idx // 2, idx % 2]
    x = np.arange(len(fractals))
    width = 0.18

    for i, res in enumerate(resolutions):
        simd_times = [simd_data[res][frac][thread] for frac in fractals]
        regular_times = [regular_data[res][frac][thread] for frac in fractals]

        offset = (i - 1.5) * width
        ax.bar(x + offset - width / 2, simd_times, width * 0.9,
               label=f'{res} SIMD', alpha=0.85)
        ax.bar(x + offset + width / 2, regular_times, width * 0.9,
               label=f'{res} Regular', alpha=0.65, hatch='//')

    ax.set_xlabel('Fractal Pattern', fontweight='bold', fontsize=11)
    ax.set_ylabel('Render Time (ms)', fontweight='bold', fontsize=11)
    ax.set_title(f'{thread} Threads', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(fractals, rotation=30, ha='right')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('fractal_complexity_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: fractal_complexity_comparison.png")

# Plot 2: Speedup Radar Charts
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(projection='polar'))
fig2.suptitle('SIMD Speedup Radar Charts by Resolution', fontsize=16, fontweight='bold')

angles = np.linspace(0, 2 * np.pi, len(fractals), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for idx, res in enumerate(resolutions):
    ax = axes2[idx // 2, idx % 2]

    for thread in threads:
        speedups = []
        for frac in fractals:
            speedup = regular_data[res][frac][thread] / simd_data[res][frac][thread]
            speedups.append(speedup)
        speedups += speedups[:1]  # Complete the circle

        ax.plot(angles, speedups, 'o-', linewidth=2.5, markersize=8,
                label=f'{thread}T', alpha=0.8)
        ax.fill(angles, speedups, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(fractals, fontsize=10)
    ax.set_ylim(0, max([regular_data[res][frac][t] / simd_data[res][frac][t]
                        for frac in fractals for t in threads]) * 1.1)
    ax.set_title(f'{res} Resolution', fontweight='bold', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)

    # Add reference circle at 1x
    ref_circle = [1] * len(angles)
    ax.plot(angles, ref_circle, 'r--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('speedup_radar_charts.png', dpi=300, bbox_inches='tight')
print("Saved: speedup_radar_charts.png")

# Plot 3: Resolution Scaling Analysis
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle('Resolution Scaling: How Render Time Grows', fontsize=16, fontweight='bold')

# First row: SIMD
for idx, frac in enumerate(fractals):
    ax = axes3[0, idx] if idx < 3 else axes3[1, idx - 3]

    for thread in threads:
        simd_times = [simd_data[res][frac][thread] for res in resolutions]
        regular_times = [regular_data[res][frac][thread] for res in resolutions]

        ax.plot(resolutions, simd_times, marker='o', linewidth=2.5,
                markersize=9, label=f'{thread}T SIMD')
        ax.plot(resolutions, regular_times, marker='s', linewidth=2,
                markersize=8, label=f'{thread}T Reg', linestyle='--', alpha=0.7)

    ax.set_xlabel('Resolution', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title(f'{frac.capitalize()}', fontweight='bold', fontsize=11)
    if idx == 0:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

# Hide the 6th subplot (only 5 fractals)
axes3[1, 2].axis('off')

plt.tight_layout()
plt.savefig('resolution_scaling.png', dpi=300, bbox_inches='tight')
print("Saved: resolution_scaling.png")

# Plot 4: Pixels Per Second (Throughput)
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle('Rendering Throughput: Megapixels per Second', fontsize=16, fontweight='bold')

# SIMD Throughput
ax_simd = axes4[0]
x = np.arange(len(resolutions))
width = 0.15

for i, thread in enumerate(threads):
    throughputs = []
    for res in resolutions:
        avg_time = np.mean([simd_data[res][frac][thread] for frac in fractals])
        megapixels = res_pixels[res] / 1_000_000
        throughput = megapixels / (avg_time / 1000)  # MP/s
        throughputs.append(throughput)

    ax_simd.bar(x + (i - 1.5) * width, throughputs, width,
                label=f'{thread} Threads', alpha=0.85)

ax_simd.set_xlabel('Resolution', fontweight='bold', fontsize=12)
ax_simd.set_ylabel('Megapixels/Second', fontweight='bold', fontsize=12)
ax_simd.set_title('SIMD Implementation', fontweight='bold', fontsize=13)
ax_simd.set_xticks(x)
ax_simd.set_xticklabels(resolutions)
ax_simd.legend()
ax_simd.grid(True, alpha=0.3, axis='y')

# Regular Throughput
ax_regular = axes4[1]
for i, thread in enumerate(threads):
    throughputs = []
    for res in resolutions:
        avg_time = np.mean([regular_data[res][frac][thread] for frac in fractals])
        megapixels = res_pixels[res] / 1_000_000
        throughput = megapixels / (avg_time / 1000)  # MP/s
        throughputs.append(throughput)

    ax_regular.bar(x + (i - 1.5) * width, throughputs, width,
                   label=f'{thread} Threads', alpha=0.85)

ax_regular.set_xlabel('Resolution', fontweight='bold', fontsize=12)
ax_regular.set_ylabel('Megapixels/Second', fontweight='bold', fontsize=12)
ax_regular.set_title('Regular Implementation', fontweight='bold', fontsize=13)
ax_regular.set_xticks(x)
ax_regular.set_xticklabels(resolutions)
ax_regular.legend()
ax_regular.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('throughput_megapixels.png', dpi=300, bbox_inches='tight')
print("Saved: throughput_megapixels.png")

# Plot 5: Performance Matrix (Heatmaps)
fig5, axes5 = plt.subplots(2, 4, figsize=(20, 10))
fig5.suptitle('Performance Heatmaps: Speedup Factor by Configuration', fontsize=16, fontweight='bold')

for idx, res in enumerate(resolutions):
    # Speedup heatmap
    ax_speedup = axes5[0, idx]
    speedup_matrix = np.zeros((len(fractals), len(threads)))

    for i, frac in enumerate(fractals):
        for j, thread in enumerate(threads):
            speedup = regular_data[res][frac][thread] / simd_data[res][frac][thread]
            speedup_matrix[i, j] = speedup

    im1 = ax_speedup.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto',
                            vmin=4, vmax=12)
    ax_speedup.set_xticks(np.arange(len(threads)))
    ax_speedup.set_yticks(np.arange(len(fractals)))
    ax_speedup.set_xticklabels(threads)
    ax_speedup.set_yticklabels(fractals)
    ax_speedup.set_xlabel('Threads', fontweight='bold')
    if idx == 0:
        ax_speedup.set_ylabel('Fractal', fontweight='bold')
    ax_speedup.set_title(f'{res} - Speedup', fontweight='bold')

    for i in range(len(fractals)):
        for j in range(len(threads)):
            val = speedup_matrix[i, j]
            text_color = 'white' if val > 8 else 'black'
            ax_speedup.text(j, i, f'{val:.1f}x', ha="center", va="center",
                            color=text_color, fontsize=9, fontweight='bold')

    if idx == 3:
        plt.colorbar(im1, ax=ax_speedup, label='Speedup')

    # Absolute time heatmap (SIMD)
    ax_time = axes5[1, idx]
    time_matrix = np.zeros((len(fractals), len(threads)))

    for i, frac in enumerate(fractals):
        for j, thread in enumerate(threads):
            time_matrix[i, j] = simd_data[res][frac][thread]

    im2 = ax_time.imshow(time_matrix, cmap='plasma', aspect='auto',
                         norm=plt.matplotlib.colors.LogNorm())
    ax_time.set_xticks(np.arange(len(threads)))
    ax_time.set_yticks(np.arange(len(fractals)))
    ax_time.set_xticklabels(threads)
    ax_time.set_yticklabels(fractals)
    ax_time.set_xlabel('Threads', fontweight='bold')
    if idx == 0:
        ax_time.set_ylabel('Fractal', fontweight='bold')
    ax_time.set_title(f'{res} - SIMD Time (ms)', fontweight='bold')

    if idx == 3:
        plt.colorbar(im2, ax=ax_time, label='Time (ms)')

plt.tight_layout()
plt.savefig('performance_heatmaps.png', dpi=300, bbox_inches='tight')
print("Saved: performance_heatmaps.png")

# Plot 6: Fractal Complexity Ranking
fig6, ax6 = plt.subplots(figsize=(14, 8))
fig6.suptitle('Fractal Computational Complexity Ranking', fontsize=16, fontweight='bold')

# Calculate average render time for each fractal (normalized by resolution)
complexity_scores = {}
for frac in fractals:
    total_time = 0
    count = 0
    for res in resolutions:
        for thread in threads:
            # Normalize by pixel count
            time_per_mpixel = simd_data[res][frac][thread] / (res_pixels[res] / 1_000_000)
            total_time += time_per_mpixel
            count += 1
    complexity_scores[frac] = total_time / count

sorted_fractals = sorted(complexity_scores.items(), key=lambda x: x[1])

# Create grouped bar chart
x = np.arange(len(sorted_fractals))
width = 0.2

for i, res in enumerate(resolutions):
    simd_times = []
    for frac, _ in sorted_fractals:
        avg_time = np.mean([simd_data[res][frac][t] for t in threads])
        simd_times.append(avg_time)

    ax6.bar(x + (i - 1.5) * width, simd_times, width,
            label=f'{res}', alpha=0.85)

ax6.set_xlabel('Fractal (Sorted by Complexity)', fontweight='bold', fontsize=12)
ax6.set_ylabel('Average SIMD Render Time (ms)', fontweight='bold', fontsize=12)
ax6.set_xticks(x)
ax6.set_xticklabels([f[0].capitalize() for f in sorted_fractals])
ax6.legend(title='Resolution', fontsize=11)
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_yscale('log')

# Add complexity score annotations
for i, (frac, score) in enumerate(sorted_fractals):
    ax6.text(i, ax6.get_ylim()[0] * 1.5, f'{score:.1f}',
             ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fractal_complexity_ranking.png', dpi=300, bbox_inches='tight')
print("Saved: fractal_complexity_ranking.png")

# Print comprehensive statistics
print("\n" + "=" * 80)
print("MANDELBROT RENDERING PERFORMANCE STATISTICS")
print("=" * 80)

for res in resolutions:
    print(f"\n{res} Resolution ({res_pixels[res]:,} pixels):")
    print("-" * 80)

    for frac in fractals:
        print(f"\n  {frac.upper()}:")
        for thread in threads:
            simd_time = simd_data[res][frac][thread]
            regular_time = regular_data[res][frac][thread]
            speedup = regular_time / simd_time
            fps_simd = 1000 / simd_time if simd_time > 0 else 0
            fps_regular = 1000 / regular_time if regular_time > 0 else 0

            print(f"    {thread:2d}T - SIMD: {simd_time:5.0f}ms ({fps_simd:5.1f} FPS) | "
                  f"Regular: {regular_time:5.0f}ms ({fps_regular:5.1f} FPS) | "
                  f"Speedup: {speedup:5.2f}x")

print("\n" + "=" * 80)
print("Average Speedup by Thread Count:")
print("-" * 80)
for thread in threads:
    total_speedup = 0
    count = 0
    for res in resolutions:
        for frac in fractals:
            speedup = regular_data[res][frac][thread] / simd_data[res][frac][thread]
            total_speedup += speedup
            count += 1
    avg_speedup = total_speedup / count
    print(f"  {thread:2d} Threads: {avg_speedup:.2f}x average speedup")

print("\n" + "=" * 80)
print("All plots generated successfully!")
print("=" * 80)

plt.show()
