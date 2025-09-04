import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Data
simd_data = {
    "1K": {  # 1920x1080
        "mandelbrot": {2: 286, 4: 216, 8: 128, 12: 89},
        "shells": {2: 491, 4: 260, 8: 155, 12: 127},
        "seastar": {2: 131, 4: 71, 8: 38, 12: 36},
        "stuff": {2: 648, 4: 337, 8: 190, 12: 135},
        "galaxy": {2: 709, 4: 368, 8: 210, 12: 153},
    },
    "4K": {  # 3840x2160
        "mandelbrot": {2: 1132, 4: 862, 8: 499, 12: 345},
        "shells": {2: 1951, 4: 1042, 8: 617, 12: 438},
        "seastar": {2: 543, 4: 283, 8: 152, 12: 126},
        "stuff": {2: 2668, 4: 1359, 8: 809, 12: 548},
        "galaxy": {2: 2860, 4: 1483, 8: 847, 12: 617},
    },
    "6K": {  # 5760x3240
        "mandelbrot": {2: 2658, 4: 2012, 8: 1175, 12: 816},
        "shells": {2: 4540, 4: 2382, 8: 1438, 12: 1018},
        "seastar": {2: 1247, 4: 664, 8: 396, 12: 304},
        "stuff": {2: 6037, 4: 3105, 8: 1778, 12: 1269},
        "galaxy": {2: 6516, 4: 3339, 8: 1934, 12: 1388},
    },
    "8K": {  # 7680x4320
        "mandelbrot": {2: 4616, 4: 3537, 8: 2088, 12: 1454},
        "shells": {2: 8092, 4: 4236, 8: 2541, 12: 1787},
        "seastar": {2: 2133, 4: 1179, 8: 662, 12: 515},
        "stuff": {2: 10577, 4: 5521, 8: 3131, 12: 2293},
        "galaxy": {2: 11410, 4: 5952, 8: 3440, 12: 2406},
    },
}

regular_data = {
    "1K": {  # 1920x1080
        "mandelbrot": {2: 977, 4: 735, 8: 425, 12: 289},
        "shells": {2: 1646, 4: 851, 8: 469, 12: 353},
        "seastar": {2: 399, 4: 212, 8: 113, 12: 100},
        "stuff": {2: 2133, 4: 1102, 8: 610, 12: 421},
        "galaxy": {2: 2254, 4: 1158, 8: 600, 12: 455},
    },
    "4K": {  # 3840x2160
        "mandelbrot": {2: 3851, 4: 2936, 8: 1699, 12: 1163},
        "shells": {2: 6568, 4: 3421, 8: 1940, 12: 1400},
        "seastar": {2: 1596, 4: 852, 8: 449, 12: 319},
        "stuff": {2: 8549, 4: 4432, 8: 2383, 12: 1714},
        "galaxy": {2: 9011, 4: 4647, 8: 2489, 12: 1798},
    },
    "6K": {  # 5760x3240
        "mandelbrot": {2: 8707, 4: 6638, 8: 3876, 12: 2673},
        "shells": {2: 14882, 4: 7732, 8: 4366, 12: 3076},
        "seastar": {2: 3622, 4: 1960, 8: 1030, 12: 821},
        "stuff": {2: 19372, 4: 10021, 8: 5418, 12: 4151},
        "galaxy": {2: 20492, 4: 10529, 8: 5701, 12: 4037},
    },
    "8K": {  # 7680x4320
        "mandelbrot": {2: 15458, 4: 11874, 8: 6897, 12: 4780},
        "shells": {2: 26406, 4: 13840, 8: 7739, 12: 5455},
        "seastar": {2: 6431, 4: 3497, 8: 1828, 12: 1419},
        "stuff": {2: 34409, 4: 17852, 8: 9874, 12: 7158},
        "galaxy": {2: 36344, 4: 18732, 8: 9774, 12: 7240},
    },
}


def calculate_speedup(regular, simd):
    """Calculate speedup factor (regular_time / simd_time)"""
    return {
        resolution: {
            zoom: {threads: regular[resolution][zoom][threads] / simd[resolution][zoom][threads]
                   for threads in regular[resolution][zoom]}
            for zoom in regular[resolution]}
        for resolution in regular}


# Calculate speedup data
speedup_data = calculate_speedup(regular_data, simd_data)

# Create 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

resolutions = ["1K", "4K", "6K", "8K"]
zoom_types = list(simd_data["1K"].keys())
threads = [2, 4, 8, 12]
colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))

# 1. Speedup by Zoom Type (grouped bar chart)
ax = axes[0, 0]
x = np.arange(len(zoom_types))
width = 0.2

for i, res in enumerate(resolutions):
    speedups = []
    for zoom in zoom_types:
        # Average speedup across all thread counts
        avg_speedup = np.mean([speedup_data[res][zoom][t] for t in threads])
        speedups.append(avg_speedup)

    ax.bar(x + i * width, speedups, width, label=f'{res}', alpha=0.8)

ax.set_xlabel('Zoom Type', fontsize=12)
ax.set_ylabel('Average SIMD Speedup (×)', fontsize=12)
ax.set_title('SIMD Speedup by Zoom Type and Resolution', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([z.title() for z in zoom_types])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 2. Speedup Heatmap by Zoom Type (darker colors)
ax = axes[0, 1]
speedup_matrix = np.zeros((len(zoom_types), len(resolutions)))

for i, zoom in enumerate(zoom_types):
    for j, res in enumerate(resolutions):
        # Average speedup across all thread counts for this zoom/resolution
        speedups = [speedup_data[res][zoom][t] for t in threads]
        speedup_matrix[i, j] = np.mean(speedups)

# Use a darker colormap
im = ax.imshow(speedup_matrix, cmap='plasma', aspect='auto', vmin=1.5, vmax=4.5)
ax.set_xticks(range(len(resolutions)))
ax.set_xticklabels(resolutions)
ax.set_yticks(range(len(zoom_types)))
ax.set_yticklabels([z.title() for z in zoom_types])
ax.set_title('SIMD Speedup Heatmap', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(zoom_types)):
    for j in range(len(resolutions)):
        text = ax.text(j, i, f'{speedup_matrix[i, j]:.1f}×',
                       ha="center", va="center", color="white", fontweight='bold', fontsize=10)

plt.colorbar(im, ax=ax, label='Speedup Factor', shrink=0.8)

# 3. Resolution Impact Analysis
ax = axes[1, 0]
pixels = {"1K": 1920 * 1080, "4K": 3840 * 2160, "6K": 5760 * 3240, "8K": 7680 * 4320}

# Calculate average speedup per resolution across all zoom types and thread counts
avg_speedups = []
pixel_counts = []

for res in resolutions:
    all_speedups = []
    for zoom in zoom_types:
        for t in threads:
            all_speedups.append(speedup_data[res][zoom][t])
    avg_speedups.append(np.mean(all_speedups))
    pixel_counts.append(pixels[res] / 1e6)  # Convert to megapixels

ax.scatter(pixel_counts, avg_speedups, s=200, alpha=0.7, c=colors)
for i, res in enumerate(resolutions):
    ax.annotate(res, (pixel_counts[i], avg_speedups[i]),
                xytext=(5, 5), textcoords='offset points', fontweight='bold')

# Fit a trend line
z = np.polyfit(pixel_counts, avg_speedups, 1)
p = np.poly1d(z)
ax.plot(pixel_counts, p(pixel_counts), "r--", alpha=0.8, linewidth=2)

ax.set_xlabel('Image Size (Megapixels)', fontsize=12)
ax.set_ylabel('Average SIMD Speedup (×)', fontsize=12)
ax.set_title('SIMD Speedup vs Image Resolution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 4. Thread Scaling Efficiency (better colors)
ax = axes[1, 1]
# Focus on 4K resolution for clarity
res = "4K"
thread_colors = ['#2E8B57', '#FF6347', '#4682B4', '#DAA520', '#9932CC']

for i, zoom in enumerate(zoom_types):
    times = [simd_data[res][zoom][t] for t in threads]
    # Calculate efficiency relative to perfect scaling from 2 threads
    base_time = times[0]  # 2-thread time
    efficiency = []

    for j, t in enumerate(threads):
        ideal_time = base_time / (t / 2)  # Perfect scaling from 2 threads
        actual_time = times[j]
        eff = (ideal_time / actual_time) * 100
        efficiency.append(eff)

    ax.plot(threads, efficiency, 'o-', linewidth=2, markersize=6,
            color=thread_colors[i], label=zoom.title())

ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Perfect Scaling')
ax.set_xlabel('Number of Threads', fontsize=12)
ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
ax.set_title('SIMD Thread Scaling Efficiency (4K)', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(threads)
ax.set_ylim(0, 110)

plt.tight_layout()
plt.show()
