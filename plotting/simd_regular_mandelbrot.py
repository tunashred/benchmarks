import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

simd_data = {
    "1K": {  # 1920x1080
        "mandelbrot": 455,
        "shells": 946,
        "seastar": 251,
        "stuff": 1104,
        "galaxy": 1302,
    },
    "4K": {  # 3840x2160
        "mandelbrot": 1820,
        "shells": 3786,
        "seastar": 982,
        "stuff": 4385,
        "galaxy": 5216,
    },
    "6K": {  # 5760x3240
        "mandelbrot": 4032,
        "shells": 8443,
        "seastar": 2242,
        "stuff": 9983,
        "galaxy": 11647,
    },
    "8K": {  # 7680x4320
        "mandelbrot": 7218,
        "shells": 14954,
        "seastar": 3943,
        "stuff": 17428,
        "galaxy": 20599,
    },
}

regular_data = {
    "1K": {  # 1920x1080
        "mandelbrot": 17,
        "shells": 17483,
        "seastar": 1422,
        "stuff": 17414,
        "galaxy": 7578,
    },
    "4K": {  # 3840x2160
        "mandelbrot": 68,
        "shells": 69630,
        "seastar": 5677,
        "stuff": 69475,
        "galaxy": 30232,
    },
    "6K": {  # 5760x3240
        "mandelbrot": 146,
        "shells": 156131,
        "seastar": 12884,
        "stuff": 156955,
        "galaxy": 68158,
    },
    "8K": {  # 7680x4320
        "mandelbrot": 259,
        "shells": 278087,
        "seastar": 22984,
        "stuff": 277286,
        "galaxy": 120818,
    },
}

# Extract all zoom names and resolutions
zooms = list(next(iter(simd_data.values())).keys())  # ["mandelbrot", "shells", ...]
resolutions = list(simd_data.keys())  # ["1K", "4K", "6K", "8K"]

# Define plot style
sns.set_style("whitegrid")
colors = ["#1f77b4", "#ff7f0e"]  # SIMD = blue, Regular = orange

# Create subplots: one per zoom level
fig, axes = plt.subplots(1, len(zooms), figsize=(22, 6), sharey=True)
fig.suptitle("Mandelbrot Generation: SIMD vs Regular Performance", fontsize=18, fontweight="bold")

for idx, zoom in enumerate(zooms):
    ax = axes[idx]

    # Get data for this zoom
    simd_times = [simd_data[res][zoom] for res in resolutions]
    reg_times = [regular_data[res][zoom] for res in resolutions]

    # Set bar positions
    x = np.arange(len(resolutions))
    width = 0.35  # Width of bars

    # Plot bars
    ax.bar(x - width / 2, simd_times, width, label="SIMD", color=colors[0])
    ax.bar(x + width / 2, reg_times, width, label="Regular", color=colors[1])

    # Titles and labels
    ax.set_title(zoom.capitalize(), fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.set_xlabel("Resolution", fontsize=12)

    if idx == 0:
        ax.set_ylabel("Time (ms)", fontsize=12)

    # Logarithmic y-axis for better readability
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Show legend only once (on first subplot)
    if idx == 0:
        ax.legend()

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
