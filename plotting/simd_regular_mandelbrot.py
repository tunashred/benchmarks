import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Given data
simd_data = {
    "1K": {"mandelbrot": 477, "shells": 998, "seastar": 262, "stuff": 1170, "galaxy": 1376},
    "4K": {"mandelbrot": 1926, "shells": 4010, "seastar": 1050, "stuff": 4730, "galaxy": 5542},
    "6K": {"mandelbrot": 4303, "shells": 8877, "seastar": 2349, "stuff": 10461, "galaxy": 12264},
    "8K": {"mandelbrot": 7620, "shells": 15755, "seastar": 4155, "stuff": 18583, "galaxy": 21778},
}

regular_data = {
    "1K": {"mandelbrot": 17, "shells": 7143, "seastar": 602, "stuff": 7121, "galaxy": 3126},
    "4K": {"mandelbrot": 59, "shells": 28530, "seastar": 2369, "stuff": 28521, "galaxy": 12527},
    "6K": {"mandelbrot": 132, "shells": 65296, "seastar": 5313, "stuff": 64377, "galaxy": 28272},
    "8K": {"mandelbrot": 239, "shells": 114358, "seastar": 9474, "stuff": 115628, "galaxy": 51102},
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
