import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# Data extracted from the GTest logs
raw_data = {
    "int": {
        "NaiveMul": [
            (512, 114), (1024, 1194), (2048, 17478), (4096, 162754), (8192, 1661269)
        ],
        "OptimizedMul": [
            (512, 21), (1024, 162), (2048, 1705), (4096, 14327), (8192, 128903)
        ],
        "BlockMul": [
            (512, 26), (1024, 223), (2048, 1820), (4096, 14901), (8192, 124236)
        ],
    },
    "long": {
        "NaiveMul": [
            (512, 114), (1024, 1006), (2048, 17842), (4096, 229797), (8192, 2427316)
        ],
        "OptimizedMul": [
            (512, 50), (1024, 598), (2048, 4309), (4096, 34837), (8192, 276389)
        ],
        "BlockMul": [
            (512, 57), (1024, 479), (2048, 3767), (4096, 30782), (8192, 260056)
        ],
    },
    "double": {
        "NaiveMul": [
            (512, 157), (1024, 1742), (2048, 24907), (4096, 282413), (8192, 2724742)
        ],
        "OptimizedMul": [
            (512, 74), (1024, 580), (2048, 5157), (4096, 43817), (8192, 354093)
        ],
        "BlockMul": [
            (512, 82), (1024, 666), (2048, 5307), (4096, 42912), (8192, 356599)
        ],
    }
}


def smart_format(x, pos):
    """Custom formatter for y-axis that shows values in scientific notation or regular format"""
    if x == 0:
        return '0'
    elif x < 1:
        return f'{x:.1f}'
    elif x < 1000:
        return f'{x:.0f}'
    elif x < 1000000:
        return f'{x / 1000:.0f}K'
    else:
        # Use scientific notation for large numbers
        exp = int(np.log10(x))
        mantissa = x / (10 ** exp)
        if mantissa == 1:
            return f'10$^{{{exp}}}$'
        else:
            return f'{mantissa:.1f}×10$^{{{exp}}}$'


def setup_y_axis(ax, times):
    """Setup y-axis with smart tick placement and formatting"""
    # Get the range of data
    min_time = min([t for t in times if t > 0] + [1])  # Avoid log(0)
    max_time = max(times)

    # Create custom tick positions
    if max_time > 0:
        # Generate ticks at powers of 10 and some intermediate values
        log_min = np.floor(np.log10(min_time))
        log_max = np.ceil(np.log10(max_time))

        ticks = []
        for exp in range(int(log_min), int(log_max) + 1):
            base = 10 ** exp
            # Add main power of 10
            ticks.append(base)
            # Add some intermediate values if there's room
            if exp < log_max:
                ticks.extend([2 * base, 5 * base])

        # Filter ticks to be within our data range
        ticks = [t for t in ticks if min_time / 2 <= t <= max_time * 2]
        ticks = sorted(set(ticks))

        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(smart_format))


def setup_x_axis(ax):
    # exact matrix sizes we want on the x axis
    matrix_sizes = [512, 1024, 2048, 4096, 8192]

    # ensure log scale (safe to call even if you already did plt.xscale("log"))
    ax.set_xscale('log')

    # force only those ticks to appear
    ax.xaxis.set_major_locator(FixedLocator(matrix_sizes))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    # hide minor ticks (these were producing the 6×10^2, 2×10^3, etc)
    ax.xaxis.set_minor_locator(NullLocator())


def add_smart_annotations(ax, annotations):
    """Add annotations with smart positioning to avoid overlaps"""
    if not annotations:
        return

    # Convert to log space for better distance calculation on log plots
    for ann in annotations:
        ann['log_x'] = np.log10(ann['x'])
        ann['log_y'] = np.log10(max(ann['y'], 0.1))  # Avoid log(0)

    # Group annotations by proximity in log space
    proximity_threshold = 0.15  # Adjust this to control grouping sensitivity
    groups = []
    used = set()

    for i, ann in enumerate(annotations):
        if i in used:
            continue

        group = [ann]
        used.add(i)

        # Find nearby annotations
        for j, other in enumerate(annotations):
            if j in used or j <= i:
                continue

            # Calculate distance in log space
            dx = abs(ann['log_x'] - other['log_x'])
            dy = abs(ann['log_y'] - other['log_y'])
            distance = np.sqrt(dx * dx + dy * dy)

            if distance < proximity_threshold:
                group.append(other)
                used.add(j)

        groups.append(group)

    # Position annotations for each group
    for group in groups:
        if len(group) == 1:
            # Single annotation - place above point
            ann = group[0]
            ax.annotate(ann['text'],
                        (ann['x'], ann['y']),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha='center',
                        fontsize=8,
                        color=ann['color'],
                        alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor='white',
                                  alpha=0.8,
                                  edgecolor='none'))
        else:
            # Multiple annotations - arrange in a smart pattern
            group.sort(key=lambda a: a['y'])  # Sort by y value

            # Calculate positions in a more spread out pattern
            n = len(group)
            positions = []

            if n == 2:
                positions = [(-20, 15), (20, 15)]
            elif n == 3:
                positions = [(-25, 20), (0, 25), (25, 20)]
            elif n == 4:
                positions = [(-30, 15), (-10, 25), (10, 25), (30, 15)]
            else:
                # For more than 4, use a circular arrangement
                for i in range(n):
                    angle = 2 * np.pi * i / n - np.pi / 2  # Start from top
                    radius = 25
                    x_offset = radius * np.cos(angle)
                    y_offset = radius * np.sin(angle) + 20  # Offset upward
                    positions.append((x_offset, y_offset))

            for i, ann in enumerate(group):
                x_offset, y_offset = positions[i] if i < len(positions) else (0, 15)

                ax.annotate(ann['text'],
                            (ann['x'], ann['y']),
                            textcoords="offset points",
                            xytext=(x_offset, y_offset),
                            ha='center',
                            fontsize=7,
                            color=ann['color'],
                            alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor='white',
                                      alpha=0.8,
                                      edgecolor=ann['color'],
                                      linewidth=0.5),
                            arrowprops=dict(arrowstyle='->',
                                            connectionstyle='arc3,rad=0.1',
                                            color=ann['color'],
                                            alpha=0.6,
                                            lw=0.8))


def plot_benchmarks(data):
    """Plot individual benchmark comparisons for each data type"""
    for dtype, algorithms in data.items():
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        all_times = []
        annotations = []

        for algorithm, values in algorithms.items():
            sizes = [x[0] for x in values]
            times = [x[1] for x in values]
            all_times.extend(times)

            # Plot line and markers
            line = plt.plot(sizes, times, marker='o', label=algorithm, linewidth=2, markersize=6)

            # Collect annotations
            for i, (size, time) in enumerate(values):
                if time > 0:  # Only annotate non-zero values
                    annotations.append({
                        'x': size, 'y': time, 'text': f'{time}',
                        'color': line[0].get_color(), 'algorithm': algorithm
                    })

        # Add smart annotations
        add_smart_annotations(ax, annotations)

        plt.title(f"Matrix Multiplication Algorithms - {dtype.upper()} Matrix", fontsize=14, fontweight='bold')
        plt.xlabel("Matrix Size (NxN)", fontsize=12)
        plt.ylabel("Time (ms)", fontsize=12)
        plt.xscale("log")
        plt.yscale("log")

        # Setup custom axes
        setup_y_axis(ax, all_times)
        setup_x_axis(ax)

        plt.grid(True, which="major", linestyle='--', linewidth=0.5, alpha=0.7)
        plt.grid(True, which="minor", linestyle=':', linewidth=0.3, alpha=0.4)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"matrix_benchmark_{dtype}.png", dpi=300, bbox_inches='tight')
        plt.show()


def plot_combined(data, algorithm):
    """Plot combined comparison of all data types for a specific algorithm"""
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    all_times = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    # Collect all annotation positions to avoid overlaps
    annotations = []

    for i, (dtype, algorithms) in enumerate(data.items()):
        if algorithm in algorithms:
            sizes = [x[0] for x in algorithms[algorithm]]
            times = [x[1] for x in algorithms[algorithm]]
            all_times.extend(times)

            # Plot line and markers
            line = plt.plot(sizes, times, marker='o', label=dtype,
                            linewidth=2, markersize=6, color=colors[i])

            # Collect annotations for this line
            for j, (size, time) in enumerate(algorithms[algorithm]):
                if time > 0:  # Annotate all non-zero values for matrix data
                    annotations.append({
                        'x': size, 'y': time, 'text': f'{time}',
                        'color': colors[i], 'dtype': dtype
                    })

    # Add annotations with smart positioning to avoid overlaps
    add_smart_annotations(ax, annotations)

    plt.title(f"{algorithm} - All Data Types", fontsize=14, fontweight='bold')
    plt.xlabel("Matrix Size (NxN)", fontsize=12)
    plt.ylabel("Time (ms)", fontsize=12)
    plt.xscale("log")
    plt.yscale("log")

    # Setup custom axes
    setup_y_axis(ax, all_times)
    setup_x_axis(ax)

    plt.grid(True, which="major", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.grid(True, which="minor", linestyle=':', linewidth=0.3, alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"matrix_benchmark_all_{algorithm.lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_speedup_comparison(data):
    """Plot speedup comparison showing how much faster optimized algorithms are vs naive"""
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    all_speedups = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    annotations = []

    for i, (dtype, algorithms) in enumerate(data.items()):
        naive_times = dict(algorithms["NaiveMul"])

        for alg_name in ["OptimizedMul", "BlockMul"]:
            if alg_name in algorithms:
                sizes = []
                speedups = []

                for size, time in algorithms[alg_name]:
                    if size in naive_times and naive_times[size] > 0:
                        speedup = naive_times[size] / time
                        sizes.append(size)
                        speedups.append(speedup)

                all_speedups.extend(speedups)

                # Plot line and markers
                label = f"{dtype} - {alg_name.replace('Mul', '')}"
                line = plt.plot(sizes, speedups, marker='o', label=label,
                                linewidth=2, markersize=6,
                                color=colors[i],
                                linestyle='-' if alg_name == "OptimizedMul" else '--')

                # Collect annotations
                for j, (size, speedup) in enumerate(zip(sizes, speedups)):
                    annotations.append({
                        'x': size, 'y': speedup, 'text': f'{speedup:.1f}x',
                        'color': colors[i], 'algorithm': alg_name
                    })

    # Add annotations
    add_smart_annotations(ax, annotations)

    plt.title("Speedup vs Naive Implementation", fontsize=14, fontweight='bold')
    plt.xlabel("Matrix Size (NxN)", fontsize=12)
    plt.ylabel("Speedup Factor", fontsize=12)
    plt.xscale("log")
    plt.yscale("log")

    # Setup custom axes
    setup_y_axis(ax, all_speedups)
    setup_x_axis(ax)

    plt.grid(True, which="major", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.grid(True, which="minor", linestyle=':', linewidth=0.3, alpha=0.4)
    plt.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig("matrix_benchmark_speedup.png", dpi=300, bbox_inches='tight')
    plt.show()


# Generate all plots
if __name__ == "__main__":
    # Plot combined comparisons for each algorithm
    plot_combined(raw_data, "NaiveMul")
    plot_combined(raw_data, "OptimizedMul")
    plot_combined(raw_data, "BlockMul")

    # Plot individual benchmarks for each data type
    plot_benchmarks(raw_data)

    # Plot speedup comparison
    plot_speedup_comparison(raw_data)