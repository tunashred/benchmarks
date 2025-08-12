import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Data extracted from the GTest logs
raw_data = {
    "int": {
        "Sequential": [
            (10, 0), (100, 2), (1000, 16), (10000, 131),
            (100000, 1382), (1000000, 13782), (2000000, 31189)
        ],
        "Jump": [
            (10, 2), (100, 6), (1000, 51), (10000, 945),
            (100000, 11832), (1000000, 158561), (2000000, 327873)
        ],
    },
    "long": {
        "Sequential": [
            (10, 0), (100, 2), (1000, 28), (10000, 266),
            (100000, 2815), (1000000, 28604), (2000000, 146210)
        ],
        "Jump": [
            (10, 1), (100, 5), (1000, 51), (10000, 1038),
            (100000, 15165), (1000000, 166890), (2000000, 1076490)
        ],
    },
    "float": {
        "Sequential": [
            (10, 0), (100, 1), (1000, 12), (10000, 127),
            (100000, 1302), (1000000, 13352), (2000000, 27942)
        ],
        "Jump": [
            (10, 2), (100, 6), (1000, 53), (10000, 999),
            (100000, 11159), (1000000, 148445), (2000000, 329537)
        ],
    },
    "double": {
        "Sequential": [
            (10, 0), (100, 2), (1000, 27), (10000, 257),
            (100000, 2656), (1000000, 29799), (2000000, 165417)
        ],
        "Jump": [
            (10, 1), (100, 5), (1000, 53), (10000, 1047),
            (100000, 15545), (1000000, 187580), (2000000, 2253729)
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
    for dtype, modes in data.items():
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        all_times = []
        annotations = []

        for mode, values in modes.items():
            sizes = [x[0] for x in values]
            times = [x[1] for x in values]
            all_times.extend(times)

            # Plot line and markers
            line = plt.plot(sizes, times, marker='o', label=mode, linewidth=2, markersize=6)

            # Collect annotations
            for i, (size, time) in enumerate(values):
                if time > 0:  # Only annotate non-zero values
                    annotations.append({
                        'x': size, 'y': time, 'text': f'{time}',
                        'color': line[0].get_color(), 'mode': mode
                    })

        # Add smart annotations
        add_smart_annotations(ax, annotations)

        plt.title(f"Sequential vs Jump Iterate - {dtype} Array", fontsize=14, fontweight='bold')
        plt.xlabel("Array Size", fontsize=12)
        plt.ylabel("Time (ms)", fontsize=12)
        plt.xscale("log")
        plt.yscale("log")

        # Setup custom y-axis
        setup_y_axis(ax, all_times)

        plt.grid(True, which="major", linestyle='--', linewidth=0.5, alpha=0.7)
        plt.grid(True, which="minor", linestyle=':', linewidth=0.3, alpha=0.4)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"benchmark_{dtype}.png", dpi=300, bbox_inches='tight')
        plt.show()

def plot_combined(data, mode):
    """Plot combined comparison of all data types for a specific mode"""
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    all_times = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    # Collect all annotation positions to avoid overlaps
    annotations = []

    for i, (dtype, modes) in enumerate(data.items()):
        if mode in modes:
            sizes = [x[0] for x in modes[mode]]
            times = [x[1] for x in modes[mode]]
            all_times.extend(times)

            # Plot line and markers
            line = plt.plot(sizes, times, marker='o', label=dtype,
                            linewidth=2, markersize=6, color=colors[i])

            # Collect annotations for this line
            for j, (size, time) in enumerate(modes[mode]):
                if time > 0 and j % 2 == 0:  # Annotate every other point and non-zero values
                    annotations.append({
                        'x': size, 'y': time, 'text': f'{time}',
                        'color': colors[i], 'dtype': dtype
                    })

    # Add annotations with smart positioning to avoid overlaps
    add_smart_annotations(ax, annotations)

    plt.title(f"{mode} Iterate - All Types", fontsize=14, fontweight='bold')
    plt.xlabel("Array Size", fontsize=12)
    plt.ylabel("Time (ms)", fontsize=12)
    plt.xscale("log")
    plt.yscale("log")

    # Setup custom y-axis
    setup_y_axis(ax, all_times)

    plt.grid(True, which="major", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.grid(True, which="minor", linestyle=':', linewidth=0.3, alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    # plt.savefig(f"benchmark_all_{mode.lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()


plot_combined(raw_data, "Sequential")
plot_combined(raw_data, "Jump")

plot_benchmarks(raw_data)
