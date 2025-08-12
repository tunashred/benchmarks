import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# Data extracted from the GTest logs (BlockMul removed)
raw_data = {
    "int": {
        "NaiveMul": [
            (512, 114), (1024, 1194), (2048, 17478), (4096, 162754), (8192, 1661269)
        ],
        "OptimizedMul": [
            (512, 21), (1024, 162), (2048, 1705), (4096, 14327), (8192, 128903)
        ]
    },
    "long": {
        "NaiveMul": [
            (512, 114), (1024, 1006), (2048, 17842), (4096, 229797), (8192, 2427316)
        ],
        "OptimizedMul": [
            (512, 50), (1024, 598), (2048, 4309), (4096, 34837), (8192, 276389)
        ]
    },
    "double": {
        "NaiveMul": [
            (512, 157), (1024, 1742), (2048, 24907), (4096, 282413), (8192, 2724742)
        ],
        "OptimizedMul": [
            (512, 74), (1024, 580), (2048, 5157), (4096, 43817), (8192, 354093)
        ]
    }
}

def smart_format(x, pos):
    if x == 0:
        return '0'
    elif x < 1:
        return f'{x:.1f}'
    elif x < 1000:
        return f'{x:.0f}'
    elif x < 1000000:
        return f'{x / 1000:.0f}K'
    else:
        exp = int(np.log10(x))
        mantissa = x / (10 ** exp)
        return f'{mantissa:.1f}×10$^{{{exp}}}$' if mantissa != 1 else f'10$^{{{exp}}}$'

def setup_y_axis(ax, times):
    min_time = min([t for t in times if t > 0] + [1])
    max_time = max(times)
    if max_time > 0:
        log_min = np.floor(np.log10(min_time))
        log_max = np.ceil(np.log10(max_time))
        ticks = []
        for exp in range(int(log_min), int(log_max) + 1):
            base = 10 ** exp
            ticks.append(base)
            if exp < log_max:
                ticks.extend([2 * base, 5 * base])
        ticks = [t for t in ticks if min_time / 2 <= t <= max_time * 2]
        ax.set_yticks(sorted(set(ticks)))
        ax.yaxis.set_major_formatter(FuncFormatter(smart_format))

def setup_x_axis(ax):
    matrix_sizes = [512, 1024, 2048, 4096, 8192]
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(matrix_sizes))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))
    ax.xaxis.set_minor_locator(NullLocator())

def add_smart_annotations(ax, annotations):
    if not annotations:
        return
    for ann in annotations:
        ann['log_x'] = np.log10(ann['x'])
        ann['log_y'] = np.log10(max(ann['y'], 0.1))
    proximity_threshold = 0.15
    groups = []
    used = set()
    for i, ann in enumerate(annotations):
        if i in used:
            continue
        group = [ann]
        used.add(i)
        for j, other in enumerate(annotations):
            if j in used or j <= i:
                continue
            dx = abs(ann['log_x'] - other['log_x'])
            dy = abs(ann['log_y'] - other['log_y'])
            if np.sqrt(dx * dx + dy * dy) < proximity_threshold:
                group.append(other)
                used.add(j)
        groups.append(group)
    for group in groups:
        if len(group) == 1:
            ann = group[0]
            ax.annotate(ann['text'], (ann['x'], ann['y']), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=8, color=ann['color'],
                        alpha=0.9, bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                        alpha=0.8, edgecolor='none'))
        else:
            group.sort(key=lambda a: a['y'])
            n = len(group)
            positions = [(-20, 15), (20, 15)] if n == 2 else [(-25, 20), (0, 25), (25, 20)]
            for i, ann in enumerate(group):
                x_offset, y_offset = positions[i] if i < len(positions) else (0, 15)
                ax.annotate(ann['text'], (ann['x'], ann['y']), textcoords="offset points",
                            xytext=(x_offset, y_offset), ha='center', fontsize=7,
                            color=ann['color'], alpha=0.9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                      alpha=0.8, edgecolor=ann['color'], linewidth=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                            color=ann['color'], alpha=0.6, lw=0.8))

def plot_benchmarks(data):
    for dtype, algorithms in data.items():
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        all_times = []
        annotations = []
        for algorithm, values in algorithms.items():
            sizes = [x[0] for x in values]
            times = [x[1] for x in values]
            all_times.extend(times)
            line = plt.plot(sizes, times, marker='o', label=algorithm, linewidth=2, markersize=6)
            for size, time in values:
                if time > 0:
                    annotations.append({'x': size, 'y': time, 'text': f'{time}',
                                        'color': line[0].get_color(), 'algorithm': algorithm})
        add_smart_annotations(ax, annotations)
        plt.title(f"Matrix Multiplication - {dtype.upper()}", fontsize=14, fontweight='bold')
        plt.xlabel("Matrix Size (NxN)", fontsize=12)
        plt.ylabel("Time (ms)", fontsize=12)
        plt.xscale("log")
        plt.yscale("log")
        setup_y_axis(ax, all_times)
        setup_x_axis(ax)
        plt.grid(True, which="major", linestyle='--', linewidth=0.5, alpha=0.7)
        plt.grid(True, which="minor", linestyle=':', linewidth=0.3, alpha=0.4)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"matrix_benchmark_{dtype}.png", dpi=300, bbox_inches='tight')
        plt.show()

def plot_combined(data, algorithm):
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    all_times = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    annotations = []
    for i, (dtype, algorithms) in enumerate(data.items()):
        if algorithm in algorithms:
            sizes = [x[0] for x in algorithms[algorithm]]
            times = [x[1] for x in algorithms[algorithm]]
            all_times.extend(times)
            line = plt.plot(sizes, times, marker='o', label=dtype, linewidth=2, markersize=6, color=colors[i])
            for size, time in algorithms[algorithm]:
                if time > 0:
                    annotations.append({'x': size, 'y': time, 'text': f'{time}', 'color': colors[i], 'dtype': dtype})
    add_smart_annotations(ax, annotations)
    plt.title(f"{algorithm} - All Data Types", fontsize=14, fontweight='bold')
    plt.xlabel("Matrix Size (NxN)", fontsize=12)
    plt.ylabel("Time (ms)", fontsize=12)
    plt.xscale("log")
    plt.yscale("log")
    setup_y_axis(ax, all_times)
    setup_x_axis(ax)
    plt.grid(True, which="major", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.grid(True, which="minor", linestyle=':', linewidth=0.3, alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"matrix_benchmark_all_{algorithm.lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_speedup_comparison(data):
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    all_speedups = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    annotations = []
    for i, (dtype, algorithms) in enumerate(data.items()):
        naive_times = dict(algorithms["NaiveMul"])
        alg_name = "OptimizedMul"
        if alg_name in algorithms:
            sizes = []
            speedups = []
            for size, time in algorithms[alg_name]:
                if size in naive_times and naive_times[size] > 0:
                    speedup = naive_times[size] / time
                    sizes.append(size)
                    speedups.append(speedup)
            all_speedups.extend(speedups)
            label = f"{dtype} - {alg_name.replace('Mul', '')}"
            line = plt.plot(sizes, speedups, marker='o', label=label, linewidth=2, markersize=6, color=colors[i])
            for size, speedup in zip(sizes, speedups):
                annotations.append({'x': size, 'y': speedup, 'text': f'{speedup:.1f}x', 'color': colors[i]})
    add_smart_annotations(ax, annotations)
    plt.title("Speedup vs Naive", fontsize=14, fontweight='bold')
    plt.xlabel("Matrix Size (NxN)", fontsize=12)
    plt.ylabel("Speedup Factor", fontsize=12)
    plt.xscale("log")
    plt.yscale("log")
    setup_y_axis(ax, all_speedups)
    setup_x_axis(ax)
    plt.grid(True, which="major", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.grid(True, which="minor", linestyle=':', linewidth=0.3, alpha=0.4)
    plt.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig("matrix_benchmark_speedup.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_combined(raw_data, "NaiveMul")
    plot_combined(raw_data, "OptimizedMul")
    plot_benchmarks(raw_data)
    plot_speedup_comparison(raw_data)
