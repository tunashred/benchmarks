import matplotlib.pyplot as plt

# Raw benchmark data manually extracted from the logs
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

# Plotting
def plot_benchmarks(data):
    for dtype, modes in data.items():
        plt.figure(figsize=(10, 6))
        for mode, values in modes.items():
            sizes = [x[0] for x in values]
            times = [x[1] for x in values]
            plt.plot(sizes, times, marker='o', label=mode)
        plt.title(f"Sequential vs Jump Iterate - {dtype} Array")
        plt.xlabel("Array Size")
        plt.ylabel("Time (ms)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, which="major", linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"benchmark_{dtype}.png")
        plt.show()

def plot_combined(data, mode):
    plt.figure(figsize=(10, 6))

    for dtype, modes in data.items():
        if mode in modes:
            sizes = [x[0] for x in modes[mode]]
            times = [x[1] for x in modes[mode]]
            plt.plot(sizes, times, marker='o', label=dtype)

    plt.title(f"{mode} Iterate - All Types")
    plt.xlabel("Array Size")
    plt.ylabel("Time (ms)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="major", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"benchmark_all_{mode.lower()}.png")
    plt.show()

plot_combined(raw_data, "Sequential")
plot_combined(raw_data, "Jump")

plot_benchmarks(raw_data)
