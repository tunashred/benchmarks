import matplotlib.pyplot as plt

# Given data
simd_data = {
    "Int": {
        "SequentialIterate": [
            (12, 0), (100, 1), (1000, 8), (10000, 99),
            (100000, 1258), (1000000, 13042), (2000000, 34350)
        ]
    },
    "Long": {
        "SequentialIterate": [
            (12, 0), (100, 1), (1000, 14), (10000, 235),
            (100000, 2785), (1000000, 51207), (2000000, 572381)
        ]
    },
    "Double": {
        "SequentialIterate": [
            (12, 0), (100, 2), (1000, 15), (10000, 248),
            (100000, 3110), (1000000, 172095), (2000000, 669552)
        ]
    }
}

regular_data = {
    "int": {
        "Sequential": [
            (12, 0), (100, 2), (1000, 16), (10000, 131),
            (100000, 1382), (1000000, 13782), (2000000, 31189)
        ]
    },
    "long": {
        "Sequential": [
            (12, 0), (100, 2), (1000, 28), (10000, 266),
            (100000, 2815), (1000000, 28604), (2000000, 146210)
        ]
    },
    "double": {
        "Sequential": [
            (12, 0), (100, 2), (1000, 27), (10000, 257),
            (100000, 2656), (1000000, 29799), (2000000, 165417)
        ]
    }
}

# Mapping between SIMD and regular dataset keys
mapping = {
    "Int": "int",
    "Long": "long",
    "Double": "double"
}

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SIMD vs Regular Iteration Times", fontsize=16, fontweight='bold')

colors = ["#1f77b4", "#ff7f0e"]  # blue for SIMD, orange for Regular

# Iterate through each data type
for idx, dtype in enumerate(["Int", "Long", "Double"]):
    ax = axes[idx]

    # Extract SIMD and regular data
    simd = simd_data[dtype]["SequentialIterate"]
    regular = regular_data[mapping[dtype]]["Sequential"]

    # Unpack data into X and Y
    simd_x, simd_y = zip(*simd)
    reg_x, reg_y = zip(*regular)

    # Plot data
    ax.plot(simd_x, simd_y, marker="o", color=colors[0], label="SIMD")
    ax.plot(reg_x, reg_y, marker="s", color=colors[1], label="Regular")

    # Set logarithmic scale for X-axis
    ax.set_xscale("log")

    # Labels and title
    ax.set_title(dtype, fontsize=14, fontweight='bold')
    ax.set_xlabel("Array Size", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6)
    ax.legend()

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
