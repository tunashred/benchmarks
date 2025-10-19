import matplotlib.pyplot as plt

# Given data
simd_data = {
    "Int": {
        "SequentialIterate": [
            (12, 0), (100, 0), (1000, 9), (10000, 108),
            (100000, 1254), (1000000, 13027), (2000000, 28804), (3000000, 46179)
        ]
    },
    "Long": {
        "SequentialIterate": [
            (12, 0), (100, 1), (1000, 19), (10000, 249),
            (100000, 2634), (1000000, 27338), (2000000, 92348), (3000000, 281257)
        ]
    },
    "Double": {
        "SequentialIterate": [
            (12, 0), (100, 1), (1000, 19), (10000, 247),
            (100000, 2636), (1000000, 27959), (2000000, 94089), (3000000, 280796)
        ]
    }
}

regular_data = {
    "int": {
        "Sequential": [
            (12, 0), (100, 3), (1000, 30), (10000, 299),
            (100000, 2960), (1000000, 29528), (2000000, 59818), (3000000, 94396)
        ]
    },
    "long": {
        "Sequential": [
            (12, 0), (100, 3), (1000, 30), (10000, 314),
            (100000, 3228), (1000000, 32278), (2000000, 112508), (3000000, 314590)
        ]
    },
    "double": {
        "Sequential": [
            (12, 1), (100, 7), (1000, 78), (10000, 765),
            (100000, 7681), (1000000, 76370), (2000000, 171045), (3000000, 339641)
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
