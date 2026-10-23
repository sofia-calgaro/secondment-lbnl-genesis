
import numpy as np
import matplotlib.pyplot as plt

# Voxel counts from grid generation
data = {
    "5 mm": {
        "Inside ULAr": 49_224_964,
        "Fully overlapping (removed)": 1_189_262,
        "Partially overlapping (kept)": 711_935,
        "No overlap (kept)": 48_035_702,
    },
    "10 mm": {
        "Inside ULAr": 6_102_076,
        "Fully overlapping (removed)": 115_277,
        "Partially overlapping (kept)": 183_252,
        "No overlap (kept)": 5_986_810,
    },
}

categories = [
    "Fully overlapping\n(removed)",
    "Partially overlapping\n(kept)",
    "No overlap\n(kept)",
]

keys = [
    "Fully overlapping (removed)",
    "Partially overlapping (kept)",
    "No overlap (kept)",
]

voxel_sizes = list(data.keys())
counts = np.array([
    [data[size][key] for key in keys]
    for size in voxel_sizes
])

# Percentages relative to all voxels inside ULAr
percentages = np.array([
    [data[size][key] / data[size]["Inside ULAr"] * 100 for key in keys]
    for size in voxel_sizes
])

x = np.arange(len(categories))
width = 0.35

# --- Percentages of voxels inside ULAr ---
fig, ax = plt.subplots(figsize=(5.5, 3.2))

bars1 = ax.bar(x - width / 2, percentages[0], width, label="5 mm")
bars2 = ax.bar(x + width / 2, percentages[1], width, label="10 mm")

ax.set_ylabel("Fraction of voxels inside ULAr (%)")
ax.set_title("Dead Ge + Cu rods overlap")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc="upper left")

ax.bar_label(bars1, labels=[f"{v:.2f}%" for v in percentages[0]],
             padding=3, fontsize=9)
ax.bar_label(bars2, labels=[f"{v:.2f}%" for v in percentages[1]],
             padding=3, fontsize=9)

ax.set_ylim(0, max(percentages.max() * 1.15, 5))
fig.tight_layout()
fig.savefig("dead_mass_overlap_percentages.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

print("Voxel overlap summary")
print("-" * 75)

for i, size in enumerate(voxel_sizes):
    print(f"\n{size} voxels (Inside ULAr: {data[size]['Inside ULAr']:,})")
    for key, count in zip(keys, counts[i]):
        fraction = count / data[size]["Inside ULAr"] * 100
        print(f"  {key:35s}: {count:>12,} ({fraction:5.2f}%)")