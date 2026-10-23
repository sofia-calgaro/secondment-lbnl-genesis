import numpy as np
import matplotlib.pyplot as plt

attenuation_length_mm = 300  # 30 cm
z0_mm = 350                  # SiPMs z-position

z_mm = np.linspace(325, 2000, 2000)

# Distance traveled from the SiPM position
distance_mm = np.maximum(z_mm - z0_mm, 0)

# Expected surviving fraction [%]
survival = np.exp(-distance_mm / attenuation_length_mm) * 100

# Positions where survival reaches each threshold
z_10 = z0_mm + attenuation_length_mm * np.log(100 / 10)
z_1 = z0_mm + attenuation_length_mm * np.log(100 / 1)

print(f"10% survival reached at z = {z_10:.1f} mm")
print(f"1% survival reached at z = {z_1:.1f} mm")

plt.figure(figsize=(8, 5))
plt.plot(z_mm, survival, linewidth=2)

plt.axvline(z0_mm, linestyle="--", color="gray",
            label=f"SiPMs z-position ({z0_mm} mm)")
plt.axvline(z_10, linestyle="--", color="darkorange",
            label=f"10% survival (z = {z_10:.0f} mm)")
plt.axvline(z_1, linestyle="--", color="red",
            label=f"1% survival (z = {z_1:.0f} mm)")

plt.axhline(10, linestyle=":", color="darkorange")
plt.axhline(1, linestyle=":", color="red")

plt.xlabel("z [mm]")
plt.ylabel("Expected surviving fraction [%]")
plt.title("Optical photon survival (attenuation length = 30 cm)")
plt.xlim(325, 2000)
plt.ylim(0.1, 200)
plt.yscale("log")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("optical_photon_survival.png", dpi=200)
plt.show()

z_target = 1200+350
prob = np.exp(-(z_target - z0_mm) / attenuation_length_mm) * 100

print(f"Survival at z = {z_target} mm: {prob:.2f}%")