import numpy as np
import matplotlib.pyplot as plt
from noma_dataset_generator import (
    NOMAConfig,
    compute_alpha_double_prime
)

# --------------------------------------------------
# Fixed representative channel
# --------------------------------------------------
M = 3
beta = 2.0

distances = np.array([2.0, 4.5, 7.5])   # near, mid, far

np.random.seed(7)
h_real = np.random.randn(M)
h_imag = np.random.randn(M)
h_complex = (h_real + 1j*h_imag) / np.sqrt(2)

h_complex = h_complex / (distances ** (beta / 2))
h_gains = np.abs(h_complex)**2
h_gains = np.sort(h_gains)[::-1]

# --------------------------------------------------
# VERY IMPORTANT: small P region
# --------------------------------------------------
P_values = np.linspace(0.01, 0.8, 20)

alpha_dd = np.zeros((len(P_values), M))

for i, P in enumerate(P_values):
    config = NOMAConfig(M=M, P=P, Pg=0.01)
    alpha_dd[i, :] = compute_alpha_double_prime(h_gains, config)

# --------------------------------------------------
# Plot: Fig. 2 (PROPOSED – SIC-bound behaviour)
# --------------------------------------------------
plt.figure(figsize=(7, 5))

plt.plot(P_values, alpha_dd[:, 0],
         marker="s", linewidth=2, label="User 1")

plt.plot(P_values, alpha_dd[:, 1],
         marker="^", linewidth=2, label="User 2")

plt.plot(P_values, alpha_dd[:, 2],
         marker="o", linewidth=2.5, label="User 3")

plt.xlabel("Transmit Power P (W)")
plt.ylabel("Power Allocation Coefficient (SIC-bound)")
plt.title("PA Coefficients vs Transmit Power (Proposed Method – Fig. 2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
