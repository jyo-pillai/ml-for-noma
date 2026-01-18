import numpy as np
import matplotlib.pyplot as plt
from noma_dataset_generator import (
    NOMAConfig,
    compute_power_allocation,
    compute_sum_rate
)

# --------------------------------------------------
# Simulation Parameters (Paper-consistent)
# --------------------------------------------------
M = 3
beta = 2.0                  # Path loss exponent
N_trials = 3000             # Monte Carlo runs per P
P_values = np.linspace(0.1, 3.0, 12)

# Distance range (near → far users)
d_min = 1.0
d_max = 10.0

avg_sum_rate = []

# --------------------------------------------------
# Monte Carlo Sweep over Transmit Power
# --------------------------------------------------
for P in P_values:
    print(f"Simulating for P = {P:.2f} W")
    config = NOMAConfig(M=M, P=P)
    sum_rate_trials = []

    for _ in range(N_trials):

        # ------------------------------------------
        # Generate distance-aware Rayleigh channels
        # ------------------------------------------
        distances = np.random.uniform(d_min, d_max, M)

        h_real = np.random.randn(M)
        h_imag = np.random.randn(M)
        h_complex = (h_real + 1j * h_imag) / np.sqrt(2)

        # Apply path loss: h ~ CN(0, d^{-beta})
        h_complex = h_complex / (distances ** (beta / 2))
        h_gains = np.abs(h_complex) ** 2

        # Sort strongest → weakest (SIC order)
        h_gains = np.sort(h_gains)[::-1]

        # ------------------------------------------
        # Compute optimal PA (Algorithm 1)
        # ------------------------------------------
        alpha_star = compute_power_allocation(h_gains, config)

        # ------------------------------------------
        # Compute sum rate
        # ------------------------------------------
        sum_rate = compute_sum_rate(h_gains, alpha_star, config)
        sum_rate_trials.append(sum_rate)

    avg_sum_rate.append(np.mean(sum_rate_trials))

# --------------------------------------------------
# Plot: Fig. 3 (Corrected)
# --------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(P_values, avg_sum_rate, "o-", linewidth=2)

plt.xlabel("Transmit Power P (W)")
plt.ylabel("Average Sum Rate (bps/Hz)")
plt.title("Sum Rate vs Transmit Power (NOMA with Path Loss)")
plt.grid(True)
plt.tight_layout()
plt.show()
