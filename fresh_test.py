#!/usr/bin/env python3
import sys
import importlib

# Force reload
if 'noma_dataset_generator' in sys.modules:
    importlib.reload(sys.modules['noma_dataset_generator'])

import numpy as np
from noma_dataset_generator import NOMAConfig, compute_alpha_double_prime

print("Fresh test with forced reload")
h_gains = np.array([2.0, 1.0, 0.5])
config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)

result = compute_alpha_double_prime(h_gains, config)
print(f'Result: {result}')
print(f'Any negative: {np.any(result < 0) if result is not None else "None"}')