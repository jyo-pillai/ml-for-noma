#!/usr/bin/env python3

import numpy as np

# Test the corrected implementation
h_gains = np.array([2.0, 1.0, 0.5])

# Import and test
try:
    from noma_dataset_generator import NOMAConfig, compute_alpha_double_prime, compute_power_allocation
    
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Testing corrected implementation:")
    print(f"Channel gains: {h_gains}")
    
    alpha_double_prime = compute_alpha_double_prime(h_gains, config)
    print(f"α'': {alpha_double_prime}")
    
    if alpha_double_prime is not None and not np.any(alpha_double_prime < 0):
        alpha_star = compute_power_allocation(h_gains, config)
        print(f"α*: {alpha_star}")
        print(f"Varies from 1/M? {not np.allclose(alpha_star, 1/config.M)}")
        print("SUCCESS: Implementation is working!")
    else:
        print("α'' has negative values or is None")
        
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")