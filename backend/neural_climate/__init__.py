"""
Mars GCM Neural Emulator.

A PyTorch neural network that learns to emulate the parametric Mars climate
model (mars_climate.py), providing 100-1000x faster inference for climate
predictions across (lat, lon, Ls, elevation) input space.

Architecture:
    Input:  (lat, lon_sin, lon_cos, ls_sin, ls_cos, elevation)  — 6 features
    Output: (T_mean, T_max, T_min, pressure, dust_tau, wind_mean, frost_prob) — 7 targets

Training data is generated from the existing parametric model on a dense
(lat × lon × Ls) grid, then the neural network learns the mapping.
Once trained, the emulator can replace the parametric model for bulk
climate queries (e.g., global maps, Monte Carlo site selection).
"""

__version__ = "0.1.0"
