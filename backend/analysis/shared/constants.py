"""Physical and planetary constants for SHARAD and Mars analysis.

Single source of truth — all analysis modules should import from here
instead of defining their own copies.
"""

# ── SHARAD instrument ────────────────────────────────────────────
SPEED_OF_LIGHT = 299_792_458.0                      # m/s (vacuum)
SHARAD_SAMPLE_INTERVAL_US = 3.0 / 80.0              # 0.0375 µs per range bin
SHARAD_SAMPLE_INTERVAL_S = SHARAD_SAMPLE_INTERVAL_US * 1e-6
N_SHARAD_RANGE_BINS = 667

# ── Mars ellipsoid (IAU 2000) ────────────────────────────────────
MARS_EQUATORIAL_RADIUS_M = 3_396_190.0              # equatorial radius (a)
MARS_POLAR_RADIUS_M = 3_376_200.0                   # polar radius (b)
MARS_MEAN_RADIUS_M = (2 * MARS_EQUATORIAL_RADIUS_M + MARS_POLAR_RADIUS_M) / 3.0
MARS_MEAN_RADIUS_KM = MARS_MEAN_RADIUS_M / 1000.0
MARS_AB2 = (MARS_EQUATORIAL_RADIUS_M / MARS_POLAR_RADIUS_M) ** 2  # ≈ 1.01184
