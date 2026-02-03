"""3D fractal distance estimators and ray marching utilities.

This module provides distance estimator functions for 3D fractals,
designed for use with ray marching in GPU kernels.
"""

from math import cos, log2, pi, sin, sqrt

from gpu_math import gpu_acos, gpu_atan2


# ============================================================================
# Constants for 3D Ray Marching
# ============================================================================

comptime MAX_STEPS: Int = 256
"""Maximum ray march iterations."""

comptime MAX_DIST: Float32 = 10.0
"""Maximum ray travel distance before giving up."""

comptime EPSILON: Float32 = 0.0005
"""Surface hit threshold - ray closer than this is considered a hit."""

comptime NORMAL_EPSILON: Float32 = 0.001
"""Gradient sampling offset for normal calculation."""

comptime LN2_32: Float32 = 0.693147180559945
"""Natural log of 2, for distance estimation formula."""


# ============================================================================
# Mandelbulb Distance Estimator
# ============================================================================

@always_inline
fn mandelbulb_de(
    x: Float32, y: Float32, z: Float32,
    power: Float32,
    imax: Int,
) -> Float32:
    """Compute distance estimate to Mandelbulb surface.

    Uses the triplex power formula in spherical coordinates.
    Returns 0.5 * r * ln(r) / dr where dr tracks the derivative.

    Args:
        x, y, z: Point to evaluate distance from.
        power: Mandelbulb power parameter (classic is 8.0).
        imax: Maximum iterations for convergence.

    Returns:
        Estimated distance to the Mandelbulb surface.
    """
    var zx = x
    var zy = y
    var zz = z
    var dr = Float32(1.0)
    var r = Float32(0.0)

    for _ in range(imax):
        r = sqrt(zx * zx + zy * zy + zz * zz)

        if r > Float32(2.0):
            break

        # Convert to spherical coordinates
        var theta = gpu_acos(zz / r)
        var phi = gpu_atan2(zy, zx)

        # Update derivative: dr = dr * n * r^(n-1) + 1
        # Using ** operator which works on GPU via exp2/log2
        dr = (r ** (power - Float32(1.0))) * power * dr + Float32(1.0)

        # Compute z^n using spherical coords
        var zr = r ** power
        var new_theta = theta * power
        var new_phi = phi * power

        # Convert back to Cartesian and add c (starting point)
        var sin_theta = sin(new_theta)
        zx = zr * sin_theta * cos(new_phi) + x
        zy = zr * sin_theta * sin(new_phi) + y
        zz = zr * cos(new_theta) + z

    # Distance estimate: 0.5 * r * ln(r) / dr
    return Float32(0.5) * r * log2(r) * LN2_32 / dr
