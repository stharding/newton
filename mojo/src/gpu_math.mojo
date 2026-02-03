"""GPU math helper functions for operations not in stdlib GPU intrinsics.

The stdlib math functions (sin, cos, sqrt, log2, exp2, rsqrt, pow) already have
GPU intrinsics. This module provides GPU-compatible versions of atan/atan2/acos
that would otherwise call libm (unavailable on GPU).

Note: pow (via ** operator) works on GPU - it uses exp2(exp * log2(base)).
"""

from math import acos, atan, atan2, cos, pi, sin, sqrt
from sys import is_gpu


# ============================================================================
# Constants
# ============================================================================

comptime PI32: Float32 = pi
comptime PI_2_32: Float32 = pi / 2


# ============================================================================
# GPU-compatible atan (stdlib version calls libm)
# ============================================================================

@always_inline
fn gpu_atan(x: Float32) -> Float32:
    """Compute atan(x) using polynomial approximation on GPU, standard on CPU."""
    @parameter
    if is_gpu():
        var abs_x = x if x >= Float32(0) else -x
        var sign = Float32(1.0) if x >= Float32(0) else Float32(-1.0)

        var use_recip = abs_x > Float32(1.0)
        var t = abs_x if not use_recip else Float32(1.0) / abs_x

        var t2 = t * t
        var result = t * (Float32(1.0) + t2 * (Float32(-0.333333333) + t2 * (Float32(0.2) + t2 * (Float32(-0.142857143) + t2 * Float32(0.111111111)))))

        if use_recip:
            result = PI_2_32 - result

        return sign * result
    else:
        return atan(x)


# ============================================================================
# GPU-compatible atan2 (stdlib version calls libm)
# ============================================================================

@always_inline
fn gpu_atan2(y: Float32, x: Float32) -> Float32:
    """Compute atan2(y, x) using gpu_atan on GPU, standard on CPU."""
    @parameter
    if is_gpu():
        if x > Float32(0):
            return gpu_atan(y / x)
        elif x < Float32(0):
            if y >= Float32(0):
                return gpu_atan(y / x) + PI32
            else:
                return gpu_atan(y / x) - PI32
        else:
            if y > Float32(0):
                return PI_2_32
            elif y < Float32(0):
                return -PI_2_32
            else:
                return Float32(0.0)
    else:
        return atan2(y, x)


# ============================================================================
# GPU-compatible acos (stdlib version calls libm)
# ============================================================================

@always_inline
fn gpu_acos(x: Float32) -> Float32:
    """Compute acos(x) using atan2 identity on GPU, standard on CPU."""
    @parameter
    if is_gpu():
        var clamped = x
        if clamped > Float32(1.0):
            clamped = Float32(1.0)
        elif clamped < Float32(-1.0):
            clamped = Float32(-1.0)
        var sqrt_term = sqrt(Float32(1.0) - clamped * clamped)
        return gpu_atan2(sqrt_term, clamped)
    else:
        return acos(x)
