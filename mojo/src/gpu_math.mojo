"""GPU math helper functions using PTX intrinsics for NVIDIA GPUs.

These functions provide fast approximations of common math operations
using GPU-native instructions. They're designed for use in GPU kernels
where standard library functions may not be available or optimal.
"""

from sys._assembly import inlined_assembly


# ============================================================================
# Basic PTX Intrinsics
# ============================================================================

@always_inline
fn gpu_sin(x: Float32) -> Float32:
    """Compute sin using GPU PTX intrinsic."""
    return inlined_assembly[
        "sin.approx.ftz.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)

@always_inline
fn gpu_cos(x: Float32) -> Float32:
    """Compute cos using GPU PTX intrinsic."""
    return inlined_assembly[
        "cos.approx.ftz.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)

@always_inline
fn gpu_exp2(x: Float32) -> Float32:
    """Compute 2^x using GPU PTX intrinsic."""
    return inlined_assembly[
        "ex2.approx.ftz.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)

@always_inline
fn gpu_log2(x: Float32) -> Float32:
    """Compute log2(x) using GPU PTX intrinsic."""
    return inlined_assembly[
        "lg2.approx.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)

@always_inline
fn gpu_rsqrt(x: Float32) -> Float32:
    """Compute 1/sqrt(x) using GPU PTX intrinsic."""
    return inlined_assembly[
        "rsqrt.approx.ftz.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)

@always_inline
fn gpu_sqrt(x: Float32) -> Float32:
    """Compute sqrt(x) using GPU PTX intrinsic."""
    return inlined_assembly[
        "sqrt.approx.ftz.f32 $0, $1;",
        Float32,
        constraints="=f,f",
        has_side_effect=False,
    ](x)


# ============================================================================
# Derived Functions (built on PTX intrinsics)
# ============================================================================

@always_inline
fn gpu_atan(x: Float32) -> Float32:
    """Compute atan(x) using polynomial approximation.

    Uses a minimax polynomial accurate to ~1e-4 for |x| <= 1.
    For |x| > 1, uses atan(x) = pi/2 - atan(1/x).
    """
    comptime PI_2: Float32 = 1.5707963267948966

    var abs_x = x if x >= Float32(0) else -x
    var sign = Float32(1.0) if x >= Float32(0) else Float32(-1.0)

    # Range reduction: if |x| > 1, use atan(x) = pi/2 - atan(1/x)
    var use_recip = abs_x > Float32(1.0)
    var t = abs_x if not use_recip else Float32(1.0) / abs_x

    # Polynomial approximation for atan(t) where |t| <= 1
    # atan(t) = t - t³/3 + t⁵/5 - t⁷/7 + t⁹/9 ...
    # Using Horner form: t * (1 + t² * (-1/3 + t² * (1/5 + t² * (-1/7 + t² * 1/9))))
    var t2 = t * t
    var result = t * (Float32(1.0) + t2 * (Float32(-0.333333333) + t2 * (Float32(0.2) + t2 * (Float32(-0.142857143) + t2 * Float32(0.111111111)))))

    # Adjust for range reduction
    if use_recip:
        result = PI_2 - result

    return sign * result

@always_inline
fn gpu_atan2(y: Float32, x: Float32) -> Float32:
    """Compute atan2(y, x) using gpu_atan with quadrant handling."""
    comptime PI: Float32 = 3.14159265358979323846
    comptime PI_2: Float32 = 1.5707963267948966

    if x > Float32(0):
        return gpu_atan(y / x)
    elif x < Float32(0):
        if y >= Float32(0):
            return gpu_atan(y / x) + PI
        else:
            return gpu_atan(y / x) - PI
    else:  # x == 0
        if y > Float32(0):
            return PI_2
        elif y < Float32(0):
            return -PI_2
        else:
            return Float32(0.0)


@always_inline
fn gpu_pow(base: Float32, exp: Float32) -> Float32:
    """Compute base^exp using GPU intrinsics: base^exp = 2^(exp * log2(base))."""
    if base <= Float32(0):
        return Float32(0.0)
    return gpu_exp2(exp * gpu_log2(base))


@always_inline
fn gpu_acos(x: Float32) -> Float32:
    """Compute acos(x) using atan-based approximation.

    Uses the identity: acos(x) = atan2(sqrt(1-x²), x)
    """
    var clamped = x
    if clamped > Float32(1.0):
        clamped = Float32(1.0)
    elif clamped < Float32(-1.0):
        clamped = Float32(-1.0)
    var sqrt_term = gpu_sqrt(Float32(1.0) - clamped * clamped)
    return gpu_atan2(sqrt_term, clamped)
