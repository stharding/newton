"""GPU math helper functions with CPU fallbacks.

These functions use fast PTX intrinsics on NVIDIA GPUs and fall back
to standard math on CPU.
"""

from sys._assembly import inlined_assembly
from sys import is_nvidia_gpu
from math import sin, cos, sqrt, log2, exp2, atan, atan2, acos


# ============================================================================
# Basic Math Functions (PTX on GPU, standard math on CPU)
# ============================================================================

@always_inline
fn gpu_sin(x: Float32) -> Float32:
    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "sin.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return sin(x)

@always_inline
fn gpu_cos(x: Float32) -> Float32:
    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "cos.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return cos(x)

@always_inline
fn gpu_exp2(x: Float32) -> Float32:
    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "ex2.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return exp2(x)

@always_inline
fn gpu_log2(x: Float32) -> Float32:
    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "lg2.approx.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return log2(x)

@always_inline
fn gpu_rsqrt(x: Float32) -> Float32:
    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "rsqrt.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return 1.0 / sqrt(x)

@always_inline
fn gpu_sqrt(x: Float32) -> Float32:
    @parameter
    if is_nvidia_gpu():
        return inlined_assembly[
            "sqrt.approx.ftz.f32 $0, $1;",
            Float32,
            constraints="=f,f",
            has_side_effect=False,
        ](x)
    else:
        return sqrt(x)


# ============================================================================
# Derived Functions (same implementation, use above primitives)
# ============================================================================

@always_inline
fn gpu_atan(x: Float32) -> Float32:
    """Compute atan(x) using polynomial approximation on GPU, standard on CPU."""
    @parameter
    if is_nvidia_gpu():
        # Polynomial approximation for GPU (no atan intrinsic in PTX)
        comptime PI_2: Float32 = 1.5707963267948966

        var abs_x = x if x >= Float32(0) else -x
        var sign = Float32(1.0) if x >= Float32(0) else Float32(-1.0)

        var use_recip = abs_x > Float32(1.0)
        var t = abs_x if not use_recip else Float32(1.0) / abs_x

        var t2 = t * t
        var result = t * (Float32(1.0) + t2 * (Float32(-0.333333333) + t2 * (Float32(0.2) + t2 * (Float32(-0.142857143) + t2 * Float32(0.111111111)))))

        if use_recip:
            result = PI_2 - result

        return sign * result
    else:
        return atan(x)

@always_inline
fn gpu_atan2(y: Float32, x: Float32) -> Float32:
    """Compute atan2(y, x)."""
    @parameter
    if is_nvidia_gpu():
        comptime PI: Float32 = 3.14159265358979323846
        comptime PI_2: Float32 = 1.5707963267948966

        if x > Float32(0):
            return gpu_atan(y / x)
        elif x < Float32(0):
            if y >= Float32(0):
                return gpu_atan(y / x) + PI
            else:
                return gpu_atan(y / x) - PI
        else:
            if y > Float32(0):
                return PI_2
            elif y < Float32(0):
                return -PI_2
            else:
                return Float32(0.0)
    else:
        return atan2(y, x)


@always_inline
fn gpu_pow(base: Float32, exp: Float32) -> Float32:
    """Compute base^exp using GPU intrinsics: base^exp = 2^(exp * log2(base))."""
    if base <= Float32(0):
        return Float32(0.0)
    return gpu_exp2(exp * gpu_log2(base))


@always_inline
fn gpu_acos(x: Float32) -> Float32:
    """Compute acos(x)."""
    @parameter
    if is_nvidia_gpu():
        var clamped = x
        if clamped > Float32(1.0):
            clamped = Float32(1.0)
        elif clamped < Float32(-1.0):
            clamped = Float32(-1.0)
        var sqrt_term = gpu_sqrt(Float32(1.0) - clamped * clamped)
        return gpu_atan2(sqrt_term, clamped)
    else:
        return acos(x)
