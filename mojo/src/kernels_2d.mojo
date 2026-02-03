"""2D fractal GPU kernels (Mandelbrot, Julia, Burning Ship, Tricorn)."""

from complex import ComplexFloat64
from gpu import global_idx
from layout import Layout, LayoutTensor
from math import clamp, cos, exp2, log2, pi, sin, sqrt, tau

from gpu_math import gpu_atan2


# ============================================================================
# Constants
# ============================================================================

comptime ESCAPE_RADIUS_SQ: Float64 = 256.0
"""Escape radius squared for iteration bailout."""

comptime COLOR_FREQUENCY: Float32 = 0.05
"""Frequency multiplier for smooth coloring."""

comptime SATURATION_BOOST: Float32 = 1.1
"""Saturation boost for color output."""


# ============================================================================
# Helper: Convert pixel to complex coordinate
# ============================================================================

@always_inline
fn pixel_to_complex(
    px: Int, py: Int, width: Int, height: Int,
    left: Float64, right: Float64, top: Float64, bottom: Float64,
) -> ComplexFloat64:
    """Convert pixel coordinates to complex plane coordinates."""
    var re = left + (Float64(px) / Float64(width)) * (right - left)
    var im = top + (Float64(py) / Float64(height)) * (bottom - top)
    return ComplexFloat64(re, im)


# ============================================================================
# Shared smooth coloring (cosine gradient palette)
# ============================================================================

@always_inline
fn write_smooth_color[
    layout: Layout
](
    output: LayoutTensor[DType.uint8, layout, MutAnyOrigin],
    pixel_idx: Int,
    iterations: Int,
    imax: Int,
    final_r2: Float64,
    color_seed: Float64,
    power: Float32 = 2.0,
):
    """Apply smooth coloring based on escape iteration count.

    The power parameter adjusts the smooth iteration formula for z^p fractals.
    """
    if iterations == imax:
        output[pixel_idx] = 0
        output[pixel_idx + 1] = 0
        output[pixel_idx + 2] = 0
    else:
        var log_zn = Float32(0.5) * log2(Float32(final_r2))  # log2(|z|)
        # For z^p iteration, divide by log2(p) to normalize the smoothing
        var smooth_iter = Float32(iterations) + Float32(1.0) - log2(log_zn) / log2(power)

        var t = smooth_iter * COLOR_FREQUENCY + Float32(color_seed)
        t = t - Float32(Int(t))

        comptime TAU32: Float32 = tau
        var r = Float32(0.45) + Float32(0.35) * cos(TAU32 * (t * Float32(0.8) + Float32(0.0)))
        var g = Float32(0.40) + Float32(0.30) * cos(TAU32 * (t * Float32(0.8) + Float32(0.15)))
        var b = Float32(0.55) + Float32(0.35) * cos(TAU32 * (t * Float32(0.8) + Float32(0.35)))

        # Boost saturation
        var gray = (r + g + b) / Float32(3.0)
        r = gray + (r - gray) * SATURATION_BOOST
        g = gray + (g - gray) * SATURATION_BOOST
        b = gray + (b - gray) * SATURATION_BOOST

        # Clamp to [0, 1]
        r = clamp(r, Float32(0.0), Float32(1.0))
        g = clamp(g, Float32(0.0), Float32(1.0))
        b = clamp(b, Float32(0.0), Float32(1.0))

        output[pixel_idx] = UInt8(r * Float32(255.0))
        output[pixel_idx + 1] = UInt8(g * Float32(255.0))
        output[pixel_idx + 2] = UInt8(b * Float32(255.0))


# ============================================================================
# Mandelbrot kernel: z = z² + c, c = pixel coordinate
# ============================================================================

fn mandelbrot_kernel[
    output_layout: Layout,
](
    output: LayoutTensor[DType.uint8, output_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    window_left: Float64,
    window_right: Float64,
    window_top: Float64,
    window_bottom: Float64,
    imax: Int,
    color_seed: Float64,
):
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)

    if px >= width or py >= height:
        return

    var c = pixel_to_complex(px, py, width, height, window_left, window_right, window_top, window_bottom)
    var z = ComplexFloat64(0.0, 0.0)
    var iterations = imax

    for count in range(imax):
        if z.squared_norm() > ESCAPE_RADIUS_SQ:
            iterations = count
            break
        z = z.squared_add(c)

    var pixel_idx = (py * width + px) * 3
    write_smooth_color[output_layout](output, pixel_idx, iterations, imax, z.squared_norm(), color_seed)


# ============================================================================
# Julia kernel: z = z^power + c, z starts at pixel, c is constant
# ============================================================================

fn julia_kernel[
    output_layout: Layout,
](
    output: LayoutTensor[DType.uint8, output_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    window_left: Float64,
    window_right: Float64,
    window_top: Float64,
    window_bottom: Float64,
    c_re: Float64,
    c_im: Float64,
    power_re: Float64,
    power_im: Float64,
    imax: Int,
    color_seed: Float64,
):
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)

    if px >= width or py >= height:
        return

    var z = pixel_to_complex(px, py, width, height, window_left, window_right, window_top, window_bottom)
    var c = ComplexFloat64(c_re, c_im)

    var iterations = imax
    var final_r2 = Float64(0.0)

    # Detect positive integer powers for direct computation (avoids atan2 discontinuity)
    var int_power = Int(power_re)
    var is_int_power = power_im == 0.0 and power_re == Float64(int_power) and int_power >= 2

    for count in range(imax):
        var r2 = z.squared_norm()
        if r2 > ESCAPE_RADIUS_SQ:
            iterations = count
            final_r2 = r2
            break

        if is_int_power:
            # Compute z^n via iterative complex multiplication
            var result = z
            for _ in range(int_power - 1):
                result = result * z
            z = result + c
        else:
            # General complex power: z^p = exp(p * log(z))
            var zx = Float32(z.re)
            var zy = Float32(z.im)
            var p_re = Float32(power_re)
            var p_im = Float32(power_im)

            var r = sqrt(zx * zx + zy * zy)

            if r < 1e-10:
                z = c
            else:
                comptime LN2: Float32 = 0.693147180559945
                comptime LOG2E: Float32 = 1.4426950408889634
                var theta = gpu_atan2(zy, zx)
                var ln_r = log2(r) * LN2
                var log2_mag = p_re * log2(r) - p_im * theta * LOG2E
                var mag = exp2(log2_mag)
                var angle = p_re * theta + p_im * ln_r
                z = ComplexFloat64(Float64(mag * cos(angle)), Float64(mag * sin(angle))) + c

    var pixel_idx = (py * width + px) * 3
    write_smooth_color[output_layout](output, pixel_idx, iterations, imax, final_r2, color_seed, Float32(power_re))


# ============================================================================
# Burning Ship kernel: z = (|Re(z)| + i|Im(z)|)² + c
# ============================================================================

fn burning_ship_kernel[
    output_layout: Layout,
](
    output: LayoutTensor[DType.uint8, output_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    window_left: Float64,
    window_right: Float64,
    window_top: Float64,
    window_bottom: Float64,
    imax: Int,
    color_seed: Float64,
):
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)

    if px >= width or py >= height:
        return

    var c = pixel_to_complex(px, py, width, height, window_left, window_right, window_top, window_bottom)
    var z = ComplexFloat64(0.0, 0.0)
    var iterations = imax
    var final_r2 = Float64(0.0)

    for count in range(imax):
        var r2 = z.squared_norm()
        if r2 > ESCAPE_RADIUS_SQ:
            iterations = count
            final_r2 = r2
            break
        # Take absolute values before squaring
        var abs_z = ComplexFloat64(z.re if z.re >= 0 else -z.re, z.im if z.im >= 0 else -z.im)
        z = abs_z.squared_add(c)

    var pixel_idx = (py * width + px) * 3
    write_smooth_color[output_layout](output, pixel_idx, iterations, imax, final_r2, color_seed)


# ============================================================================
# Tricorn kernel: z = conj(z)² + c (Mandelbar)
# ============================================================================

fn tricorn_kernel[
    output_layout: Layout,
](
    output: LayoutTensor[DType.uint8, output_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    window_left: Float64,
    window_right: Float64,
    window_top: Float64,
    window_bottom: Float64,
    imax: Int,
    color_seed: Float64,
):
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)

    if px >= width or py >= height:
        return

    var c = pixel_to_complex(px, py, width, height, window_left, window_right, window_top, window_bottom)
    var z = ComplexFloat64(0.0, 0.0)
    var iterations = imax
    var final_r2 = Float64(0.0)

    for count in range(imax):
        var r2 = z.squared_norm()
        if r2 > ESCAPE_RADIUS_SQ:
            iterations = count
            final_r2 = r2
            break
        # Square the conjugate: conj(z)² + c
        z = z.conj().squared_add(c)

    var pixel_idx = (py * width + px) * 3
    write_smooth_color[output_layout](output, pixel_idx, iterations, imax, final_r2, color_seed)
