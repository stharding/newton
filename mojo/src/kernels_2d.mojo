"""2D fractal GPU kernels (Mandelbrot, Julia, Burning Ship, Tricorn)."""

from gpu import global_idx
from layout import Layout, LayoutTensor
from math import sin, cos, log2, sqrt, exp2

from gpu_math import gpu_atan2


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
):
    """Apply smooth coloring based on escape iteration count."""
    if iterations == imax:
        output[pixel_idx] = 0
        output[pixel_idx + 1] = 0
        output[pixel_idx + 2] = 0
    else:
        var log_zn = Float32(0.5) * log2(Float32(final_r2))
        var smooth_iter = Float32(iterations) + Float32(1.0) - log2(log_zn)

        var t = smooth_iter * Float32(0.05) + Float32(color_seed)
        t = t - Float32(Int(t))

        var pi2 = Float32(6.28318530)
        var r = Float32(0.45) + Float32(0.35) * cos(pi2 * (t * Float32(0.8) + Float32(0.0)))
        var g = Float32(0.40) + Float32(0.30) * cos(pi2 * (t * Float32(0.8) + Float32(0.15)))
        var b = Float32(0.55) + Float32(0.35) * cos(pi2 * (t * Float32(0.8) + Float32(0.35)))

        var gray = (r + g + b) / Float32(3.0)
        r = gray + (r - gray) * Float32(1.1)
        g = gray + (g - gray) * Float32(1.1)
        b = gray + (b - gray) * Float32(1.1)

        if r < Float32(0.0): r = Float32(0.0)
        if r > Float32(1.0): r = Float32(1.0)
        if g < Float32(0.0): g = Float32(0.0)
        if g > Float32(1.0): g = Float32(1.0)
        if b < Float32(0.0): b = Float32(0.0)
        if b > Float32(1.0): b = Float32(1.0)

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

    var c_re = window_left + (Float64(px) / Float64(width)) * (window_right - window_left)
    var c_im = window_top + (Float64(py) / Float64(height)) * (window_bottom - window_top)

    var z_re = Float64(0.0)
    var z_im = Float64(0.0)
    var iterations = imax

    var z_re2 = Float64(0.0)
    var z_im2 = Float64(0.0)

    for count in range(imax):
        z_re2 = z_re * z_re
        z_im2 = z_im * z_im
        if z_re2 + z_im2 > 256.0:
            iterations = count
            break
        var new_re = z_re2 - z_im2 + c_re
        z_im = 2.0 * z_re * z_im + c_im
        z_re = new_re

    var pixel_idx = (py * width + px) * 3
    write_smooth_color[output_layout](output, pixel_idx, iterations, imax, z_re2 + z_im2, color_seed)


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

    var z_re = window_left + (Float64(px) / Float64(width)) * (window_right - window_left)
    var z_im = window_top + (Float64(py) / Float64(height)) * (window_bottom - window_top)

    var iterations = imax
    var final_r2 = Float64(0.0)

    var use_simple_square = power_im == 0.0 and power_re == 2.0

    for count in range(imax):
        var r2 = z_re * z_re + z_im * z_im
        if r2 > 256.0:
            iterations = count
            final_r2 = r2
            break

        if use_simple_square:
            var new_re = z_re * z_re - z_im * z_im + c_re
            var new_im = 2.0 * z_re * z_im + c_im
            z_re = new_re
            z_im = new_im
        else:
            var zx = Float32(z_re)
            var zy = Float32(z_im)
            var p_re = Float32(power_re)
            var p_im = Float32(power_im)

            var r = sqrt(zx * zx + zy * zy)

            if r < 1e-10:
                z_re = c_re
                z_im = c_im
            else:
                var theta = gpu_atan2(zy, zx)
                var ln_r = log2(r) * Float32(0.693147180559945)
                var log2_mag = p_re * log2(r) - p_im * theta * Float32(1.4426950408889634)
                var mag = exp2(log2_mag)
                var angle = p_re * theta + p_im * ln_r
                z_re = Float64(mag * cos(angle)) + c_re
                z_im = Float64(mag * sin(angle)) + c_im

    var pixel_idx = (py * width + px) * 3
    write_smooth_color[output_layout](output, pixel_idx, iterations, imax, final_r2, color_seed)


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

    var c_re = window_left + (Float64(px) / Float64(width)) * (window_right - window_left)
    var c_im = window_top + (Float64(py) / Float64(height)) * (window_bottom - window_top)

    var z_re = Float64(0.0)
    var z_im = Float64(0.0)
    var iterations = imax
    var final_r2 = Float64(0.0)

    for count in range(imax):
        var z_re2 = z_re * z_re
        var z_im2 = z_im * z_im
        var r2 = z_re2 + z_im2
        if r2 > 256.0:
            iterations = count
            final_r2 = r2
            break
        var abs_re = z_re if z_re >= 0 else -z_re
        var abs_im = z_im if z_im >= 0 else -z_im
        var new_re = abs_re * abs_re - abs_im * abs_im + c_re
        z_im = 2.0 * abs_re * abs_im + c_im
        z_re = new_re

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

    var c_re = window_left + (Float64(px) / Float64(width)) * (window_right - window_left)
    var c_im = window_top + (Float64(py) / Float64(height)) * (window_bottom - window_top)

    var z_re = Float64(0.0)
    var z_im = Float64(0.0)
    var iterations = imax
    var final_r2 = Float64(0.0)

    for count in range(imax):
        var z_re2 = z_re * z_re
        var z_im2 = z_im * z_im
        var r2 = z_re2 + z_im2
        if r2 > 256.0:
            iterations = count
            final_r2 = r2
            break
        var new_re = z_re2 - z_im2 + c_re
        z_im = -2.0 * z_re * z_im + c_im
        z_re = new_re

    var pixel_idx = (py * width + px) * 3
    write_smooth_color[output_layout](output, pixel_idx, iterations, imax, final_r2, color_seed)
