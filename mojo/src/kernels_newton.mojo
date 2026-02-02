"""Newton fractal GPU kernels."""

from gpu import global_idx
from layout import Layout, LayoutTensor
from math import log2


# ============================================================================
# Constants for Newton Fractal Glow Effect
# ============================================================================

comptime GLOW_BASE: Float64 = 8.0
comptime GLOW_ZOOM_SCALE: Float64 = 6.0
comptime GLOW_ZOOM_OFFSET: Float64 = 0.2
comptime GLOW_RANGE: Float64 = 12.0
comptime GLOW_MAX_ADD: Float64 = 120.0
comptime GLOW_MIN_ZOOM_SCALE: Float64 = 0.2


# ============================================================================
# Newton iteration kernel
# ============================================================================

fn newton_kernel[
    coeffs_layout: Layout,
    newton_layout: Layout,
](
    coeffs: LayoutTensor[DType.float64, coeffs_layout, MutAnyOrigin],
    num_coeffs: Int,
    output: LayoutTensor[DType.float64, newton_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    window_left: Float64,
    window_right: Float64,
    window_top: Float64,
    window_bottom: Float64,
    tolerance: Float64,
    imax: Int,
):
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)

    if px >= width or py >= height:
        return

    var z_re = window_left + (Float64(px) / Float64(width)) * (window_right - window_left)
    var z_im = window_top + (Float64(py) / Float64(height)) * (window_bottom - window_top)

    var iterations = Float64(imax)
    var zero_div = False
    var prev_diff_sq = Float64(1.0)

    for count in range(imax):
        var p_re = Float64(0.0)
        var p_im = Float64(0.0)
        var dp_re = Float64(0.0)
        var dp_im = Float64(0.0)

        for i in range(num_coeffs):
            var c = rebind[Float64](coeffs[i])
            var exp = num_coeffs - 1 - i

            var new_p_re = p_re * z_re - p_im * z_im + c
            var new_p_im = p_re * z_im + p_im * z_re
            p_re = new_p_re
            p_im = new_p_im

            if exp > 0:
                var dc = c * Float64(exp)
                var new_dp_re = dp_re * z_re - dp_im * z_im + dc
                var new_dp_im = dp_re * z_im + dp_im * z_re
                dp_re = new_dp_re
                dp_im = new_dp_im

        var dp_mag_sq = dp_re * dp_re + dp_im * dp_im
        if dp_mag_sq < 1e-20:
            zero_div = True
            break

        var ratio_re = (p_re * dp_re + p_im * dp_im) / dp_mag_sq
        var ratio_im = (p_im * dp_re - p_re * dp_im) / dp_mag_sq

        var old_re = z_re
        var old_im = z_im
        z_re = z_re - ratio_re
        z_im = z_im - ratio_im

        var diff_sq = (old_re - z_re) * (old_re - z_re) + (old_im - z_im) * (old_im - z_im)
        if diff_sq < tolerance * tolerance:
            var tol_sq = tolerance * tolerance
            var log_prev = log2(prev_diff_sq + 1e-30)
            var log_curr = log2(diff_sq + 1e-30)
            var log_tol = log2(tol_sq)
            var frac = (log_tol - log_prev) / (log_curr - log_prev + 1e-10)
            if frac < 0.0:
                frac = 0.0
            if frac > 1.0:
                frac = 1.0
            iterations = Float64(count) + frac
            break
        prev_diff_sq = diff_sq

    var pixel_idx = (py * width + px) * 3
    if zero_div:
        output[pixel_idx] = 0.0
        output[pixel_idx + 1] = 0.0
        output[pixel_idx + 2] = -1.0
    else:
        output[pixel_idx] = z_re
        output[pixel_idx + 1] = z_im
        output[pixel_idx + 2] = iterations


# ============================================================================
# Newton colorization kernel
# ============================================================================

fn colorize_kernel[
    newton_layout: Layout,
    roots_layout: Layout,
    rgb_layout: Layout,
](
    newton_output: LayoutTensor[DType.float64, newton_layout, MutAnyOrigin],
    roots: LayoutTensor[DType.float64, roots_layout, MutAnyOrigin],
    num_roots: Int,
    rgb_output: LayoutTensor[DType.uint8, rgb_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    tolerance: Float64,
    imax: Int,
    glow_intensity: Float64,
    zoom: Float64,
):
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)

    if px >= width or py >= height:
        return

    var newton_idx = (py * width + px) * 3
    var re = rebind[Float64](newton_output[newton_idx])
    var im = rebind[Float64](newton_output[newton_idx + 1])
    var iterations = rebind[Float64](newton_output[newton_idx + 2])

    var r: UInt8 = 150
    var g: UInt8 = 150
    var b: UInt8 = 150

    if iterations < 0:
        r = 128
        g = 128
        b = 128
    elif num_roots > 0:
        var root_idx = 0
        var best_dist_sq = Float64(1e10)

        for i in range(num_roots):
            var root_re = rebind[Float64](roots[i * 5])
            var root_im = rebind[Float64](roots[i * 5 + 1])
            var diff_re = root_re - re
            var diff_im = root_im - im
            var dist_sq = diff_re * diff_re + diff_im * diff_im
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                root_idx = i

        if num_roots > 0:
            var base_r = rebind[Float64](roots[root_idx * 5 + 2])
            var base_g = rebind[Float64](roots[root_idx * 5 + 3])
            var base_b = rebind[Float64](roots[root_idx * 5 + 4])

            var glow_start = GLOW_BASE + GLOW_ZOOM_SCALE / (zoom + GLOW_ZOOM_OFFSET)

            var t = (iterations - glow_start) / GLOW_RANGE
            if t < 0.0:
                t = 0.0
            if t > 1.0:
                t = 1.0

            var zoom_scale = 1.0
            if zoom > 1.0:
                zoom_scale = 1.0 / log2(zoom + 1.0)
                if zoom_scale < GLOW_MIN_ZOOM_SCALE:
                    zoom_scale = GLOW_MIN_ZOOM_SCALE

            var glow_factor = t * glow_intensity * zoom_scale

            var final_r = base_r
            var final_g = base_g
            var final_b = base_b

            var glow_add = glow_factor * GLOW_MAX_ADD
            final_r = final_r + glow_add
            final_g = final_g + glow_add
            final_b = final_b + glow_add

            if final_r > 255.0:
                final_r = 255.0
            if final_g > 255.0:
                final_g = 255.0
            if final_b > 255.0:
                final_b = 255.0

            r = UInt8(final_r)
            g = UInt8(final_g)
            b = UInt8(final_b)

    var rgb_idx = (py * width + px) * 3
    rgb_output[rgb_idx] = r
    rgb_output[rgb_idx + 1] = g
    rgb_output[rgb_idx + 2] = b
