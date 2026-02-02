"""Newton fractal GPU renderer - Python-importable module."""

from os import abort
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from random import random_ui64
from math import ceildiv, atan2, log2
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu import global_idx
from layout import Layout, LayoutTensor
from sys import has_accelerator, is_nvidia_gpu
from memory import UnsafePointer
from memory.unsafe_pointer import alloc

from gpu_math import gpu_sin, gpu_cos, gpu_exp2, gpu_log2, gpu_sqrt, gpu_rsqrt, gpu_atan, gpu_atan2, gpu_pow, gpu_acos
from fractals_3d import mandelbulb_de, MAX_STEPS, MAX_DIST, EPSILON, NORMAL_EPSILON


# ============================================================================
# Constants for Newton Fractal Glow Effect
# ============================================================================

# Glow threshold adapts to zoom: glow_start = GLOW_BASE + GLOW_ZOOM_SCALE / (zoom + GLOW_ZOOM_OFFSET)
# This gives higher thresholds when zoomed in (where iterations are high everywhere)
comptime GLOW_BASE: Float64 = 8.0        # Base iteration threshold for glow
comptime GLOW_ZOOM_SCALE: Float64 = 6.0  # How much zoom affects threshold
comptime GLOW_ZOOM_OFFSET: Float64 = 0.2 # Prevents division issues at zoom=0
comptime GLOW_RANGE: Float64 = 12.0      # Iterations over which glow ramps up
comptime GLOW_MAX_ADD: Float64 = 120.0   # Max brightness added per channel
comptime GLOW_MIN_ZOOM_SCALE: Float64 = 0.2  # Min intensity scale when zoomed out


# ============================================================================
# GPU Kernels
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
            # Smooth iteration using interpolation between prev and current step
            var tol_sq = tolerance * tolerance
            # Linear interpolation in log space
            var log_prev = log2(prev_diff_sq + 1e-30)
            var log_curr = log2(diff_sq + 1e-30)
            var log_tol = log2(tol_sq)
            # Where does tolerance fall between prev and curr in log space?
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
        # Color ALL pixels (converged or not) based on closest root
        # This prevents gray areas when zoomed out where many pixels
        # land in chaotic boundary regions
        var root_idx = 0
        var best_dist_sq = Float64(1e10)

        # Find the closest known root
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

            # Edge glow: higher iterations = closer to basin boundary
            # Threshold adapts to zoom level for consistent appearance
            var glow_start = GLOW_BASE + GLOW_ZOOM_SCALE / (zoom + GLOW_ZOOM_OFFSET)

            var t = (iterations - glow_start) / GLOW_RANGE
            if t < 0.0:
                t = 0.0
            if t > 1.0:
                t = 1.0

            # Scale intensity down when zoomed out to prevent basin wash-out
            var zoom_scale = 1.0
            if zoom > 1.0:
                zoom_scale = 1.0 / log2(zoom + 1.0)
                if zoom_scale < GLOW_MIN_ZOOM_SCALE:
                    zoom_scale = GLOW_MIN_ZOOM_SCALE

            var glow_factor = t * glow_intensity * zoom_scale

            # Start with full base colors, add glow toward white
            var final_r = base_r
            var final_g = base_g
            var final_b = base_b

            var glow_add = glow_factor * GLOW_MAX_ADD
            final_r = final_r + glow_add
            final_g = final_g + glow_add
            final_b = final_b + glow_add

            # Clamp to valid range
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
        if z_re2 + z_im2 > 256.0:  # Larger escape radius for smooth coloring
            iterations = count
            break
        var new_re = z_re2 - z_im2 + c_re
        z_im = 2.0 * z_re * z_im + c_im
        z_re = new_re

    # Color based on iteration count
    var pixel_idx = (py * width + px) * 3
    if iterations == imax:
        output[pixel_idx] = 0
        output[pixel_idx + 1] = 0
        output[pixel_idx + 2] = 0
    else:
        # Smooth iteration count: n + 1 - log2(log2(|z|))
        var mag_sq = z_re2 + z_im2
        var log_zn = 0.5 * gpu_log2(Float32(mag_sq))  # log2(|z|)
        var smooth_iter = Float32(iterations) + Float32(1.0) - gpu_log2(log_zn)

        # Use smooth iteration for coloring
        var t = smooth_iter * Float32(0.05) + Float32(color_seed)  # Slower color cycling
        t = t - Float32(Int(t))  # Wrap to 0-1

        # Elegant color palette: deep navy -> dusty rose -> cream -> teal
        # Using cosine gradients with carefully tuned parameters
        var pi2 = Float32(6.28318530)

        # Muted, sophisticated palette
        # a = base, b = amplitude, c = frequency, d = phase
        var r = Float32(0.45) + Float32(0.35) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.0)))
        var g = Float32(0.40) + Float32(0.30) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.15)))
        var b = Float32(0.55) + Float32(0.35) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.35)))

        # Subtle saturation adjustment
        var gray = (r + g + b) / Float32(3.0)
        r = gray + (r - gray) * Float32(1.1)
        g = gray + (g - gray) * Float32(1.1)
        b = gray + (b - gray) * Float32(1.1)

        # Clamp
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
# Julia kernel: z = z² + c, z starts at pixel, c is constant
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

    # Fast path for z^2 (most common case)
    var use_simple_square = power_im == 0.0 and power_re == 2.0

    for count in range(imax):
        var r2 = z_re * z_re + z_im * z_im
        if r2 > 256.0:  # Larger escape radius for smooth coloring
            iterations = count
            final_r2 = r2
            break

        if use_simple_square:
            # Direct z^2 calculation: (a+bi)^2 = a^2 - b^2 + 2abi
            var new_re = z_re * z_re - z_im * z_im + c_re
            var new_im = 2.0 * z_re * z_im + c_im
            z_re = new_re
            z_im = new_im
        else:
            # Complex power: z^(a+bi) where z = r*e^(i*theta)
            var zx = Float32(z_re)
            var zy = Float32(z_im)
            var p_re = Float32(power_re)
            var p_im = Float32(power_im)

            var r = gpu_sqrt(zx * zx + zy * zy)

            if r < 1e-10:
                z_re = c_re
                z_im = c_im
            else:
                var theta = gpu_atan2(zy, zx)
                var ln_r = gpu_log2(r) * Float32(0.693147180559945)
                var log2_mag = p_re * gpu_log2(r) - p_im * theta * Float32(1.4426950408889634)
                var mag = gpu_exp2(log2_mag)
                var angle = p_re * theta + p_im * ln_r
                z_re = Float64(mag * gpu_cos(angle)) + c_re
                z_im = Float64(mag * gpu_sin(angle)) + c_im

    # Color based on iteration count
    var pixel_idx = (py * width + px) * 3
    if iterations == imax:
        output[pixel_idx] = 0
        output[pixel_idx + 1] = 0
        output[pixel_idx + 2] = 0
    else:
        # Smooth iteration count
        var log_zn = Float32(0.5) * gpu_log2(Float32(final_r2))
        var smooth_iter = Float32(iterations) + Float32(1.0) - gpu_log2(log_zn)

        var t = smooth_iter * Float32(0.05) + Float32(color_seed)
        t = t - Float32(Int(t))

        # Elegant muted palette
        var pi2 = Float32(6.28318530)
        var r = Float32(0.45) + Float32(0.35) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.0)))
        var g = Float32(0.40) + Float32(0.30) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.15)))
        var b = Float32(0.55) + Float32(0.35) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.35)))

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
        output[pixel_idx] = UInt8(r * 255)
        output[pixel_idx + 1] = UInt8(g * 255)
        output[pixel_idx + 2] = UInt8(b * 255)


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
        # Take absolute values before squaring
        var abs_re = z_re if z_re >= 0 else -z_re
        var abs_im = z_im if z_im >= 0 else -z_im
        var new_re = abs_re * abs_re - abs_im * abs_im + c_re
        z_im = 2.0 * abs_re * abs_im + c_im
        z_re = new_re

    # Color based on iteration count
    var pixel_idx = (py * width + px) * 3
    if iterations == imax:
        output[pixel_idx] = 0
        output[pixel_idx + 1] = 0
        output[pixel_idx + 2] = 0
    else:
        # Smooth iteration count
        var log_zn = Float32(0.5) * gpu_log2(Float32(final_r2))
        var smooth_iter = Float32(iterations) + Float32(1.0) - gpu_log2(log_zn)

        var t = smooth_iter * Float32(0.05) + Float32(color_seed)
        t = t - Float32(Int(t))

        # Elegant muted palette
        var pi2 = Float32(6.28318530)
        var r = Float32(0.45) + Float32(0.35) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.0)))
        var g = Float32(0.40) + Float32(0.30) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.15)))
        var b = Float32(0.55) + Float32(0.35) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.35)))

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
        # Use conjugate: conj(z)² = (z_re - i*z_im)² = z_re² - z_im² - 2*z_re*z_im*i
        var new_re = z_re2 - z_im2 + c_re
        z_im = -2.0 * z_re * z_im + c_im  # Note the negative sign
        z_re = new_re

    # Color based on iteration count
    var pixel_idx = (py * width + px) * 3
    if iterations == imax:
        output[pixel_idx] = 0
        output[pixel_idx + 1] = 0
        output[pixel_idx + 2] = 0
    else:
        # Smooth iteration count
        var log_zn = Float32(0.5) * gpu_log2(Float32(final_r2))
        var smooth_iter = Float32(iterations) + Float32(1.0) - gpu_log2(log_zn)

        var t = smooth_iter * Float32(0.05) + Float32(color_seed)
        t = t - Float32(Int(t))

        # Elegant muted palette
        var pi2 = Float32(6.28318530)
        var r = Float32(0.45) + Float32(0.35) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.0)))
        var g = Float32(0.40) + Float32(0.30) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.15)))
        var b = Float32(0.55) + Float32(0.35) * gpu_cos(pi2 * (t * Float32(0.8) + Float32(0.35)))

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
# Mandelbulb 3D fractal kernel (ray marching)
# ============================================================================

fn mandelbulb_kernel[
    output_layout: Layout,
](
    output: LayoutTensor[DType.uint8, output_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    cam_x: Float32,
    cam_y: Float32,
    cam_z: Float32,
    cam_yaw: Float32,
    cam_pitch: Float32,
    power: Float32,
    imax: Int,
    color_seed: Float32,
):
    """Ray march to render Mandelbulb fractal.

    Camera uses FPS-style controls:
    - cam_x/y/z: Camera position
    - cam_yaw: Left/right rotation (radians)
    - cam_pitch: Up/down rotation (radians, clamped to ±π/2)
    """
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)

    if px >= width or py >= height:
        return

    # Compute ray direction from pixel coordinates and camera orientation
    var aspect = Float32(width) / Float32(height)
    var fov = Float32(1.0)  # Field of view factor

    # Normalized device coordinates (-1 to 1)
    var ndc_x = (Float32(2.0) * Float32(px) / Float32(width) - Float32(1.0)) * aspect * fov
    var ndc_y = (Float32(1.0) - Float32(2.0) * Float32(py) / Float32(height)) * fov

    # Build camera basis vectors from yaw and pitch
    var cos_yaw = gpu_cos(cam_yaw)
    var sin_yaw = gpu_sin(cam_yaw)
    var cos_pitch = gpu_cos(cam_pitch)
    var sin_pitch = gpu_sin(cam_pitch)

    # Forward direction (where camera is looking)
    var fwd_x = cos_pitch * sin_yaw
    var fwd_y = sin_pitch
    var fwd_z = cos_pitch * cos_yaw

    # Right direction (perpendicular to forward, in XZ plane)
    var right_x = cos_yaw
    var right_y = Float32(0.0)
    var right_z = -sin_yaw

    # Up direction (cross product of right and forward)
    var up_x = -sin_pitch * sin_yaw
    var up_y = cos_pitch
    var up_z = -sin_pitch * cos_yaw

    # Ray direction in world space
    var ray_dx = right_x * ndc_x + up_x * ndc_y + fwd_x
    var ray_dy = right_y * ndc_x + up_y * ndc_y + fwd_y
    var ray_dz = right_z * ndc_x + up_z * ndc_y + fwd_z

    # Normalize ray direction
    var ray_len = gpu_sqrt(ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz)
    ray_dx = ray_dx / ray_len
    ray_dy = ray_dy / ray_len
    ray_dz = ray_dz / ray_len

    # Ray marching
    var total_dist = Float32(0.0)
    var hit = False
    var steps = 0

    var pos_x = cam_x
    var pos_y = cam_y
    var pos_z = cam_z

    for step in range(MAX_STEPS):
        var dist = mandelbulb_de(pos_x, pos_y, pos_z, power, imax)

        if dist < EPSILON:
            hit = True
            steps = step
            break

        total_dist = total_dist + dist

        if total_dist > MAX_DIST:
            break

        pos_x = pos_x + ray_dx * dist
        pos_y = pos_y + ray_dy * dist
        pos_z = pos_z + ray_dz * dist
        steps = step

    var pixel_idx = (py * width + px) * 3

    if hit:
        # Compute surface normal via gradient (central differences)
        var nx = mandelbulb_de(pos_x + NORMAL_EPSILON, pos_y, pos_z, power, imax) - mandelbulb_de(pos_x - NORMAL_EPSILON, pos_y, pos_z, power, imax)
        var ny = mandelbulb_de(pos_x, pos_y + NORMAL_EPSILON, pos_z, power, imax) - mandelbulb_de(pos_x, pos_y - NORMAL_EPSILON, pos_z, power, imax)
        var nz = mandelbulb_de(pos_x, pos_y, pos_z + NORMAL_EPSILON, power, imax) - mandelbulb_de(pos_x, pos_y, pos_z - NORMAL_EPSILON, power, imax)

        # Normalize normal
        var n_len = gpu_sqrt(nx * nx + ny * ny + nz * nz)
        if n_len > Float32(1e-10):
            nx = nx / n_len
            ny = ny / n_len
            nz = nz / n_len

        # Light direction (fixed, from upper-right-front)
        var light_x = Float32(0.577)
        var light_y = Float32(0.577)
        var light_z = Float32(-0.577)

        # Diffuse lighting
        var diffuse = nx * light_x + ny * light_y + nz * light_z
        if diffuse < Float32(0.0):
            diffuse = Float32(0.0)

        # Specular lighting (Blinn-Phong)
        # Half vector between light and view direction
        var view_x = -ray_dx
        var view_y = -ray_dy
        var view_z = -ray_dz
        var half_x = light_x + view_x
        var half_y = light_y + view_y
        var half_z = light_z + view_z
        var half_len = gpu_sqrt(half_x * half_x + half_y * half_y + half_z * half_z)
        half_x = half_x / half_len
        half_y = half_y / half_len
        half_z = half_z / half_len
        var spec_dot = nx * half_x + ny * half_y + nz * half_z
        if spec_dot < Float32(0.0):
            spec_dot = Float32(0.0)
        var specular = gpu_pow(spec_dot, Float32(32.0))

        # Ambient occlusion approximation based on step count
        var ao = Float32(1.0) - Float32(steps) / Float32(MAX_STEPS) * Float32(0.5)

        # Color palette based on surface normal direction
        # Use spherical coords of normal for a gradient between complementary colors
        var n_theta = gpu_acos(ny)  # 0 to pi (top to bottom)
        var n_phi = gpu_atan2(nz, nx)  # -pi to pi (around)

        # Create a triadic color scheme using normal direction
        # Base hue from phi (horizontal angle) + color_seed
        var hue1 = (n_phi / Float32(6.28318530) + Float32(0.5) + color_seed)
        hue1 = hue1 - Float32(Int(hue1))

        # Blend factor from theta (vertical angle) - creates gradient from warm to cool
        var blend = n_theta / Float32(3.14159265)  # 0 at top, 1 at bottom

        # Two complementary base colors
        # Warm color (oranges/magentas)
        var warm_r = Float32(0.95)
        var warm_g = Float32(0.4) + hue1 * Float32(0.3)
        var warm_b = Float32(0.5) + hue1 * Float32(0.4)

        # Cool color (teals/blues)
        var cool_r = Float32(0.3) + hue1 * Float32(0.2)
        var cool_g = Float32(0.6) + hue1 * Float32(0.3)
        var cool_b = Float32(0.9)

        # Blend between warm and cool based on normal direction
        var base_r = warm_r * (Float32(1.0) - blend) + cool_r * blend
        var base_g = warm_g * (Float32(1.0) - blend) + cool_g * blend
        var base_b = warm_b * (Float32(1.0) - blend) + cool_b * blend

        # Combine lighting
        var ambient = Float32(0.2)
        var intensity = ambient + diffuse * Float32(0.65) + specular * Float32(0.4)
        intensity = intensity * ao

        var r_out = base_r * intensity
        var g_out = base_g * intensity
        var b_out = base_b * intensity

        # Clamp
        if r_out > Float32(1.0):
            r_out = Float32(1.0)
        if g_out > Float32(1.0):
            g_out = Float32(1.0)
        if b_out > Float32(1.0):
            b_out = Float32(1.0)

        output[pixel_idx] = UInt8(r_out * Float32(255.0))
        output[pixel_idx + 1] = UInt8(g_out * Float32(255.0))
        output[pixel_idx + 2] = UInt8(b_out * Float32(255.0))
    else:
        # Background gradient
        var t = Float32(py) / Float32(height)
        var bg_r = UInt8(Float32(10.0) + t * Float32(20.0))
        var bg_g = UInt8(Float32(10.0) + t * Float32(25.0))
        var bg_b = UInt8(Float32(20.0) + t * Float32(40.0))
        output[pixel_idx] = bg_r
        output[pixel_idx + 1] = bg_g
        output[pixel_idx + 2] = bg_b


# ============================================================================
# Python-exposed functions (each arg is a separate PythonObject)
# ============================================================================

@export
fn render_fractal(
    py_args: PythonObject,
) raises -> PythonObject:
    """Render Newton fractal to numpy array.

    Args is a tuple: (width, height, coeffs, left, right, top, bottom, tolerance, imax, color_seed, glow_intensity, zoom)
    """
    var width = Int(py=py_args[0])
    var height = Int(py=py_args[1])
    var py_coeffs = py_args[2]
    var left = Float64(py=py_args[3])
    var right = Float64(py=py_args[4])
    var top = Float64(py=py_args[5])
    var bottom = Float64(py=py_args[6])
    var tolerance = Float64(py=py_args[7])
    var imax = Int(py=py_args[8])
    var color_seed = Float64(py=py_args[9])
    var glow_intensity = Float64(py=py_args[10])
    var zoom = Float64(py=py_args[11])

    var num_coeffs = len(py_coeffs)

    if not has_accelerator():
        raise Error("No GPU found")

    # Create GPU context and buffers
    var ctx = DeviceContext()

    var newton_size = width * height * 3
    var rgb_size = width * height * 3
    var max_roots = 16

    var max_coeffs = 16
    var coeffs_host = ctx.enqueue_create_host_buffer[DType.float64](max_coeffs)
    var coeffs_device = ctx.enqueue_create_buffer[DType.float64](max_coeffs)
    var newton_device = ctx.enqueue_create_buffer[DType.float64](newton_size)
    var newton_host = ctx.enqueue_create_host_buffer[DType.float64](newton_size)
    var roots_host = ctx.enqueue_create_host_buffer[DType.float64](max_roots * 5)
    var roots_device = ctx.enqueue_create_buffer[DType.float64](max_roots * 5)
    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    # Upload coefficients
    for i in range(num_coeffs):
        coeffs_host[i] = Float64(py=py_coeffs[i])
    ctx.enqueue_copy(coeffs_device, coeffs_host)
    ctx.synchronize()

    # Layouts (use 16 for max polynomial degree)
    comptime coeffs_layout = Layout.row_major(16)
    comptime newton_layout = Layout.row_major(1920 * 1080 * 3)
    comptime roots_layout = Layout.row_major(16 * 5)
    comptime rgb_layout = Layout.row_major(1920 * 1080 * 3)

    var coeffs_tensor = LayoutTensor[DType.float64, coeffs_layout](coeffs_device)
    var newton_tensor = LayoutTensor[DType.float64, newton_layout](newton_device)

    comptime block_size = 16
    var grid_x = ceildiv(width, block_size)
    var grid_y = ceildiv(height, block_size)

    # Newton kernel
    ctx.enqueue_function[
        newton_kernel[coeffs_layout, newton_layout],
        newton_kernel[coeffs_layout, newton_layout],
    ](
        coeffs_tensor,
        num_coeffs,
        newton_tensor,
        width,
        height,
        left,
        right,
        top,
        bottom,
        tolerance,
        imax,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    # Discover roots
    ctx.enqueue_copy(newton_host, newton_device)
    ctx.synchronize()

    var num_roots = 0
    var step_x = width // 32
    var step_y = height // 32
    if step_x < 1:
        step_x = 1
    if step_y < 1:
        step_y = 1

    # First pass: discover roots (just coordinates)
    for py in range(0, height, step_y):
        for px in range(0, width, step_x):
            var idx = (py * width + px) * 3
            var re = Float64(newton_host[idx])
            var im = Float64(newton_host[idx + 1])
            var iterations = Float64(newton_host[idx + 2])

            if iterations >= 0 and iterations < Float64(imax):
                var found = False
                var tol_sq = tolerance * tolerance * 4
                for i in range(num_roots):
                    var root_re = Float64(roots_host[i * 5])
                    var root_im = Float64(roots_host[i * 5 + 1])
                    var diff_re = root_re - re
                    var diff_im = root_im - im
                    var dist_sq = diff_re * diff_re + diff_im * diff_im
                    if dist_sq < tol_sq:
                        found = True
                        break
                if not found and num_roots < 16:
                    roots_host[num_roots * 5] = re
                    roots_host[num_roots * 5 + 1] = im
                    num_roots += 1

    # Sort roots by angle (atan2) for consistent coloring across zoom levels
    # Simple bubble sort - only up to 16 roots
    for i in range(num_roots):
        for j in range(i + 1, num_roots):
            var re_i = Float64(roots_host[i * 5])
            var im_i = Float64(roots_host[i * 5 + 1])
            var re_j = Float64(roots_host[j * 5])
            var im_j = Float64(roots_host[j * 5 + 1])
            var angle_i = atan2(im_i, re_i)
            var angle_j = atan2(im_j, re_j)
            if angle_j < angle_i:
                # Swap
                roots_host[i * 5] = re_j
                roots_host[i * 5 + 1] = im_j
                roots_host[j * 5] = re_i
                roots_host[j * 5 + 1] = im_i

    # Assign colors based on sorted order
    for i in range(num_roots):
        var palette_idx = (i + Int(color_seed * 8)) % 8

        var r_col: Float64
        var g_col: Float64
        var b_col: Float64

        if palette_idx == 0:
            # Soft coral
            r_col = 210.0; g_col = 120.0; b_col = 120.0
        elif palette_idx == 1:
            # Dusty teal
            r_col = 95.0; g_col = 158.0; b_col = 160.0
        elif palette_idx == 2:
            # Muted gold
            r_col = 190.0; g_col = 165.0; b_col = 100.0
        elif palette_idx == 3:
            # Slate blue
            r_col = 110.0; g_col = 130.0; b_col = 180.0
        elif palette_idx == 4:
            # Sage green
            r_col = 130.0; g_col = 170.0; b_col = 130.0
        elif palette_idx == 5:
            # Dusty rose
            r_col = 180.0; g_col = 130.0; b_col = 155.0
        elif palette_idx == 6:
            # Warm gray
            r_col = 160.0; g_col = 150.0; b_col = 140.0
        else:
            # Soft lavender
            r_col = 150.0; g_col = 140.0; b_col = 180.0

        roots_host[i * 5 + 2] = r_col
        roots_host[i * 5 + 3] = g_col
        roots_host[i * 5 + 4] = b_col

    ctx.enqueue_copy(roots_device, roots_host)
    ctx.synchronize()

    # Colorize kernel
    var roots_tensor = LayoutTensor[DType.float64, roots_layout](roots_device)
    var rgb_tensor = LayoutTensor[DType.uint8, rgb_layout](rgb_device)

    ctx.enqueue_function[
        colorize_kernel[newton_layout, roots_layout, rgb_layout],
        colorize_kernel[newton_layout, roots_layout, rgb_layout],
    ](
        newton_tensor,
        roots_tensor,
        num_roots,
        rgb_tensor,
        width,
        height,
        tolerance,
        imax,
        glow_intensity,
        zoom,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    # Convert to numpy array
    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")

    var ptr = rgb_host.unsafe_ptr()
    var ptr_int = Int(ptr)

    var c_ptr = ctypes.cast(ptr_int, ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))

    return arr.copy()


@export
fn render_mandelbrot(
    py_args: PythonObject,
) raises -> PythonObject:
    """Render Mandelbrot set to numpy array.

    Args is a tuple: (width, height, left, right, top, bottom, imax, color_seed)
    """
    var width = Int(py=py_args[0])
    var height = Int(py=py_args[1])
    var left = Float64(py=py_args[2])
    var right = Float64(py=py_args[3])
    var top = Float64(py=py_args[4])
    var bottom = Float64(py=py_args[5])
    var imax = Int(py=py_args[6])
    var color_seed = Float64(py=py_args[7])

    if not has_accelerator():
        raise Error("No GPU found")

    var ctx = DeviceContext()
    var rgb_size = width * height * 3

    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    comptime rgb_layout = Layout.row_major(1920 * 1080 * 3)
    var rgb_tensor = LayoutTensor[DType.uint8, rgb_layout](rgb_device)

    comptime block_size = 16
    var grid_x = ceildiv(width, block_size)
    var grid_y = ceildiv(height, block_size)

    ctx.enqueue_function[
        mandelbrot_kernel[rgb_layout],
        mandelbrot_kernel[rgb_layout],
    ](
        rgb_tensor,
        width,
        height,
        left,
        right,
        top,
        bottom,
        imax,
        color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var ptr_int = Int(ptr)
    var c_ptr = ctypes.cast(ptr_int, ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


@export
fn render_julia(
    py_args: PythonObject,
) raises -> PythonObject:
    """Render Julia set to numpy array.

    Args is a tuple: (width, height, left, right, top, bottom, c_re, c_im, power_re, power_im, imax, color_seed)
    """
    var width = Int(py=py_args[0])
    var height = Int(py=py_args[1])
    var left = Float64(py=py_args[2])
    var right = Float64(py=py_args[3])
    var top = Float64(py=py_args[4])
    var bottom = Float64(py=py_args[5])
    var c_re = Float64(py=py_args[6])
    var c_im = Float64(py=py_args[7])
    var power_re = Float64(py=py_args[8])
    var power_im = Float64(py=py_args[9])
    var imax = Int(py=py_args[10])
    var color_seed = Float64(py=py_args[11])

    if not has_accelerator():
        raise Error("No GPU found")

    var ctx = DeviceContext()
    var rgb_size = width * height * 3

    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    comptime rgb_layout = Layout.row_major(1920 * 1080 * 3)
    var rgb_tensor = LayoutTensor[DType.uint8, rgb_layout](rgb_device)

    comptime block_size = 16
    var grid_x = ceildiv(width, block_size)
    var grid_y = ceildiv(height, block_size)

    ctx.enqueue_function[
        julia_kernel[rgb_layout],
        julia_kernel[rgb_layout],
    ](
        rgb_tensor,
        width,
        height,
        left,
        right,
        top,
        bottom,
        c_re,
        c_im,
        power_re,
        power_im,
        imax,
        color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var ptr_int = Int(ptr)
    var c_ptr = ctypes.cast(ptr_int, ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


@export
fn render_burning_ship(
    py_args: PythonObject,
) raises -> PythonObject:
    """Render Burning Ship fractal to numpy array.

    Args is a tuple: (width, height, left, right, top, bottom, imax, color_seed)
    """
    var width = Int(py=py_args[0])
    var height = Int(py=py_args[1])
    var left = Float64(py=py_args[2])
    var right = Float64(py=py_args[3])
    var top = Float64(py=py_args[4])
    var bottom = Float64(py=py_args[5])
    var imax = Int(py=py_args[6])
    var color_seed = Float64(py=py_args[7])

    if not has_accelerator():
        raise Error("No GPU found")

    var ctx = DeviceContext()
    var rgb_size = width * height * 3

    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    comptime rgb_layout = Layout.row_major(1920 * 1080 * 3)
    var rgb_tensor = LayoutTensor[DType.uint8, rgb_layout](rgb_device)

    comptime block_size = 16
    var grid_x = ceildiv(width, block_size)
    var grid_y = ceildiv(height, block_size)

    ctx.enqueue_function[
        burning_ship_kernel[rgb_layout],
        burning_ship_kernel[rgb_layout],
    ](
        rgb_tensor,
        width,
        height,
        left,
        right,
        top,
        bottom,
        imax,
        color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var ptr_int = Int(ptr)
    var c_ptr = ctypes.cast(ptr_int, ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


@export
fn render_tricorn(
    py_args: PythonObject,
) raises -> PythonObject:
    """Render Tricorn (Mandelbar) fractal to numpy array.

    Args is a tuple: (width, height, left, right, top, bottom, imax, color_seed)
    """
    var width = Int(py=py_args[0])
    var height = Int(py=py_args[1])
    var left = Float64(py=py_args[2])
    var right = Float64(py=py_args[3])
    var top = Float64(py=py_args[4])
    var bottom = Float64(py=py_args[5])
    var imax = Int(py=py_args[6])
    var color_seed = Float64(py=py_args[7])

    if not has_accelerator():
        raise Error("No GPU found")

    var ctx = DeviceContext()
    var rgb_size = width * height * 3

    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    comptime rgb_layout = Layout.row_major(1920 * 1080 * 3)
    var rgb_tensor = LayoutTensor[DType.uint8, rgb_layout](rgb_device)

    comptime block_size = 16
    var grid_x = ceildiv(width, block_size)
    var grid_y = ceildiv(height, block_size)

    ctx.enqueue_function[
        tricorn_kernel[rgb_layout],
        tricorn_kernel[rgb_layout],
    ](
        rgb_tensor,
        width,
        height,
        left,
        right,
        top,
        bottom,
        imax,
        color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var ptr_int = Int(ptr)
    var c_ptr = ctypes.cast(ptr_int, ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


@export
fn render_mandelbulb(
    py_args: PythonObject,
) raises -> PythonObject:
    """Render Mandelbulb 3D fractal to numpy array.

    Args is a tuple: (width, height, cam_x, cam_y, cam_z, cam_yaw, cam_pitch, power, imax, color_seed)
    """
    var width = Int(py=py_args[0])
    var height = Int(py=py_args[1])
    var cam_x = Float32(py=py_args[2])
    var cam_y = Float32(py=py_args[3])
    var cam_z = Float32(py=py_args[4])
    var cam_yaw = Float32(py=py_args[5])
    var cam_pitch = Float32(py=py_args[6])
    var power = Float32(py=py_args[7])
    var imax = Int(py=py_args[8])
    var color_seed = Float32(py=py_args[9])

    if not has_accelerator():
        raise Error("No GPU found")

    var ctx = DeviceContext()
    var rgb_size = width * height * 3

    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    comptime rgb_layout = Layout.row_major(1920 * 1080 * 3)
    var rgb_tensor = LayoutTensor[DType.uint8, rgb_layout](rgb_device)

    comptime block_size = 16
    var grid_x = ceildiv(width, block_size)
    var grid_y = ceildiv(height, block_size)

    ctx.enqueue_function[
        mandelbulb_kernel[rgb_layout],
        mandelbulb_kernel[rgb_layout],
    ](
        rgb_tensor,
        width,
        height,
        cam_x,
        cam_y,
        cam_z,
        cam_yaw,
        cam_pitch,
        power,
        imax,
        color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var ptr_int = Int(ptr)
    var c_ptr = ctypes.cast(ptr_int, ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


@export
fn has_gpu() -> PythonObject:
    """Check if GPU is available."""
    return PythonObject(has_accelerator())


@export
fn get_gpu_name() raises -> PythonObject:
    """Get GPU name."""
    if not has_accelerator():
        return PythonObject("No GPU")
    var ctx = DeviceContext()
    return PythonObject(ctx.name())


# ============================================================================
# Module initialization
# ============================================================================

@export
fn PyInit_newton_renderer() -> PythonObject:
    try:
        var m = PythonModuleBuilder("newton_renderer")
        m.def_function[render_fractal]("render_newton", docstring="Render Newton fractal")
        m.def_function[render_mandelbrot]("render_mandelbrot", docstring="Render Mandelbrot set")
        m.def_function[render_julia]("render_julia", docstring="Render Julia set")
        m.def_function[render_burning_ship]("render_burning_ship", docstring="Render Burning Ship fractal")
        m.def_function[render_tricorn]("render_tricorn", docstring="Render Tricorn fractal")
        m.def_function[render_mandelbulb]("render_mandelbulb", docstring="Render Mandelbulb 3D fractal")
        m.def_function[has_gpu]("has_gpu", docstring="Check GPU availability")
        m.def_function[get_gpu_name]("get_gpu_name", docstring="Get GPU name")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
