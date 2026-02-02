"""Newton fractal GPU renderer - Python-importable module."""

from os import abort
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from random import random_ui64
from math import ceildiv, atan2
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu import global_idx
from layout import Layout, LayoutTensor
from sys import has_accelerator, is_nvidia_gpu
from sys._assembly import inlined_assembly
from memory import UnsafePointer
from memory.unsafe_pointer import alloc


# ============================================================================
# GPU Math Helper Functions (using PTX intrinsics for NVIDIA)
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
            iterations = Float64(count)
            break

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
    elif iterations < Float64(imax):
        var root_idx = -1
        var tol_sq = tolerance * tolerance * 4

        for i in range(num_roots):
            var root_re = rebind[Float64](roots[i * 5])
            var root_im = rebind[Float64](roots[i * 5 + 1])
            var diff_re = root_re - re
            var diff_im = root_im - im
            var dist_sq = diff_re * diff_re + diff_im * diff_im
            if dist_sq < tol_sq:
                root_idx = i
                break

        if root_idx >= 0:
            var base_r = rebind[Float64](roots[root_idx * 5 + 2])
            var base_g = rebind[Float64](roots[root_idx * 5 + 3])
            var base_b = rebind[Float64](roots[root_idx * 5 + 4])
            var scaled = (iterations / Float64(imax)) * 105.0
            r = UInt8(base_r + scaled)
            g = UInt8(base_g + scaled)
            b = UInt8(base_b + scaled)

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

    for count in range(imax):
        var z_re2 = z_re * z_re
        var z_im2 = z_im * z_im
        if z_re2 + z_im2 > 4.0:
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
        # Smooth coloring using hue
        var t = Float64(iterations) / Float64(imax)
        var hue = t + color_seed
        hue = hue - Float64(Int(hue))
        # HSV to RGB with S=1, V=1
        var h6 = hue * 6.0
        var sector = Int(h6) % 6
        var f = h6 - Float64(Int(h6))
        var q = 1.0 - f
        var r: Float64
        var g: Float64
        var b: Float64
        if sector == 0:
            r = 1.0; g = f; b = 0.0
        elif sector == 1:
            r = q; g = 1.0; b = 0.0
        elif sector == 2:
            r = 0.0; g = 1.0; b = f
        elif sector == 3:
            r = 0.0; g = q; b = 1.0
        elif sector == 4:
            r = f; g = 0.0; b = 1.0
        else:
            r = 1.0; g = 0.0; b = q
        output[pixel_idx] = UInt8(r * 255)
        output[pixel_idx + 1] = UInt8(g * 255)
        output[pixel_idx + 2] = UInt8(b * 255)


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

    # Fast path for z^2 (most common case)
    var use_simple_square = power_im == 0.0 and power_re == 2.0

    for count in range(imax):
        var r2 = z_re * z_re + z_im * z_im
        if r2 > 4.0:
            iterations = count
            break

        if use_simple_square:
            # Direct z^2 calculation: (a+bi)^2 = a^2 - b^2 + 2abi
            var new_re = z_re * z_re - z_im * z_im + c_re
            var new_im = 2.0 * z_re * z_im + c_im
            z_re = new_re
            z_im = new_im
        else:
            # Complex power: z^(a+bi) where z = r*e^(i*theta)
            # z^(a+bi) = r^a * e^(-b*theta) * e^(i*(a*theta + b*ln(r)))
            # magnitude = r^a * e^(-b*theta)
            # angle = a*theta + b*ln(r)
            var zx = Float32(z_re)
            var zy = Float32(z_im)
            var p_re = Float32(power_re)
            var p_im = Float32(power_im)

            # r = |z| using GPU sqrt
            var r = gpu_sqrt(zx * zx + zy * zy)

            if r < 1e-10:
                z_re = c_re
                z_im = c_im
            else:
                # theta = atan2(y, x)
                var theta = gpu_atan2(zy, zx)
                var ln_r = gpu_log2(r) * Float32(0.693147180559945)  # log2(r) * ln(2) = ln(r)

                # magnitude = r^a * e^(-b*theta) = 2^(a*log2(r)) * 2^(-b*theta/ln(2))
                var log2_mag = p_re * gpu_log2(r) - p_im * theta * Float32(1.4426950408889634)  # 1/ln(2)
                var mag = gpu_exp2(log2_mag)

                # angle = a*theta + b*ln(r)
                var angle = p_re * theta + p_im * ln_r

                z_re = Float64(mag * gpu_cos(angle)) + c_re
                z_im = Float64(mag * gpu_sin(angle)) + c_im

    # Color based on iteration count (same as Mandelbrot)
    var pixel_idx = (py * width + px) * 3
    if iterations == imax:
        output[pixel_idx] = 0
        output[pixel_idx + 1] = 0
        output[pixel_idx + 2] = 0
    else:
        var t = Float64(iterations) / Float64(imax)
        var hue = t + color_seed
        hue = hue - Float64(Int(hue))
        var h6 = hue * 6.0
        var sector = Int(h6) % 6
        var f = h6 - Float64(Int(h6))
        var q = 1.0 - f
        var r: Float64
        var g: Float64
        var b: Float64
        if sector == 0:
            r = 1.0; g = f; b = 0.0
        elif sector == 1:
            r = q; g = 1.0; b = 0.0
        elif sector == 2:
            r = 0.0; g = 1.0; b = f
        elif sector == 3:
            r = 0.0; g = q; b = 1.0
        elif sector == 4:
            r = f; g = 0.0; b = 1.0
        else:
            r = 1.0; g = 0.0; b = q
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

    for count in range(imax):
        var z_re2 = z_re * z_re
        var z_im2 = z_im * z_im
        if z_re2 + z_im2 > 4.0:
            iterations = count
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
        var t = Float64(iterations) / Float64(imax)
        var hue = t + color_seed
        hue = hue - Float64(Int(hue))
        var h6 = hue * 6.0
        var sector = Int(h6) % 6
        var f = h6 - Float64(Int(h6))
        var q = 1.0 - f
        var r: Float64
        var g: Float64
        var b: Float64
        if sector == 0:
            r = 1.0; g = f; b = 0.0
        elif sector == 1:
            r = q; g = 1.0; b = 0.0
        elif sector == 2:
            r = 0.0; g = 1.0; b = f
        elif sector == 3:
            r = 0.0; g = q; b = 1.0
        elif sector == 4:
            r = f; g = 0.0; b = 1.0
        else:
            r = 1.0; g = 0.0; b = q
        output[pixel_idx] = UInt8(r * 255)
        output[pixel_idx + 1] = UInt8(g * 255)
        output[pixel_idx + 2] = UInt8(b * 255)


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

    for count in range(imax):
        var z_re2 = z_re * z_re
        var z_im2 = z_im * z_im
        if z_re2 + z_im2 > 4.0:
            iterations = count
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
        var t = Float64(iterations) / Float64(imax)
        var hue = t + color_seed
        hue = hue - Float64(Int(hue))
        var h6 = hue * 6.0
        var sector = Int(h6) % 6
        var f = h6 - Float64(Int(h6))
        var q = 1.0 - f
        var r: Float64
        var g: Float64
        var b: Float64
        if sector == 0:
            r = 1.0; g = f; b = 0.0
        elif sector == 1:
            r = q; g = 1.0; b = 0.0
        elif sector == 2:
            r = 0.0; g = 1.0; b = f
        elif sector == 3:
            r = 0.0; g = q; b = 1.0
        elif sector == 4:
            r = f; g = 0.0; b = 1.0
        else:
            r = 1.0; g = 0.0; b = q
        output[pixel_idx] = UInt8(r * 255)
        output[pixel_idx + 1] = UInt8(g * 255)
        output[pixel_idx + 2] = UInt8(b * 255)


# ============================================================================
# Python-exposed functions (each arg is a separate PythonObject)
# ============================================================================

@export
fn render_fractal(
    py_args: PythonObject,
) raises -> PythonObject:
    """Render Newton fractal to numpy array.

    Args is a tuple: (width, height, coeffs, left, right, top, bottom, tolerance, imax, color_seed)
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
                    # Deterministic colors based on root position + seed
                    # Use angle in complex plane for hue
                    var angle = atan2(im, re)  # -pi to pi
                    var hue = (angle + 3.14159265) / 6.28318530 + color_seed  # 0 to 1 + offset
                    hue = hue - Float64(Int(hue))  # wrap to 0-1
                    # Convert hue to RGB (simplified HSV with S=1, V=0.9)
                    var h6 = hue * 6.0
                    var sector = Int(h6) % 6
                    var f = h6 - Float64(Int(h6))
                    var v = 150.0
                    var p = 0.0
                    var q = v * (1.0 - f)
                    var t = v * f
                    var r_col: Float64
                    var g_col: Float64
                    var b_col: Float64
                    if sector == 0:
                        r_col = v; g_col = t; b_col = p
                    elif sector == 1:
                        r_col = q; g_col = v; b_col = p
                    elif sector == 2:
                        r_col = p; g_col = v; b_col = t
                    elif sector == 3:
                        r_col = p; g_col = q; b_col = v
                    elif sector == 4:
                        r_col = t; g_col = p; b_col = v
                    else:
                        r_col = v; g_col = p; b_col = q
                    roots_host[num_roots * 5 + 2] = r_col
                    roots_host[num_roots * 5 + 3] = g_col
                    roots_host[num_roots * 5 + 4] = b_col
                    num_roots += 1

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
        m.def_function[has_gpu]("has_gpu", docstring="Check GPU availability")
        m.def_function[get_gpu_name]("get_gpu_name", docstring="Get GPU name")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
