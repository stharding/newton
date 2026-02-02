"""Fractal GPU renderer - Python-importable module."""

from os import abort
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from random import random_ui64
from math import ceildiv, atan2
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from sys import has_accelerator

from kernels_newton import newton_kernel, colorize_kernel
from kernels_2d import mandelbrot_kernel, julia_kernel, burning_ship_kernel, tricorn_kernel
from kernels_3d import mandelbulb_kernel


# ============================================================================
# Python-exposed render functions
# ============================================================================

@export
fn render_fractal(py_args: PythonObject) raises -> PythonObject:
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

    for i in range(num_coeffs):
        coeffs_host[i] = Float64(py=py_coeffs[i])
    ctx.enqueue_copy(coeffs_device, coeffs_host)
    ctx.synchronize()

    comptime coeffs_layout = Layout.row_major(16)
    comptime newton_layout = Layout.row_major(1920 * 1080 * 3)
    comptime roots_layout = Layout.row_major(16 * 5)
    comptime rgb_layout = Layout.row_major(1920 * 1080 * 3)

    var coeffs_tensor = LayoutTensor[DType.float64, coeffs_layout](coeffs_device)
    var newton_tensor = LayoutTensor[DType.float64, newton_layout](newton_device)

    comptime block_size = 16
    var grid_x = ceildiv(width, block_size)
    var grid_y = ceildiv(height, block_size)

    ctx.enqueue_function[
        newton_kernel[coeffs_layout, newton_layout],
        newton_kernel[coeffs_layout, newton_layout],
    ](
        coeffs_tensor, num_coeffs, newton_tensor,
        width, height, left, right, top, bottom, tolerance, imax,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(newton_host, newton_device)
    ctx.synchronize()

    # Discover roots
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
                    num_roots += 1

    # Sort roots by angle for consistent coloring
    for i in range(num_roots):
        for j in range(i + 1, num_roots):
            var re_i = Float64(roots_host[i * 5])
            var im_i = Float64(roots_host[i * 5 + 1])
            var re_j = Float64(roots_host[j * 5])
            var im_j = Float64(roots_host[j * 5 + 1])
            var angle_i = atan2(im_i, re_i)
            var angle_j = atan2(im_j, re_j)
            if angle_j < angle_i:
                roots_host[i * 5] = re_j
                roots_host[i * 5 + 1] = im_j
                roots_host[j * 5] = re_i
                roots_host[j * 5 + 1] = im_i

    # Assign colors
    for i in range(num_roots):
        var palette_idx = (i + Int(color_seed * 8)) % 8
        var r_col: Float64
        var g_col: Float64
        var b_col: Float64

        if palette_idx == 0:
            r_col = 210.0; g_col = 120.0; b_col = 120.0
        elif palette_idx == 1:
            r_col = 95.0; g_col = 158.0; b_col = 160.0
        elif palette_idx == 2:
            r_col = 190.0; g_col = 165.0; b_col = 100.0
        elif palette_idx == 3:
            r_col = 110.0; g_col = 130.0; b_col = 180.0
        elif palette_idx == 4:
            r_col = 130.0; g_col = 170.0; b_col = 130.0
        elif palette_idx == 5:
            r_col = 180.0; g_col = 130.0; b_col = 155.0
        elif palette_idx == 6:
            r_col = 160.0; g_col = 150.0; b_col = 140.0
        else:
            r_col = 150.0; g_col = 140.0; b_col = 180.0

        roots_host[i * 5 + 2] = r_col
        roots_host[i * 5 + 3] = g_col
        roots_host[i * 5 + 4] = b_col

    ctx.enqueue_copy(roots_device, roots_host)
    ctx.synchronize()

    var roots_tensor = LayoutTensor[DType.float64, roots_layout](roots_device)
    var rgb_tensor = LayoutTensor[DType.uint8, rgb_layout](rgb_device)

    ctx.enqueue_function[
        colorize_kernel[newton_layout, roots_layout, rgb_layout],
        colorize_kernel[newton_layout, roots_layout, rgb_layout],
    ](
        newton_tensor, roots_tensor, num_roots, rgb_tensor,
        width, height, tolerance, imax, glow_intensity, zoom,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var c_ptr = ctypes.cast(Int(ptr), ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


@export
fn render_mandelbrot(py_args: PythonObject) raises -> PythonObject:
    """Render Mandelbrot set to numpy array."""
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
        rgb_tensor, width, height, left, right, top, bottom, imax, color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var c_ptr = ctypes.cast(Int(ptr), ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


@export
fn render_julia(py_args: PythonObject) raises -> PythonObject:
    """Render Julia set to numpy array."""
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
        rgb_tensor, width, height, left, right, top, bottom,
        c_re, c_im, power_re, power_im, imax, color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var c_ptr = ctypes.cast(Int(ptr), ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


@export
fn render_burning_ship(py_args: PythonObject) raises -> PythonObject:
    """Render Burning Ship fractal to numpy array."""
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
        rgb_tensor, width, height, left, right, top, bottom, imax, color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var c_ptr = ctypes.cast(Int(ptr), ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


@export
fn render_tricorn(py_args: PythonObject) raises -> PythonObject:
    """Render Tricorn (Mandelbar) fractal to numpy array."""
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
        rgb_tensor, width, height, left, right, top, bottom, imax, color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var c_ptr = ctypes.cast(Int(ptr), ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


@export
fn render_mandelbulb(py_args: PythonObject) raises -> PythonObject:
    """Render Mandelbulb 3D fractal to numpy array."""
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
        rgb_tensor, width, height,
        cam_x, cam_y, cam_z, cam_yaw, cam_pitch,
        power, imax, color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var c_ptr = ctypes.cast(Int(ptr), ctypes.POINTER(ctypes.c_uint8))
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
fn PyInit_renderer() -> PythonObject:
    try:
        var m = PythonModuleBuilder("renderer")
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
