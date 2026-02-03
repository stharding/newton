"""Fractal GPU renderer - Python-importable module."""

from os import abort
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from math import atan2, ceildiv
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from sys import has_accelerator

from kernels_newton import newton_kernel, colorize_kernel
from kernels_2d import mandelbrot_kernel, julia_kernel, burning_ship_kernel, tricorn_kernel
from kernels_3d import mandelbulb_kernel


# ============================================================================
# Constants
# ============================================================================

comptime L = Layout.row_major(1)
"""Single layout for all buffers - size parameter unused for 1D row-major."""

comptime BLOCK_SIZE: Int = 16
"""GPU thread block size for 2D kernels."""

comptime MAX_ROOTS: Int = 16
"""Maximum number of roots to track for Newton fractal."""

comptime MAX_COEFFS: Int = 16
"""Maximum polynomial coefficients for Newton fractal."""


# ============================================================================
# Helper: Convert host buffer to numpy array
# ============================================================================

fn host_buffer_to_numpy(
    rgb_host: HostBuffer[DType.uint8], width: Int, height: Int
) raises -> PythonObject:
    """Convert a host RGB buffer to a numpy array."""
    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var ptr = rgb_host.unsafe_ptr()
    var c_ptr = ctypes.cast(Int(ptr), ctypes.POINTER(ctypes.c_uint8))
    var arr = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(height, width, 3))
    return arr.copy()


# ============================================================================
# Helper: Compute grid dimensions
# ============================================================================

@always_inline
fn compute_grid(width: Int, height: Int) -> Tuple[Int, Int]:
    """Compute grid dimensions for GPU kernel launch."""
    return Tuple(ceildiv(width, BLOCK_SIZE), ceildiv(height, BLOCK_SIZE))


# ============================================================================
# Newton fractal renderer (special case - two-pass with root discovery)
# ============================================================================

@export
fn render_newton(py_args: PythonObject) raises -> PythonObject:
    """Render Newton fractal to numpy array.

    Args tuple: (width, height, coeffs, left, right, top, bottom, tolerance, imax, color_seed, glow_intensity, zoom)
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
    var ctx = DeviceContext()
    var rgb_size = width * height * 3
    var newton_size = width * height * 3

    # Create buffers
    var coeffs_host = ctx.enqueue_create_host_buffer[DType.float64](MAX_COEFFS)
    var coeffs_device = ctx.enqueue_create_buffer[DType.float64](MAX_COEFFS)
    var newton_device = ctx.enqueue_create_buffer[DType.float64](newton_size)
    var newton_host = ctx.enqueue_create_host_buffer[DType.float64](newton_size)
    var roots_host = ctx.enqueue_create_host_buffer[DType.float64](MAX_ROOTS * 5)
    var roots_device = ctx.enqueue_create_buffer[DType.float64](MAX_ROOTS * 5)
    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    # Copy coefficients to GPU
    for i in range(num_coeffs):
        coeffs_host[i] = Float64(py=py_coeffs[i])
    ctx.enqueue_copy(coeffs_device, coeffs_host)
    ctx.synchronize()

    var coeffs_tensor = LayoutTensor[DType.float64, L](coeffs_device)
    var newton_tensor = LayoutTensor[DType.float64, L](newton_device)

    var grid_x: Int
    var grid_y: Int
    grid_x, grid_y = compute_grid(width, height)

    # Pass 1: Newton iteration
    ctx.enqueue_function[
        newton_kernel[L, L],
        newton_kernel[L, L],
    ](
        coeffs_tensor, num_coeffs, newton_tensor,
        width, height, left, right, top, bottom, tolerance, imax,
        grid_dim=(grid_x, grid_y),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    ctx.enqueue_copy(newton_host, newton_device)
    ctx.synchronize()

    # Discover and sort roots (CPU)
    var num_roots = _discover_roots(newton_host, roots_host, width, height, imax, tolerance)
    _sort_roots_by_angle(roots_host, num_roots)
    _assign_root_colors(roots_host, num_roots, color_seed)

    ctx.enqueue_copy(roots_device, roots_host)
    ctx.synchronize()

    # Pass 2: Colorization
    var roots_tensor = LayoutTensor[DType.float64, L](roots_device)
    var rgb_tensor = LayoutTensor[DType.uint8, L](rgb_device)

    ctx.enqueue_function[
        colorize_kernel[L, L, L],
        colorize_kernel[L, L, L],
    ](
        newton_tensor, roots_tensor, num_roots, rgb_tensor,
        width, height, tolerance, imax, glow_intensity, zoom,
        grid_dim=(grid_x, grid_y),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    return host_buffer_to_numpy(rgb_host, width, height)


fn _discover_roots(
    newton_host: HostBuffer[DType.float64],
    roots_host: HostBuffer[DType.float64],
    width: Int, height: Int, imax: Int, tolerance: Float64,
) -> Int:
    """Discover unique roots from Newton iteration output."""
    var num_roots = 0
    var step_x = width // 32
    var step_y = height // 32
    if step_x < 1:
        step_x = 1
    if step_y < 1:
        step_y = 1

    var tol_sq = tolerance * tolerance * 4

    for py in range(0, height, step_y):
        for px in range(0, width, step_x):
            var idx = (py * width + px) * 3
            var re = Float64(newton_host[idx])
            var im = Float64(newton_host[idx + 1])
            var iterations = Float64(newton_host[idx + 2])

            if iterations >= 0 and iterations < Float64(imax):
                var found = False
                for i in range(num_roots):
                    var root_re = Float64(roots_host[i * 5])
                    var root_im = Float64(roots_host[i * 5 + 1])
                    var diff_re = root_re - re
                    var diff_im = root_im - im
                    var dist_sq = diff_re * diff_re + diff_im * diff_im
                    if dist_sq < tol_sq:
                        found = True
                        break
                if not found and num_roots < MAX_ROOTS:
                    roots_host[num_roots * 5] = re
                    roots_host[num_roots * 5 + 1] = im
                    num_roots += 1

    return num_roots


fn _sort_roots_by_angle(roots_host: HostBuffer[DType.float64], num_roots: Int):
    """Sort roots by angle for consistent coloring."""
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


fn _get_palette_color(idx: Int) -> Tuple[Float64, Float64, Float64]:
    """Get RGB color from palette by index (mod 8)."""
    var i = idx % 8
    if i == 0:
        return Tuple(Float64(210.0), Float64(120.0), Float64(120.0))  # Rose
    elif i == 1:
        return Tuple(Float64(95.0), Float64(158.0), Float64(160.0))   # Teal
    elif i == 2:
        return Tuple(Float64(190.0), Float64(165.0), Float64(100.0))  # Gold
    elif i == 3:
        return Tuple(Float64(110.0), Float64(130.0), Float64(180.0))  # Slate blue
    elif i == 4:
        return Tuple(Float64(130.0), Float64(170.0), Float64(130.0))  # Sage
    elif i == 5:
        return Tuple(Float64(180.0), Float64(130.0), Float64(155.0))  # Mauve
    elif i == 6:
        return Tuple(Float64(160.0), Float64(150.0), Float64(140.0))  # Stone
    else:
        return Tuple(Float64(150.0), Float64(140.0), Float64(180.0))  # Lavender


fn _assign_root_colors(roots_host: HostBuffer[DType.float64], num_roots: Int, color_seed: Float64):
    """Assign colors to each root based on palette index."""
    for i in range(num_roots):
        var palette_idx = i + Int(color_seed * 8)
        var color = _get_palette_color(palette_idx)
        roots_host[i * 5 + 2] = color[0]
        roots_host[i * 5 + 3] = color[1]
        roots_host[i * 5 + 4] = color[2]


# ============================================================================
# 2D Fractal renderers (single-pass)
# ============================================================================

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

    var ctx = DeviceContext()
    var rgb_size = width * height * 3
    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    var rgb_tensor = LayoutTensor[DType.uint8, L](rgb_device)
    var grid_x: Int
    var grid_y: Int
    grid_x, grid_y = compute_grid(width, height)

    ctx.enqueue_function[
        mandelbrot_kernel[L],
        mandelbrot_kernel[L],
    ](
        rgb_tensor, width, height, left, right, top, bottom, imax, color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    return host_buffer_to_numpy(rgb_host, width, height)


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

    var ctx = DeviceContext()
    var rgb_size = width * height * 3
    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    var rgb_tensor = LayoutTensor[DType.uint8, L](rgb_device)
    var grid_x: Int
    var grid_y: Int
    grid_x, grid_y = compute_grid(width, height)

    ctx.enqueue_function[
        julia_kernel[L],
        julia_kernel[L],
    ](
        rgb_tensor, width, height, left, right, top, bottom,
        c_re, c_im, power_re, power_im, imax, color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    return host_buffer_to_numpy(rgb_host, width, height)


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

    var ctx = DeviceContext()
    var rgb_size = width * height * 3
    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    var rgb_tensor = LayoutTensor[DType.uint8, L](rgb_device)
    var grid_x: Int
    var grid_y: Int
    grid_x, grid_y = compute_grid(width, height)

    ctx.enqueue_function[
        burning_ship_kernel[L],
        burning_ship_kernel[L],
    ](
        rgb_tensor, width, height, left, right, top, bottom, imax, color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    return host_buffer_to_numpy(rgb_host, width, height)


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

    var ctx = DeviceContext()
    var rgb_size = width * height * 3
    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    var rgb_tensor = LayoutTensor[DType.uint8, L](rgb_device)
    var grid_x: Int
    var grid_y: Int
    grid_x, grid_y = compute_grid(width, height)

    ctx.enqueue_function[
        tricorn_kernel[L],
        tricorn_kernel[L],
    ](
        rgb_tensor, width, height, left, right, top, bottom, imax, color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    return host_buffer_to_numpy(rgb_host, width, height)


# ============================================================================
# 3D Fractal renderer
# ============================================================================

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

    var ctx = DeviceContext()
    var rgb_size = width * height * 3
    var rgb_device = ctx.enqueue_create_buffer[DType.uint8](rgb_size)
    var rgb_host = ctx.enqueue_create_host_buffer[DType.uint8](rgb_size)
    ctx.synchronize()

    var rgb_tensor = LayoutTensor[DType.uint8, L](rgb_device)
    var grid_x: Int
    var grid_y: Int
    grid_x, grid_y = compute_grid(width, height)

    ctx.enqueue_function[
        mandelbulb_kernel[L],
        mandelbulb_kernel[L],
    ](
        rgb_tensor, width, height,
        cam_x, cam_y, cam_z, cam_yaw, cam_pitch,
        power, imax, color_seed,
        grid_dim=(grid_x, grid_y),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )

    ctx.enqueue_copy(rgb_host, rgb_device)
    ctx.synchronize()

    return host_buffer_to_numpy(rgb_host, width, height)


# ============================================================================
# GPU info functions
# ============================================================================

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
        m.def_function[render_newton]("render_newton", docstring="Render Newton fractal")
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
