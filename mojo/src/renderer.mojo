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
# JuliaParams - Struct for Julia fractal parameters
# ============================================================================

@fieldwise_init
struct JuliaParams(TrivialRegisterType, Representable):
    """Parameters for rendering a Julia set fractal.

    All fields are trivial types, so TrivialRegisterType auto-generates
    copy, move, and destructor.
    """
    var width: Int
    var height: Int
    var left: Float64
    var right: Float64
    var top: Float64
    var bottom: Float64
    var c_re: Float64
    var c_im: Float64
    var power_re: Float64
    var power_im: Float64
    var imax: Int
    var color_seed: Float64

    fn __repr__(self) -> String:
        """String representation for Python."""
        return String("JuliaParams(width=") + String(self.width) + ", height=" + String(self.height) + ", ...)"


fn _init_julia_params(out self: JuliaParams, args: PythonObject, kwargs: PythonObject) raises:
    """Initialize JuliaParams from Python keyword arguments with defaults."""
    # Default values
    var width = 800
    var height = 600
    var left = Float64(-2.0)
    var right = Float64(2.0)
    var top = Float64(-1.5)
    var bottom = Float64(1.5)
    var c_re = Float64(-0.7)
    var c_im = Float64(0.27015)
    var power_re = Float64(2.0)
    var power_im = Float64(0.0)
    var imax = 256
    var color_seed = Float64(0.0)

    # Override with any provided kwargs
    if "width" in kwargs:
        width = Int(py=kwargs["width"])
    if "height" in kwargs:
        height = Int(py=kwargs["height"])
    if "left" in kwargs:
        left = Float64(py=kwargs["left"])
    if "right" in kwargs:
        right = Float64(py=kwargs["right"])
    if "top" in kwargs:
        top = Float64(py=kwargs["top"])
    if "bottom" in kwargs:
        bottom = Float64(py=kwargs["bottom"])
    if "c_re" in kwargs:
        c_re = Float64(py=kwargs["c_re"])
    if "c_im" in kwargs:
        c_im = Float64(py=kwargs["c_im"])
    if "power_re" in kwargs:
        power_re = Float64(py=kwargs["power_re"])
    if "power_im" in kwargs:
        power_im = Float64(py=kwargs["power_im"])
    if "imax" in kwargs:
        imax = Int(py=kwargs["imax"])
    if "color_seed" in kwargs:
        color_seed = Float64(py=kwargs["color_seed"])

    # Construct with fieldwise init
    self = JuliaParams(width, height, left, right, top, bottom,
                       c_re, c_im, power_re, power_im, imax, color_seed)


# ============================================================================
# ViewParams - View window for 2D fractals
# ============================================================================

@fieldwise_init
struct ViewParams(TrivialRegisterType, Representable):
    """View window parameters for 2D fractals (Mandelbrot, BurningShip, Tricorn).

    All fields are trivial types, so TrivialRegisterType auto-generates
    copy, move, and destructor.
    """
    var width: Int
    var height: Int
    var left: Float64
    var right: Float64
    var top: Float64
    var bottom: Float64
    var imax: Int
    var color_seed: Float64

    fn __repr__(self) -> String:
        """String representation for Python."""
        return String("ViewParams(width=") + String(self.width) + ", height=" + String(self.height) + ", ...)"


fn _init_view_params(out self: ViewParams, args: PythonObject, kwargs: PythonObject) raises:
    """Initialize ViewParams from Python keyword arguments with defaults."""
    # Default values
    var width = 800
    var height = 600
    var left = Float64(-2.5)
    var right = Float64(1.0)
    var top = Float64(-1.2)
    var bottom = Float64(1.2)
    var imax = 256
    var color_seed = Float64(0.0)

    # Override with any provided kwargs
    if "width" in kwargs:
        width = Int(py=kwargs["width"])
    if "height" in kwargs:
        height = Int(py=kwargs["height"])
    if "left" in kwargs:
        left = Float64(py=kwargs["left"])
    if "right" in kwargs:
        right = Float64(py=kwargs["right"])
    if "top" in kwargs:
        top = Float64(py=kwargs["top"])
    if "bottom" in kwargs:
        bottom = Float64(py=kwargs["bottom"])
    if "imax" in kwargs:
        imax = Int(py=kwargs["imax"])
    if "color_seed" in kwargs:
        color_seed = Float64(py=kwargs["color_seed"])

    self = ViewParams(width, height, left, right, top, bottom, imax, color_seed)


# ============================================================================
# MandelbulbParams - Struct for 3D Mandelbulb parameters
# ============================================================================

@fieldwise_init
struct MandelbulbParams(TrivialRegisterType, Representable):
    """Parameters for rendering Mandelbulb 3D fractal."""
    var width: Int
    var height: Int
    var cam_x: Float32
    var cam_y: Float32
    var cam_z: Float32
    var cam_yaw: Float32
    var cam_pitch: Float32
    var power: Float32
    var imax: Int
    var color_seed: Float32

    fn __repr__(self) -> String:
        """String representation for Python."""
        return String("MandelbulbParams(width=") + String(self.width) + ", height=" + String(self.height) + ", ...)"


fn _init_mandelbulb_params(out self: MandelbulbParams, args: PythonObject, kwargs: PythonObject) raises:
    """Initialize MandelbulbParams from Python keyword arguments with defaults."""
    # Default values
    var width = 800
    var height = 600
    var cam_x = Float32(0.0)
    var cam_y = Float32(0.0)
    var cam_z = Float32(-3.0)
    var cam_yaw = Float32(0.0)
    var cam_pitch = Float32(0.0)
    var power = Float32(8.0)
    var imax = 256
    var color_seed = Float32(0.0)

    # Override with any provided kwargs
    if "width" in kwargs:
        width = Int(py=kwargs["width"])
    if "height" in kwargs:
        height = Int(py=kwargs["height"])
    if "cam_x" in kwargs:
        cam_x = Float32(py=kwargs["cam_x"])
    if "cam_y" in kwargs:
        cam_y = Float32(py=kwargs["cam_y"])
    if "cam_z" in kwargs:
        cam_z = Float32(py=kwargs["cam_z"])
    if "cam_yaw" in kwargs:
        cam_yaw = Float32(py=kwargs["cam_yaw"])
    if "cam_pitch" in kwargs:
        cam_pitch = Float32(py=kwargs["cam_pitch"])
    if "power" in kwargs:
        power = Float32(py=kwargs["power"])
    if "imax" in kwargs:
        imax = Int(py=kwargs["imax"])
    if "color_seed" in kwargs:
        color_seed = Float32(py=kwargs["color_seed"])

    self = MandelbulbParams(width, height, cam_x, cam_y, cam_z, cam_yaw, cam_pitch, power, imax, color_seed)


# ============================================================================
# NewtonParams - Struct for Newton fractal parameters (coeffs passed separately)
# ============================================================================

@fieldwise_init
struct NewtonParams(TrivialRegisterType, Representable):
    """Parameters for rendering Newton fractal.

    Note: coeffs is NOT included here because it's a Python list (not trivial).
    Pass coeffs as a separate argument to render_newton_v2(params, coeffs).
    """
    var width: Int
    var height: Int
    var left: Float64
    var right: Float64
    var top: Float64
    var bottom: Float64
    var tolerance: Float64
    var imax: Int
    var color_seed: Float64
    var glow_intensity: Float64
    var zoom: Float64

    fn __repr__(self) -> String:
        """String representation for Python."""
        return String("NewtonParams(width=") + String(self.width) + ", height=" + String(self.height) + ", ...)"


fn _init_newton_params(out self: NewtonParams, args: PythonObject, kwargs: PythonObject) raises:
    """Initialize NewtonParams from Python keyword arguments with defaults."""
    # Default values
    var width = 800
    var height = 600
    var left = Float64(-2.0)
    var right = Float64(2.0)
    var top = Float64(-2.0)
    var bottom = Float64(2.0)
    var tolerance = Float64(0.0001)
    var imax = 30
    var color_seed = Float64(0.0)
    var glow_intensity = Float64(0.15)
    var zoom = Float64(1.0)

    # Override with any provided kwargs
    if "width" in kwargs:
        width = Int(py=kwargs["width"])
    if "height" in kwargs:
        height = Int(py=kwargs["height"])
    if "left" in kwargs:
        left = Float64(py=kwargs["left"])
    if "right" in kwargs:
        right = Float64(py=kwargs["right"])
    if "top" in kwargs:
        top = Float64(py=kwargs["top"])
    if "bottom" in kwargs:
        bottom = Float64(py=kwargs["bottom"])
    if "tolerance" in kwargs:
        tolerance = Float64(py=kwargs["tolerance"])
    if "imax" in kwargs:
        imax = Int(py=kwargs["imax"])
    if "color_seed" in kwargs:
        color_seed = Float64(py=kwargs["color_seed"])
    if "glow_intensity" in kwargs:
        glow_intensity = Float64(py=kwargs["glow_intensity"])
    if "zoom" in kwargs:
        zoom = Float64(py=kwargs["zoom"])

    self = NewtonParams(width, height, left, right, top, bottom, tolerance, imax, color_seed, glow_intensity, zoom)


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

fn render_newton(params_obj: PythonObject, coeffs: PythonObject) raises -> PythonObject:
    """Render Newton fractal to numpy array.

    Args:
        params_obj: A NewtonParams instance with all rendering parameters.
        coeffs: Python list of polynomial coefficients (highest degree first).
    """
    var params_ptr = params_obj.downcast_value_ptr[NewtonParams]()
    return _render_newton_impl(
        params_ptr[].width, params_ptr[].height, coeffs,
        params_ptr[].left, params_ptr[].right, params_ptr[].top, params_ptr[].bottom,
        params_ptr[].tolerance, params_ptr[].imax, params_ptr[].color_seed,
        params_ptr[].glow_intensity, params_ptr[].zoom,
    )


fn _render_newton_impl(
    width: Int, height: Int, py_coeffs: PythonObject,
    left: Float64, right: Float64, top: Float64, bottom: Float64,
    tolerance: Float64, imax: Int, color_seed: Float64,
    glow_intensity: Float64, zoom: Float64,
) raises -> PythonObject:
    """Internal implementation shared by both Newton render APIs."""
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

fn render_mandelbrot(params_obj: PythonObject) raises -> PythonObject:
    """Render Mandelbrot set to numpy array."""
    var params_ptr = params_obj.downcast_value_ptr[ViewParams]()
    return _render_mandelbrot_impl(
        params_ptr[].width, params_ptr[].height,
        params_ptr[].left, params_ptr[].right, params_ptr[].top, params_ptr[].bottom,
        params_ptr[].imax, params_ptr[].color_seed,
    )


fn _render_mandelbrot_impl(
    width: Int, height: Int,
    left: Float64, right: Float64, top: Float64, bottom: Float64,
    imax: Int, color_seed: Float64,
) raises -> PythonObject:
    """Internal implementation shared by both Mandelbrot render APIs."""
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


fn render_julia(params_obj: PythonObject) raises -> PythonObject:
    """Render Julia set to numpy array.

    Args:
        params_obj: A JuliaParams instance with all rendering parameters.
    """
    var params_ptr = params_obj.downcast_value_ptr[JuliaParams]()
    return _render_julia_impl(
        params_ptr[].width, params_ptr[].height,
        params_ptr[].left, params_ptr[].right, params_ptr[].top, params_ptr[].bottom,
        params_ptr[].c_re, params_ptr[].c_im, params_ptr[].power_re, params_ptr[].power_im,
        params_ptr[].imax, params_ptr[].color_seed,
    )


fn _render_julia_impl(
    width: Int, height: Int,
    left: Float64, right: Float64, top: Float64, bottom: Float64,
    c_re: Float64, c_im: Float64, power_re: Float64, power_im: Float64,
    imax: Int, color_seed: Float64,
) raises -> PythonObject:
    """Internal implementation shared by both Julia render APIs."""
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


fn render_burning_ship(params_obj: PythonObject) raises -> PythonObject:
    """Render Burning Ship fractal to numpy array."""
    var params_ptr = params_obj.downcast_value_ptr[ViewParams]()
    return _render_burning_ship_impl(
        params_ptr[].width, params_ptr[].height,
        params_ptr[].left, params_ptr[].right, params_ptr[].top, params_ptr[].bottom,
        params_ptr[].imax, params_ptr[].color_seed,
    )


fn _render_burning_ship_impl(
    width: Int, height: Int,
    left: Float64, right: Float64, top: Float64, bottom: Float64,
    imax: Int, color_seed: Float64,
) raises -> PythonObject:
    """Internal implementation shared by both Burning Ship render APIs."""
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


fn render_tricorn(params_obj: PythonObject) raises -> PythonObject:
    """Render Tricorn (Mandelbar) fractal to numpy array."""
    var params_ptr = params_obj.downcast_value_ptr[ViewParams]()
    return _render_tricorn_impl(
        params_ptr[].width, params_ptr[].height,
        params_ptr[].left, params_ptr[].right, params_ptr[].top, params_ptr[].bottom,
        params_ptr[].imax, params_ptr[].color_seed,
    )


fn _render_tricorn_impl(
    width: Int, height: Int,
    left: Float64, right: Float64, top: Float64, bottom: Float64,
    imax: Int, color_seed: Float64,
) raises -> PythonObject:
    """Internal implementation shared by both Tricorn render APIs."""
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

fn render_mandelbulb(params_obj: PythonObject) raises -> PythonObject:
    """Render Mandelbulb 3D fractal to numpy array."""
    var params_ptr = params_obj.downcast_value_ptr[MandelbulbParams]()
    return _render_mandelbulb_impl(
        params_ptr[].width, params_ptr[].height,
        params_ptr[].cam_x, params_ptr[].cam_y, params_ptr[].cam_z,
        params_ptr[].cam_yaw, params_ptr[].cam_pitch,
        params_ptr[].power, params_ptr[].imax, params_ptr[].color_seed,
    )


fn _render_mandelbulb_impl(
    width: Int, height: Int,
    cam_x: Float32, cam_y: Float32, cam_z: Float32,
    cam_yaw: Float32, cam_pitch: Float32,
    power: Float32, imax: Int, color_seed: Float32,
) raises -> PythonObject:
    """Internal implementation shared by both Mandelbulb render APIs."""
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

        # Register param types with their __init__ methods
        # Note: ViewParams is used for Mandelbrot, BurningShip, and Tricorn (same fields)
        m.add_type[JuliaParams]("JuliaParams").def_py_init[_init_julia_params]()
        m.add_type[ViewParams]("ViewParams").def_py_init[_init_view_params]()
        m.add_type[MandelbulbParams]("MandelbulbParams").def_py_init[_init_mandelbulb_params]()
        m.add_type[NewtonParams]("NewtonParams").def_py_init[_init_newton_params]()

        # Register render functions
        m.def_function[render_newton]("render_newton", docstring="Render Newton fractal")
        m.def_function[render_mandelbrot]("render_mandelbrot", docstring="Render Mandelbrot set")
        m.def_function[render_julia]("render_julia", docstring="Render Julia set")
        m.def_function[render_burning_ship]("render_burning_ship", docstring="Render Burning Ship fractal")
        m.def_function[render_tricorn]("render_tricorn", docstring="Render Tricorn fractal")
        m.def_function[render_mandelbulb]("render_mandelbulb", docstring="Render Mandelbulb 3D fractal")

        # Utility functions
        m.def_function[has_gpu]("has_gpu", docstring="Check GPU availability")
        m.def_function[get_gpu_name]("get_gpu_name", docstring="Get GPU name")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
