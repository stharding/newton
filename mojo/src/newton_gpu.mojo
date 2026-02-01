"""Newton fractal generator - GPU accelerated version."""

from python import Python, PythonObject
from random import random_ui64
from math import sqrt, ceildiv
from gpu.host import DeviceContext, HostBuffer
from gpu import global_idx
from layout import Layout, LayoutTensor
from sys import has_accelerator
from time import perf_counter_ns
from moclap import cli_parse


# ============================================================================
# CLI Config
# ============================================================================

@fieldwise_init
struct Config(Copyable, Movable, Defaultable, Writable):
    """CLI configuration for Newton fractal generator."""
    var coefficients: String
    var out_file: String
    var tolerance: Float64
    var imax: Int
    var width: Int
    var height: Int
    var window_left: Float64
    var window_right: Float64
    var window_top: Float64
    var window_bottom: Float64

    fn __init__(out self):
        self.coefficients = "1,0,0,-1"
        self.out_file = "out.png"
        self.tolerance = 0.0001
        self.imax = 30
        self.width = 500
        self.height = 500
        self.window_left = -1.0
        self.window_right = 1.0
        self.window_top = 1.0
        self.window_bottom = -1.0


fn parse_coefficients(s: String) raises -> List[Float64]:
    var coeffs = List[Float64]()
    var parts = s.split(",")
    for i in range(len(parts)):
        var part = parts[i].strip()
        if len(part) > 0:
            coeffs.append(atof(part))
    return coeffs^


# ============================================================================
# GPU Kernel
# ============================================================================

fn newton_kernel[
    num_coeffs: Int,
    width: Int,
    height: Int,
    coeffs_layout: Layout,
    output_layout: Layout,
](
    coeffs: LayoutTensor[DType.float64, coeffs_layout, MutAnyOrigin],
    output: LayoutTensor[DType.float64, output_layout, MutAnyOrigin],
    window_left: Float64,
    window_right: Float64,
    window_top: Float64,
    window_bottom: Float64,
    tolerance: Float64,
    imax: Int,
):
    """GPU kernel: each thread computes one pixel."""
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)

    if px >= width or py >= height:
        return

    # Map pixel to complex plane
    var z_re = window_left + (Float64(px) / Float64(width)) * (window_right - window_left)
    var z_im = window_top + (Float64(py) / Float64(height)) * (window_bottom - window_top)

    var iterations = Float64(imax)
    var zero_div = False

    # Newton iteration
    for count in range(imax):
        # Evaluate polynomial p(z) and derivative p'(z) using Horner's method
        var p_re = Float64(0.0)
        var p_im = Float64(0.0)
        var dp_re = Float64(0.0)
        var dp_im = Float64(0.0)

        # Manual unroll for small coefficient counts
        @parameter
        for i in range(num_coeffs):
            # Get coefficient and exponent - extract scalar from SIMD
            var c = rebind[Float64](coeffs[i])
            var exp = num_coeffs - 1 - i

            # p = p * z + c  (complex multiplication then add)
            var new_p_re = p_re * z_re - p_im * z_im + c
            var new_p_im = p_re * z_im + p_im * z_re
            p_re = new_p_re
            p_im = new_p_im

            # dp = dp * z + c*exp (for derivative)
            if exp > 0:
                var dc = c * Float64(exp)
                var new_dp_re = dp_re * z_re - dp_im * z_im + dc
                var new_dp_im = dp_re * z_im + dp_im * z_re
                dp_re = new_dp_re
                dp_im = new_dp_im

        # Check for zero derivative
        var dp_mag_sq = dp_re * dp_re + dp_im * dp_im
        if dp_mag_sq < 1e-20:
            zero_div = True
            break

        # Newton step: z = z - p(z) / p'(z)
        var ratio_re = (p_re * dp_re + p_im * dp_im) / dp_mag_sq
        var ratio_im = (p_im * dp_re - p_re * dp_im) / dp_mag_sq

        var old_re = z_re
        var old_im = z_im
        z_re = z_re - ratio_re
        z_im = z_im - ratio_im

        # Check convergence (use squared distance to avoid sqrt)
        var diff_sq = (old_re - z_re) * (old_re - z_re) + (old_im - z_im) * (old_im - z_im)
        if diff_sq < tolerance * tolerance:
            iterations = Float64(count)
            break

    # Store result: (re, im, iterations)
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
# Root clustering and coloring (CPU)
# ============================================================================

@fieldwise_init
struct Root(Copyable, Movable):
    var re: Float64
    var im: Float64
    var r: Int
    var g: Int
    var b: Int

    fn __init__(out self, re: Float64, im: Float64):
        self.re = re
        self.im = im
        self.r = Int(random_ui64(0, 150))
        self.g = Int(random_ui64(0, 150))
        self.b = Int(random_ui64(0, 150))

    fn matches(self, re: Float64, im: Float64, tolerance: Float64) -> Bool:
        var dist = sqrt((self.re - re) * (self.re - re) + (self.im - im) * (self.im - im))
        return dist < tolerance * 2


fn find_matching_root(roots: List[Root], re: Float64, im: Float64, tolerance: Float64) -> Int:
    for i in range(len(roots)):
        if roots[i].matches(re, im, tolerance):
            return i
    return -1


fn colorize_and_save(
    output_host: HostBuffer[DType.float64],
    width: Int,
    height: Int,
    imax: Int,
    tolerance: Float64,
    filename: String,
) raises:
    var PIL = Python.import_module("PIL.Image")
    var roots = List[Root]()
    var pixels = Python.list()

    for py in range(height):
        for px in range(width):
            var idx = (py * width + px) * 3
            var re = Float64(output_host[idx])
            var im = Float64(output_host[idx + 1])
            var iterations = Float64(output_host[idx + 2])

            var r: Int = 150
            var g: Int = 150
            var b: Int = 150

            if iterations < 0:
                r = 128
                g = 128
                b = 128
            elif iterations < Float64(imax):
                var root_idx = find_matching_root(roots, re, im, tolerance)
                if root_idx < 0:
                    roots.append(Root(re, im))
                    root_idx = len(roots) - 1

                var scaled_count = Int((iterations / Float64(imax)) * 105.0)
                r = roots[root_idx].r + scaled_count
                g = roots[root_idx].g + scaled_count
                b = roots[root_idx].b + scaled_count

            _ = pixels.append(Python.tuple(r, g, b))

    print(String(len(roots)) + " roots found")

    var size = Python.tuple(width, height)
    var img = PIL.new("RGB", size)
    _ = img.putdata(pixels)
    _ = img.save(filename)
    print("Saved to " + filename)


# ============================================================================
# Main
# ============================================================================

fn run_gpu[num_coeffs: Int, width: Int, height: Int](
    coeffs: List[Float64],
    config: Config,
) raises:
    var ctx = DeviceContext()
    print("Using GPU: " + ctx.name())

    comptime coeffs_layout = Layout.row_major(num_coeffs)
    comptime output_size = width * height * 3
    comptime output_layout = Layout.row_major(output_size)

    var coeffs_host = ctx.enqueue_create_host_buffer[DType.float64](num_coeffs)
    var coeffs_device = ctx.enqueue_create_buffer[DType.float64](num_coeffs)
    var output_device = ctx.enqueue_create_buffer[DType.float64](output_size)
    var output_host = ctx.enqueue_create_host_buffer[DType.float64](output_size)
    ctx.synchronize()

    for i in range(num_coeffs):
        coeffs_host[i] = coeffs[i]

    ctx.enqueue_copy(coeffs_device, coeffs_host)
    ctx.synchronize()

    var coeffs_tensor = LayoutTensor[DType.float64, coeffs_layout](coeffs_device)
    var output_tensor = LayoutTensor[DType.float64, output_layout](output_device)

    comptime block_size = 16
    var grid_x = ceildiv(width, block_size)
    var grid_y = ceildiv(height, block_size)

    print("Launching: grid=(" + String(grid_x) + "," + String(grid_y) +
          ") block=(" + String(block_size) + "," + String(block_size) + ")")

    var t0 = perf_counter_ns()
    ctx.enqueue_function[
        newton_kernel[num_coeffs, width, height, coeffs_layout, output_layout],
        newton_kernel[num_coeffs, width, height, coeffs_layout, output_layout],
    ](
        coeffs_tensor,
        output_tensor,
        config.window_left,
        config.window_right,
        config.window_top,
        config.window_bottom,
        config.tolerance,
        config.imax,
        grid_dim=(grid_x, grid_y),
        block_dim=(block_size, block_size),
    )

    ctx.enqueue_copy(output_host, output_device)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var gpu_ms = Float64(t1 - t0) / 1_000_000.0
    print("GPU compute + copy: " + String(gpu_ms) + " ms")

    var t2 = perf_counter_ns()
    colorize_and_save(output_host, width, height, config.imax, config.tolerance, config.out_file)
    var t3 = perf_counter_ns()
    var color_ms = Float64(t3 - t2) / 1_000_000.0
    print("Colorization: " + String(color_ms) + " ms")


fn main() raises:
    var config = cli_parse[Config]()

    if not has_accelerator():
        print("No GPU found! Use newton.mojo for CPU version.")
        return

    var coeffs = parse_coefficients(config.coefficients)
    var num_coeffs = len(coeffs)

    print("Newton fractal (GPU)")
    print("dims: " + String(config.width) + "x" + String(config.height))
    print("coefficients: " + config.coefficients)

    # Dispatch to compile-time specialized versions
    if num_coeffs == 4 and config.width == 500 and config.height == 500:
        run_gpu[4, 500, 500](coeffs, config)
    elif num_coeffs == 4 and config.width == 1000 and config.height == 1000:
        run_gpu[4, 1000, 1000](coeffs, config)
    elif num_coeffs == 4 and config.width == 2000 and config.height == 2000:
        run_gpu[4, 2000, 2000](coeffs, config)
    elif num_coeffs == 4 and config.width == 4000 and config.height == 4000:
        run_gpu[4, 4000, 4000](coeffs, config)
    elif num_coeffs == 5 and config.width == 500 and config.height == 500:
        run_gpu[5, 500, 500](coeffs, config)
    elif num_coeffs == 5 and config.width == 1000 and config.height == 1000:
        run_gpu[5, 1000, 1000](coeffs, config)
    elif num_coeffs == 5 and config.width == 2000 and config.height == 2000:
        run_gpu[5, 2000, 2000](coeffs, config)
    elif num_coeffs == 5 and config.width == 4000 and config.height == 4000:
        run_gpu[5, 4000, 4000](coeffs, config)
    else:
        print("Unsupported config. Supported: coeffs=4-5, dims=500/1000/2000/4000 square")
