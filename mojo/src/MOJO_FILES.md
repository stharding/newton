# Mojo Source Files Documentation

This document describes each Mojo file in the fractal viewer project.

---

## Core Files

### `newton_renderer.mojo`
**Purpose:** Python-importable GPU module for the interactive fractal viewer (`viewer.py`).

**What it does:**
- Exports functions callable from Python via `mojo.importer`
- Renders Newton, Mandelbrot, Julia, Burning Ship, Tricorn, and Mandelbulb fractals on the GPU
- Returns numpy arrays for display in pygame

**How it works:**
1. **GPU Kernels:** Each fractal type has its own kernel function:
   - `newton_kernel` - Newton-Raphson root finding with Horner's method
   - `mandelbrot_kernel` - Classic z = z² + c iteration
   - `julia_kernel` - Julia set with complex power support (z^(a+bi))
   - `burning_ship_kernel` - z = (|Re(z)| + i|Im(z)|)² + c
   - `tricorn_kernel` - z = conj(z)² + c (Mandelbar)
   - `mandelbulb_kernel` - 3D ray-marched Mandelbulb fractal

2. **Imports helper modules:**
   - `gpu_math` - PTX intrinsics for fast GPU math
   - `fractals_3d` - 3D fractal distance estimators

3. **Python Module Registration:** `PyInit_newton_renderer()` uses `PythonModuleBuilder` to expose functions.

4. **Memory Flow:**
   ```
   Python args → DeviceBuffer → GPU kernel → HostBuffer → numpy array → Python
   ```

---

### `gpu_math.mojo`
**Purpose:** GPU math helper functions using PTX intrinsics for NVIDIA GPUs.

**What it does:**
- Provides fast approximations of common math operations using GPU-native instructions
- Designed for use in GPU kernels where standard library functions may not be optimal

**Functions:**
- **Basic PTX intrinsics:**
  - `gpu_sin`, `gpu_cos` - Using `sin.approx.ftz.f32`, `cos.approx.ftz.f32`
  - `gpu_exp2`, `gpu_log2` - Using `ex2.approx.ftz.f32`, `lg2.approx.f32`
  - `gpu_sqrt`, `gpu_rsqrt` - Using `sqrt.approx.ftz.f32`, `rsqrt.approx.ftz.f32`
- **Derived functions (built on PTX intrinsics):**
  - `gpu_atan` - Polynomial approximation with range reduction
  - `gpu_atan2` - Full quadrant-aware arctangent
  - `gpu_pow` - Using `2^(exp * log2(base))`
  - `gpu_acos` - Using `atan2(sqrt(1-x²), x)`

---

### `fractals_3d.mojo`
**Purpose:** 3D fractal distance estimators and ray marching utilities.

**What it does:**
- Provides distance estimator functions for 3D fractals
- Defines ray marching constants

**Contents:**
- **Constants:** `MAX_STEPS`, `MAX_DIST`, `EPSILON`, `NORMAL_EPSILON`
- **Functions:**
  - `mandelbulb_de` - Distance estimator for Mandelbulb using triplex power formula in spherical coordinates

---

### `moclap.mojo`
**Purpose:** Command-line argument parser using Mojo's reflection system.

**What it does:**
- Parses CLI arguments into a user-defined struct
- Auto-generates `--help` output from struct field names/types
- Supports strings, booleans, integers, and floats

**How it works:**
- Uses `struct_field_names`, `struct_field_types` from the `reflection` module
- Iterates over `argv()` matching `--field_name` patterns
- Uses `__struct_field_ref` to get mutable references to struct fields

**Usage:**
```mojo
@fieldwise_init
struct Config(Copyable, Movable, Defaultable, Writable):
    var width: Int
    fn __init__(out self):
        self.width = 500  # default

fn main() raises:
    var config = cli_parse[Config]()
```

---

### `newton.mojo`
**Purpose:** CPU-only Newton fractal generator (baseline implementation).

**What it does:**
- Generates Newton fractal images using pure Mojo on CPU
- Outputs PNG via Python PIL

**How it works:**
1. **Polynomial struct:** Stores coefficients, evaluates using Horner's method, computes derivative
2. **Newton iteration:** `z = z - p(z)/p'(z)` until convergence
3. **Root tracking:** Discovers roots and assigns random colors
4. **Output:** Uses PIL's `putdata()` for bulk pixel write

---

### `newton_gpu.mojo`
**Purpose:** GPU implementation - Newton kernel on GPU, colorization on CPU.

**What it does:**
- GPU computes Newton iteration for all pixels in parallel
- CPU scans results to find roots and assign colors
- Outputs PNG via PIL

**How it works:**
1. **GPU Kernel:** Each thread computes one pixel's final z-value and iteration count
   - Uses `@parameter for` to unroll coefficient loop at compile time
2. **CPU Colorization:** Sequential scan of GPU output to discover roots and color pixels

---

### `gpu_test.mojo`
**Purpose:** Minimal GPU test to verify setup works.

**What it does:**
- Simple kernel where each thread writes its index
- Verifies DeviceContext, buffer allocation, kernel launch, and copy-back

---

## File Relationships

```
viewer.py (Python)
    │
    └── imports → newton_renderer.mojo (GPU kernels + Python module)
                      │
                      ├── imports → gpu_math.mojo (PTX intrinsics)
                      │
                      └── imports → fractals_3d.mojo (3D distance estimators)
                                        │
                                        └── imports → gpu_math.mojo

Standalone executables:
    newton.mojo      (CPU baseline)
    newton_gpu.mojo  (GPU implementation)
    gpu_test.mojo    (GPU verification)
```

---

## Common Patterns

### GPU Kernel Structure
```mojo
fn my_kernel[layout: Layout](
    output: LayoutTensor[DType.uint8, layout, MutAnyOrigin],
    width: Int,
    height: Int,
):
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)
    if px >= width or py >= height:
        return
    # computation...
    output[(py * width + px) * 3] = result
```

### Buffer Allocation & Kernel Launch
```mojo
var ctx = DeviceContext()
var device_buf = ctx.enqueue_create_buffer[DType.float64](size)
var host_buf = ctx.enqueue_create_host_buffer[DType.float64](size)
ctx.synchronize()

var tensor = LayoutTensor[DType.float64, layout](device_buf)

ctx.enqueue_function[my_kernel[layout], my_kernel[layout]](
    tensor, width, height,
    grid_dim=(ceildiv(width, 16), ceildiv(height, 16)),
    block_dim=(16, 16),
)

ctx.enqueue_copy(host_buf, device_buf)
ctx.synchronize()
```

### Polynomial Evaluation (Horner's Method)
```mojo
# p(z) = a_n*z^n + ... + a_0
var p_re = 0.0
var p_im = 0.0
for i in range(num_coeffs):
    var c = coeffs[i]
    var new_re = p_re * z_re - p_im * z_im + c
    var new_im = p_re * z_im + p_im * z_re
    p_re, p_im = new_re, new_im
```
