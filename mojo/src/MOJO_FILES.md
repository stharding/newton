# Mojo Source Files Documentation

This document describes each Mojo file in the fractal viewer project.

---

## Core Files

### `renderer.mojo`
**Purpose:** Python-importable GPU module for the interactive fractal viewer (`viewer.py`).

**What it does:**
- Exports functions callable from Python via `mojo.importer`
- Imports kernels from separate modules and exposes them as Python functions
- Handles GPU buffer allocation, kernel dispatch, and numpy array conversion

**Exports:**
- `render_newton` - Newton fractal with root discovery and glow effect
- `render_mandelbrot` - Classic Mandelbrot set
- `render_julia` - Julia set with complex power support
- `render_burning_ship` - Burning Ship fractal
- `render_tricorn` - Tricorn (Mandelbar) fractal
- `render_mandelbulb` - 3D ray-marched Mandelbulb
- `has_gpu`, `get_gpu_name` - GPU availability checks

---

### `kernels_2d.mojo`
**Purpose:** 2D escape-time fractal kernels.

**Kernels:**
- `mandelbrot_kernel` - z = z² + c iteration
- `julia_kernel` - z = z^power + c with complex power support
- `burning_ship_kernel` - z = (|Re(z)| + i|Im(z)|)² + c
- `tricorn_kernel` - z = conj(z)² + c

All use smooth iteration coloring with cosine gradient palette.

---

### `kernels_newton.mojo`
**Purpose:** Newton fractal kernels (special two-pass rendering).

**Kernels:**
- `newton_kernel` - Newton-Raphson iteration, outputs (z_re, z_im, iterations) per pixel
- `colorize_kernel` - Colors pixels based on closest root with zoom-adaptive glow

**Constants:** Glow effect parameters (`GLOW_BASE`, `GLOW_RANGE`, etc.)

---

### `kernels_3d.mojo`
**Purpose:** 3D ray-marched fractal kernels.

**Kernels:**
- `mandelbulb_kernel` - Full ray marching with camera controls, surface normals, Phong shading

---

### `gpu_math.mojo`
**Purpose:** GPU math helper functions using PTX intrinsics for NVIDIA GPUs.

**Functions:**
- **Basic PTX intrinsics:** `gpu_sin`, `gpu_cos`, `gpu_exp2`, `gpu_log2`, `gpu_sqrt`, `gpu_rsqrt`
- **Derived functions:** `gpu_atan`, `gpu_atan2`, `gpu_pow`, `gpu_acos`

---

### `fractals_3d.mojo`
**Purpose:** 3D fractal distance estimators and ray marching constants.

**Contents:**
- **Constants:** `MAX_STEPS`, `MAX_DIST`, `EPSILON`, `NORMAL_EPSILON`
- **Functions:** `mandelbulb_de` - Distance estimator using triplex power formula

---

### `moclap.mojo`
**Purpose:** Command-line argument parser using Mojo's reflection system.

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

---

### `newton_gpu.mojo`
**Purpose:** Standalone GPU Newton fractal generator (outputs PNG).

---

### `gpu_test.mojo`
**Purpose:** Minimal GPU test to verify setup works.

---

## File Relationships

```
viewer.py (Python)
    │
    └── imports → renderer.mojo (Python exports, buffer management)
                      │
                      ├── imports → kernels_newton.mojo
                      ├── imports → kernels_2d.mojo
                      ├── imports → kernels_3d.mojo
                      │                 │
                      │                 └── imports → fractals_3d.mojo
                      │                                   │
                      │                                   └── imports → gpu_math.mojo
                      │
                      └── (kernels import gpu_math.mojo as needed)

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
