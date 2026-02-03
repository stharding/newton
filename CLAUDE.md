# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Newton fractal generator with four implementations: Python, Cython, Rust, and Mojo. Visualizes Newton-Raphson root-finding convergence for polynomials, where colors represent different roots and intensity reflects iteration count.

## Build & Run Commands

### Python (with Cython)
```bash
cd python
pip install -e .           # Install in editable mode (builds Cython extension)
pynewton --help            # Show CLI options
pynewton --coefficients 1 0 0 -1 --dims 500 500  # Generate fractal for x³-1
pynewton --cython          # Use optimized Cython version
```

### Rust
```bash
cd rust
cargo build --release
cargo run --release
```

### Mojo
```bash
cd mojo/src
pixi run mojo build newton.mojo -o newton   # Compile
pixi run ./newton --help                     # Show CLI options
pixi run ./newton --coefficients "1,0,0,-1" --width 500 --height 500
```

### Interactive Viewer (Mojo GPU)
```bash
cd mojo/src
pixi run python viewer.py   # Launch interactive fractal viewer
```

**Building newton_renderer.mojo:** This module is compiled automatically on first import via Mojo's Python interop (`mojo.importer`). The compiled `.so` is cached in `__mojocache__/`. To force recompilation after editing:
```bash
cd mojo/src
rm -rf __mojocache__
pixi run python viewer.py   # Recompiles on import
```

## Architecture

**Core Algorithm**: Newton-Raphson iteration `x_next = x - f(x)/f'(x)` applied to each pixel mapped from complex plane via affine transformation.

**Polynomial Class** (all implementations):
- Coefficients stored as list (highest degree first)
- Uses Horner's method for efficient evaluation
- `derivative()` returns new Polynomial object

**Four Implementations**:
- `python/pynewton/pynewton.py`: Pure Python with PIL
- `python/pynewton/cynewton.pyx`: Cython with typed variables and C++ vectors
- `rust/src/main.rs`: Native Rust with raster/clap
- `mojo/src/newton.mojo`: Mojo with moclap CLI + PIL via Python interop

**Zero denominator handling**: Pixels where derivative is zero render as grey.

**Interactive Viewer** (`mojo/src/viewer.py`):
- `FractalViewer` class with dataclasses for state (View2D, Camera3D, JuliaParams, etc.)
- Event dispatch via `_EVENT_HANDLERS` dict mapping pygame event types to methods
- Key dispatch via `_KEY_HANDLERS` dict mapping keys to actions
- Main loop is 5 lines: handle events → handle continuous input → render if needed

## Mojo-Specific Notes

- Uses pixi for environment management (conda-based, includes `max` package)
- CLI parsing via vendored moclap (reflection-based, like Rust's clap derive)
- Python interop for PIL image output: `from python import Python, PythonObject`
- Batched pixel writes via `img.putdata()` for performance
- Run from `mojo/src/` directory so moclap import resolves

## Mojo Syntax Reference (nightly 2025+)

**If you need to verify current syntax**, the stdlib source is at `~/newton/modular/mojo/stdlib/std/`. Key files:
- `complex/complex.mojo` - ComplexSIMD patterns
- `collections/inline_array.mojo` - InlineArray usage
- `benchmark/benchmark.mojo` - struct with TrivialRegisterType example

### Struct Definitions
```mojo
# OLD (removed):
@value
@register_passable("trivial")
struct Vec3:
    var x: Float32

# CURRENT:
@fieldwise_init
struct Vec3(TrivialRegisterType):
    var x: Float32
```

### Tuple Returns
```mojo
# OLD (doesn't work):
fn foo() -> (Int, Int):
    return (1, 2)

# CURRENT:
fn foo() -> Tuple[Int, Int]:
    return Tuple(1, 2)
```

### Init Methods
```mojo
# Use `out self` not `inout self`:
fn __init__(out self, x: Float32):
    self.x = x
```

### InlineArray
```mojo
# Requires keyword argument:
var arr = InlineArray[Int, 4](fill=0)
var arr = InlineArray[Int, 4](uninitialized=True)

# Or list literal (comptime only):
comptime arr: InlineArray[Int, 4] = [1, 2, 3, 4]
```

### GPU Math Compatibility
**Works on GPU** (has intrinsics):
- `sin`, `cos`, `sqrt`, `log2`, `exp2`, `rsqrt`
- `clamp` (from math module)
- `**` operator for pow (uses `exp2(exp * log2(base))`)

**Does NOT work on GPU** (calls libm):
- `atan`, `atan2`, `acos` - need custom implementations (see `gpu_math.mojo`)

### Stdlib Complex Type
```mojo
from complex import ComplexFloat64  # or ComplexFloat32, ComplexSIMD

var z = ComplexFloat64(0.0, 0.0)
var c = ComplexFloat64(re, im)
z = z.squared_add(c)      # z = z² + c (Mandelbrot iteration)
var mag_sq = z.squared_norm()  # |z|² without sqrt
var conj = z.conj()       # complex conjugate
```

## Mojo Refactoring Retrospective (Feb 2026)

### What Was Planned vs What Happened

| Proposal | Status | Notes |
|----------|--------|-------|
| Use `ComplexSIMD` for 2D fractals | ✅ Done | `squared_add()` and `squared_norm()` work great |
| Use `clamp` from stdlib | ✅ Done | Direct replacement |
| Use math constants (`pi`, `tau`) | ✅ Done | Direct replacement |
| Remove `gpu_pow` | ✅ Done | `**` operator works on GPU |
| Keep `gpu_atan2` | ✅ Kept | Confirmed stdlib calls libm (won't work on GPU) |
| Add `Vec3` struct for 3D | ✅ Done | But hit syntax snags (see below) |
| Extract `pixel_to_complex` helper | ✅ Done | Returns `ComplexFloat64` |
| Extract renderer helpers | ✅ Partial | `host_buffer_to_numpy`, `compute_grid` extracted. Full `RenderContext` struct deferred |
| Remove redundant `num_roots` check | ✅ Done | Simple fix |
| Named constants for magic numbers | ✅ Done | `ESCAPE_RADIUS_SQ`, `SPECULAR_EXPONENT`, etc. |
| Keyword args for Python interop | ❌ Skipped | Works but adds complexity; tuple unpacking is fine |
| Extract Newton root functions | ✅ Done | `_discover_roots`, `_sort_roots_by_angle`, `_get_palette_color` |

### Syntax Gotchas Encountered

1. **`@value` removed** - Had to change `@value @register_passable("trivial")` to `@fieldwise_init` + `TrivialRegisterType` trait

2. **Tuple returns changed** - `-> (Int, Int)` doesn't work anymore, need `-> Tuple[Int, Int]` and return `Tuple(a, b)`

3. **`StaticTuple` unavailable** - Tried to use for color palette, ended up using a simple function with if/elif instead

4. **`InlineArray` init** - Can't pass positional args directly, needs `fill=` or `uninitialized=` keyword

### What Worked Well

- Stdlib `ComplexFloat64` is a perfect fit for Mandelbrot/Julia - cleaner than manual real/imag math
- `clamp` is a direct drop-in, no surprises
- Extracting helpers improved readability significantly
- Constants with docstrings make the code self-documenting

### What To Do Differently Next Time

- **Check stdlib source FIRST** before assuming what's available
- **Test compile early** - syntax changes fast, don't write 500 lines before testing
- **Keep the modular stdlib handy** at `~/newton/modular/mojo/stdlib/std/`

## CLI Arguments

### Python
- `--coefficients`: Space-separated (e.g., `1 0 0 -1`)
- `--dims`: Two values for width/height
- `--cython`: Use Cython version

### Mojo
- `--coefficients`: Comma-separated string (e.g., `"1,0,0,-1"`)
- `--width`, `--height`: Separate arguments
- `--window_left`, `--window_right`, `--window_top`, `--window_bottom`: Separate arguments

### Common
- `--tolerance`: Convergence threshold (default: 0.0001)
- `--imax`: Max iterations (default: 30)
- `--out_file` (Mojo) / `-o` (Python): Output filename

## Performance (1000x1000, x^4-1)
- Pure Python: ~4.9s
- Mojo (compiled): ~0.52s
- Cython: ~0.62s
- Rust: (comparable to Cython)
