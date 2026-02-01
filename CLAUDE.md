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

## Mojo-Specific Notes

- Uses pixi for environment management (conda-based, includes `max` package)
- CLI parsing via vendored moclap (reflection-based, like Rust's clap derive)
- Python interop for PIL image output: `from python import Python, PythonObject`
- Batched pixel writes via `img.putdata()` for performance
- Run from `mojo/src/` directory so moclap import resolves

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
