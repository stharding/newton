"""Deep zoom Mandelbrot GPU kernel using perturbation theory.

This kernel renders Mandelbrot at arbitrary zoom depths by computing
small perturbations from a reference orbit computed at the view center.

Core formula: δ_{n+1} = 2·Z_n·δ_n + δ_n² + Δc

Where:
- Z_n is the precomputed reference orbit (arbitrary precision, converted to Float64)
- δ_n is the perturbation for this pixel (starts at Δc)
- Δc is the pixel's offset from the reference center
"""

from complex import ComplexFloat64
from gpu import global_idx
from layout import Layout, LayoutTensor
from math import clamp, cos, exp2, log2, sin, sqrt, tau

# Import shared constants and coloring from kernels_2d
from kernels_2d import ESCAPE_RADIUS_SQ, COLOR_FREQUENCY, SATURATION_BOOST, write_smooth_color


# ============================================================================
# Constants
# ============================================================================

comptime GLITCH_THRESHOLD: Float64 = 1e6
"""Perturbation is considered glitched when |δ|² > |Z+δ|² × GLITCH_THRESHOLD."""

comptime REBASING_THRESHOLD: Float64 = 1e-4
"""Rebase when |δ|² < |Z|² × REBASING_THRESHOLD (perturbation much smaller than reference)."""


# ============================================================================
# Helper: Convert pixel to delta from center
# ============================================================================

@always_inline
fn pixel_to_delta(
    px: Int, py: Int, width: Int, height: Int, delta_per_pixel: Float64,
) -> ComplexFloat64:
    """Convert pixel coordinates to complex delta from view center.

    The delta represents the offset from the reference orbit center.
    At deep zooms, this delta is very small but still representable in Float64.
    """
    # Pixel offset from center (center is at width/2, height/2)
    var dx = Float64(px) - Float64(width) / 2.0
    var dy = Float64(py) - Float64(height) / 2.0

    # Convert to complex offset (y is negated for standard complex plane orientation)
    var re = dx * delta_per_pixel
    var im = -dy * delta_per_pixel

    return ComplexFloat64(re, im)


# ============================================================================
# Perturbation kernel
# ============================================================================

fn mandelbrot_perturbation_kernel[
    output_layout: Layout,
    orbit_layout: Layout,
](
    output: LayoutTensor[DType.uint8, output_layout, MutAnyOrigin],
    ref_orbit_re: LayoutTensor[DType.float64, orbit_layout, MutAnyOrigin],
    ref_orbit_im: LayoutTensor[DType.float64, orbit_layout, MutAnyOrigin],
    orbit_length: Int,
    width: Int,
    height: Int,
    delta_per_pixel: Float64,
    ref_offset_re: Float64,
    ref_offset_im: Float64,
    imax: Int,
    color_seed: Float64,
):
    """GPU kernel for deep zoom Mandelbrot using perturbation theory.

    Each pixel computes a small perturbation δ from the precomputed reference
    orbit Z. This allows accurate rendering at zoom levels far beyond Float64
    limits, since the perturbations remain small enough to be representable.

    ref_offset_re/im is the offset from VIEW CENTER to REFERENCE ORBIT center.
    This allows the reference orbit to be at a different location than the view center.

    Glitched pixels (where perturbation grows too large relative to orbit)
    are marked with iterations=-1 for potential second-pass rendering.
    """
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)

    if px >= width or py >= height:
        return

    # Compute this pixel's offset from VIEW center
    var pixel_delta = pixel_to_delta(px, py, width, height, delta_per_pixel)

    # Total delta from REFERENCE orbit = pixel_delta - ref_offset
    # (because ref_offset is view_center -> orbit_center, we subtract)
    var delta_c = ComplexFloat64(pixel_delta.re - ref_offset_re, pixel_delta.im - ref_offset_im)

    # Initialize perturbation: δ₀ = 0 (because z₀ = 0 for Mandelbrot)
    # After first iteration, δ₁ = Δc via the recurrence formula
    var delta = ComplexFloat64(0.0, 0.0)

    var iterations = imax
    var final_r2 = Float64(0.0)
    var glitched = False

    # Perturbation iteration
    # Formula: δ_{n+1} = 2·Z_n·δ_n + δ_n² + Δc
    for n in range(imax):
        # Get reference orbit value Z_n
        # If we've exceeded reference orbit length, continue with standard iteration
        if n >= orbit_length:
            # Fallback: treat remaining iterations as standard Mandelbrot
            # This shouldn't happen if reference orbit is computed correctly
            break

        var Z_n = ComplexFloat64(rebind[Float64](ref_orbit_re[n]), rebind[Float64](ref_orbit_im[n]))

        # Full position: z_n = Z_n + δ_n
        var z_full = Z_n + delta

        # Check escape condition on full position
        var z_norm_sq = z_full.squared_norm()
        if z_norm_sq > ESCAPE_RADIUS_SQ:
            iterations = n
            final_r2 = z_norm_sq
            break

        # Check for glitch: perturbation too large relative to full value
        var delta_norm_sq = delta.squared_norm()
        if delta_norm_sq > z_norm_sq * GLITCH_THRESHOLD:
            glitched = True
            iterations = -1  # Mark as glitched
            break

        # Perturbation update: δ_{n+1} = 2·Z_n·δ_n + δ_n² + Δc
        # = (2·Z_n + δ_n)·δ_n + Δc
        var two_Z_plus_delta = ComplexFloat64(2.0 * Z_n.re + delta.re, 2.0 * Z_n.im + delta.im)
        delta = two_Z_plus_delta * delta + delta_c

    # Write output
    var pixel_idx = (py * width + px) * 3

    if glitched:
        # Mark glitched pixels with a distinct color (gray)
        output[pixel_idx] = 64
        output[pixel_idx + 1] = 64
        output[pixel_idx + 2] = 64
    else:
        write_smooth_color[output_layout](output, pixel_idx, iterations, imax, final_r2, color_seed)
