#!/usr/bin/env python3
"""Render a specific deep zoom location to an image file."""
import mojo.importer
import renderer
from deep_zoom import ReferenceOrbit, compute_delta_per_pixel
from PIL import Image
import numpy as np

# Location from screenshot
CENTER_RE = "-1.9073395970641375017156346454"
CENTER_IM = "0.00062538602748309027910015455"
LOG2_ZOOM = 45.5
IMAX = 1000
WIDTH = 1200
HEIGHT = 900

def render_deep_zoom():
    print(f"Rendering at zoom 2^{LOG2_ZOOM}...")
    print(f"Center: {CENTER_RE} + {CENTER_IM}i")

    precision = ReferenceOrbit.precision_for_zoom(LOG2_ZOOM)
    print(f"Using {precision} digits of precision")

    # Compute reference orbit at center
    ref_orbit = ReferenceOrbit(CENTER_RE, CENTER_IM, precision)
    orbit_len = ref_orbit.compute(IMAX)
    print(f"Center orbit: {orbit_len} iterations, escaped={ref_orbit.escaped}")

    # If center escapes, search for better reference
    if ref_orbit.escaped:
        print("Searching for better reference (Mojo CPU)...")
        from decimal import Decimal, getcontext
        getcontext().prec = precision

        delta_per_pixel = compute_delta_per_pixel(LOG2_ZOOM, HEIGHT)
        center_re_f = float(CENTER_RE)
        center_im_f = float(CENTER_IM)

        # Use Mojo CPU search (fast)
        params = (center_re_f, center_im_f, WIDTH, HEIGHT, delta_per_pixel, IMAX, 8)
        best_px, best_py, best_iters = renderer.find_best_reference_deep(params)
        print(f"Mojo found: pixel ({best_px}, {best_py}) with {best_iters} iters")

        # Convert pixel to arbitrary precision coordinates and verify
        dx = best_px - WIDTH / 2
        dy = best_py - HEIGHT / 2
        offset_re = dx * delta_per_pixel
        offset_im = -dy * delta_per_pixel

        best_re = str(Decimal(CENTER_RE) + Decimal(str(offset_re)))
        best_im = str(Decimal(CENTER_IM) + Decimal(str(offset_im)))

        candidate = ReferenceOrbit(best_re, best_im, precision)
        candidate_len = candidate.compute(IMAX)
        print(f"Verified with arbitrary precision: {candidate_len} iters, escaped={candidate.escaped}")

        if candidate_len > orbit_len:
            print(f"Using better reference: {candidate_len} iterations")
            ref_orbit = candidate
            orbit_len = candidate_len
        else:
            print("Mojo candidate not better, trying Python fallback...")
            # Fallback to Python arbitrary precision search
            best_orbit = ref_orbit
            best_len = orbit_len
            sample_grid = 12

            for gy in range(sample_grid):
                for gx in range(sample_grid):
                    px = (gx / (sample_grid - 1) - 0.5) * WIDTH
                    py = (gy / (sample_grid - 1) - 0.5) * HEIGHT

                    offset_re = px * delta_per_pixel
                    offset_im = -py * delta_per_pixel

                    sample_re = str(Decimal(CENTER_RE) + Decimal(str(offset_re)))
                    sample_im = str(Decimal(CENTER_IM) + Decimal(str(offset_im)))

                    cand = ReferenceOrbit(sample_re, sample_im, precision)
                    cand_len = cand.compute(IMAX)

                    if cand_len > best_len:
                        best_orbit = cand
                        best_len = cand_len
                        print(f"  Found better: {cand_len} iters at grid ({gx}, {gy})")
                        if not cand.escaped:
                            print("  Found non-escaping!")
                            break
                else:
                    continue
                break

            if best_len > orbit_len:
                ref_orbit = best_orbit
                orbit_len = best_len

    # Render with perturbation
    ref_re, ref_im = ref_orbit.get_orbit_arrays()
    delta_per_pixel = compute_delta_per_pixel(LOG2_ZOOM, HEIGHT)

    params = renderer.DeepZoomParams(
        width=WIDTH,
        height=HEIGHT,
        delta_per_pixel=delta_per_pixel,
        imax=IMAX,
        color_seed=0.0,
    )

    print("Rendering...")
    rgb = renderer.render_mandelbrot_deep(params, ref_re, ref_im)

    # Save as PNG
    img = Image.fromarray(rgb.astype(np.uint8))
    filename = "deep_zoom_location.png"
    img.save(filename)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    render_deep_zoom()
