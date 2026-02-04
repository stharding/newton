#!/usr/bin/env python3
"""Test script to render and compare deep zoom vs standard Mandelbrot.

Usage:
    python test_render_deep.py                    # Run comparison tests
    python test_render_deep.py render ZOOM        # Render single image at zoom level
    python test_render_deep.py render ZOOM RE IM  # Render at specific center
"""

import argparse
import numpy as np
from PIL import Image
import sys

import mojo.importer
import renderer
from deep_zoom import ReferenceOrbit, compute_delta_per_pixel


def render_standard(center_re, center_im, zoom, width, height, imax, color_seed=0.0):
    """Render using standard Mandelbrot kernel."""
    aspect = width / height
    half_w = zoom * aspect
    half_h = zoom

    params = renderer.ViewParams(
        width=width, height=height,
        left=center_re - half_w,
        right=center_re + half_w,
        top=center_im + half_h,
        bottom=center_im - half_h,
        imax=imax,
        color_seed=color_seed,
    )
    return renderer.render_mandelbrot(params)


def render_deep(center_re_str, center_im_str, log2_zoom, width, height, imax, color_seed=0.0):
    """Render using deep zoom perturbation kernel."""
    precision = ReferenceOrbit.precision_for_zoom(log2_zoom)
    orbit = ReferenceOrbit(center_re_str, center_im_str, precision)
    orbit_len = orbit.compute(imax)

    print(f"  Reference orbit: {orbit_len} pts, escaped={orbit.escaped}")

    ref_re, ref_im = orbit.get_orbit_arrays()
    delta_per_pixel = compute_delta_per_pixel(log2_zoom, height)

    params = renderer.DeepZoomParams(
        width=width, height=height,
        delta_per_pixel=delta_per_pixel,
        imax=imax,
        color_seed=color_seed,
    )

    return renderer.render_mandelbrot_deep(params, ref_re, ref_im)


def log2_zoom_to_half_height(log2_zoom):
    """Convert log2_zoom to the half-height parameter used by standard renderer."""
    return 2.0 / (2 ** log2_zoom)


def compare_renders(name, center_re, center_im, log2_zoom, width=400, height=300, imax=256):
    """Render both methods and save comparison images."""
    print(f"\n=== {name} ===")
    print(f"Center: {center_re}, {center_im}")
    print(f"Zoom: 2^{log2_zoom} = {2**log2_zoom:.2f}x")

    half_h = log2_zoom_to_half_height(log2_zoom)
    rgb_std = render_standard(float(center_re), float(center_im), half_h, width, height, imax)
    rgb_deep = render_deep(center_re, center_im, log2_zoom, width, height, imax)

    img_std = Image.fromarray(rgb_std)
    img_deep = Image.fromarray(rgb_deep)

    img_std.save(f"test_{name}_standard.png")
    img_deep.save(f"test_{name}_deep.png")

    diff = np.abs(rgb_std.astype(np.int16) - rgb_deep.astype(np.int16)).astype(np.uint8)
    diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)
    Image.fromarray(diff_amplified).save(f"test_{name}_diff.png")

    print(f"  Max diff: {np.max(diff)}, Mean diff: {np.mean(diff):.2f}")
    return rgb_std, rgb_deep


def render_single(log2_zoom, center_re="-0.75", center_im="0.0", width=800, height=600, imax=1000):
    """Render a single deep zoom image."""
    print(f"Rendering at zoom 2^{log2_zoom} ({2**log2_zoom:.2e}x)")
    print(f"Center: {center_re}, {center_im}")
    print(f"Size: {width}x{height}, imax: {imax}")

    rgb = render_deep(center_re, center_im, log2_zoom, width, height, imax)

    filename = f"mandelbrot_2e{int(log2_zoom)}.png"
    Image.fromarray(rgb).save(filename)
    print(f"Saved: {filename}")
    return rgb


def run_comparison_tests():
    """Run standard comparison tests."""
    print("Testing deep zoom renderer...")
    print(f"GPU: {renderer.get_gpu_name()}")

    compare_renders("default", "-0.75", "0.0", log2_zoom=2.0, imax=256)
    compare_renders("cardioid_edge", "-0.75", "0.0", log2_zoom=5.0, imax=256)
    compare_renders("seahorse", "-0.743643887037151", "0.131825904205330", log2_zoom=3.0, imax=500)

    print("\n=== Tests complete ===")
    print("Check the generated test_*.png files")


def main():
    parser = argparse.ArgumentParser(description="Test deep zoom Mandelbrot renderer")
    parser.add_argument("command", nargs="?", default="test", choices=["test", "render"])
    parser.add_argument("zoom", nargs="?", type=float, default=10.0, help="log2 zoom level")
    parser.add_argument("re", nargs="?", default="-0.75", help="center real part")
    parser.add_argument("im", nargs="?", default="0.0", help="center imaginary part")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--imax", type=int, default=1000)

    args = parser.parse_args()

    if args.command == "test":
        run_comparison_tests()
    else:
        render_single(args.zoom, args.re, args.im, args.width, args.height, args.imax)


if __name__ == "__main__":
    main()
