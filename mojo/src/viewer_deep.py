#!/usr/bin/env python3
"""Deep zoom Mandelbrot viewer with hybrid rendering.

Uses standard Float64 rendering at shallow zoom levels, automatically
switching to perturbation theory when zoomed beyond Float64 limits (~10^13).
This gives a smooth click-and-drag experience at all zoom levels.

Controls:
    Mouse wheel     Zoom in/out at cursor
    Click           Center on clicked point
    Arrow keys      Fine pan
    +/-            Adjust max iterations
    R              Randomize colors
    D              Reset to default view
    F              Toggle FPS display
    H/?            Toggle help
    Q/ESC          Quit
"""

from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import Optional
import gc
import os
import sys
import time

# Force X11 backend for proper window decorations on Wayland
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

import numpy as np
import pygame

import mojo.importer
import renderer
from deep_zoom import ReferenceOrbit, compute_delta_per_pixel


# =============================================================================
# Constants
# =============================================================================

ZOOM_FACTOR = 1.5
PAN_SPEED = 0.1  # As fraction of view width
KEY_REPEAT_INTERVAL = 0.1
FONT_SIZE = 24
PADDING = 10
HELP_OVERLAY_ALPHA = 200

# Default starting location
DEFAULT_CENTER_RE = "-0.75"
DEFAULT_CENTER_IM = "0.0"
DEFAULT_LOG2_ZOOM = 2.0

# Zoom threshold for switching to perturbation rendering
# Below this, standard Float64 rendering works fine
# Above this, we need perturbation theory
PERTURBATION_THRESHOLD = 40.0  # ~10^12 zoom

# Interesting locations for testing deep zoom
DEEP_ZOOM_LOCATIONS = [
    # Seahorse valley
    ("-0.743643887037151", "0.131825904205330", 50),
    # Elephant valley
    ("0.250006", "0.0", 30),
    # Mini Mandelbrot
    ("-1.768778833", "-0.001738996", 40),
    # Another mini
    ("-0.16070135", "1.0375665", 35),
]


# =============================================================================
# State
# =============================================================================

@dataclass
class DragState:
    """Mouse drag state for panning."""
    active: bool = False
    start_pos: Optional[tuple] = None
    start_center_re: Optional[str] = None
    start_center_im: Optional[str] = None


@dataclass
class DeepZoomState:
    """State for deep zoom viewer."""
    # Center coordinates as arbitrary-precision strings
    center_re: str = DEFAULT_CENTER_RE
    center_im: str = DEFAULT_CENTER_IM

    # Zoom level as log2 (zoom factor = 2^log2_zoom)
    log2_zoom: float = DEFAULT_LOG2_ZOOM

    # Rendering parameters
    imax: int = 1000
    color_seed: float = 0.0

    # Cached reference orbit (only used for perturbation mode)
    ref_orbit: Optional[ReferenceOrbit] = None
    ref_orbit_center_re: Optional[str] = None  # Center where orbit was computed
    ref_orbit_center_im: Optional[str] = None
    needs_recompute: bool = True

    # Timing info
    last_orbit_time_ms: float = 0.0
    last_render_time_ms: float = 0.0

    # Rendering mode
    using_perturbation: bool = False

    # Zoom behavior: True = zoom at center, False = zoom at cursor
    zoom_to_center: bool = False


# =============================================================================
# Help Text
# =============================================================================

HELP_LINES = [
    "Deep Zoom Mandelbrot Viewer",
    "",
    "Navigation:",
    "  Left drag      Pan view",
    "  Right click    Center on point",
    "  Scroll         Zoom in/out",
    "  C              Toggle zoom at center/cursor",
    "  Arrows         Pan",
    "",
    "Parameters:",
    "  +/-            Max iterations (+/- 100)",
    "  Shift +/-      Max iterations (+/- 1000)",
    "  R              Randomize colors",
    "",
    "Presets:",
    "  1-4            Jump to deep zoom location",
    "  D              Reset to default view",
    "",
    "Display:",
    "  F              Toggle info display",
    "  H/?            Toggle help",
    "  Q/ESC          Quit",
    "",
    "Rendering switches to perturbation",
    f"mode at zoom > 2^{int(PERTURBATION_THRESHOLD)} (~10^12)",
]


# =============================================================================
# Arbitrary-Precision Math Helpers
# =============================================================================

def add_decimal_strings(a: str, b: str, precision: int = 100) -> str:
    """Add two decimal strings with arbitrary precision."""
    getcontext().prec = precision
    result = Decimal(a) + Decimal(b)
    return str(result)


def sub_decimal_strings(a: str, b: str, precision: int = 100) -> str:
    """Subtract two decimal strings with arbitrary precision."""
    getcontext().prec = precision
    result = Decimal(a) - Decimal(b)
    return str(result)


def mul_decimal_string(a: str, factor: float, precision: int = 100) -> str:
    """Multiply decimal string by a float."""
    getcontext().prec = precision
    result = Decimal(a) * Decimal(str(factor))
    return str(result)


# =============================================================================
# Main Viewer Class
# =============================================================================

class DeepZoomViewer:
    """Interactive deep zoom Mandelbrot viewer."""

    def __init__(self, width: int = 1200, height: int = 900):
        self.width = width
        self.height = height
        self.state = DeepZoomState()
        self.drag = DragState()

        # UI state
        self.show_info = True
        self.show_help = False
        self.running = True
        self.needs_render = True

        # Timing
        self.frame_times: list[float] = []
        self.last_fps = 0.0
        self.last_key_time = 0.0

        # Pygame objects
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None

    def run(self):
        """Main entry point."""
        if not self._init_pygame():
            return

        while self.running:
            self._handle_events()
            self._handle_continuous_input()
            self._render_if_needed()
            self.clock.tick(60)

        self._print_stats()
        pygame.quit()

    def _init_pygame(self) -> bool:
        """Initialize pygame and check GPU."""
        print("Checking GPU...")
        if not renderer.has_gpu():
            print("No GPU found!")
            return False
        print(f"GPU: {renderer.get_gpu_name()}")

        pygame.init()
        pygame.key.set_repeat(150, 25)
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Deep Zoom Mandelbrot")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", FONT_SIZE)
        return True

    def _print_stats(self):
        """Print rendering statistics on exit."""
        if self.frame_times:
            avg_ms = sum(self.frame_times) / len(self.frame_times)
            print(f"\nRendered {len(self.frame_times)} frames")
            print(f"Average frame time: {avg_ms:.1f}ms ({1000/avg_ms:.0f} FPS)")

    # =========================================================================
    # Event Handling
    # =========================================================================

    def _handle_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._on_keydown(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._on_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._on_mouse_up(event)
            elif event.type == pygame.MOUSEMOTION:
                self._on_mouse_motion(event)
            elif event.type == pygame.MOUSEWHEEL:
                self._on_mouse_wheel(event)
            elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWRESIZED):
                self.needs_render = True

    def _on_keydown(self, event):
        """Handle key press."""
        key = event.key
        mods = pygame.key.get_mods()

        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.running = False
        elif key == pygame.K_f:
            self.show_info = not self.show_info
            self.needs_render = True
        elif key in (pygame.K_h, pygame.K_QUESTION, pygame.K_SLASH):
            self.show_help = not self.show_help
            self.needs_render = True
        elif key == pygame.K_d:
            self._reset_view()
        elif key == pygame.K_r:
            import random
            self.state.color_seed = random.random()
            self.needs_render = True
        elif key == pygame.K_c:
            self.state.zoom_to_center = not self.state.zoom_to_center
            mode = "center" if self.state.zoom_to_center else "cursor"
            print(f"Zoom mode: {mode}")
            self.needs_render = True
        elif key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
            delta = 1000 if mods & pygame.KMOD_SHIFT else 100
            self.state.imax += delta
            self.state.needs_recompute = True
            self.needs_render = True
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            delta = 1000 if mods & pygame.KMOD_SHIFT else 100
            self.state.imax = max(100, self.state.imax - delta)
            self.state.needs_recompute = True
            self.needs_render = True
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            idx = key - pygame.K_1
            if idx < len(DEEP_ZOOM_LOCATIONS):
                loc = DEEP_ZOOM_LOCATIONS[idx]
                self.state.center_re = loc[0]
                self.state.center_im = loc[1]
                self.state.log2_zoom = loc[2]
                self.state.needs_recompute = True
                self.needs_render = True
                print(f"Jumped to location {idx + 1}: zoom 2^{loc[2]}")

    def _on_mouse_down(self, event):
        """Handle mouse button press - left drag, right center."""
        if event.button == 1:
            # Left click: start drag
            self.drag.active = True
            self.drag.start_pos = event.pos
            self.drag.start_center_re = self.state.center_re
            self.drag.start_center_im = self.state.center_im

        elif event.button == 3:
            # Right click: center on clicked point
            self._center_on_point(event.pos)

    def _center_on_point(self, pos):
        """Center the view on the clicked point."""
        width, height = self.screen.get_size()
        mx, my = pos

        # Convert click position to complex offset from current center
        delta_per_pixel = compute_delta_per_pixel(self.state.log2_zoom, height)
        dx = mx - width / 2
        dy = my - height / 2

        offset_re = dx * delta_per_pixel
        offset_im = -dy * delta_per_pixel  # Negate for standard orientation

        # Add offset to center
        precision = ReferenceOrbit.precision_for_zoom(self.state.log2_zoom)
        self.state.center_re = add_decimal_strings(
            self.state.center_re, str(offset_re), precision
        )
        self.state.center_im = add_decimal_strings(
            self.state.center_im, str(offset_im), precision
        )

        self.state.needs_recompute = True
        self.needs_render = True

    def _on_mouse_up(self, event):
        """Handle mouse button release - end drag."""
        if event.button == 1:
            self.drag.active = False

    def _on_mouse_motion(self, event):
        """Handle mouse movement - pan if dragging."""
        if not self.drag.active or self.drag.start_pos is None:
            return

        width, height = self.screen.get_size()
        dx = event.pos[0] - self.drag.start_pos[0]
        dy = event.pos[1] - self.drag.start_pos[1]

        # Convert pixel delta to complex delta
        delta_per_pixel = compute_delta_per_pixel(self.state.log2_zoom, height)
        offset_re = -dx * delta_per_pixel  # Negative: drag right = move left in complex plane
        offset_im = dy * delta_per_pixel   # Positive: drag down = move up in complex plane

        # Update center from start position (not current position)
        precision = ReferenceOrbit.precision_for_zoom(self.state.log2_zoom)
        self.state.center_re = add_decimal_strings(
            self.drag.start_center_re, str(offset_re), precision
        )
        self.state.center_im = add_decimal_strings(
            self.drag.start_center_im, str(offset_im), precision
        )

        self.state.needs_recompute = True
        self.needs_render = True

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel - zoom at cursor or center."""
        # Apply zoom
        if event.y > 0:
            self.state.log2_zoom += 0.5
        else:
            self.state.log2_zoom = max(0, self.state.log2_zoom - 0.5)

        # If zooming to center, no need to adjust - just zoom
        if not self.state.zoom_to_center:
            # Zoom at cursor: adjust center to keep cursor point fixed
            width, height = self.screen.get_size()
            mx, my = pygame.mouse.get_pos()
            dx = mx - width / 2
            dy = my - height / 2

            # Get cursor offset before and after zoom
            old_zoom = self.state.log2_zoom - (0.5 if event.y > 0 else -0.5)
            old_delta = compute_delta_per_pixel(old_zoom, height)
            new_delta = compute_delta_per_pixel(self.state.log2_zoom, height)

            cursor_offset_re = dx * old_delta
            cursor_offset_im = -dy * old_delta
            new_cursor_offset_re = dx * new_delta
            new_cursor_offset_im = -dy * new_delta

            adjustment_re = cursor_offset_re - new_cursor_offset_re
            adjustment_im = cursor_offset_im - new_cursor_offset_im

            precision = ReferenceOrbit.precision_for_zoom(self.state.log2_zoom)
            self.state.center_re = add_decimal_strings(
                self.state.center_re, str(adjustment_re), precision
            )
            self.state.center_im = add_decimal_strings(
                self.state.center_im, str(adjustment_im), precision
            )

        self.state.needs_recompute = True
        self.needs_render = True

    def _handle_continuous_input(self):
        """Handle held keys for panning."""
        current_time = time.perf_counter()
        if current_time - self.last_key_time < KEY_REPEAT_INTERVAL:
            return

        keys = pygame.key.get_pressed()
        width, height = self.screen.get_size()
        delta_per_pixel = compute_delta_per_pixel(self.state.log2_zoom, height)
        pan_amount = delta_per_pixel * width * PAN_SPEED

        moved = False
        precision = ReferenceOrbit.precision_for_zoom(self.state.log2_zoom)

        if keys[pygame.K_UP]:
            self.state.center_im = add_decimal_strings(
                self.state.center_im, str(pan_amount), precision
            )
            moved = True
        if keys[pygame.K_DOWN]:
            self.state.center_im = sub_decimal_strings(
                self.state.center_im, str(pan_amount), precision
            )
            moved = True
        if keys[pygame.K_LEFT]:
            self.state.center_re = sub_decimal_strings(
                self.state.center_re, str(pan_amount), precision
            )
            moved = True
        if keys[pygame.K_RIGHT]:
            self.state.center_re = add_decimal_strings(
                self.state.center_re, str(pan_amount), precision
            )
            moved = True

        if moved:
            self.state.needs_recompute = True
            self.needs_render = True
            self.last_key_time = current_time

    def _reset_view(self):
        """Reset to default view."""
        self.state.center_re = DEFAULT_CENTER_RE
        self.state.center_im = DEFAULT_CENTER_IM
        self.state.log2_zoom = DEFAULT_LOG2_ZOOM
        self.state.imax = 1000
        self.state.needs_recompute = True
        self.needs_render = True
        print("Reset to default view")

    def _find_best_reference_mojo(self, width: int, height: int) -> tuple[int, int, int]:
        """Find best reference point using Mojo CPU search.

        Uses delta-from-center coordinates which stay accurate at deep zoom.

        Returns:
            Tuple of (px, py, iterations) for the pixel with highest iteration count.
        """
        center_re = float(self.state.center_re)
        center_im = float(self.state.center_im)
        delta_per_pixel = compute_delta_per_pixel(self.state.log2_zoom, height)

        params = (center_re, center_im, width, height, delta_per_pixel, self.state.imax, 8)
        result = renderer.find_best_reference_deep(params)
        return int(result[0]), int(result[1]), int(result[2])

    # =========================================================================
    # Rendering
    # =========================================================================

    def _render_standard(self, width: int, height: int) -> np.ndarray:
        """Render using standard Float64 Mandelbrot (for shallow zooms)."""
        # Convert log2_zoom to half-height parameter
        half_h = 2.0 / (2 ** self.state.log2_zoom)
        aspect = width / height
        half_w = half_h * aspect

        center_re = float(self.state.center_re)
        center_im = float(self.state.center_im)

        params = renderer.ViewParams(
            width=width,
            height=height,
            left=center_re - half_w,
            right=center_re + half_w,
            top=center_im + half_h,
            bottom=center_im - half_h,
            imax=self.state.imax,
            color_seed=self.state.color_seed,
        )
        return renderer.render_mandelbrot(params)

    def _render_perturbation(self, width: int, height: int) -> tuple[np.ndarray, bool]:
        """Render using perturbation theory (for deep zooms).

        Returns:
            Tuple of (rgb_array, used_perturbation). If reference escapes too
            early, falls back to standard rendering and returns False.
        """
        # Recompute reference orbit if needed
        if self.state.needs_recompute:
            t_orbit = time.perf_counter()
            precision = ReferenceOrbit.precision_for_zoom(self.state.log2_zoom)
            delta_per_pixel = compute_delta_per_pixel(self.state.log2_zoom, height)

            # Save previous orbit for perturbation search (before computing new one)
            prev_orbit = self.state.ref_orbit
            prev_orbit_center_re = self.state.ref_orbit_center_re
            prev_orbit_center_im = self.state.ref_orbit_center_im

            # First try the center point
            self.state.ref_orbit = ReferenceOrbit(
                self.state.center_re,
                self.state.center_im,
                precision
            )
            self.state.ref_orbit.compute(self.state.imax)

            # If center escapes, search for a better reference point
            if self.state.ref_orbit.escaped:
                original_orbit_length = self.state.ref_orbit.orbit_length
                print(f"Reference escapes at {original_orbit_length} iters, searching...")

                found_better = False

                # Try perturbation search if we have a previous orbit
                if (prev_orbit is not None and
                    prev_orbit_center_re is not None and
                    prev_orbit_center_im is not None and
                    prev_orbit.orbit_length > 50):  # Only use if previous orbit was decent
                    print(f"Using perturbation search with previous orbit ({prev_orbit.orbit_length} pts)...")

                    # Compute offset from OLD reference center to NEW view center
                    center_offset_re = float(sub_decimal_strings(
                        self.state.center_re,
                        prev_orbit_center_re,
                        precision
                    ))
                    center_offset_im = float(sub_decimal_strings(
                        self.state.center_im,
                        prev_orbit_center_im,
                        precision
                    ))
                    print(f"  Center offset: ({center_offset_re:.6e}, {center_offset_im:.6e})")
                    print(f"  Delta per pixel: {delta_per_pixel:.6e}")

                    # Get previous orbit arrays
                    prev_re, prev_im = prev_orbit.get_orbit_arrays()

                    # Search using perturbation
                    params = (center_offset_re, center_offset_im, width, height, delta_per_pixel, self.state.imax, 8)
                    best_px, best_py, best_iters = renderer.find_best_reference_perturbation(params, prev_re, prev_im)
                    print(f"Perturbation search found: pixel ({best_px}, {best_py}) with {best_iters} iters")

                    if best_iters > original_orbit_length:
                        # Convert pixel to arbitrary precision coordinates
                        dx = best_px - width / 2
                        dy = best_py - height / 2
                        offset_re = dx * delta_per_pixel
                        offset_im = -dy * delta_per_pixel

                        best_re = add_decimal_strings(self.state.center_re, str(offset_re), precision)
                        best_im = add_decimal_strings(self.state.center_im, str(offset_im), precision)

                        candidate_orbit = ReferenceOrbit(best_re, best_im, precision)
                        candidate_orbit.compute(self.state.imax)

                        if candidate_orbit.orbit_length > original_orbit_length:
                            print(f"Verified: {candidate_orbit.orbit_length} iters, escaped={candidate_orbit.escaped}")
                            self.state.ref_orbit = candidate_orbit
                            found_better = True

                # Fallback to Mojo CPU search (Float64, may lose precision)
                if not found_better:
                    best_px, best_py, best_iters = self._find_best_reference_mojo(width, height)
                    print(f"Mojo CPU search found: {best_iters} iters")

                    if best_iters > original_orbit_length:
                        dx = best_px - width / 2
                        dy = best_py - height / 2
                        offset_re = dx * delta_per_pixel
                        offset_im = -dy * delta_per_pixel

                        best_re = add_decimal_strings(self.state.center_re, str(offset_re), precision)
                        best_im = add_decimal_strings(self.state.center_im, str(offset_im), precision)

                        candidate_orbit = ReferenceOrbit(best_re, best_im, precision)
                        candidate_orbit.compute(self.state.imax)

                        if candidate_orbit.orbit_length > original_orbit_length:
                            print(f"Verified: {candidate_orbit.orbit_length} iters")
                            self.state.ref_orbit = candidate_orbit
                            found_better = True

                # Final fallback: arbitrary precision search (slow but accurate)
                if not found_better or self.state.ref_orbit.escaped:
                    print("Trying arbitrary precision search...")
                    best_orbit = self.state.ref_orbit

                    for gy in range(8):
                        for gx in range(8):
                            px = (gx / 7 - 0.5) * width
                            py = (gy / 7 - 0.5) * height
                            offset_re = px * delta_per_pixel
                            offset_im = -py * delta_per_pixel

                            sample_re = add_decimal_strings(self.state.center_re, str(offset_re), precision)
                            sample_im = add_decimal_strings(self.state.center_im, str(offset_im), precision)

                            candidate = ReferenceOrbit(sample_re, sample_im, precision)
                            candidate.compute(self.state.imax)

                            if candidate.orbit_length > best_orbit.orbit_length:
                                best_orbit = candidate
                                if not candidate.escaped:
                                    print(f"Found non-escaping at grid ({gx}, {gy})")
                                    break
                        else:
                            continue
                        break

                    if best_orbit.orbit_length > self.state.ref_orbit.orbit_length:
                        self.state.ref_orbit = best_orbit
                        print(f"Using: {best_orbit.orbit_length} iters, escaped={best_orbit.escaped}")

            # Store the actual reference orbit center (may differ from view center if search found better point)
            self.state.ref_orbit_center_re = self.state.ref_orbit.center_re_str
            self.state.ref_orbit_center_im = self.state.ref_orbit.center_im_str

            self.state.last_orbit_time_ms = (time.perf_counter() - t_orbit) * 1000
            self.state.needs_recompute = False

        # If still escaping too early even after search, fall back to standard
        # Only fall back if orbit is really short (< 20 iterations)
        if self.state.ref_orbit.escaped and self.state.ref_orbit.orbit_length < 20:
            return self._render_standard(width, height), False

        ref_re, ref_im = self.state.ref_orbit.get_orbit_arrays()
        delta_per_pixel = compute_delta_per_pixel(self.state.log2_zoom, height)

        # Compute offset from view center to orbit center
        precision = ReferenceOrbit.precision_for_zoom(self.state.log2_zoom)
        ref_offset_re = float(sub_decimal_strings(
            self.state.ref_orbit.center_re_str,
            self.state.center_re,
            precision
        ))
        ref_offset_im = float(sub_decimal_strings(
            self.state.ref_orbit.center_im_str,
            self.state.center_im,
            precision
        ))

        params = renderer.DeepZoomParams(
            width=width,
            height=height,
            delta_per_pixel=delta_per_pixel,
            ref_offset_re=ref_offset_re,
            ref_offset_im=ref_offset_im,
            imax=self.state.imax,
            color_seed=self.state.color_seed,
        )
        return renderer.render_mandelbrot_deep(params, ref_re, ref_im), True

    def _render_if_needed(self):
        """Render frame if needed."""
        if not self.needs_render:
            return

        t0 = time.perf_counter()

        # Get current window size
        self.screen = pygame.display.get_surface()
        width, height = self.screen.get_size()

        # Choose rendering method based on zoom level
        # Use standard rendering for shallow zooms (faster, works everywhere)
        # Use perturbation for deep zooms (needed beyond Float64 limits)
        want_perturbation = self.state.log2_zoom >= PERTURBATION_THRESHOLD

        t_render = time.perf_counter()
        if want_perturbation:
            rgb_array, actually_used_perturbation = self._render_perturbation(width, height)
            self.state.using_perturbation = actually_used_perturbation
        else:
            self.state.last_orbit_time_ms = 0.0  # No orbit computation needed
            rgb_array = self._render_standard(width, height)
            self.state.using_perturbation = False
        self.state.last_render_time_ms = (time.perf_counter() - t_render) * 1000

        # Display
        surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        self.last_fps = 1000 / total_ms if total_ms > 0 else 0

        # Draw overlays
        if self.show_info:
            self._draw_info_overlay(width)
        if self.show_help:
            self._draw_help_overlay()

        pygame.display.flip()

        self.frame_times.append(total_ms)
        self.needs_render = False
        gc.collect()

    def _draw_info_overlay(self, width: int):
        """Draw info overlay with zoom level and timing."""
        y = PADDING // 2

        # Format zoom level nicely
        zoom_factor = 2 ** self.state.log2_zoom
        if self.state.log2_zoom < 10:
            zoom_str = f"{zoom_factor:.0f}x"
        else:
            # Express as 10^N for very large zooms
            log10_zoom = self.state.log2_zoom * 0.301  # log10(2) ≈ 0.301
            zoom_str = f"10^{log10_zoom:.1f}"

        # Rendering mode
        if self.state.using_perturbation:
            mode_str = "perturbation"
        elif self.state.log2_zoom >= PERTURBATION_THRESHOLD:
            mode_str = "standard (fallback)"  # Would use perturbation but reference escaped
        else:
            mode_str = "standard"

        # Zoom target indicator
        zoom_target = "[C]" if self.state.zoom_to_center else ""

        # Main info line
        line1 = f"Zoom: 2^{self.state.log2_zoom:.1f} ({zoom_str}) {zoom_target}| Mode: {mode_str} | imax: {self.state.imax}"
        self._draw_text(line1, PADDING, y)
        y += self.font.get_linesize()

        # Timing info
        if self.state.using_perturbation:
            orbit_len = self.state.ref_orbit.orbit_length if self.state.ref_orbit else 0
            line2 = f"Orbit: {self.state.last_orbit_time_ms:.0f}ms ({orbit_len} pts) | Render: {self.state.last_render_time_ms:.0f}ms | {self.last_fps:.0f} FPS"
        else:
            line2 = f"Render: {self.state.last_render_time_ms:.0f}ms | {self.last_fps:.0f} FPS"
        self._draw_text(line2, PADDING, y)
        y += self.font.get_linesize()

        # Center coordinates (truncated for display)
        center_re_display = self.state.center_re[:30] + "..." if len(self.state.center_re) > 30 else self.state.center_re
        center_im_display = self.state.center_im[:30] + "..." if len(self.state.center_im) > 30 else self.state.center_im
        line3 = f"Center: {center_re_display} + {center_im_display}i"
        self._draw_text(line3, PADDING, y)

    def _draw_text(self, text: str, x: int, y: int, color=(255, 255, 255)):
        """Render text with background."""
        surf = self.font.render(text, True, color, (0, 0, 0))
        self.screen.blit(surf, (x, y))

    def _draw_help_overlay(self):
        """Draw help overlay."""
        line_height = self.font.get_linesize()
        help_width = max(self.font.size(line)[0] for line in HELP_LINES) + PADDING * 2
        help_height = len(HELP_LINES) * line_height + PADDING * 2

        # Semi-transparent background
        help_bg = pygame.Surface((help_width, help_height))
        help_bg.set_alpha(HELP_OVERLAY_ALPHA)
        help_bg.fill((0, 0, 0))
        help_y = line_height * 3 + PADDING
        self.screen.blit(help_bg, (PADDING, help_y))

        # Render help text
        for i, line in enumerate(HELP_LINES):
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (PADDING * 2, help_y + PADDING + i * line_height))


# =============================================================================
# Entry Point
# =============================================================================

def main():
    viewer = DeepZoomViewer()
    viewer.run()


if __name__ == "__main__":
    main()
