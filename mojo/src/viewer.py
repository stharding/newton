#!/usr/bin/env python3
"""Fractal interactive viewer - Python frontend with Mojo GPU backend."""

import os
import sys
import time

# Force X11 backend for proper window decorations on Wayland
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

# Import Mojo module
import mojo.importer
import newton_renderer

import pygame
import numpy as np


# Fractal types
NEWTON = "newton"
MANDELBROT = "mandelbrot"
JULIA = "julia"
BURNING_SHIP = "burning_ship"
TRICORN = "tricorn"

# Preset Julia constants (interesting values)
JULIA_PRESETS = [
    (-0.7, 0.27015),      # Classic
    (-0.8, 0.156),        # Spiral
    (-0.4, 0.6),          # Dendrite
    (0.285, 0.01),        # Galaxy
    (-0.70176, -0.3842),  # Lightning
    (0.355, 0.355),       # Rabbit
    (-0.54, 0.54),        # Dragon
]


def main():
    # Configuration
    width, height = 1600, 1200
    coefficients = [1.0, 0.0, 0.0, -1.0]  # z^3 - 1
    tolerance = 0.0001
    imax = 30

    # Fractal state
    fractal_type = NEWTON
    julia_preset_idx = 0
    julia_c = JULIA_PRESETS[julia_preset_idx]
    julia_power = (2.0, 0.0)  # Complex exponent (real, imaginary)

    # Saved view states for switching between fractals
    saved_views = {
        NEWTON: {"center": (0.0, 0.0), "zoom": 1.0},
        MANDELBROT: {"center": (-0.5, 0.0), "zoom": 1.5},
        JULIA: {"center": (0.0, 0.0), "zoom": 1.5},
        BURNING_SHIP: {"center": (-0.4, -0.5), "zoom": 1.5},
        TRICORN: {"center": (-0.3, 0.0), "zoom": 1.5},
    }

    # Parse command line args
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "mandelbrot":
            fractal_type = MANDELBROT
        elif arg == "julia":
            fractal_type = JULIA
        elif arg == "newton":
            fractal_type = NEWTON
        else:
            coefficients = [float(c) for c in arg.split(",")]

    # Check GPU
    print("Checking GPU...")
    if not newton_renderer.has_gpu():
        print("No GPU found!")
        return
    print(f"GPU: {newton_renderer.get_gpu_name()}")

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("Newton Fractal")
    clock = pygame.time.Clock()

    # View state
    center_re, center_im = 0.0, 0.0
    zoom = 1.0  # 1.0 = view from -1 to 1 on the imaginary axis
    aspect = width / height

    # Control speeds
    pan_speed = 0.02
    zoom_factor = 1.1

    running = True
    needs_render = True
    frame_times = []
    last_fps = 0
    last_render_ms = 0
    show_fps = True
    show_help = False
    color_seed = 0.0

    # Font for on-screen display
    font_size = 28
    font = pygame.font.SysFont("monospace", font_size)
    line_height = font.get_linesize()
    padding = 10

    # Help text
    help_lines = [
        "Keybindings:",
        "",
        "Navigation:",
        "  Drag           Pan",
        "  Scroll         Zoom (at cursor)",
        "  Arrows         Pan",
        "",
        "Fractal type:",
        "  N              Newton fractal",
        "  M              Mandelbrot set",
        "  J              Julia set",
        "  B              Burning Ship",
        "  T              Tricorn (Mandelbar)",
        "  [ / ]          Cycle Julia presets",
        "  D              Reset Julia defaults",
        "",
        "Modifiers:",
        "  Shift+drag     Adjust Julia c",
        "  Alt+drag       Adjust exponent",
        "  Shift+click    Mandelbrot -> Julia",
        "",
        "Parameters:",
        "  2-9            Newton: z^n - 1",
        "  + / -          Max iterations (shift=10x)",
        "  R              Randomize colors",
        "",
        "Display:",
        "  F              Toggle FPS",
        "  H / ?          This help",
        "  0              Reset view",
        "  Q / ESC        Quit",
    ]

    # Mouse drag state
    dragging = False
    drag_start_pos = None
    drag_start_center = None

    while running:
        # Event handling (native Python - fast!)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_f:
                    # Toggle FPS display
                    show_fps = not show_fps
                    needs_render = True
                elif event.key in (pygame.K_h, pygame.K_QUESTION, pygame.K_SLASH):
                    # Toggle help overlay
                    show_help = not show_help
                    needs_render = True
                elif event.key == pygame.K_r:
                    # Randomize root colors
                    import random

                    color_seed = random.random()
                    needs_render = True
                elif event.key == pygame.K_0:
                    # Reset view
                    center_re, center_im = 0.0, 0.0
                    zoom = 1.0
                    needs_render = True
                elif event.key in (
                    pygame.K_2,
                    pygame.K_3,
                    pygame.K_4,
                    pygame.K_5,
                    pygame.K_6,
                    pygame.K_7,
                    pygame.K_8,
                    pygame.K_9,
                ):
                    # Change polynomial degree: z^n - 1
                    n = event.key - pygame.K_0
                    coefficients = [1.0] + [0.0] * (n - 1) + [-1.0]
                    print(f"Polynomial: z^{n} - 1")
                    needs_render = True
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    # Increase max iterations (shift for 10x)
                    delta = 100 if event.mod & pygame.KMOD_SHIFT else 10
                    imax += delta
                    print(f"Max iterations: {imax}")
                    needs_render = True
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    # Decrease max iterations (shift for 10x)
                    delta = 100 if event.mod & pygame.KMOD_SHIFT else 10
                    imax = max(imax - delta, 10)
                    print(f"Max iterations: {imax}")
                    needs_render = True
                elif event.key == pygame.K_n:
                    # Save current view, switch to Newton
                    saved_views[fractal_type] = {"center": (center_re, center_im), "zoom": zoom}
                    fractal_type = NEWTON
                    center_re, center_im = saved_views[NEWTON]["center"]
                    zoom = saved_views[NEWTON]["zoom"]
                    print("Fractal: Newton")
                    needs_render = True
                elif event.key == pygame.K_m:
                    # Save current view, switch to Mandelbrot
                    saved_views[fractal_type] = {"center": (center_re, center_im), "zoom": zoom}
                    fractal_type = MANDELBROT
                    center_re, center_im = saved_views[MANDELBROT]["center"]
                    zoom = saved_views[MANDELBROT]["zoom"]
                    print("Fractal: Mandelbrot")
                    needs_render = True
                elif event.key == pygame.K_j:
                    # Save current view, switch to Julia
                    saved_views[fractal_type] = {"center": (center_re, center_im), "zoom": zoom}
                    fractal_type = JULIA
                    center_re, center_im = saved_views[JULIA]["center"]
                    zoom = saved_views[JULIA]["zoom"]
                    print(f"Fractal: Julia (c = {julia_c[0]:.3f} + {julia_c[1]:.3f}i)")
                    needs_render = True
                elif event.key == pygame.K_b:
                    # Save current view, switch to Burning Ship
                    saved_views[fractal_type] = {"center": (center_re, center_im), "zoom": zoom}
                    fractal_type = BURNING_SHIP
                    center_re, center_im = saved_views[BURNING_SHIP]["center"]
                    zoom = saved_views[BURNING_SHIP]["zoom"]
                    print("Fractal: Burning Ship")
                    needs_render = True
                elif event.key == pygame.K_t:
                    # Save current view, switch to Tricorn
                    saved_views[fractal_type] = {"center": (center_re, center_im), "zoom": zoom}
                    fractal_type = TRICORN
                    center_re, center_im = saved_views[TRICORN]["center"]
                    zoom = saved_views[TRICORN]["zoom"]
                    print("Fractal: Tricorn")
                    needs_render = True
                elif event.key == pygame.K_RIGHTBRACKET:
                    # Next Julia preset
                    julia_preset_idx = (julia_preset_idx + 1) % len(JULIA_PRESETS)
                    julia_c = JULIA_PRESETS[julia_preset_idx]
                    print(f"Julia preset {julia_preset_idx + 1}: c = {julia_c[0]:.3f} + {julia_c[1]:.3f}i")
                    needs_render = True
                elif event.key == pygame.K_LEFTBRACKET:
                    # Previous Julia preset
                    julia_preset_idx = (julia_preset_idx - 1) % len(JULIA_PRESETS)
                    julia_c = JULIA_PRESETS[julia_preset_idx]
                    print(f"Julia preset {julia_preset_idx + 1}: c = {julia_c[0]:.3f} + {julia_c[1]:.3f}i")
                    needs_render = True
                elif event.key == pygame.K_d:
                    # Reset Julia to defaults
                    julia_preset_idx = 0
                    julia_c = JULIA_PRESETS[julia_preset_idx]
                    julia_power = (2.0, 0.0)
                    if fractal_type == JULIA:
                        center_re, center_im = 0.0, 0.0
                        zoom = 1.5
                    print("Julia reset to defaults")
                    needs_render = True
            elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWRESIZED):
                needs_render = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mods = pygame.key.get_mods()
                    if (mods & pygame.KMOD_SHIFT) and fractal_type == MANDELBROT:
                        # Shift+click on Mandelbrot: jump to Julia at that c
                        mouse_x, mouse_y = event.pos
                        half_w = zoom * aspect
                        half_h = zoom
                        click_re = center_re + (mouse_x / width - 0.5) * 2 * half_w
                        click_im = center_im - (mouse_y / height - 0.5) * 2 * half_h
                        julia_c = (click_re, click_im)
                        # Save Mandelbrot view before switching
                        saved_views[MANDELBROT] = {"center": (center_re, center_im), "zoom": zoom}
                        fractal_type = JULIA
                        center_re, center_im = saved_views[JULIA]["center"]
                        zoom = saved_views[JULIA]["zoom"]
                        print(f"Julia at c = {julia_c[0]:.4f} + {julia_c[1]:.4f}i")
                        needs_render = True
                    else:
                        dragging = True
                        drag_start_pos = event.pos
                        drag_start_center = (center_re, center_im)
                        drag_start_julia_c = julia_c
                        drag_start_power = julia_power
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging and drag_start_pos is not None:
                    dx = event.pos[0] - drag_start_pos[0]
                    dy = event.pos[1] - drag_start_pos[1]
                    mods = pygame.key.get_mods()

                    if mods & pygame.KMOD_SHIFT:
                        # Shift+drag: modify Julia c parameter
                        julia_c = (
                            drag_start_julia_c[0] + dx * 0.005,
                            drag_start_julia_c[1] - dy * 0.005,
                        )
                        if fractal_type != JULIA:
                            fractal_type = JULIA
                            center_re, center_im = 0.0, 0.0
                            zoom = 1.5
                        needs_render = True
                    elif mods & (pygame.KMOD_ALT | pygame.KMOD_CTRL):
                        # Alt/Ctrl+drag: adjust complex power exponent
                        # X-axis = real part, Y-axis = imaginary part
                        julia_power = (
                            max(1.5, min(10.0, drag_start_power[0] + dx * 0.02)),
                            drag_start_power[1] - dy * 0.02,
                        )
                        if fractal_type != JULIA:
                            saved_views[fractal_type] = {"center": (center_re, center_im), "zoom": zoom}
                            fractal_type = JULIA
                            center_re, center_im = saved_views[JULIA]["center"]
                            zoom = saved_views[JULIA]["zoom"]
                        needs_render = True
                    else:
                        # Normal drag: pan
                        re_per_pixel = (zoom * aspect * 2) / width
                        im_per_pixel = (zoom * 2) / height
                        center_re = drag_start_center[0] - dx * re_per_pixel
                        center_im = drag_start_center[1] + dy * im_per_pixel
                        needs_render = True
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom with scroll wheel (zoom towards mouse position)
                mouse_x, mouse_y = pygame.mouse.get_pos()
                # Convert mouse to complex coordinates before zoom
                half_w = zoom * aspect
                half_h = zoom
                mouse_re = center_re + (mouse_x / width - 0.5) * 2 * half_w
                mouse_im = center_im - (mouse_y / height - 0.5) * 2 * half_h

                if event.y > 0:
                    zoom /= zoom_factor
                else:
                    zoom *= zoom_factor

                # Adjust center to keep mouse point fixed
                half_w = zoom * aspect
                half_h = zoom
                new_mouse_re = center_re + (mouse_x / width - 0.5) * 2 * half_w
                new_mouse_im = center_im - (mouse_y / height - 0.5) * 2 * half_h
                center_re += mouse_re - new_mouse_re
                center_im += mouse_im - new_mouse_im

                needs_render = True

        # Continuous key input for smooth panning/zooming
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            center_im += pan_speed * zoom
            needs_render = True
        if keys[pygame.K_DOWN]:
            center_im -= pan_speed * zoom
            needs_render = True
        if keys[pygame.K_LEFT]:
            center_re -= pan_speed * zoom
            needs_render = True
        if keys[pygame.K_RIGHT]:
            center_re += pan_speed * zoom
            needs_render = True

        # Render if needed
        if needs_render:
            t0 = time.perf_counter()

            # Get actual current window size
            screen = pygame.display.get_surface()
            width, height = screen.get_size()
            aspect = width / height

            # Calculate window bounds
            half_w = zoom * aspect
            half_h = zoom
            left = center_re - half_w
            right = center_re + half_w
            top = center_im + half_h
            bottom = center_im - half_h

            # Single Mojo call for GPU rendering
            if fractal_type == NEWTON:
                rgb_array = newton_renderer.render_newton(
                    (
                        width,
                        height,
                        coefficients,
                        left,
                        right,
                        top,
                        bottom,
                        tolerance,
                        imax,
                        color_seed,
                    )
                )
            elif fractal_type == MANDELBROT:
                rgb_array = newton_renderer.render_mandelbrot(
                    (
                        width,
                        height,
                        left,
                        right,
                        top,
                        bottom,
                        imax,
                        color_seed,
                    )
                )
            elif fractal_type == JULIA:
                rgb_array = newton_renderer.render_julia(
                    (
                        width,
                        height,
                        left,
                        right,
                        top,
                        bottom,
                        julia_c[0],
                        julia_c[1],
                        julia_power[0],
                        julia_power[1],
                        imax,
                        color_seed,
                    )
                )
            elif fractal_type == BURNING_SHIP:
                rgb_array = newton_renderer.render_burning_ship(
                    (
                        width,
                        height,
                        left,
                        right,
                        top,
                        bottom,
                        imax,
                        color_seed,
                    )
                )
            elif fractal_type == TRICORN:
                rgb_array = newton_renderer.render_tricorn(
                    (
                        width,
                        height,
                        left,
                        right,
                        top,
                        bottom,
                        imax,
                        color_seed,
                    )
                )

            t1 = time.perf_counter()
            render_ms = (t1 - t0) * 1000

            # Display with pygame (need to swap axes for pygame)
            surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
            screen.blit(surface, (0, 0))

            t2 = time.perf_counter()
            total_ms = (t2 - t0) * 1000
            last_fps = 1000 / total_ms if total_ms > 0 else 0
            last_render_ms = render_ms

            # Draw FPS overlay if enabled
            if show_fps:
                if fractal_type == JULIA:
                    if julia_power[1] == 0:
                        power_str = f"z^{julia_power[0]:.2f}"
                    else:
                        power_str = f"z^({julia_power[0]:.2f}{julia_power[1]:+.2f}i)"
                    fps_text = f"{fractal_type} c={julia_c[0]:.3f}{julia_c[1]:+.3f}i {power_str} | {last_render_ms:.1f}ms | {last_fps:.0f} FPS"
                else:
                    fps_text = f"{fractal_type} | {last_render_ms:.1f}ms | {last_fps:.0f} FPS | zoom: {zoom:.2e} | imax: {imax}"
                fps_surface = font.render(fps_text, True, (255, 255, 255), (0, 0, 0))
                screen.blit(fps_surface, (padding, padding // 2))

            # Draw help overlay if enabled
            if show_help:
                # Calculate dimensions from content
                help_width = (
                    max(font.size(line)[0] for line in help_lines) + padding * 2
                )
                help_height = len(help_lines) * line_height + padding * 2
                # Semi-transparent background
                help_bg = pygame.Surface((help_width, help_height))
                help_bg.set_alpha(200)
                help_bg.fill((0, 0, 0))
                help_y = line_height + padding
                screen.blit(help_bg, (padding, help_y))
                # Render help text
                for i, line in enumerate(help_lines):
                    text_surface = font.render(line, True, (255, 255, 255))
                    screen.blit(
                        text_surface, (padding * 2, help_y + padding + i * line_height)
                    )

            pygame.display.flip()

            frame_times.append(total_ms)
            needs_render = False

        # Cap at 60 FPS when idle
        clock.tick(60)

    # Print stats
    if frame_times:
        avg_ms = sum(frame_times) / len(frame_times)
        print(f"\nRendered {len(frame_times)} frames")
        print(f"Average frame time: {avg_ms:.1f}ms ({1000/avg_ms:.0f} FPS)")

    pygame.quit()


if __name__ == "__main__":
    main()
