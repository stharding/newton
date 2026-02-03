#!/usr/bin/env python3
"""Fractal interactive viewer - Python frontend with Mojo GPU backend."""

from dataclasses import dataclass, field
from typing import Callable, Optional
import gc
import math
import os
import random
import sys
import time

# Force X11 backend for proper window decorations on Wayland
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

import mojo.importer
import renderer

import pygame
import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Fractal types
NEWTON = "newton"
MANDELBROT = "mandelbrot"
JULIA = "julia"
BURNING_SHIP = "burning_ship"
TRICORN = "tricorn"
MANDELBULB = "mandelbulb"

# 3D fractals use different control scheme
FRACTALS_3D = {MANDELBULB}

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

# Control constants
PAN_SPEED = 0.02
ZOOM_FACTOR = 1.1
MOVE_SPEED_3D = 0.05
LOOK_SPEED_3D = 0.003
MOUSE_SENSITIVITY = 0.005
POWER_ADJUST_SENSITIVITY = 0.02
KEY_REPEAT_INTERVAL = 0.1

# Display constants
FONT_SIZE = 28
PADDING = 10
HELP_OVERLAY_ALPHA = 200


# =============================================================================
# State Dataclasses
# =============================================================================

@dataclass
class View2D:
    """2D fractal view state."""
    center_re: float = 0.0
    center_im: float = 0.0
    zoom: float = 1.0


@dataclass
class Camera3D:
    """3D fractal camera state."""
    x: float = 0.0
    y: float = 0.0
    z: float = -2.5
    yaw: float = 0.0
    pitch: float = 0.0

    def get_vectors(self):
        """Compute forward, right, and up direction vectors from yaw/pitch."""
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        cos_pitch = math.cos(self.pitch)
        sin_pitch = math.sin(self.pitch)

        forward = (cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw)
        right = (cos_yaw, 0, -sin_yaw)
        up = (-sin_pitch * sin_yaw, cos_pitch, -sin_pitch * cos_yaw)
        return forward, right, up

    def move_forward(self, amount: float):
        forward, _, _ = self.get_vectors()
        self.x += forward[0] * amount
        self.y += forward[1] * amount
        self.z += forward[2] * amount

    def move_right(self, amount: float):
        _, right, _ = self.get_vectors()
        self.x += right[0] * amount
        self.z += right[2] * amount

    def move_up(self, amount: float):
        self.y += amount

    def reset(self):
        self.x, self.y, self.z = 0.0, 0.0, -2.5
        self.yaw, self.pitch = 0.0, 0.0


@dataclass
class JuliaParams:
    """Julia set parameters."""
    c_re: float = -0.7
    c_im: float = 0.27015
    power_re: float = 2.0
    power_im: float = 0.0
    preset_idx: int = 0

    def next_preset(self):
        self.preset_idx = (self.preset_idx + 1) % len(JULIA_PRESETS)
        self.c_re, self.c_im = JULIA_PRESETS[self.preset_idx]

    def prev_preset(self):
        self.preset_idx = (self.preset_idx - 1) % len(JULIA_PRESETS)
        self.c_re, self.c_im = JULIA_PRESETS[self.preset_idx]

    def reset(self):
        self.preset_idx = 0
        self.c_re, self.c_im = JULIA_PRESETS[0]
        self.power_re, self.power_im = 2.0, 0.0


@dataclass
class DragState:
    """Mouse drag state."""
    active: bool = False
    start_pos: Optional[tuple] = None
    start_view: Optional[View2D] = None
    start_camera: Optional[Camera3D] = None
    start_julia_c: Optional[tuple] = None
    start_power: Optional[tuple] = None


@dataclass
class EditState:
    """Text input state for editable values."""
    active_field: Optional[str] = None  # None, "c_re", "c_im", "power_re", "power_im"
    text: str = ""
    field_rects: dict = field(default_factory=dict)

    FIELDS = ["c_re", "c_im", "power_re", "power_im"]

    def start_editing(self, field_name: str, initial_text: str):
        self.active_field = field_name
        self.text = initial_text
        pygame.key.start_text_input()

    def stop_editing(self):
        self.active_field = None
        self.text = ""
        pygame.key.stop_text_input()

    def next_field(self, reverse: bool = False):
        if self.active_field is None:
            return None
        idx = self.FIELDS.index(self.active_field)
        idx = (idx - 1 if reverse else idx + 1) % len(self.FIELDS)
        return self.FIELDS[idx]


# =============================================================================
# Help Text
# =============================================================================

HELP_LINES = [
    "Keybindings:",
    "",
    "2D Navigation:",
    "  Drag           Pan",
    "  Scroll         Zoom (at cursor)",
    "  Arrows         Pan",
    "",
    "3D Navigation (Mandelbulb):",
    "  Up/Down        Move forward/back",
    "  Left/Right     Strafe left/right",
    "  Space/Ctrl     Move up/down",
    "  Drag           Look around",
    "  Shift+drag     Pan (no rotation)",
    "  Scroll         Move forward/back",
    "",
    "Fractal type:",
    "  N              Newton fractal",
    "  M              Mandelbrot set",
    "  J              Julia set",
    "  B              Burning Ship",
    "  T              Tricorn (Mandelbar)",
    "  P              Mandelbulb (3D)",
    "  [ / ]          Cycle presets/power",
    "  D              Reset defaults",
    "",
    "Modifiers:",
    "  Shift+drag     Adjust Julia c",
    "  Alt+drag       Adjust exponent",
    "  Shift+click    Mandelbrot -> Julia",
    "  Click values   Edit Julia params (Tab=next)",
    "",
    "Parameters:",
    "  2-9            Power (Julia: z^n, Newton: z^n-1)",
    "  + / -          Max iterations (shift=10x)",
    "  G / Shift+G    Glow intensity +/- (Newton)",
    "  R              Randomize colors",
    "",
    "Display:",
    "  F              Toggle FPS",
    "  H / ?          This help",
    "  0              Reset view",
    "  Q / ESC        Quit",
]


# =============================================================================
# Renderer Dispatch
# =============================================================================

class FractalRenderer:
    """Handles GPU rendering dispatch for different fractal types."""

    @staticmethod
    def render(fractal_type: str, width: int, height: int,
               view: View2D, camera: Camera3D, julia: JuliaParams,
               coefficients: list, tolerance: float, imax: int,
               color_seed: float, glow_intensity: float,
               mandelbulb_power: float) -> np.ndarray:
        """Render the appropriate fractal type and return RGB array."""
        aspect = width / height
        half_w = view.zoom * aspect
        half_h = view.zoom
        left = view.center_re - half_w
        right = view.center_re + half_w
        top = view.center_im + half_h
        bottom = view.center_im - half_h

        if fractal_type == NEWTON:
            params = renderer.NewtonParams(
                width=width, height=height,
                left=left, right=right, top=top, bottom=bottom,
                tolerance=tolerance, imax=imax, color_seed=color_seed,
                glow_intensity=glow_intensity, zoom=view.zoom,
            )
            return renderer.render_newton(params, coefficients)
        elif fractal_type == MANDELBROT:
            params = renderer.ViewParams(
                width=width, height=height,
                left=left, right=right, top=top, bottom=bottom,
                imax=imax, color_seed=color_seed,
            )
            return renderer.render_mandelbrot(params)
        elif fractal_type == JULIA:
            params = renderer.JuliaParams(
                width=width, height=height,
                left=left, right=right, top=top, bottom=bottom,
                c_re=julia.c_re, c_im=julia.c_im,
                power_re=julia.power_re, power_im=julia.power_im,
                imax=imax, color_seed=color_seed,
            )
            return renderer.render_julia(params)
        elif fractal_type == BURNING_SHIP:
            params = renderer.ViewParams(
                width=width, height=height,
                left=left, right=right, top=top, bottom=bottom,
                imax=imax, color_seed=color_seed,
            )
            return renderer.render_burning_ship(params)
        elif fractal_type == TRICORN:
            params = renderer.ViewParams(
                width=width, height=height,
                left=left, right=right, top=top, bottom=bottom,
                imax=imax, color_seed=color_seed,
            )
            return renderer.render_tricorn(params)
        elif fractal_type == MANDELBULB:
            params = renderer.MandelbulbParams(
                width=width, height=height,
                cam_x=camera.x, cam_y=camera.y, cam_z=camera.z,
                cam_yaw=camera.yaw, cam_pitch=camera.pitch,
                power=mandelbulb_power, imax=imax, color_seed=color_seed,
            )
            return renderer.render_mandelbulb(params)
        else:
            raise ValueError(f"Unknown fractal type: {fractal_type}")


# =============================================================================
# Main Viewer Class
# =============================================================================

class FractalViewer:
    """Interactive fractal viewer with pygame UI and Mojo GPU backend."""

    def __init__(self, width: int = 1600, height: int = 1200):
        self.width = width
        self.height = height

        # Fractal parameters
        self.fractal_type = NEWTON
        self.coefficients = [1.0, 0.0, 0.0, -1.0]  # z^3 - 1
        self.tolerance = 0.0001
        self.imax = 30
        self.color_seed = 0.0
        self.glow_intensity = 1.0
        self.mandelbulb_power = 8.0

        # View state
        self.view = View2D()
        self.camera = Camera3D()
        self.julia = JuliaParams()

        # Saved views for switching between fractals
        self.saved_views = {
            NEWTON: View2D(0.0, 0.0, 1.0),
            MANDELBROT: View2D(-0.5, 0.0, 1.5),
            JULIA: View2D(0.0, 0.0, 1.5),
            BURNING_SHIP: View2D(-0.4, -0.5, 1.5),
            TRICORN: View2D(-0.3, 0.0, 1.5),
        }
        self.saved_cameras = {
            MANDELBULB: Camera3D(0.0, 0.0, -2.5, 0.0, 0.0),
        }

        # UI state
        self.drag = DragState()
        self.edit = EditState()
        self.show_fps = True
        self.show_help = False
        self.needs_render = True
        self.running = True

        # Timing
        self.frame_times = []
        self.last_fps = 0
        self.last_render_ms = 0
        self.last_key_repeat_time = 0.0

        # Pygame objects (initialized in run())
        self.screen = None
        self.clock = None
        self.font = None

    def run(self):
        """Main entry point - initialize pygame and run the event loop."""
        if not self._init_pygame():
            return

        self._parse_args()

        while self.running:
            self._handle_events()
            self._handle_continuous_input()
            self._render_if_needed()
            self.clock.tick(60)

        self._print_stats()
        pygame.quit()

    def _init_pygame(self) -> bool:
        """Initialize pygame and check GPU. Returns False if no GPU."""
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
        pygame.display.set_caption("Newton Fractal")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", FONT_SIZE)
        return True

    def _parse_args(self):
        """Parse command line arguments."""
        if len(sys.argv) > 1:
            arg = sys.argv[1].lower()
            type_map = {
                "mandelbrot": MANDELBROT,
                "julia": JULIA,
                "newton": NEWTON,
                "mandelbulb": MANDELBULB,
                "burning_ship": BURNING_SHIP,
                "tricorn": TRICORN,
            }
            if arg in type_map:
                self._switch_fractal(type_map[arg])
            else:
                self.coefficients = [float(c) for c in arg.split(",")]

    def _print_stats(self):
        """Print rendering statistics on exit."""
        if self.frame_times:
            avg_ms = sum(self.frame_times) / len(self.frame_times)
            print(f"\nRendered {len(self.frame_times)} frames")
            print(f"Average frame time: {avg_ms:.1f}ms ({1000/avg_ms:.0f} FPS)")

    # =========================================================================
    # Fractal Switching
    # =========================================================================

    def _switch_fractal(self, target: str):
        """Save current view and switch to target fractal type."""
        # Save current view
        if self.fractal_type in FRACTALS_3D:
            self.saved_cameras[self.fractal_type] = Camera3D(
                self.camera.x, self.camera.y, self.camera.z,
                self.camera.yaw, self.camera.pitch
            )
        else:
            self.saved_views[self.fractal_type] = View2D(
                self.view.center_re, self.view.center_im, self.view.zoom
            )

        # Switch to target
        self.fractal_type = target

        # Load target view
        if target in FRACTALS_3D:
            saved = self.saved_cameras.get(target, Camera3D())
            self.camera = Camera3D(saved.x, saved.y, saved.z, saved.yaw, saved.pitch)
        else:
            saved = self.saved_views.get(target, View2D())
            self.view = View2D(saved.center_re, saved.center_im, saved.zoom)

        print(f"Fractal: {target}")

    # =========================================================================
    # Event Handling
    # =========================================================================

    def _handle_events(self):
        """Process all pygame events."""
        for event in pygame.event.get():
            handler = self._event_handlers.get(event.type)
            if handler:
                handler(self, event)

    @property
    def _event_handlers(self) -> dict:
        """Map event types to handler methods."""
        return {
            pygame.QUIT: lambda self, e: setattr(self, 'running', False),
            pygame.TEXTINPUT: FractalViewer._on_text_input,
            pygame.KEYDOWN: FractalViewer._on_keydown,
            pygame.VIDEORESIZE: lambda self, e: setattr(self, 'needs_render', True),
            pygame.WINDOWRESIZED: lambda self, e: setattr(self, 'needs_render', True),
            pygame.MOUSEBUTTONDOWN: FractalViewer._on_mouse_down,
            pygame.MOUSEBUTTONUP: FractalViewer._on_mouse_up,
            pygame.MOUSEMOTION: FractalViewer._on_mouse_motion,
            pygame.MOUSEWHEEL: FractalViewer._on_mouse_wheel,
        }

    def _on_text_input(self, event):
        """Handle text input for field editing."""
        if self.edit.active_field is not None:
            self.edit.text += event.text
            self.needs_render = True

    def _on_keydown(self, event):
        """Handle key press events."""
        # Text editing mode
        if self.edit.active_field is not None:
            if self._handle_edit_key(event):
                return

        # Regular key handlers
        handler = self._key_handlers.get(event.key)
        if handler:
            handler(self, event)

    def _handle_edit_key(self, event) -> bool:
        """Handle keys while editing a field. Returns True if handled."""
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._apply_edit()
            self.edit.stop_editing()
            self.needs_render = True
            return True
        elif event.key == pygame.K_ESCAPE:
            self.edit.stop_editing()
            self.needs_render = True
            return True
        elif event.key == pygame.K_BACKSPACE:
            self.edit.text = self.edit.text[:-1]
            self.needs_render = True
            return True
        elif event.key == pygame.K_TAB:
            self._apply_edit()
            mods = pygame.key.get_mods()
            next_field = self.edit.next_field(reverse=bool(mods & pygame.KMOD_SHIFT))
            self.edit.active_field = next_field
            self.edit.text = self._get_field_text(next_field)
            self.needs_render = True
            return True
        return False

    def _apply_edit(self):
        """Apply the current edit text to the appropriate field."""
        try:
            val = float(self.edit.text)
            field = self.edit.active_field
            if field == "c_re":
                self.julia.c_re = val
            elif field == "c_im":
                self.julia.c_im = val
            elif field == "power_re":
                self.julia.power_re = val
            elif field == "power_im":
                self.julia.power_im = val
            print(f"Set {field} = {val}")
        except ValueError:
            print(f"Invalid number: {self.edit.text}")

    def _get_field_text(self, field: str) -> str:
        """Get formatted display text for a Julia parameter field."""
        if field == "c_re":
            return f"{self.julia.c_re:.4f}"
        elif field == "c_im":
            return f"{self.julia.c_im:.4f}"
        elif field == "power_re":
            return f"{self.julia.power_re:.2f}"
        elif field == "power_im":
            return f"{self.julia.power_im:.2f}"
        return ""

    @property
    def _key_handlers(self) -> dict:
        """Map keys to handler methods."""
        return {
            pygame.K_ESCAPE: lambda s, e: setattr(s, 'running', False),
            pygame.K_q: lambda s, e: setattr(s, 'running', False),
            pygame.K_f: FractalViewer._toggle_fps,
            pygame.K_h: FractalViewer._toggle_help,
            pygame.K_QUESTION: FractalViewer._toggle_help,
            pygame.K_SLASH: FractalViewer._toggle_help,
            pygame.K_r: FractalViewer._randomize_colors,
            pygame.K_0: FractalViewer._reset_view,
            pygame.K_n: lambda s, e: (s._switch_fractal(NEWTON), setattr(s, 'needs_render', True)),
            pygame.K_m: lambda s, e: (s._switch_fractal(MANDELBROT), setattr(s, 'needs_render', True)),
            pygame.K_j: lambda s, e: (s._switch_fractal(JULIA), setattr(s, 'needs_render', True)),
            pygame.K_b: lambda s, e: (s._switch_fractal(BURNING_SHIP), setattr(s, 'needs_render', True)),
            pygame.K_t: lambda s, e: (s._switch_fractal(TRICORN), setattr(s, 'needs_render', True)),
            pygame.K_p: lambda s, e: (s._switch_fractal(MANDELBULB), setattr(s, 'needs_render', True)),
            pygame.K_RIGHTBRACKET: FractalViewer._next_preset,
            pygame.K_LEFTBRACKET: FractalViewer._prev_preset,
            pygame.K_d: FractalViewer._reset_defaults,
            pygame.K_2: lambda s, e: s._set_power(2),
            pygame.K_3: lambda s, e: s._set_power(3),
            pygame.K_4: lambda s, e: s._set_power(4),
            pygame.K_5: lambda s, e: s._set_power(5),
            pygame.K_6: lambda s, e: s._set_power(6),
            pygame.K_7: lambda s, e: s._set_power(7),
            pygame.K_8: lambda s, e: s._set_power(8),
            pygame.K_9: lambda s, e: s._set_power(9),
        }

    def _toggle_fps(self, event):
        self.show_fps = not self.show_fps
        self.needs_render = True

    def _toggle_help(self, event):
        self.show_help = not self.show_help
        self.needs_render = True

    def _randomize_colors(self, event):
        self.color_seed = random.random()
        self.needs_render = True

    def _reset_view(self, event):
        if self.fractal_type in FRACTALS_3D:
            self.camera.reset()
        else:
            self.view = View2D()
        self.needs_render = True

    def _next_preset(self, event):
        if self.fractal_type == MANDELBULB:
            self.mandelbulb_power = min(self.mandelbulb_power + 0.5, 16.0)
            print(f"Mandelbulb power: {self.mandelbulb_power}")
        else:
            self.julia.next_preset()
            print(f"Julia preset {self.julia.preset_idx + 1}: "
                  f"c = {self.julia.c_re:.3f} + {self.julia.c_im:.3f}i")
        self.needs_render = True

    def _prev_preset(self, event):
        if self.fractal_type == MANDELBULB:
            self.mandelbulb_power = max(self.mandelbulb_power - 0.5, 2.0)
            print(f"Mandelbulb power: {self.mandelbulb_power}")
        else:
            self.julia.prev_preset()
            print(f"Julia preset {self.julia.preset_idx + 1}: "
                  f"c = {self.julia.c_re:.3f} + {self.julia.c_im:.3f}i")
        self.needs_render = True

    def _reset_defaults(self, event):
        if self.fractal_type == MANDELBULB:
            self.camera.reset()
            self.mandelbulb_power = 8.0
            print("Mandelbulb reset to defaults")
        else:
            self.julia.reset()
            if self.fractal_type == JULIA:
                self.view = View2D(0.0, 0.0, 1.5)
            print("Julia reset to defaults")
        self.needs_render = True

    def _set_power(self, n: int):
        if self.fractal_type == JULIA:
            self.julia.power_re, self.julia.power_im = float(n), 0.0
            print(f"Julia power: z^{n}")
        else:
            self.coefficients = [1.0] + [0.0] * (n - 1) + [-1.0]
            print(f"Polynomial: z^{n} - 1")
        self.needs_render = True

    # =========================================================================
    # Mouse Handling
    # =========================================================================

    def _on_mouse_down(self, event):
        """Handle mouse button press."""
        if event.button != 1:  # Left click only
            return

        # Check for field click in Julia mode
        if self.fractal_type == JULIA and self.show_fps:
            for field_name, rect in self.edit.field_rects.items():
                if rect.collidepoint(event.pos):
                    self.edit.start_editing(field_name, self._get_field_text(field_name))
                    self.needs_render = True
                    return

        # Cancel any active editing
        if self.edit.active_field is not None:
            self.edit.stop_editing()
            self.needs_render = True

        mods = pygame.key.get_mods()

        # Shift+click on Mandelbrot: jump to Julia
        if (mods & pygame.KMOD_SHIFT) and self.fractal_type == MANDELBROT:
            self._jump_to_julia(event.pos)
            return

        # Start drag
        self.drag.active = True
        self.drag.start_pos = event.pos
        self.drag.start_view = View2D(
            self.view.center_re, self.view.center_im, self.view.zoom
        )
        self.drag.start_camera = Camera3D(
            self.camera.x, self.camera.y, self.camera.z,
            self.camera.yaw, self.camera.pitch
        )
        self.drag.start_julia_c = (self.julia.c_re, self.julia.c_im)
        self.drag.start_power = (self.julia.power_re, self.julia.power_im)

    def _jump_to_julia(self, pos):
        """Jump from Mandelbrot to Julia at clicked position."""
        width, height = self.screen.get_size()
        aspect = width / height
        half_w = self.view.zoom * aspect
        half_h = self.view.zoom
        click_re = self.view.center_re + (pos[0] / width - 0.5) * 2 * half_w
        click_im = self.view.center_im - (pos[1] / height - 0.5) * 2 * half_h
        self.julia.c_re, self.julia.c_im = click_re, click_im
        self._switch_fractal(JULIA)
        self.needs_render = True

    def _on_mouse_up(self, event):
        """Handle mouse button release."""
        if event.button == 1:
            self.drag.active = False

    def _on_mouse_motion(self, event):
        """Handle mouse movement."""
        if not self.drag.active or self.drag.start_pos is None:
            return

        dx = event.pos[0] - self.drag.start_pos[0]
        dy = event.pos[1] - self.drag.start_pos[1]
        mods = pygame.key.get_mods()

        if self.fractal_type in FRACTALS_3D:
            self._handle_3d_drag(dx, dy, mods)
        elif mods & pygame.KMOD_SHIFT:
            self._handle_julia_c_drag(dx, dy)
        elif mods & (pygame.KMOD_ALT | pygame.KMOD_CTRL):
            self._handle_power_drag(dx, dy)
        else:
            self._handle_pan_drag(dx, dy)

    def _handle_3d_drag(self, dx: int, dy: int, mods: int):
        """Handle 3D camera drag."""
        if mods & pygame.KMOD_SHIFT:
            # Shift+drag: move camera perpendicular to view direction
            _, right, up = self.camera.get_vectors()
            move_scale = MOUSE_SENSITIVITY
            self.camera.x = self.drag.start_camera.x - dx * right[0] * move_scale + dy * up[0] * move_scale
            self.camera.y = self.drag.start_camera.y + dy * up[1] * move_scale
            self.camera.z = self.drag.start_camera.z - dx * right[2] * move_scale + dy * up[2] * move_scale
        else:
            # Normal drag: rotate camera
            self.camera.yaw = self.drag.start_camera.yaw + dx * LOOK_SPEED_3D
            self.camera.pitch = self.drag.start_camera.pitch - dy * LOOK_SPEED_3D
            # Clamp pitch to avoid gimbal lock
            self.camera.pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self.camera.pitch))
        self.needs_render = True

    def _handle_julia_c_drag(self, dx: int, dy: int):
        """Handle Julia c parameter drag."""
        self.julia.c_re = self.drag.start_julia_c[0] + dx * MOUSE_SENSITIVITY
        self.julia.c_im = self.drag.start_julia_c[1] - dy * MOUSE_SENSITIVITY
        if self.fractal_type != JULIA:
            self._switch_fractal(JULIA)
        self.needs_render = True

    def _handle_power_drag(self, dx: int, dy: int):
        """Handle power exponent drag."""
        self.julia.power_re = max(1.5, min(10.0, self.drag.start_power[0] + dx * POWER_ADJUST_SENSITIVITY))
        self.julia.power_im = self.drag.start_power[1] - dy * POWER_ADJUST_SENSITIVITY
        if self.fractal_type != JULIA:
            self._switch_fractal(JULIA)
        self.needs_render = True

    def _handle_pan_drag(self, dx: int, dy: int):
        """Handle 2D pan drag."""
        width, height = self.screen.get_size()
        aspect = width / height
        re_per_pixel = (self.view.zoom * aspect * 2) / width
        im_per_pixel = (self.view.zoom * 2) / height
        self.view.center_re = self.drag.start_view.center_re - dx * re_per_pixel
        self.view.center_im = self.drag.start_view.center_im + dy * im_per_pixel
        self.needs_render = True

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel scrolling."""
        if self.fractal_type in FRACTALS_3D:
            self.camera.move_forward(MOVE_SPEED_3D * 3 * event.y)
        else:
            self._zoom_at_cursor(event.y)
        self.needs_render = True

    def _zoom_at_cursor(self, direction: int):
        """Zoom towards/away from cursor position."""
        width, height = self.screen.get_size()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        aspect = width / height

        # Convert mouse to complex coordinates before zoom
        half_w = self.view.zoom * aspect
        half_h = self.view.zoom
        mouse_re = self.view.center_re + (mouse_x / width - 0.5) * 2 * half_w
        mouse_im = self.view.center_im - (mouse_y / height - 0.5) * 2 * half_h

        # Apply zoom
        if direction > 0:
            self.view.zoom /= ZOOM_FACTOR
        else:
            self.view.zoom *= ZOOM_FACTOR

        # Adjust center to keep mouse point fixed
        half_w = self.view.zoom * aspect
        half_h = self.view.zoom
        new_mouse_re = self.view.center_re + (mouse_x / width - 0.5) * 2 * half_w
        new_mouse_im = self.view.center_im - (mouse_y / height - 0.5) * 2 * half_h
        self.view.center_re += mouse_re - new_mouse_re
        self.view.center_im += mouse_im - new_mouse_im

    # =========================================================================
    # Continuous Input
    # =========================================================================

    def _handle_continuous_input(self):
        """Handle held keys for smooth movement."""
        keys = pygame.key.get_pressed()

        if self.fractal_type in FRACTALS_3D:
            self._handle_3d_movement(keys)
        else:
            self._handle_2d_movement(keys)

        self._handle_parameter_keys(keys)

    def _handle_3d_movement(self, keys):
        """Handle 3D camera movement with arrow keys."""
        if keys[pygame.K_UP]:
            self.camera.move_forward(MOVE_SPEED_3D)
            self.needs_render = True
        if keys[pygame.K_DOWN]:
            self.camera.move_forward(-MOVE_SPEED_3D)
            self.needs_render = True
        if keys[pygame.K_LEFT]:
            self.camera.move_right(-MOVE_SPEED_3D)
            self.needs_render = True
        if keys[pygame.K_RIGHT]:
            self.camera.move_right(MOVE_SPEED_3D)
            self.needs_render = True
        if keys[pygame.K_SPACE]:
            self.camera.move_up(MOVE_SPEED_3D)
            self.needs_render = True
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            self.camera.move_up(-MOVE_SPEED_3D)
            self.needs_render = True

    def _handle_2d_movement(self, keys):
        """Handle 2D panning with arrow keys."""
        if keys[pygame.K_UP]:
            self.view.center_im += PAN_SPEED * self.view.zoom
            self.needs_render = True
        if keys[pygame.K_DOWN]:
            self.view.center_im -= PAN_SPEED * self.view.zoom
            self.needs_render = True
        if keys[pygame.K_LEFT]:
            self.view.center_re -= PAN_SPEED * self.view.zoom
            self.needs_render = True
        if keys[pygame.K_RIGHT]:
            self.view.center_re += PAN_SPEED * self.view.zoom
            self.needs_render = True

    def _handle_parameter_keys(self, keys):
        """Handle rate-limited parameter adjustment keys."""
        current_time = time.perf_counter()
        if current_time - self.last_key_repeat_time <= KEY_REPEAT_INTERVAL:
            return

        mods = pygame.key.get_mods()
        key_acted = False

        if keys[pygame.K_g]:
            if mods & pygame.KMOD_SHIFT:
                self.glow_intensity = max(0.0, self.glow_intensity - 0.1)
            else:
                self.glow_intensity += 0.1
            self.needs_render = True
            key_acted = True

        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
            delta = 100 if mods & pygame.KMOD_SHIFT else 10
            self.imax += delta
            self.needs_render = True
            key_acted = True

        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            delta = 100 if mods & pygame.KMOD_SHIFT else 10
            self.imax = max(self.imax - delta, 10)
            self.needs_render = True
            key_acted = True

        if key_acted:
            self.last_key_repeat_time = current_time

    # =========================================================================
    # Rendering
    # =========================================================================

    def _render_if_needed(self):
        """Render frame if state has changed."""
        if not self.needs_render:
            return

        t0 = time.perf_counter()

        # Get current window size
        self.screen = pygame.display.get_surface()
        width, height = self.screen.get_size()

        # Render fractal
        rgb_array = FractalRenderer.render(
            self.fractal_type, width, height,
            self.view, self.camera, self.julia,
            self.coefficients, self.tolerance, self.imax,
            self.color_seed, self.glow_intensity, self.mandelbulb_power
        )

        t1 = time.perf_counter()
        render_ms = (t1 - t0) * 1000

        # Display with pygame
        surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))

        t2 = time.perf_counter()
        total_ms = (t2 - t0) * 1000
        self.last_fps = 1000 / total_ms if total_ms > 0 else 0
        self.last_render_ms = render_ms

        # Draw overlays
        if self.show_fps:
            self._draw_fps_overlay(width)
        if self.show_help:
            self._draw_help_overlay()

        pygame.display.flip()

        self.frame_times.append(total_ms)
        self.needs_render = False
        gc.collect()

    def _draw_fps_overlay(self, width: int):
        """Draw FPS and parameter info overlay."""
        if self.fractal_type == JULIA:
            self._draw_julia_overlay()
        elif self.fractal_type == MANDELBULB:
            text = f"{self.fractal_type} power={self.mandelbulb_power:.1f} | {self.last_render_ms:.1f}ms | {self.last_fps:.0f} FPS | imax: {self.imax}"
            self._draw_text(text, PADDING, PADDING // 2)
        elif self.fractal_type == NEWTON:
            text = f"{self.fractal_type} | {self.last_render_ms:.1f}ms | {self.last_fps:.0f} FPS | zoom: {self.view.zoom:.2f} | glow: {self.glow_intensity:.1f} | imax: {self.imax}"
            self._draw_text(text, PADDING, PADDING // 2)
        else:
            text = f"{self.fractal_type} | {self.last_render_ms:.1f}ms | {self.last_fps:.0f} FPS | zoom: {self.view.zoom:.2e} | imax: {self.imax}"
            self._draw_text(text, PADDING, PADDING // 2)

    def _draw_julia_overlay(self):
        """Draw Julia-specific overlay with editable fields."""
        self.edit.field_rects.clear()
        x_pos = PADDING
        y_pos = PADDING // 2

        x_pos = self._draw_text("julia c=", x_pos, y_pos)
        x_pos = self._draw_editable_field("c_re", x_pos, y_pos)
        x_pos = self._draw_text("+" if self.julia.c_im >= 0 else "", x_pos, y_pos)
        x_pos = self._draw_editable_field("c_im", x_pos, y_pos)
        x_pos = self._draw_text("i z^", x_pos, y_pos)
        x_pos = self._draw_editable_field("power_re", x_pos, y_pos)
        x_pos = self._draw_text("+" if self.julia.power_im >= 0 else "", x_pos, y_pos)
        x_pos = self._draw_editable_field("power_im", x_pos, y_pos)
        x_pos = self._draw_text("i", x_pos, y_pos)
        self._draw_text(f" | {self.last_render_ms:.1f}ms | {self.last_fps:.0f} FPS | imax: {self.imax}", x_pos, y_pos)

    def _draw_text(self, text: str, x: int, y: int, color=(255, 255, 255)) -> int:
        """Render text at position and return new x position."""
        surf = self.font.render(text, True, color, (0, 0, 0))
        self.screen.blit(surf, (x, y))
        return x + surf.get_width()

    def _draw_editable_field(self, field: str, x: int, y: int) -> int:
        """Render an editable field and return new x position."""
        is_editing = self.edit.active_field == field
        text = (self.edit.text + "|") if is_editing else self._get_field_text(field)
        color = (255, 255, 0) if is_editing else (100, 200, 255)
        surf = self.font.render(text, True, color, (0, 0, 0))
        self.edit.field_rects[field] = pygame.Rect(x, y, surf.get_width(), surf.get_height())
        self.screen.blit(surf, (x, y))
        return x + surf.get_width()

    def _draw_help_overlay(self):
        """Draw help text overlay."""
        line_height = self.font.get_linesize()
        help_width = max(self.font.size(line)[0] for line in HELP_LINES) + PADDING * 2
        help_height = len(HELP_LINES) * line_height + PADDING * 2

        # Semi-transparent background
        help_bg = pygame.Surface((help_width, help_height))
        help_bg.set_alpha(HELP_OVERLAY_ALPHA)
        help_bg.fill((0, 0, 0))
        help_y = line_height + PADDING
        self.screen.blit(help_bg, (PADDING, help_y))

        # Render help text
        for i, line in enumerate(HELP_LINES):
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (PADDING * 2, help_y + PADDING + i * line_height))


# =============================================================================
# Entry Point
# =============================================================================

def main():
    viewer = FractalViewer()
    viewer.run()


if __name__ == "__main__":
    main()
