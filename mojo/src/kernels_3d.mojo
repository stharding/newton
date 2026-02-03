"""3D fractal GPU kernels (Mandelbulb)."""

from gpu import global_idx
from layout import Layout, LayoutTensor
from math import clamp, cos, pi, sin, sqrt, tau

from gpu_math import gpu_acos, gpu_atan2
from fractals_3d import mandelbulb_de, MAX_STEPS, MAX_DIST, EPSILON, NORMAL_EPSILON


# ============================================================================
# Constants for Lighting
# ============================================================================

comptime AMBIENT_INTENSITY: Float32 = 0.2
"""Base ambient light level."""

comptime DIFFUSE_INTENSITY: Float32 = 0.65
"""Diffuse lighting contribution."""

comptime SPECULAR_INTENSITY: Float32 = 0.4
"""Specular highlight contribution."""

comptime SPECULAR_EXPONENT: Float32 = 32.0
"""Shininess for Blinn-Phong specular."""

comptime AO_STRENGTH: Float32 = 0.5
"""Ambient occlusion strength based on step count."""


# ============================================================================
# Vec3 - Simple 3D vector type
# ============================================================================

@fieldwise_init
struct Vec3(TrivialRegisterType):
    """A simple 3D vector for ray marching calculations."""
    var x: Float32
    var y: Float32
    var z: Float32

    @always_inline
    fn __add__(self, other: Self) -> Self:
        return Self(self.x + other.x, self.y + other.y, self.z + other.z)

    @always_inline
    fn __sub__(self, other: Self) -> Self:
        return Self(self.x - other.x, self.y - other.y, self.z - other.z)

    @always_inline
    fn __mul__(self, scalar: Float32) -> Self:
        return Self(self.x * scalar, self.y * scalar, self.z * scalar)

    @always_inline
    fn __neg__(self) -> Self:
        return Self(-self.x, -self.y, -self.z)

    @always_inline
    fn dot(self, other: Self) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z

    @always_inline
    fn length(self) -> Float32:
        return sqrt(self.dot(self))

    @always_inline
    fn normalize(self) -> Self:
        var len = self.length()
        if len > Float32(1e-10):
            return Self(self.x / len, self.y / len, self.z / len)
        return self


# ============================================================================
# Camera utilities
# ============================================================================

@always_inline
fn camera_basis(yaw: Float32, pitch: Float32) -> Tuple[Vec3, Vec3, Vec3]:
    """Compute camera basis vectors (forward, right, up) from yaw and pitch."""
    var cos_yaw = cos(yaw)
    var sin_yaw = sin(yaw)
    var cos_pitch = cos(pitch)
    var sin_pitch = sin(pitch)

    var forward = Vec3(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw)
    var right = Vec3(cos_yaw, Float32(0.0), -sin_yaw)
    var up = Vec3(-sin_pitch * sin_yaw, cos_pitch, -sin_pitch * cos_yaw)

    return Tuple(forward, right, up)


@always_inline
fn compute_normal(pos: Vec3, power: Float32, imax: Int) -> Vec3:
    """Compute surface normal via gradient of distance estimator."""
    var nx = mandelbulb_de(pos.x + NORMAL_EPSILON, pos.y, pos.z, power, imax) - mandelbulb_de(pos.x - NORMAL_EPSILON, pos.y, pos.z, power, imax)
    var ny = mandelbulb_de(pos.x, pos.y + NORMAL_EPSILON, pos.z, power, imax) - mandelbulb_de(pos.x, pos.y - NORMAL_EPSILON, pos.z, power, imax)
    var nz = mandelbulb_de(pos.x, pos.y, pos.z + NORMAL_EPSILON, power, imax) - mandelbulb_de(pos.x, pos.y, pos.z - NORMAL_EPSILON, power, imax)
    return Vec3(nx, ny, nz).normalize()


# ============================================================================
# Lighting calculation
# ============================================================================

@always_inline
fn compute_lighting(
    normal: Vec3,
    ray_dir: Vec3,
    steps: Int,
    color_seed: Float32,
) -> Vec3:
    """Compute lit color using Blinn-Phong shading model."""
    # Light direction (normalized (1, 1, -1))
    var light = Vec3(Float32(0.577), Float32(0.577), Float32(-0.577))

    # Diffuse
    var diffuse = normal.dot(light)
    if diffuse < Float32(0.0):
        diffuse = Float32(0.0)

    # Specular (Blinn-Phong)
    var view = -ray_dir
    var half_vec = (light + view).normalize()
    var spec_dot = normal.dot(half_vec)
    if spec_dot < Float32(0.0):
        spec_dot = Float32(0.0)
    var specular = spec_dot ** SPECULAR_EXPONENT

    # Ambient occlusion from step count
    var ao = Float32(1.0) - Float32(steps) / Float32(MAX_STEPS) * AO_STRENGTH

    # Color from normal direction (warm/cool based on orientation)
    var n_theta = gpu_acos(normal.y)
    var n_phi = gpu_atan2(normal.z, normal.x)

    comptime TAU32: Float32 = tau
    var hue = (n_phi / TAU32 + Float32(0.5) + color_seed)
    hue = hue - Float32(Int(hue))

    var blend = n_theta / Float32(pi)

    # Warm color palette
    var warm = Vec3(
        Float32(0.95),
        Float32(0.4) + hue * Float32(0.3),
        Float32(0.5) + hue * Float32(0.4),
    )

    # Cool color palette
    var cool = Vec3(
        Float32(0.3) + hue * Float32(0.2),
        Float32(0.6) + hue * Float32(0.3),
        Float32(0.9),
    )

    # Blend warm and cool
    var base_r = warm.x * (Float32(1.0) - blend) + cool.x * blend
    var base_g = warm.y * (Float32(1.0) - blend) + cool.y * blend
    var base_b = warm.z * (Float32(1.0) - blend) + cool.z * blend

    # Combine lighting
    var intensity = AMBIENT_INTENSITY + diffuse * DIFFUSE_INTENSITY + specular * SPECULAR_INTENSITY
    intensity = intensity * ao

    return Vec3(
        clamp(base_r * intensity, Float32(0.0), Float32(1.0)),
        clamp(base_g * intensity, Float32(0.0), Float32(1.0)),
        clamp(base_b * intensity, Float32(0.0), Float32(1.0)),
    )


# ============================================================================
# Mandelbulb 3D fractal kernel (ray marching)
# ============================================================================

fn mandelbulb_kernel[
    output_layout: Layout,
](
    output: LayoutTensor[DType.uint8, output_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    cam_x: Float32,
    cam_y: Float32,
    cam_z: Float32,
    cam_yaw: Float32,
    cam_pitch: Float32,
    power: Float32,
    imax: Int,
    color_seed: Float32,
):
    """Ray march to render Mandelbulb fractal."""
    var px = Int(global_idx.x)
    var py = Int(global_idx.y)

    if px >= width or py >= height:
        return

    # Compute ray direction from pixel coordinates and camera orientation
    var aspect = Float32(width) / Float32(height)
    var fov = Float32(1.0)

    var ndc_x = (Float32(2.0) * Float32(px) / Float32(width) - Float32(1.0)) * aspect * fov
    var ndc_y = (Float32(1.0) - Float32(2.0) * Float32(py) / Float32(height)) * fov

    # Get camera basis vectors
    var forward: Vec3
    var right: Vec3
    var up: Vec3
    forward, right, up = camera_basis(cam_yaw, cam_pitch)

    # Ray direction in world space
    var ray_dir = (right * ndc_x + up * ndc_y + forward).normalize()

    # Ray marching
    var pos = Vec3(cam_x, cam_y, cam_z)
    var total_dist = Float32(0.0)
    var hit = False
    var steps = 0

    for step in range(MAX_STEPS):
        var dist = mandelbulb_de(pos.x, pos.y, pos.z, power, imax)

        if dist < EPSILON:
            hit = True
            steps = step
            break

        total_dist = total_dist + dist

        if total_dist > MAX_DIST:
            break

        pos = pos + ray_dir * dist
        steps = step

    var pixel_idx = (py * width + px) * 3

    if hit:
        var normal = compute_normal(pos, power, imax)
        var color = compute_lighting(normal, ray_dir, steps, color_seed)

        output[pixel_idx] = UInt8(color.x * Float32(255.0))
        output[pixel_idx + 1] = UInt8(color.y * Float32(255.0))
        output[pixel_idx + 2] = UInt8(color.z * Float32(255.0))
    else:
        # Background gradient
        var t = Float32(py) / Float32(height)
        output[pixel_idx] = UInt8(Float32(10.0) + t * Float32(20.0))
        output[pixel_idx + 1] = UInt8(Float32(10.0) + t * Float32(25.0))
        output[pixel_idx + 2] = UInt8(Float32(20.0) + t * Float32(40.0))
