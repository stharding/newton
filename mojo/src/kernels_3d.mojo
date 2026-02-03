"""3D fractal GPU kernels (Mandelbulb)."""

from gpu import global_idx
from layout import Layout, LayoutTensor
from math import sin, cos, sqrt

from gpu_math import gpu_pow, gpu_acos, gpu_atan2
from fractals_3d import mandelbulb_de, MAX_STEPS, MAX_DIST, EPSILON, NORMAL_EPSILON


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

    # Build camera basis vectors from yaw and pitch
    var cos_yaw = cos(cam_yaw)
    var sin_yaw = sin(cam_yaw)
    var cos_pitch = cos(cam_pitch)
    var sin_pitch = sin(cam_pitch)

    # Forward direction
    var fwd_x = cos_pitch * sin_yaw
    var fwd_y = sin_pitch
    var fwd_z = cos_pitch * cos_yaw

    # Right direction
    var right_x = cos_yaw
    var right_y = Float32(0.0)
    var right_z = -sin_yaw

    # Up direction
    var up_x = -sin_pitch * sin_yaw
    var up_y = cos_pitch
    var up_z = -sin_pitch * cos_yaw

    # Ray direction in world space
    var ray_dx = right_x * ndc_x + up_x * ndc_y + fwd_x
    var ray_dy = right_y * ndc_x + up_y * ndc_y + fwd_y
    var ray_dz = right_z * ndc_x + up_z * ndc_y + fwd_z

    # Normalize ray direction
    var ray_len = sqrt(ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz)
    ray_dx = ray_dx / ray_len
    ray_dy = ray_dy / ray_len
    ray_dz = ray_dz / ray_len

    # Ray marching
    var total_dist = Float32(0.0)
    var hit = False
    var steps = 0

    var pos_x = cam_x
    var pos_y = cam_y
    var pos_z = cam_z

    for step in range(MAX_STEPS):
        var dist = mandelbulb_de(pos_x, pos_y, pos_z, power, imax)

        if dist < EPSILON:
            hit = True
            steps = step
            break

        total_dist = total_dist + dist

        if total_dist > MAX_DIST:
            break

        pos_x = pos_x + ray_dx * dist
        pos_y = pos_y + ray_dy * dist
        pos_z = pos_z + ray_dz * dist
        steps = step

    var pixel_idx = (py * width + px) * 3

    if hit:
        # Compute surface normal via gradient
        var nx = mandelbulb_de(pos_x + NORMAL_EPSILON, pos_y, pos_z, power, imax) - mandelbulb_de(pos_x - NORMAL_EPSILON, pos_y, pos_z, power, imax)
        var ny = mandelbulb_de(pos_x, pos_y + NORMAL_EPSILON, pos_z, power, imax) - mandelbulb_de(pos_x, pos_y - NORMAL_EPSILON, pos_z, power, imax)
        var nz = mandelbulb_de(pos_x, pos_y, pos_z + NORMAL_EPSILON, power, imax) - mandelbulb_de(pos_x, pos_y, pos_z - NORMAL_EPSILON, power, imax)

        var n_len = sqrt(nx * nx + ny * ny + nz * nz)
        if n_len > Float32(1e-10):
            nx = nx / n_len
            ny = ny / n_len
            nz = nz / n_len

        # Light direction
        var light_x = Float32(0.577)
        var light_y = Float32(0.577)
        var light_z = Float32(-0.577)

        # Diffuse lighting
        var diffuse = nx * light_x + ny * light_y + nz * light_z
        if diffuse < Float32(0.0):
            diffuse = Float32(0.0)

        # Specular lighting (Blinn-Phong)
        var view_x = -ray_dx
        var view_y = -ray_dy
        var view_z = -ray_dz
        var half_x = light_x + view_x
        var half_y = light_y + view_y
        var half_z = light_z + view_z
        var half_len = sqrt(half_x * half_x + half_y * half_y + half_z * half_z)
        half_x = half_x / half_len
        half_y = half_y / half_len
        half_z = half_z / half_len
        var spec_dot = nx * half_x + ny * half_y + nz * half_z
        if spec_dot < Float32(0.0):
            spec_dot = Float32(0.0)
        var specular = gpu_pow(spec_dot, Float32(32.0))

        # Ambient occlusion
        var ao = Float32(1.0) - Float32(steps) / Float32(MAX_STEPS) * Float32(0.5)

        # Color from normal direction
        var n_theta = gpu_acos(ny)
        var n_phi = gpu_atan2(nz, nx)

        var hue1 = (n_phi / Float32(6.28318530) + Float32(0.5) + color_seed)
        hue1 = hue1 - Float32(Int(hue1))

        var blend = n_theta / Float32(3.14159265)

        # Warm color
        var warm_r = Float32(0.95)
        var warm_g = Float32(0.4) + hue1 * Float32(0.3)
        var warm_b = Float32(0.5) + hue1 * Float32(0.4)

        # Cool color
        var cool_r = Float32(0.3) + hue1 * Float32(0.2)
        var cool_g = Float32(0.6) + hue1 * Float32(0.3)
        var cool_b = Float32(0.9)

        var base_r = warm_r * (Float32(1.0) - blend) + cool_r * blend
        var base_g = warm_g * (Float32(1.0) - blend) + cool_g * blend
        var base_b = warm_b * (Float32(1.0) - blend) + cool_b * blend

        var ambient = Float32(0.2)
        var intensity = ambient + diffuse * Float32(0.65) + specular * Float32(0.4)
        intensity = intensity * ao

        var r_out = base_r * intensity
        var g_out = base_g * intensity
        var b_out = base_b * intensity

        if r_out > Float32(1.0):
            r_out = Float32(1.0)
        if g_out > Float32(1.0):
            g_out = Float32(1.0)
        if b_out > Float32(1.0):
            b_out = Float32(1.0)

        output[pixel_idx] = UInt8(r_out * Float32(255.0))
        output[pixel_idx + 1] = UInt8(g_out * Float32(255.0))
        output[pixel_idx + 2] = UInt8(b_out * Float32(255.0))
    else:
        # Background gradient
        var t = Float32(py) / Float32(height)
        var bg_r = UInt8(Float32(10.0) + t * Float32(20.0))
        var bg_g = UInt8(Float32(10.0) + t * Float32(25.0))
        var bg_b = UInt8(Float32(20.0) + t * Float32(40.0))
        output[pixel_idx] = bg_r
        output[pixel_idx + 1] = bg_g
        output[pixel_idx + 2] = bg_b
