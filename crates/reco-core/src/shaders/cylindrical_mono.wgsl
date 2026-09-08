// Reco v2 -- Single-input cylindrical projection.
//
// Models the video as a texture painted on the inside surface of a
// cylinder of radius `focal_length`. The virtual camera sits on the
// cylinder axis and looks outward; pan/tilt/zoom rotate the camera
// and scale the FOV. Based on the projection used by
// gilbertchen/actionstitch-player (180-degree cylindrical video
// player, MIT license; see projection.rs attribution).
//
// Geometry summary:
//   - Cylinder radius:          focal_length (world units)
//   - Angular sweep:            angular_sweep_rad (e.g. PI for 180deg)
//   - theta_start:              PI/2 - angular_sweep_rad/2 (centered)
//   - Video pixel (px, py)  ->  cylinder point (theta, y_world) with
//                               theta = theta_start + (px/width)*sweep
//                               y_world = py - height/2
//
// Fragment dispatch:
//   For each output pixel (viewport UV in [0,1]):
//     1. Build a ray from the virtual camera through the pixel.
//     2. Rotate the ray by the pose (yaw/pitch) plus the optional
//        screen_rotation_rad tilt around the view axis.
//     3. Intersect the ray with the cylinder x^2 + z^2 = focal_length^2
//        and pick the positive-t hit.
//     4. Map the hit back to video UV via theta = atan2(z, x) and
//        v = (y_world + height/2) / height.
//     5. Sample the video texture; return transparent black when the
//        ray misses the cylinder or falls outside the video bounds.
//
// Notes vs. the L-shape projection:
//   - Single input texture (camera_count = 1).
//   - No fisheye undistortion: source video is already flat.
//   - Works for any angular_sweep in (0, 2*PI]; 180deg is the
//     typical action-camera case but 360deg is supported too.

struct CylUniforms {
    // Model-view-projection / virtual-camera rotation.
    view: mat4x4<f32>,
    // Cylinder radius in world units.
    focal_length: f32,
    // Full horizontal angular sweep of the cylinder in radians.
    angular_sweep: f32,
    // Screen rotation around the view axis in radians (tilt correction).
    screen_rotation: f32,
    // Video height in world units (matches `videoHeight` in source).
    video_height: f32,
    // Vertical FOV of the virtual camera in radians.
    v_fov: f32,
    // Viewport aspect ratio (output_width / output_height).
    aspect: f32,
    // 1.0 when the source plane(s) are NV12 (interleaved UV in t_u);
    // 0.0 for YUV420P (separate t_u/t_v). Mirrors `fisheye.wgsl`'s
    // `flags.y` format switch, minus the BGRA/zero-copy variants that
    // path supports - mono only serves CPU-decoded file input today.
    is_nv12: f32,
    // Padding for std140 alignment (mat4x4 + 8 f32 = 96 bytes, 16-aligned).
    _pad0: f32,
};

// YUV420P plane textures (Y = full res R8Unorm, U/V = half res R8Unorm),
// or NV12 (t_y = R8Unorm, t_u = interleaved Rg8Unorm, t_v unused dummy) -
// same texture-plane convention as `fisheye.wgsl`, so the Rust-side bind
// group layout and upload helpers are directly shared.
@group(0) @binding(0) var t_y: texture_2d<f32>;
@group(0) @binding(1) var t_u: texture_2d<f32>;
@group(0) @binding(2) var t_v: texture_2d<f32>;
@group(0) @binding(3) var s_video: sampler;
@group(1) @binding(0) var<uniform> u: CylUniforms;

// BT.709 limited-range (16-235/16-240) YCbCr -> full-range sRGB RGB.
// Same matrix and range convention as `fisheye.wgsl`'s `sample_yuv`,
// so mono and stereo output match visually. NV12 vs YUV420P selection
// only (no BGRA/zero-copy variant - mono only serves CPU-decoded file
// input today, see the module doc comment in `projection/mod.rs`).
fn sample_yuv(uv: vec2<f32>) -> vec3<f32> {
    let y_raw = textureSample(t_y, s_video, uv).r;
    var u_raw: f32;
    var v_raw: f32;
    if u.is_nv12 > 0.5 {
        let uv_sample = textureSample(t_u, s_video, uv);
        u_raw = uv_sample.r;
        v_raw = uv_sample.g;
    } else {
        u_raw = textureSample(t_u, s_video, uv).r;
        v_raw = textureSample(t_v, s_video, uv).r;
    }
    let y = (y_raw - 16.0 / 255.0) * (255.0 / 219.0);
    let cb = (u_raw - 128.0 / 255.0) * (255.0 / 224.0);
    let cr = (v_raw - 128.0 / 255.0) * (255.0 / 224.0);
    let r = y + 1.5748 * cr;
    let g = y - 0.1873 * cb - 0.4681 * cr;
    let b = y + 1.8556 * cb;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    // True clip-space NDC (-1..+1, Y-up), carried through unchanged for
    // per-pixel ray construction below. NOT a texture UV - this shader
    // never samples the video texture at a rasterizer-interpolated
    // coordinate (that happens via theta_norm/v_norm instead), so there
    // is no video-texture "V=0 at top" convention to bake in here. An
    // earlier version emitted such a texture-convention UV and reused
    // it as if it were NDC, which silently flipped ray.y's sign and
    // rendered the output upside-down.
    @location(0) ndc: vec2<f32>,
};

// Full-screen triangle vertex shader (no vertex buffer needed; dispatch
// with 3 vertices and index into this table).
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    out.ndc = positions[vid];
    return out;
}

@fragment
fn fs_cylindrical_mono(in: VsOut) -> @location(0) vec4<f32> {
    // Step 1: already true NDC (-1..+1) - see VsOut::ndc's doc comment.
    let ndc = in.ndc;

    // Step 2: build a ray in view space from the camera at the origin.
    // Horizontal FOV derived from aspect.
    let tan_half_v = tan(u.v_fov * 0.5);
    let tan_half_h = tan_half_v * u.aspect;
    // Forward is -Z in our camera convention; y is up.
    var ray = vec3<f32>(ndc.x * tan_half_h, ndc.y * tan_half_v, -1.0);

    // Step 2b: screen rotation around the view axis (tilts the cylinder
    // left/right). Rotate in the XY plane before world-space rotation.
    let cr = cos(u.screen_rotation);
    let sr = sin(u.screen_rotation);
    ray = vec3<f32>(cr * ray.x - sr * ray.y, sr * ray.x + cr * ray.y, ray.z);

    // Step 3: rotate ray into world space by the view matrix.
    let ray_w = (u.view * vec4<f32>(ray, 0.0)).xyz;

    // Step 4: intersect the ray with the cylinder x^2 + z^2 = r^2.
    // Camera origin is on the axis (x = z = 0), so:
    //   (t * rw.x)^2 + (t * rw.z)^2 = r^2
    //   t = r / sqrt(rw.x^2 + rw.z^2).
    // Guard against a pure-vertical ray (singularity).
    let horiz = sqrt(ray_w.x * ray_w.x + ray_w.z * ray_w.z);
    if (horiz < 1e-6) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let t = u.focal_length / horiz;
    let hit = ray_w * t;

    // Step 5: map hit to cylinder parametrization (theta, y_world).
    let theta = atan2(hit.z, hit.x);
    let half_sweep = u.angular_sweep * 0.5;
    // theta_start = PI/2 - half_sweep.
    let theta_start = 1.5707963 - half_sweep;
    let theta_norm = (theta - theta_start) / u.angular_sweep;
    let v_norm = (hit.y + u.video_height * 0.5) / u.video_height;

    // Outside the painted region -> transparent.
    if (theta_norm < 0.0 || theta_norm > 1.0 || v_norm < 0.0 || v_norm > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Step 6: sample the video planes. Flip v so (0,0) is top-left as
    // is conventional for video textures.
    let uv_sample = vec2<f32>(theta_norm, 1.0 - v_norm);
    return vec4<f32>(sample_yuv(uv_sample), 1.0);
}
