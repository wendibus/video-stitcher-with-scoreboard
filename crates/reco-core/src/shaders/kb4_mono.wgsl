// Reco v2 -- Single-input KB4 fisheye projection.
//
// Renders a virtual pan/tilt/zoom crop directly from ONE raw,
// uncorrected fisheye camera - no stitching, no second plane. Replaces
// `cylindrical_mono.wgsl`, which assumed the input was *already*
// unwrapped onto a cylinder (actionstitch-player's model, meant for
// pre-stitched panorama files); a raw camera recording is not that,
// and applying the cylindrical model to it bowed straight court lines
// into arcs. See `crate::projection::project_world_point`'s doc
// comment and `crate::calibration::CameraParams` for the model this
// shader mirrors.
//
// Fragment dispatch:
//   For each output pixel:
//     1. Build a ray from the virtual camera through the pixel
//        (same full-screen-triangle NDC ray as the cylindrical
//        shader - vs_fullscreen/VsOut are unchanged).
//     2. Rotate the ray into the real camera's own optical frame via
//        the pose (yaw/pitch/rig_tilt/rig_roll) view matrix - after
//        this rotation, `ray_w` is expressed in the same frame the
//        camera's own KB4 intrinsics were calibrated in (mono has no
//        second camera to place in a scene, so unlike the L-shape
//        path there is no separate "world" frame here at all).
//     3. Forward-project `ray_w` through the KB4 polynomial (the same
//        SYNC_WITH'd polynomial as `fisheye.wgsl`'s `fs_main` and
//        `reco_core::lens::kb4::theta_d`, generalized from that
//        shader's `atan(r)` "z=1 plane" form to `atan2(r, z)` on the
//        full ray - the plane form cannot represent theta >= 90
//        degrees, but this mono path's self-calibration explicitly
//        searches lenses up to 110 degrees half-FOV).
//     4. Sample the video texture at the resulting distorted pixel;
//        return transparent black when the ray falls outside the
//        camera's calibrated field of view or the image bounds.

struct Kb4MonoUniforms {
    // View-to-world (really view-to-camera-optical-frame) rotation.
    view: mat4x4<f32>,
    // [fx/width, fy/height, cx/width, cy/height] - same normalized
    // convention as `fisheye.wgsl`'s `u.intrinsics`.
    intrinsics: vec4<f32>,
    // KB4 distortion coefficients [k1, k2, k3, k4].
    dist: vec4<f32>,
    // Vertical FOV of the virtual output camera, radians.
    v_fov: f32,
    // Viewport aspect ratio (output_width / output_height).
    aspect: f32,
    // 1.0 when the source plane is NV12; 0.0 for YUV420P.
    is_nv12: f32,
    // Max angle (radians) from the real camera's optical axis this
    // calibration is trusted for - beyond it, KB4's polynomial isn't
    // guaranteed monotonic (could fold back and alias a wrong pixel),
    // so reject rather than trust an extrapolated sample.
    max_theta: f32,
};

@group(0) @binding(0) var t_y: texture_2d<f32>;
@group(0) @binding(1) var t_u: texture_2d<f32>;
@group(0) @binding(2) var t_v: texture_2d<f32>;
@group(0) @binding(3) var s_video: sampler;
@group(1) @binding(0) var<uniform> u: Kb4MonoUniforms;

// BT.709 limited-range YCbCr -> full-range sRGB RGB - identical to
// `fisheye.wgsl`/`cylindrical_mono.wgsl`'s `sample_yuv`.
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
    // True clip-space NDC (-1..+1, Y-up) - see cylindrical_mono.wgsl's
    // VsOut for why this is passed raw rather than through a texture-
    // convention UV.
    @location(0) ndc: vec2<f32>,
};

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
fn fs_kb4_mono(in: VsOut) -> @location(0) vec4<f32> {
    let ndc = in.ndc;

    let tan_half_v = tan(u.v_fov * 0.5);
    let tan_half_h = tan_half_v * u.aspect;
    // Forward is -Z in view space; y is up (same convention as
    // cylindrical_mono.wgsl, no screen_rotation term here - mono
    // self-calibration reports rig_tilt/rig_roll instead, applied on
    // the Rust side via the same view_matrix() the cylindrical path
    // used).
    let ray = vec3<f32>(ndc.x * tan_half_h, ndc.y * tan_half_v, -1.0);

    // Rotate into the real camera's optical frame. `ray_w` comes out
    // in this codebase's graphics convention (+Y = up, same as
    // view_matrix/yaw/pitch everywhere else) - but `CameraParams`'
    // cy/fy (and the KB4 model generally) use the standard computer-
    // vision convention (+Y = down, row-major from the top), the same
    // convention `project_world_point`'s solved rotation was fit
    // against real (Y-down) clicked pixel coordinates in. Negate Y
    // once, here, to bridge the two - a fixed reflection, not
    // something rig_tilt/rig_roll (pure rotations) could express.
    let ray_w_gfx = (u.view * vec4<f32>(ray, 0.0)).xyz;
    let ray_w = vec3<f32>(ray_w_gfx.x, -ray_w_gfx.y, ray_w_gfx.z);

    let r_xy = sqrt(ray_w.x * ray_w.x + ray_w.y * ray_w.y);
    let theta = atan2(r_xy, ray_w.z);
    if theta < 0.0 || theta > u.max_theta {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let t2 = theta * theta;
    let t4 = t2 * t2;
    let t6 = t4 * t2;
    let t8 = t4 * t4;
    let theta_d = theta * (1.0 + u.dist.x * t2 + u.dist.y * t4 + u.dist.z * t6 + u.dist.w * t8);

    var scale = 0.0;
    if r_xy > 1e-9 {
        scale = theta_d / r_xy;
    }

    let fx = u.intrinsics.x;
    let fy = u.intrinsics.y;
    let cx = u.intrinsics.z;
    let cy = u.intrinsics.w;
    let distorted_uv = vec2<f32>(
        fx * ray_w.x * scale + cx,
        fy * ray_w.y * scale + cy,
    );

    if distorted_uv.x < 0.0 || distorted_uv.x > 1.0 ||
       distorted_uv.y < 0.0 || distorted_uv.y > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    return vec4<f32>(sample_yuv(distorted_uv), 1.0);
}
