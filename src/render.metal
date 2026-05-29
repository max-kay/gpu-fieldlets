#include <metal_stdlib>
using namespace metal;

struct Intersection {
  float3 normal;
  float dist;
  bool ok;
};

struct Ray {
  float3 root;
  float3 dir;
};

struct FrameSpec {
  uint2 dims;
  uint particle_count;
  uint oversamples;
  float4 cam_root;
  float4 cam_s1, cam_s2, cam_dir;
  float4 ell_axes, ell_color;
  float4 light_dir;
  float4 bg_color;
  float ambient_light;
};

float3x3 get_rotation_matrix(float3 d) {
  float3 from = normalize(d);
  float3 to = float3(0.0, 0.0, 1.0);

  float cos_theta = dot(from, to);

  if (cos_theta > 0.999999) {
    return float3x3(1.0);
  }
  if (cos_theta < -0.999999) {
    return float3x3(float3(1, 0, 0), float3(0, -1, 0), float3(0, 0, -1));
  }

  float3 v = cross(from, to);
  float k = 1.0 / (1.0 + cos_theta);

  float3x3 R;
  R[0][0] = 1.0 - k * (v.y * v.y + v.z * v.z);
  R[1][0] = -v.z + k * v.x * v.y;
  R[2][0] = v.y + k * v.x * v.z;

  R[0][1] = v.z + k * v.x * v.y;
  R[1][1] = 1.0 - k * (v.x * v.x + v.z * v.z);
  R[2][1] = -v.x + k * v.y * v.z;

  R[0][2] = -v.y + k * v.x * v.z;
  R[1][2] = v.x + k * v.y * v.z;
  R[2][2] = 1.0 - k * (v.x * v.x + v.y * v.y);

  return R;
}

Intersection intersect(Ray ray, float3 center, float3 orient, float3 ell_axes) {
  float3x3 R = get_rotation_matrix(orient);
  float3 a_dir = R * ray.dir;
  float3 a_root = R * (ray.root - center);

  float3 t_dir = a_dir / ell_axes;
  float3 t_root = a_root / ell_axes;

  float dot_val = dot(t_dir, t_root);
  float t_dir_norm2 = dot(t_dir, t_dir);
  float discriminant =
      dot_val * dot_val - t_dir_norm2 * (dot(t_root, t_root) - 1.0);

  if (discriminant < 0) {
    return (Intersection){float3(0), 0.0, false};
  }

  float t_inter = -(dot_val + sqrt(discriminant)) / t_dir_norm2;
  float3 inter_point = t_root + t_inter * t_dir;

  float3 local_normal = inter_point / ell_axes;
  float3 world_normal = transpose(R) * local_normal;

  return (Intersection){
      normalize(world_normal),
      t_inter * length(ray.dir),
      true,
  };
}

float3 get_color(float3 view_dir, float3 light_dir, float3 normal,
                 float3 base_color, float ambient) {
  float diffuse = max(dot(normal, light_dir), 0.0);

  float3 V = -view_dir;
  float3 H = normalize(light_dir + V);

  float spec_dot = dot(normal, H);
  float specular = 0.0;
  if (spec_dot > 0.0) {
    specular = pow(spec_dot, 80.0) * 0.6;
  }

  return min(base_color * (ambient + diffuse) + specular, 1.0);
}

kernel void render_kernel(device uint *output [[buffer(0)]],
                          device const float4 *centers [[buffer(1)]],
                          device const float4 *directions [[buffer(2)]],
                          constant FrameSpec &frame [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= frame.dims.x || gid.y >= frame.dims.y)
    return;

  float3 total_color = float3(0.0);
  float3 cam_root = frame.cam_root.xyz;
  float3 cam_dir = frame.cam_dir.xyz;
  float3 s1 = frame.cam_s1.xyz;
  float3 s2 = frame.cam_s2.xyz;

  uint total_samples = frame.oversamples * frame.oversamples;
  float subpixel_step = 1.0 / (float)frame.oversamples;

  for (uint j = 0; j < frame.oversamples; j++) {
    for (uint i = 0; i < frame.oversamples; i++) {
      float x_off = (float)gid.x + ((float)i + 0.5f) * subpixel_step -
                    (float)frame.dims.x / 2.0f;
      float y_off = (float)gid.y + ((float)j + 0.5f) * subpixel_step -
                    (float)frame.dims.y / 2.0f;

      Ray ray = (Ray){
          .root = cam_root,
          .dir = cam_dir + x_off * s1 + y_off * s2,
      };

      bool has_intersected = false;
      Intersection closest_intersection;

      for (uint p = 0; p < frame.particle_count; p++) {
        Intersection inter = intersect(ray, centers[p].xyz, directions[p].xyz,
                                       frame.ell_axes.xyz);
        if (inter.ok) {
          if (!has_intersected || inter.dist < closest_intersection.dist) {
            closest_intersection = inter;
            has_intersected = true;
          }
        }
      }

      if (has_intersected) {
        total_color += get_color(normalize(ray.dir), frame.light_dir.xyz,
                                 closest_intersection.normal,
                                 frame.ell_color.rgb, frame.ambient_light);
      } else {
        total_color += frame.bg_color.rgb;
      }
    }
  }

  float3 final_color = total_color / (float)total_samples;

  uint r = (uint)(saturate(final_color.r) * 255.0f);
  uint g = (uint)(saturate(final_color.g) * 255.0f);
  uint b = (uint)(saturate(final_color.b) * 255.0f);
  uint a = 255;

  output[gid.y * frame.dims.x + gid.x] =
      (a << 24) | (b << 16) | (g << 8) | (r << 0);
}
