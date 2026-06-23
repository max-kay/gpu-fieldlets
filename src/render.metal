#include <metal_stdlib>
using namespace metal;

#define ELL_COLOR (float3(0.55, 0.08, 0.6))
#define H_COLOR (float3(0.1, 0.1, 0.8))
#define E_COLOR (float3(0.8, 0.1, 0.1))
#define VECTOR_RADIUS ((float)0.02)
#define LINE_RADIUS ((float)0.005)
#define LINE_COLOR (float3(0.2))
#define BG_COLOR (float3(1.0))

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
  uint2 sub_img_dims;

  uint particle_count;
  uint oversamples;
  float ambient_light;
  float culling_radius;

  float4 cam_root;
  float4 cam_s1, cam_s2, cam_dir;
  float4 ell_axes;
  float4 light_dir;
  float4 h_field;
  float4 e_field;
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

Intersection intersect_line(Ray ray, float3 start, float3 end, float radius) {
  float3 line_dir = end - start;
  float3 root_diff = ray.root - start;

  float3 v = normalize(line_dir);

  // Express ray properties relative to the cylinder axis
  float3 q_dir = ray.dir - dot(ray.dir, v) * v;
  float3 q_root = root_diff - dot(root_diff, v) * v;

  // Coefficients for the quadratic equation: A*t^2 + 2*B*t + C = 0
  float A = length_squared(q_dir);
  float B = dot(q_root, q_dir);
  float C = length_squared(q_root) - (radius * radius);

  // Check for parallel/degenerate cases
  if (abs(A) < 1e-6) {
    return (Intersection){float3(0.0), 0.0, false};
  }

  float discriminant = B * B - A * C;
  if (discriminant < 0.0) {
    return (Intersection){float3(0.0), 0.0, false};
  }

  // Find the closest intersection point along the ray
  float t_ray = (-B - sqrt(discriminant)) / A;
  if (t_ray < 0.0) {
    return (Intersection){float3(0.0), 0.0, false};
  }

  // Calculate exact hit position on the cylinder surface
  float3 hit_point = ray.root + ray.dir * t_ray;

  // Project hit point onto the finite line segment to find its longitudinal
  // position (t_line)
  float t_line = dot(hit_point - start, line_dir) / length_squared(line_dir);

  if (t_line < 0.0 || t_line > 1.0) {
    return (Intersection){float3(0.0), 0.0, false};
  }

  float3 axis_point = start + t_line * line_dir;
  float3 surface_normal = normalize(hit_point - axis_point);

  return (Intersection){
      .normal = surface_normal,
      .dist = t_ray,
      .ok = true,
  };
}

Intersection intersect_sphere(Ray ray, float3 center, float radius) {
  float3 oc = ray.root - center;

  float b = dot(oc, ray.dir);
  float c = length_squared(oc) - radius * radius;

  float discriminant = b * b - c;

  if (discriminant < 0.0) {
    return (Intersection){float3(0.0), 0.0, false};
  }

  float t = -b - sqrt(discriminant);

  if (t < 0.0) {
    t = -b + sqrt(discriminant);
    if (t < 0.0) {
      return (Intersection){float3(0.0), 0.0, false};
    }
  }

  float3 hit_point = ray.root + ray.dir * t;
  float3 surface_normal = normalize(hit_point - center);

  return (Intersection){
      .normal = surface_normal,
      .dist = t,
      .ok = true,
  };
}

Intersection intersect_ell(Ray ray, float3 center, float3 orient,
                           float3 ell_axes) {
  float3x3 R = get_rotation_matrix(orient);
  float3 a_dir = R * ray.dir;
  float3 a_root = R * (ray.root - center);

  float3 t_dir = a_dir / ell_axes;
  float3 t_root = a_root / ell_axes;

  float dot_val = dot(t_dir, t_root);
  float t_dir_norm2 = length_squared(t_dir);
  float discriminant =
      dot_val * dot_val - t_dir_norm2 * (length_squared(t_root) - 1.0);

  if (discriminant < 0) {
    return (Intersection){float3(0), 0.0, false};
  }

  float t_inter = -(dot_val + sqrt(discriminant)) / t_dir_norm2;
  float3 inter_point = t_root + t_inter * t_dir;

  float3 local_normal = inter_point / ell_axes;
  float3 world_normal = transpose(R) * local_normal;

  return (Intersection){
      normalize(world_normal),
      t_inter,
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

uint compute_visible_edges(float3 view_dir) {
  bool face_visible[6] = {
      dot(view_dir, float3(0.0, 0.0, 1.0)) >= 0.0,  // Top
      dot(view_dir, float3(0.0, 0.0, -1.0)) >= 0.0, // Bottom
      dot(view_dir, float3(0.0, 1.0, 0.0)) >= 0.0,  // Front
      dot(view_dir, float3(0.0, -1.0, 0.0)) >= 0.0, // Back
      dot(view_dir, float3(1.0, 0.0, 0.0)) >= 0.0,  // Right
      dot(view_dir, float3(-1.0, 0.0, 0.0)) >= 0.0  // Left
  };

  uint mask = 0;
  if (face_visible[0] || face_visible[3])
    mask |= (1 << 0); // top-back
  if (face_visible[0] || face_visible[5])
    mask |= (1 << 1); // top-left
  if (face_visible[0] || face_visible[2])
    mask |= (1 << 2); // top-front
  if (face_visible[0] || face_visible[4])
    mask |= (1 << 3); // top-right
  if (face_visible[3] || face_visible[4])
    mask |= (1 << 4); // mantle-back-right
  if (face_visible[3] || face_visible[5])
    mask |= (1 << 5); // mantle-back-left
  if (face_visible[2] || face_visible[5])
    mask |= (1 << 6); // mantle-front-left
  if (face_visible[2] || face_visible[4])
    mask |= (1 << 7); // mantle-front-right
  if (face_visible[1] || face_visible[3])
    mask |= (1 << 8); // bottom-back
  if (face_visible[1] || face_visible[5])
    mask |= (1 << 9); // bottom-left
  if (face_visible[1] || face_visible[2])
    mask |= (1 << 10); // bottom-front
  if (face_visible[1] || face_visible[4])
    mask |= (1 << 11); // bottom-right

  return mask;
}

constant float3 BOX_LINES[24] = {
    // top (edges 0, 1, 2, 3)
    float3(0.5, -0.5, 0.5),
    float3(-0.5, -0.5, 0.5),
    float3(-0.5, -0.5, 0.5),
    float3(-0.5, 0.5, 0.5),
    float3(-0.5, 0.5, 0.5),
    float3(0.5, 0.5, 0.5),
    float3(0.5, 0.5, 0.5),
    float3(0.5, -0.5, 0.5),
    // mantle (edges 4, 5, 6, 7)
    float3(0.5, -0.5, -0.5),
    float3(0.5, -0.5, 0.5),
    float3(-0.5, -0.5, -0.5),
    float3(-0.5, -0.5, 0.5),
    float3(-0.5, 0.5, -0.5),
    float3(-0.5, 0.5, 0.5),
    float3(0.5, 0.5, -0.5),
    float3(0.5, 0.5, 0.5),
    // bottom (edges 8, 9, 10, 11)
    float3(0.5, -0.5, -0.5),
    float3(-0.5, -0.5, -0.5),
    float3(-0.5, -0.5, -0.5),
    float3(-0.5, 0.5, -0.5),
    float3(-0.5, 0.5, -0.5),
    float3(0.5, 0.5, -0.5),
    float3(0.5, 0.5, -0.5),
    float3(0.5, -0.5, -0.5),
};

inline Ray generate_ray(float2 window_coord, uint2 dims, uint2 sub_pixel,
                        constant FrameSpec &frame) {
  float2 off = window_coord + (float2(sub_pixel) + float2(0.5)) /
                                  (float2(dims) * frame.oversamples);

  return (Ray){.root = frame.cam_root.xyz,
               .dir = normalize(frame.cam_dir.xyz + off.x * frame.cam_s1.xyz -
                                off.y * frame.cam_s2.xyz)};
}

float3 trace_main_scene(Ray ray, device const float4 *centers,
                        device const float4 *directions,
                        constant FrameSpec &frame) {
  bool has_intersected = false;
  Intersection closest_inter;
  float3 color = BG_COLOR;

  // Ellipsoid Particles
  for (uint p = 0; p < frame.particle_count; p++) {
    Intersection inter = intersect_ell(ray, centers[p].xyz, directions[p].xyz,
                                       frame.ell_axes.xyz);
    if (inter.ok && (!has_intersected || inter.dist < closest_inter.dist)) {
      closest_inter = inter;
      has_intersected = true;
      color = ELL_COLOR;
    }
  }

  uint edge_mask = compute_visible_edges(frame.cam_dir.xyz);

  // Wireframe Box Lines
  for (int l = 0; l < 12; l++) {
    if ((edge_mask & (1 << l)) == 0)
      continue;

    Intersection inter = intersect_line(ray, BOX_LINES[2 * l],
                                        BOX_LINES[2 * l + 1], LINE_RADIUS);
    if (inter.ok && (!has_intersected || inter.dist < closest_inter.dist)) {
      closest_inter = inter;
      has_intersected = true;
      color = LINE_COLOR;
    }
  }

  // Wireframe Box Line Caps
  for (int c = 0; c < 24; c++) {
    if ((edge_mask & (1 << (c / 2))) == 0)
      continue;

    Intersection inter = intersect_sphere(ray, BOX_LINES[c], LINE_RADIUS);
    if (inter.ok && (!has_intersected || inter.dist < closest_inter.dist)) {
      closest_inter = inter;
      has_intersected = true;
      color = LINE_COLOR;
    }
  }

  if (has_intersected) {
    return get_color(ray.dir, frame.light_dir.xyz, closest_inter.normal, color,
                     frame.ambient_light);
  }
  return BG_COLOR;
}

float3 trace_vector_field(Ray ray, float3 field_val, float3 field_color,
                          constant FrameSpec &frame) {
  Intersection inter =
      intersect_line(ray, float3(0.0), field_val * 0.5f, VECTOR_RADIUS);
  if (inter.ok) {
    return get_color(ray.dir, frame.light_dir.xyz, inter.normal, field_color,
                     frame.ambient_light);
  }
  return BG_COLOR;
}

// --- MAIN KERNEL ---
kernel void render_kernel(device uint *output [[buffer(0)]],
                          device const float4 *centers [[buffer(1)]],
                          device const float4 *directions [[buffer(2)]],
                          constant FrameSpec &frame [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {

  if (gid.x >= frame.dims.x || gid.y >= frame.dims.y)
    return;

  float aspect = (float)frame.dims.x / (float)frame.dims.y;
  float3 total_color = float3(0.0);

  // Coordinates mapping for different view windows
  float2 centered_main =
      float2(gid.xy) * 2.0f / float2(frame.dims) - float2(1.0f);

  float2 centered_h_window =
      float2(gid.x, gid.y - frame.dims.y + frame.sub_img_dims.y) * 2.0f /
          float2(frame.sub_img_dims) -
      float2(1.0f);

  // BUG FIXED HERE: Corrected viewport coordinate shifts
  float2 centered_e_window =
      float2(gid.x - frame.dims.x + frame.sub_img_dims.x,
             gid.y - frame.dims.y + frame.sub_img_dims.y) *
          2.0f / float2(frame.sub_img_dims) -
      float2(1.0f);

  // Determine which viewport region the current thread hits
  if (length(centered_main * float2(aspect, 1.0f)) < frame.culling_radius) {
    for (uint j = 0; j < frame.oversamples; j++) {
      for (uint i = 0; i < frame.oversamples; i++) {
        Ray ray = generate_ray(centered_main, frame.dims, uint2(i, j), frame);
        total_color += trace_main_scene(ray, centers, directions, frame);
      }
    }
  } else if (length(centered_h_window * float2(aspect, 1.0f)) <
                 frame.culling_radius &&
             length(frame.h_field.xyz) > 0.2) {
    for (uint j = 0; j < frame.oversamples; j++) {
      for (uint i = 0; i < frame.oversamples; i++) {
        Ray ray = generate_ray(centered_h_window, frame.sub_img_dims,
                               uint2(i, j), frame);
        total_color +=
            trace_vector_field(ray, frame.h_field.xyz, H_COLOR, frame);
      }
    }
  } else if (length(centered_e_window * float2(aspect, 1.0f)) <
                 frame.culling_radius &&
             length(frame.e_field.xyz) > 0.2) {
    for (uint j = 0; j < frame.oversamples; j++) {
      for (uint i = 0; i < frame.oversamples; i++) {
        Ray ray = generate_ray(centered_e_window, frame.sub_img_dims,
                               uint2(i, j), frame);
        total_color +=
            trace_vector_field(ray, frame.e_field.xyz, E_COLOR, frame);
      }
    }
  } else {
    total_color = BG_COLOR * (frame.oversamples * frame.oversamples);
  }

  float3 final_color =
      total_color / (float)(frame.oversamples * frame.oversamples);

  uint r = (uint)(saturate(final_color.r) * 255.0f);
  uint g = (uint)(saturate(final_color.g) * 255.0f);
  uint b = (uint)(saturate(final_color.b) * 255.0f);
  uint a = 255;

  output[gid.y * frame.dims.x + gid.x] = (a << 24) | (b << 16) | (g << 8) | r;
}
