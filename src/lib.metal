#include <metal_stdlib>
using namespace metal;

#define EPSILON_0 8.8541878188e-12
#define MU_0 1.25663706127e-6

struct GPUParams {
  uint particle_number;
  float rve_side_len;
  float epsilon_mat;
  float mag_dipole;
  float particle_vol;
  float e_sus_x;
  float e_sus_z;
  float radius_eq;
  float repulsion_factor;
  float t_drag;
  float r_drag;
  float4 ext_e_field;
  float4 ext_h_field;
};

float3 mod_rve(float3 r, float side_len) {
  return r - round(r / side_len) * side_len;
}

kernel void update_e_field(device const float4 *positions [[buffer(0)]],
                           device const float4 *el_dipole_moments [[buffer(1)]],
                           device float4 *e_field [[buffer(2)]],
                           constant GPUParams &params [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 e_field_i = params.ext_e_field.xyz;
  float3 pos_i = positions[i].xyz;

  for (uint j = 0; j < params.particle_number; j++) {
    if (i == j)
      continue;

    float3 r_ji = mod_rve(pos_i - positions[j].xyz, params.rve_side_len);
    float dist = length(r_ji);
    float3 r_ji_hat = r_ji / dist;

    float prefactor =
        1.0 / (4.0 * M_PI_F * EPSILON_0 * params.epsilon_mat) / pow(dist, 3);
    float3 e_ji =
        prefactor * (3.0 * dot(el_dipole_moments[j].xyz, r_ji_hat) * r_ji_hat -
                     el_dipole_moments[j].xyz);
    e_field_i += e_ji;
  }
  e_field[i] = float4(e_field_i, 0.0);
}

kernel void update_h_field(device const float4 *positions [[buffer(0)]],
                           device const float4 *directions [[buffer(1)]],
                           device float4 *h_field [[buffer(2)]],
                           constant GPUParams &params [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 h_field_i = params.ext_h_field.xyz;
  float3 pos_i = positions[i].xyz;

  for (uint j = 0; j < params.particle_number; j++) {
    if (i == j)
      continue;

    float3 r_ji = mod_rve(pos_i - positions[j].xyz, params.rve_side_len);
    float dist = length(r_ji);
    float3 r_ji_hat = r_ji / dist;

    float prefactor = params.mag_dipole / (4.0 * M_PI_F) / pow(dist, 3);
    float3 h_ji =
        prefactor *
        (3.0 * dot(directions[j].xyz, r_ji_hat) * r_ji_hat - directions[j].xyz);
    h_field_i += h_ji;
  }
  h_field[i] = float4(h_field_i, 0.0);
}

kernel void update_el_dipoles(device const float4 *e_field [[buffer(0)]],
                              device const float4 *directions [[buffer(1)]],
                              device float4 *el_dipole_moments [[buffer(2)]],
                              constant GPUParams &params [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 e_i = e_field[i].xyz;
  float3 d_i = directions[i].xyz;

  el_dipole_moments[i] =
      float4(params.particle_vol * EPSILON_0 *
                 (params.e_sus_x * e_i +
                  (params.e_sus_z - params.e_sus_x) * dot(d_i, e_i) * d_i),
             0.0);
}

kernel void update_p_vels(device const float4 *positions [[buffer(0)]],
                          device const float4 *directions [[buffer(1)]],
                          device const float4 *el_dipole_moments [[buffer(2)]],
                          device float4 *pos_vel [[buffer(3)]],
                          constant GPUParams &params [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 pos_i = positions[i].xyz;
  float3 dir_i = directions[i].xyz;
  float3 dipole_i = el_dipole_moments[i].xyz;
  float3 total_f = float3(0.0);

  for (uint j = 0; j < params.particle_number; j++) {
    if (i == j)
      continue;

    float3 r_ji = mod_rve(pos_i - positions[j].xyz, params.rve_side_len);
    float dist = length(r_ji);
    float3 r_ji_hat = r_ji / dist;

    // magnetic
    float f_h1_dot =
        dot(dir_i, directions[j].xyz) -
        5.0 * dot(r_ji_hat, directions[j].xyz) * dot(r_ji_hat, dir_i);
    float3 f_h1 = f_h1_dot * r_ji_hat;
    float3 f_h2 = dot(r_ji_hat, dir_i) * directions[j].xyz +
                  dot(r_ji_hat, directions[j].xyz) * dir_i;
    float3 f_h = 3.0 * MU_0 * pow(params.mag_dipole, 2) / 4.0 / M_PI_F /
                 pow(dist, 4) * (f_h1 + f_h2);

    // electric
    float f_e1_dot =
        dot(dipole_i, el_dipole_moments[j].xyz) -
        5.0 * dot(r_ji_hat, el_dipole_moments[j].xyz) * dot(r_ji_hat, dipole_i);
    float3 f_e1 = f_e1_dot * r_ji_hat;
    float3 f_e2 = dot(r_ji_hat, dipole_i) * el_dipole_moments[j].xyz +
                  dot(r_ji_hat, el_dipole_moments[j].xyz) * dipole_i;
    float3 f_e = 3.0 / EPSILON_0 / params.epsilon_mat / 2.0 / M_PI_F /
                 pow(dist, 4) * (f_e1 + f_e2);

    // repulsive
    float3 f_r = 3.0 * MU_0 * pow(params.mag_dipole, 2) /
                 (2.0 * M_PI_F * pow(2.0 * params.radius_eq, 4)) *
                 (exp(-params.repulsion_factor *
                      (dist / (2.0 * params.radius_eq) - 1.0)) *
                  r_ji_hat);

    total_f += f_h + f_e + f_r;
  }

  pos_vel[i] = float4(total_f / params.t_drag, 0.0);
}

kernel void update_positions(device float4 *positions [[buffer(0)]],
                             device const float4 *pos_vel [[buffer(1)]],
                             constant float &delta_t [[buffer(2)]],
                             constant GPUParams &params [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  positions[i] = float4(
      mod_rve(positions[i].xyz + pos_vel[i].xyz * delta_t, params.rve_side_len),
      0.0);
}

kernel void update_directions(device float4 *directions [[buffer(0)]],
                              device const float4 *h_field [[buffer(1)]],
                              device const float4 *e_field [[buffer(2)]],
                              constant float &delta_t [[buffer(3)]],
                              constant GPUParams &params [[buffer(4)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 h_i = h_field[i].xyz;
  float3 e_i = e_field[i].xyz;
  float3 d_i = directions[i].xyz;

  float3 magnetic = MU_0 * params.mag_dipole * (h_i - d_i * dot(h_i, d_i));
  float3 electric = params.particle_vol * EPSILON_0 *
                    (params.e_sus_z - params.e_sus_x) * dot(e_i, d_i) *
                    (e_i - d_i * dot(e_i, d_i));

  float3 dir_vel = (magnetic + electric) / params.r_drag;

  directions[i] = float4(normalize(d_i + delta_t * dir_vel), 0.0);
}

kernel void
check_finite_and_max_vel(device const float4 *el_dipole_moments [[buffer(0)]],
                         device const float4 *e_field [[buffer(1)]],
                         device const float4 *h_field [[buffer(2)]],
                         device const float4 *positions [[buffer(3)]],
                         device const float4 *directions [[buffer(4)]],
                         device const float4 *pos_vel [[buffer(5)]],
                         device float *output [[buffer(6)]],
                         constant GPUParams &params [[buffer(7)]],
                         uint i [[thread_position_in_grid]]) {
  if (i != 0)
    return;

  float max_v = 0.0;
  bool finite = true;

  for (uint j = 0; j < params.particle_number; j++) {
    max_v = max(max_v, length(pos_vel[j].xyz));

    finite &= all(isfinite(el_dipole_moments[j]));
    finite &= all(isfinite(e_field[j]));
    finite &= all(isfinite(h_field[j]));
    finite &= all(isfinite(positions[j]));
    finite &= all(isfinite(directions[j]));
    finite &= all(isfinite(pos_vel[j]));

    if (!finite)
      break;
  }

  output[0] = max_v;
  output[1] = finite ? 1.0 : 0.0;
}
