#include <metal_stdlib>
using namespace metal;

struct GPUParams {
  float4 ext_h_field;
  float4 ext_e_field;

  uint particle_number;
  float h_field_prefactor;
  float e_field_prefactor;
  float right_dipole_prefactor;

  float e_force_prefactor;
  float r_force_prefactor;
  float h_torque_prefactor;
  float e_torque_prefactor;

  float rve_side_len;
  float repulsion_factor;
  float radius_eq;
  float h_force_prefactor;
};

float pow3(float val) { return val * val * val; }
float pow4(float val) { return val * val * val * val; }

float3 mod_one(float3 r) { return r - round(r); }

kernel void update_e_dipole(device const float4 *e_field [[buffer(0)]],
                            device const float4 *direction [[buffer(1)]],
                            device float4 *e_dipole [[buffer(2)]],
                            constant GPUParams &params [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 e_i = e_field[i].xyz;
  float3 d_i = direction[i].xyz;

  float3 el_dipole = e_i + params.right_dipole_prefactor * dot(d_i, e_i) * d_i;

  e_dipole[i] = float4(el_dipole, 0.0);
}

float3 field_bracket_term(float3 rji, float3 dj) {
  return 3.0 * dot(dj, rji) * rji - dj;
}

kernel void update_e_field(device const float4 *position [[buffer(0)]],
                           device const float4 *e_dipole [[buffer(1)]],
                           device float4 *e_field [[buffer(2)]],
                           constant GPUParams &params [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 e_field_i = params.ext_e_field.xyz;
  float3 pos_i = position[i].xyz;

  for (uint j = 0; j < params.particle_number; j++) {
    if (i == j)
      continue;

    float3 r_ji = mod_one(pos_i - position[j].xyz);
    float dist = length(r_ji);
    float3 r_ji_hat = r_ji / dist;

    e_field_i += params.e_field_prefactor / pow3(dist) *
                 field_bracket_term(r_ji_hat, e_dipole[j].xyz);
  }
  e_field[i] = float4(e_field_i, 0.0);
}

kernel void update_h_field(device const float4 *position [[buffer(0)]],
                           device const float4 *direction [[buffer(1)]],
                           device float4 *h_field [[buffer(2)]],
                           constant GPUParams &params [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 h_field_i = params.ext_h_field.xyz;
  float3 pos_i = position[i].xyz;

  for (uint j = 0; j < params.particle_number; j++) {
    if (i == j)
      continue;

    float3 r_ji = mod_one(pos_i - position[j].xyz);
    float dist = length(r_ji);
    float3 r_ji_hat = r_ji / dist;

    h_field_i += params.h_field_prefactor / pow3(dist) *
                 field_bracket_term(r_ji_hat, direction[j].xyz);
  }
  h_field[i] = float4(h_field_i, 0.0);
}

float3 force_bracket_term(float3 rji, float3 di, float3 dj) {
  float f1_dot = dot(di, dj) - 5.0 * dot(rji, dj) * dot(rji, di);
  float3 f1 = f1_dot * rji;
  float3 f2 = dot(rji, di) * dj + dot(rji, dj) * di;
  return f1 + f2;
}

kernel void update_velocity(device const float4 *position [[buffer(0)]],
                            device const float4 *direction [[buffer(1)]],
                            device const float4 *e_dipole [[buffer(2)]],
                            device float4 *velocity [[buffer(3)]],
                            constant GPUParams &params [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 pos_i = position[i].xyz;
  float3 dir_i = direction[i].xyz;
  float3 dipole_i = e_dipole[i].xyz;

  float3 total_vel = float3(0.0);

  for (uint j = 0; j < params.particle_number; j++) {
    if (i == j)
      continue;

    float3 r_ji = mod_one(pos_i - position[j].xyz);
    float dist = length(r_ji);
    float3 r_ji_hat = r_ji / dist;

    // magnetic
    float3 f_h = (params.h_force_prefactor / pow4(dist)) *
                 force_bracket_term(r_ji_hat, dir_i, direction[j].xyz);

    // electric
    float3 f_e = (params.e_force_prefactor / pow4(dist)) *
                 force_bracket_term(r_ji_hat, dipole_i, e_dipole[j].xyz);

    // repulsive
    float exponent =
        -params.repulsion_factor *
        (dist * params.rve_side_len / (2.0 * params.radius_eq) - 1.0);
    float3 f_r = params.r_force_prefactor * (exp(exponent) * r_ji_hat);

    total_vel += f_h + f_e + f_r;
  }

  velocity[i] = float4(total_vel, 0.0);
}

kernel void update_position(device float4 *position [[buffer(0)]],
                            device const float4 *pos_vel [[buffer(1)]],
                            constant float &delta_t [[buffer(2)]],
                            constant GPUParams &params [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 new_pos = position[i].xyz + pos_vel[i].xyz * delta_t;
  position[i] = float4(mod_one(new_pos), 0.0);
}

kernel void update_direction(device float4 *direction [[buffer(0)]],
                             device const float4 *h_field [[buffer(1)]],
                             device const float4 *e_field [[buffer(2)]],
                             constant float &delta_t [[buffer(3)]],
                             constant GPUParams &params [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
  if (i >= params.particle_number)
    return;

  float3 h_i = h_field[i].xyz;
  float3 e_i = e_field[i].xyz;
  float3 d_i = direction[i].xyz;

  float3 magnetic = params.h_torque_prefactor * (h_i - d_i * dot(h_i, d_i));
  float3 electric =
      params.e_torque_prefactor * dot(e_i, d_i) * (e_i - d_i * dot(e_i, d_i));

  float3 dir_vel = magnetic + electric;

  direction[i] = float4(normalize(d_i + delta_t * dir_vel), 0.0);
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
