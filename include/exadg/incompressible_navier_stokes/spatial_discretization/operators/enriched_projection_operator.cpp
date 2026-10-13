/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */


#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/enriched_projection_operator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
EnrichedProjectionOperator<dim, Number>::EnrichedProjectionOperator()
  : projection_operator(nullptr),
    degree_velocity(1),
    use_divergence_penalty(false),
    use_continuity_penalty(false),
    divergence_penalty_factor(1.0),
    continuity_penalty_factor(1.0),
    time_step_size(0.0)
{
}

template<int dim, typename Number>
void
EnrichedProjectionOperator<dim, Number>::initialize(
  ProjectionOperator<dim, Number> const &          projection_operator_in,
  std::shared_ptr<FunctionEnrichment<dim, Number>> function_enrichment_in,
  unsigned int const                               degree_velocity_in,
  bool const                                       use_divergence_penalty_in,
  bool const                                       use_continuity_penalty_in,
  double const                                     divergence_penalty_factor_in,
  double const                                     continuity_penalty_factor_in)
{
  projection_operator       = &projection_operator_in;
  function_enrichment       = function_enrichment_in;
  degree_velocity           = degree_velocity_in;
  use_divergence_penalty    = use_divergence_penalty_in;
  use_continuity_penalty    = use_continuity_penalty_in;
  divergence_penalty_factor = divergence_penalty_factor_in;
  continuity_penalty_factor = continuity_penalty_factor_in;
}

template<int dim, typename Number>
void
EnrichedProjectionOperator<dim, Number>::update(VectorType const & velocity_total,
                                                double const       time_step_size_in)
{
  time_step_size = time_step_size_in;

  auto const & mf = *function_enrichment->get_matrix_free();

  // full velocity in shadow layout (enriched + halo cells)
  VectorType u_shadow;
  mf.initialize_dof_vector(u_shadow, function_enrichment->get_dof_index_shadow_vector());
  function_enrichment->copy_global_dg_to_wall_layout(velocity_total, u_shadow);
  u_shadow.update_ghost_values();

  unsigned int const n_batches = mf.n_cell_batches() + mf.n_ghost_cell_batches();
  tau_divergence.resize(n_batches);
  tau_continuity.resize(n_batches);

  unsigned int const dof_index  = function_enrichment->get_dof_index_shadow_vector();
  unsigned int const quad_index = function_enrichment->get_quad_index();

  // same formulas as ExaDG's ConvectiveTerm penalty parameters:
  //   tau_D = factor_D * mean|u| * h_eff,  h_eff = volume^(1/dim) / (k+1)
  //   tau_C = factor_C * mean|u|
  for(unsigned int fe_index = 1; fe_index <= 2; ++fe_index)
  {
    CellIntegratorU phi(mf, dof_index, quad_index, 0, fe_index);

    for(unsigned int cell = 0; cell < n_batches; ++cell)
    {
      if(mf.get_cell_iterator(cell, 0)->active_fe_index() != fe_index)
        continue;

      phi.reinit(cell);
      phi.gather_evaluate(u_shadow, dealii::EvaluationFlags::values);

      scalar volume      = dealii::make_vectorized_array<Number>(0.0);
      scalar norm_U_mean = dealii::make_vectorized_array<Number>(0.0);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        volume += phi.JxW(q);
        norm_U_mean += phi.JxW(q) * phi.get_value(q).norm();
      }
      norm_U_mean /= volume;

      scalar h_eff;
      for(unsigned int l = 0; l < scalar::size(); ++l)
        h_eff[l] = std::pow(volume[l], 1.0 / (double)dim) / (double)(degree_velocity + 1);

      tau_divergence[cell] = divergence_penalty_factor * norm_U_mean * h_eff;
      tau_continuity[cell] = continuity_penalty_factor * norm_U_mean;
    }
  }

  u_shadow.zero_out_ghost_values();
}

template<int dim, typename Number>
void
EnrichedProjectionOperator<dim, Number>::cell_loop(VectorType const & src_bar,
                                                   VectorType const & src_tilde,
                                                   VectorType *       dst_bar,
                                                   VectorType &       dst_tilde) const
{
  auto const & mf = *function_enrichment->get_matrix_free();

  EnrichedCellIntegrator phi(mf, *function_enrichment);

  for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
  {
    if(mf.get_cell_iterator(cell, 0)->active_fe_index() != 1)
      continue;

    phi.reinit(cell);
    phi.gather_evaluate(src_bar, src_tilde,
                        dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    scalar const tau = time_step_size * tau_divergence[cell];

    for(unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      // mass: u_h . v ;  divergence penalty: dt tau_D div(u_h) div(v) = (dt tau_D div(u_h) I) : grad(v)
      vector const value_flux = phi.get_value(q);
      tensor       gradient_flux;
      if(use_divergence_penalty)
      {
        scalar const divergence = phi.get_divergence(q);
        for(unsigned int d = 0; d < dim; ++d)
          gradient_flux[d][d] = tau * divergence;
      }
      phi.submit_value_and_gradient(value_flux, gradient_flux, q);
    }

    phi.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    if(dst_bar != nullptr)
      phi.phi_dg.distribute_local_to_global(*dst_bar);
    phi.phi_en.distribute_local_to_global(dst_tilde);
  }
}

template<int dim, typename Number>
void
EnrichedProjectionOperator<dim, Number>::inner_face_loop(VectorType const & src_bar,
                                                         VectorType const & src_tilde,
                                                         VectorType *       dst_bar,
                                                         VectorType &       dst_tilde) const
{
  auto const & mf = *function_enrichment->get_matrix_free();

  unsigned int const dof_index  = function_enrichment->get_dof_index_shadow_vector();
  unsigned int const quad_index = function_enrichment->get_quad_index();

  // values of u_h and tau_C on one side of a face
  auto const evaluate_side = [&](bool const               interior,
                                 unsigned int const       fe_index,
                                 unsigned int const       face,
                                 std::vector<vector> &    values,
                                 scalar &                 tau) {
    if(fe_index == 1)
    {
      EnrichedFaceIntegrator phi(mf, *function_enrichment, interior, 1);
      phi.reinit(face);
      phi.gather_evaluate(src_bar, src_tilde, dealii::EvaluationFlags::values);
      values.resize(phi.n_q_points);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        values[q] = phi.get_value(q);
      tau = phi.phi_dg.read_cell_data(tau_continuity);
    }
    else // halo: U_bar only
    {
      FaceIntegratorU phi(mf, interior, dof_index, quad_index, 0, fe_index);
      phi.reinit(face);
      phi.gather_evaluate(src_bar, dealii::EvaluationFlags::values);
      values.resize(phi.n_q_points);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        values[q] = phi.get_value(q);
      tau = phi.read_cell_data(tau_continuity);
    }
  };

  // test one side with the given value flux and scatter
  auto const test_side = [&](bool const                  interior,
                             unsigned int const          fe_index,
                             unsigned int const          face,
                             std::vector<vector> const & flux) {
    if(fe_index == 1)
    {
      EnrichedFaceIntegrator phi(mf, *function_enrichment, interior, 1);
      phi.reinit(face);
      phi.gather_evaluate(src_bar, src_tilde, dealii::EvaluationFlags::values); // y, u_tau -> psi
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(flux[q], q);
      phi.integrate(dealii::EvaluationFlags::values);
      if(dst_bar != nullptr)
        phi.phi_dg.distribute_local_to_global(*dst_bar);
      phi.phi_en.distribute_local_to_global(dst_tilde);
    }
    else if(dst_bar != nullptr) // halo: U_bar rows only
    {
      FaceIntegratorU phi(mf, interior, dof_index, quad_index, 0, fe_index);
      phi.reinit(face);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(flux[q], q);
      phi.integrate(dealii::EvaluationFlags::values);
      phi.distribute_local_to_global(*dst_bar);
    }
  };

  std::vector<vector> values_m, values_p, flux_m, flux_p;

  for(unsigned int face = 0; face < mf.n_inner_face_batches(); ++face)
  {
    auto const         range = std::make_pair(face, face + 1);
    unsigned int const fe_m  = mf.get_face_active_fe_index(range, true);
    unsigned int const fe_p  = mf.get_face_active_fe_index(range, false);

    if(fe_m != 1 and fe_p != 1)
      continue;
    if(fe_m == 0 or fe_p == 0)
      continue;

    scalar tau_m, tau_p;
    evaluate_side(true, fe_m, face, values_m, tau_m);
    evaluate_side(false, fe_p, face, values_p, tau_p);
    scalar const tau = time_step_size * 0.5 * (tau_m + tau_p);

    // normal vector (interior side)
    FaceIntegratorU normal_evaluator(mf, true, dof_index, quad_index, 0, fe_m);
    normal_evaluator.reinit(face);

    flux_m.resize(values_m.size());
    flux_p.resize(values_m.size());
    for(unsigned int q = 0; q < values_m.size(); ++q)
    {
      vector const n    = normal_evaluator.get_normal_vector(q);
      scalar const jump = (values_m[q] - values_p[q]) * n; // normal component of the jump
      flux_m[q]         = tau * jump * n;
      flux_p[q]         = -flux_m[q];
    }

    test_side(true, fe_m, face, flux_m);
    test_side(false, fe_p, face, flux_p);
  }
}

template<int dim, typename Number>
void
EnrichedProjectionOperator<dim, Number>::vmult(BlockVectorType & dst, BlockVectorType const & src) const
{
  auto const & mf = *function_enrichment->get_matrix_free();

  dst.block(0).zero_out_ghost_values();
  dst.block(1).zero_out_ghost_values();

  // (1) K_bar_bar U_bar on the whole domain: ExaDG's projection operator
  projection_operator->vmult(dst.block(0), src.block(0));

  // (2) U_bar in shadow layout, zero vectors for the two passes
  VectorType u_bar, zero_bar, zero_tilde, correction_bar;
  mf.initialize_dof_vector(u_bar, function_enrichment->get_dof_index_shadow_vector());
  mf.initialize_dof_vector(zero_bar, function_enrichment->get_dof_index_shadow_vector());
  mf.initialize_dof_vector(zero_tilde, function_enrichment->get_dof_index_en());
  mf.initialize_dof_vector(correction_bar, function_enrichment->get_dof_index_shadow_vector());

  function_enrichment->copy_global_dg_to_wall_layout(src.block(0), u_bar);
  u_bar.update_ghost_values();
  src.block(1).update_ghost_values();
  zero_bar.update_ghost_values();
  zero_tilde.update_ghost_values();

  dst.block(1) = 0.0;

  // Pass A: U_tilde only -> K_bar_tilde U_tilde (correction_bar), K_tilde_tilde U_tilde
  cell_loop(zero_bar, src.block(1), &correction_bar, dst.block(1));
  if(use_continuity_penalty)
    inner_face_loop(zero_bar, src.block(1), &correction_bar, dst.block(1));

  // Pass B: U_bar only -> K_tilde_bar U_bar (U_bar rows already in step 1)
  cell_loop(u_bar, zero_tilde, nullptr, dst.block(1));
  if(use_continuity_penalty)
    inner_face_loop(u_bar, zero_tilde, nullptr, dst.block(1));

  correction_bar.compress(dealii::VectorOperation::add);
  dst.block(1).compress(dealii::VectorOperation::add);

  function_enrichment->add_wall_layout_to_global_dg(correction_bar, dst.block(0));

  src.block(1).zero_out_ghost_values();
}

template class EnrichedProjectionOperator<2, float>;
template class EnrichedProjectionOperator<2, double>;
template class EnrichedProjectionOperator<3, float>;
template class EnrichedProjectionOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
