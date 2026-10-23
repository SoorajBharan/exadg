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
#include <exadg/utilities/section_profile.h>

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

  AssertThrow(not use_divergence_penalty or projection_operator->get_divergence_kernel_data().type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm, dealii::ExcMessage("Enriched penalty step: only TypePenaltyParameter::ConvectiveTerm."));
  AssertThrow(not use_continuity_penalty or projection_operator->get_continuity_kernel_data().type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm,  dealii::ExcMessage("Enriched penalty step: only TypePenaltyParameter::ConvectiveTerm."));
  time_step_size = time_step_size_in;


  // (1) tau per active cell, on ExaDG's MatrixFree with ExaDG's quadrature
  auto const &       mf_std   = projection_operator->get_matrix_free();
  unsigned int const dof_std  = projection_operator->get_dof_index();
  unsigned int const quad_std = projection_operator->get_quad_index();
  auto const &       tria     = 
    mf_std.get_dof_handler(dof_std).get_triangulation();

  std::vector<double> tau_divergence_cell(tria.n_active_cells(), 0.0);
  std::vector<double> tau_continuity_cell(tria.n_active_cells(), 0.0);

  velocity_total.update_ghost_values();

  CellIntegratorU    phi_std(mf_std, dof_std, quad_std);

  unsigned int const n_batches_std = mf_std.n_cell_batches() + mf_std.n_ghost_cell_batches();
  for(unsigned int cell = 0; cell < n_batches_std; ++cell)
  {
    phi_std.reinit(cell);
    phi_std.read_dof_values(velocity_total);
    phi_std.evaluate(dealii::EvaluationFlags::values);

    scalar volume      = dealii::make_vectorized_array<Number>(0.0);
    scalar norm_U_mean = dealii::make_vectorized_array<Number>(0.0);
    for(unsigned int q = 0; q < phi_std.n_q_points; ++q)
    {
      volume += phi_std.JxW(q);
      norm_U_mean += phi_std.JxW(q) * phi_std.get_value(q).norm();
    }
    norm_U_mean /= volume;

    for(unsigned int l = 0; l < mf_std.n_active_entries_per_cell_batch(cell);
    ++l)
    {
      unsigned int const index = mf_std.get_cell_iterator(cell,
                                                          l)->active_cell_index();
      double const h_eff = std::pow(volume[l], 1.0 / (double)dim) /
        (double)(degree_velocity + 1);

      tau_divergence_cell[index] = divergence_penalty_factor * norm_U_mean[l] * h_eff;
      tau_continuity_cell[index] = continuity_penalty_factor * norm_U_mean[l];
    } 
  } 

  // (2) the same values on the cell batches of matrix_free_en (same 
  auto const &       mf        = *function_enrichment->get_matrix_free();
  unsigned int const n_batches = mf.n_cell_batches() + mf.n_ghost_cell_batches();
  tau_divergence.resize(n_batches);
  tau_continuity.resize(n_batches);

  for(unsigned int cell = 0; cell < n_batches; ++cell)
  {
    tau_divergence[cell] = dealii::make_vectorized_array<Number>(0.0);
    tau_continuity[cell] = dealii::make_vectorized_array<Number>(0.0);
    for(unsigned int l = 0; l < mf.n_active_entries_per_cell_batch(cell); ++l)
    {
      unsigned int const index = mf.get_cell_iterator(cell, l)->active_cell_index();
      tau_divergence[cell][l]  = tau_divergence_cell[index];
      tau_continuity[cell][l]  = tau_continuity_cell[index];
    }
  }
}


template<int dim, typename Number> 
void
EnrichedProjectionOperator<dim, Number>::cell_loop(VectorType const & src_bar,
                                                   VectorType const & src_tilde,
                                                   VectorType *       dst_bar,
                                                   VectorType &       dst_tilde)
const
{
  auto const & mf = *function_enrichment->get_matrix_free();

  EnrichedCellIntegrator phi(mf, *function_enrichment);

  for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
  {
    if(mf.get_cell_iterator(cell, 0)->active_fe_index() != 1)
      continue;

    phi.reinit(cell);
    phi.gather_evaluate(src_bar, src_tilde,
                        dealii::EvaluationFlags::values |
                        dealii::EvaluationFlags::gradients);

    scalar const tau = time_step_size * tau_divergence[cell];

    for(unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      // mass: u_h . v ;  divergence penalty: dt tau_D div(u_h) div(v) = (dt   tau_D div(u_h) I) : grad(v)
      // u_h = U_bar + psi U_tilde tested with psi v_tilde, psi U_tilde tested   with v_bar
      vector const value_flux    = phi.get_value(q);
      vector const value_flux_en = phi.get_value_enrichment(q);
      tensor       gradient_flux, gradient_flux_en;
      if(use_divergence_penalty)
      {
        scalar const divergence    = phi.get_divergence(q);
        scalar const divergence_en = phi.get_divergence_enrichment(q);
        for(unsigned int d = 0; d < dim; ++d)
        {
          gradient_flux[d][d]    = tau * divergence;
          gradient_flux_en[d][d] = tau * divergence_en;
        }
      }
      phi.submit_value_and_gradient_split(value_flux_en, gradient_flux_en,
                                          value_flux, gradient_flux, q);
    }
    phi.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    phi.phi_dg.distribute_local_to_global(*dst_bar);
    phi.phi_en.distribute_local_to_global(dst_tilde);
  }
}

template<int dim, typename Number>
void
EnrichedProjectionOperator<dim, Number>::inner_face_loop(VectorType const &
                                                         src_bar,
                                                         VectorType const &
                                                         src_tilde,
                                                         VectorType *
                                                         dst_bar,
                                                         VectorType &
                                                         dst_tilde) const
{
  auto const & mf = *function_enrichment->get_matrix_free();

  unsigned int const dof_index  =
    function_enrichment->get_dof_index_shadow_vector();
  unsigned int const quad_index = function_enrichment->get_quad_index();

  // values of u_h, of its enrichment part psi U_tilde, and tau_C on one side of a face

  auto const evaluate_side = [&](bool const            interior,
                                 unsigned int const    fe_index,
                                 unsigned int const    face,
                                 std::vector<vector> & values,
                                 std::vector<vector> & values_en,
                                 scalar &              tau) {
      if(fe_index == 1)
      {
        EnrichedFaceIntegrator phi(mf, *function_enrichment, interior, 1);
        phi.reinit(face);
        phi.gather_evaluate(src_bar, src_tilde, dealii::EvaluationFlags::values);
        values.resize(phi.n_q_points);
        values_en.resize(phi.n_q_points);
        for(unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          values[q]    = phi.get_value(q);
          values_en[q] = phi.get_value_enrichment(q);
        }
        tau = phi.phi_dg.read_cell_data(tau_continuity);
      }
      else // halo: U_bar only, no enrichment part
    {
        FaceIntegratorU phi(mf, interior, dof_index, quad_index, 0, fe_index);
        phi.reinit(face);
        phi.gather_evaluate(src_bar, dealii::EvaluationFlags::values);
        values.resize(phi.n_q_points);
        values_en.resize(phi.n_q_points);
        for(unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          values[q]    = phi.get_value(q);
          values_en[q] = vector();
        }
        tau = phi.read_cell_data(tau_continuity);
      }
    };

  // test one side: flux of u_h with psi v_tilde, flux of psi U_tilde with v_bar,  and scatter
  // (psi is read from the cache of FunctionEnrichment, no DoF values are needed)
  auto const test_side = [&](bool const                  interior,
                             unsigned int const          fe_index,
                             unsigned int const          face,
                             std::vector<vector> const & flux,
                             std::vector<vector> const & flux_en) {
      if(fe_index == 1)
      {
        EnrichedFaceIntegrator phi(mf, *function_enrichment, interior, 1);
        phi.reinit(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value_split(flux_en[q], flux[q], q);
        phi.integrate(dealii::EvaluationFlags::values);
        phi.phi_dg.distribute_local_to_global(*dst_bar);
        phi.phi_en.distribute_local_to_global(dst_tilde);
      } 
      else // halo: U_bar rows only
    {
        FaceIntegratorU phi(mf, interior, dof_index, quad_index, 0, fe_index);
        phi.reinit(face);
        for(unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(flux_en[q], q);
        phi.integrate(dealii::EvaluationFlags::values);
        phi.distribute_local_to_global(*dst_bar);
      }
    };

  std::vector<vector> values_m, values_p, values_m_en, values_p_en;
  std::vector<vector> flux_m, flux_p, flux_m_en, flux_p_en;

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
    evaluate_side(true, fe_m, face, values_m, values_m_en, tau_m);
    evaluate_side(false, fe_p, face, values_p, values_p_en, tau_p);
    scalar const tau = time_step_size * 0.5 * (tau_m + tau_p);

    // normal vector (interior side)
    FaceIntegratorU normal_evaluator(mf, true, dof_index, quad_index, 0, fe_m);
    normal_evaluator.reinit(face);

    flux_m.resize(values_m.size());
    flux_p.resize(values_m.size());
    flux_m_en.resize(values_m.size());
    flux_p_en.resize(values_m.size());

    for(unsigned int q = 0; q < values_m.size(); ++q)
    {
      vector const n = normal_evaluator.get_normal_vector(q);

      // normal component of the jump of u_h and of psi U_tilde
      scalar const jump    = (values_m[q] - values_p[q]) * n;
      scalar const jump_en = (values_m_en[q] - values_p_en[q]) * n;

      flux_m[q]    = tau * jump * n;
      flux_p[q]    = -flux_m[q];
      flux_m_en[q] = tau * jump_en * n;
      flux_p_en[q] = -flux_m_en[q];
    }

    test_side(true, fe_m, face, flux_m, flux_m_en);
    test_side(false, fe_p, face, flux_p, flux_p_en);
  }
}

template<int dim, typename Number>
void
EnrichedProjectionOperator<dim, Number>::vmult(BlockVectorType & dst, BlockVectorType const & src) const
{
#ifdef EXADG_PROFILE
  auto t_prof = SectionProfile::Clock::now();
#endif 
  auto const & mf = *function_enrichment->get_matrix_free();

  dst.block(0).zero_out_ghost_values();
  dst.block(1).zero_out_ghost_values();

  // (1) K_bar_bar U_bar on the whole domain: ExaDG's projection operator
  projection_operator->vmult(dst.block(0), src.block(0));
#ifdef EXADG_PROFILE
  SectionProfile::lap("pen vmult 1 ExaDG projection operator", t_prof);
#endif

  unsigned int const dof_index_shadow = function_enrichment->get_dof_index_shadow_vector();
  if(u_bar.get_partitioner() != mf.get_vector_partitioner(dof_index_shadow))
  {
    mf.initialize_dof_vector(u_bar, dof_index_shadow);
    mf.initialize_dof_vector(correction_bar, dof_index_shadow);
  }
  correction_bar = 0.0;

  function_enrichment->copy_global_dg_to_wall_layout(src.block(0), u_bar);
  u_bar.update_ghost_values();
  src.block(1).update_ghost_values();

  dst.block(1) = 0.0;
#ifdef EXADG_PROFILE
  SectionProfile::lap("pen vmult 2 alloc + copy global->wall + ghosts", t_prof);
#endif

  // (3) one pass, u_h = U_bar + psi U_tilde:
  //     K_bar_tilde U_tilde                       -> correction_bar (v_bar rows)
  //     K_tilde_bar U_bar + K_tilde_tilde U_tilde -> dst.block(1)   (psi v_tilde rows)
  //     (K_bar_bar U_bar is already in step 1)
  cell_loop(u_bar, src.block(1), &correction_bar, dst.block(1));
  if(use_continuity_penalty)
    inner_face_loop(u_bar, src.block(1), &correction_bar, dst.block(1));
#ifdef EXADG_PROFILE
  SectionProfile::lap("pen vmult 3 fused pass (U_bar, U_tilde)", t_prof);
#endif

  correction_bar.compress(dealii::VectorOperation::add);
  dst.block(1).compress(dealii::VectorOperation::add);

  function_enrichment->add_wall_layout_to_global_dg(correction_bar, dst.block(0));
#ifdef EXADG_PROFILE
  SectionProfile::lap("pen vmult 5 compress + add wall->global", t_prof);
#endif

  src.block(1).zero_out_ghost_values();
}

template class EnrichedProjectionOperator<2, float>;
template class EnrichedProjectionOperator<2, double>;
template class EnrichedProjectionOperator<3, float>;
template class EnrichedProjectionOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
