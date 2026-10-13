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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/enriched_momentum_operator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
EnrichedMomentumOperator<dim, Number>::EnrichedMomentumOperator()
  : momentum_operator(nullptr),
  penalty_factor(0.0),
  enrichment_aware_penalty(false),
  wall_penalty_factor(1.0)
{
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::initialize(
  MomentumOperator<dim, Number> const &            momentum_operator_in,
  std::shared_ptr<FunctionEnrichment<dim, Number>> function_enrichment_in)
{
  momentum_operator   = &momentum_operator_in;
  function_enrichment = function_enrichment_in;

  auto const & op_data = momentum_operator->get_data();
  auto const   vk_data = momentum_operator->get_viscous_kernel_data();

  AssertThrow(not op_data.convective_problem,
              dealii::ExcMessage("EnrichedMomentumOperator: convective term not supported."));
  AssertThrow(vk_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation,
              dealii::ExcMessage("EnrichedMomentumOperator: only LaplaceFormulation supported."));
  AssertThrow(not vk_data.viscosity_is_variable,
              dealii::ExcMessage("EnrichedMomentumOperator: only constant viscosity supported."));

  update_penalty_parameter();
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::update_penalty_parameter()
{
  auto const & mf_en = *function_enrichment->get_matrix_free();

  unsigned int const n_batches = mf_en.n_cell_batches() + mf_en.n_ghost_cell_batches();
  penalty_parameter.resize(n_batches);

  // surface/volume per cell: interior faces weighted 1/2, physical boundary faces weighted 1
  for(unsigned int b = 0; b < n_batches; ++b)
  {
    scalar value = dealii::make_vectorized_array<Number>(0.0);
    for(unsigned int l = 0; l < mf_en.n_active_entries_per_cell_batch(b); ++l)
    {
      auto const cell    = mf_en.get_cell_iterator(b, l);
      double     surface = 0.0;
      for(auto const f : cell->face_indices())
      {
        double const factor =
          (cell->at_boundary(f) and not cell->has_periodic_neighbor(f)) ? 1.0 : 0.5;
        surface += factor * cell->face(f)->measure();
      }
      value[l] = surface / cell->measure();
    }
    penalty_parameter[b] = value;
  }

  auto const &       dof_handler_shadow = function_enrichment->get_dof_handler_shadow_vector();
  unsigned int const degree             = dof_handler_shadow.get_fe(1).degree;

  penalty_factor =
    IP::get_penalty_factor<dim, Number>(degree,
                                        get_element_type(dof_handler_shadow.get_triangulation()),
                                        momentum_operator->get_viscous_kernel_data().IP_factor);
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::update_enriched_penalty_parameter() const
{
  auto const & mf = *function_enrichment->get_matrix_free();

  /*
   * Krank et al. (2018), Eq. (35)-(36) / thesis Eq. (3.14)-(3.15): the viscous step uses the
   * polynomial interior penalty parameter also on enriched cells,
   *   tau_IP,e = (k+1)^2 (A(dOmega_e^Gamma)/2 + A(dOmega_e^D)) / V(Omega_e),
   * which is admissible since the NIPG (s = -1) is stable for any tau_IP > 0.
   * The enrichment tau below is only computed if enrichment_aware_penalty is set.
   */
  if(not enrichment_aware_penalty)
  {
    unsigned int const n_batches_poly = mf.n_cell_batches() + mf.n_ghost_cell_batches();

    penalty_enrichment.resize(n_batches_poly);
    function_enrichment->wall_penalty.resize(n_batches_poly);
    for(unsigned int b = 0; b < n_batches_poly; ++b)
    {
      penalty_enrichment[b]               = dealii::make_vectorized_array<Number>(0.0);
      function_enrichment->wall_penalty[b] = wall_penalty_factor * penalty_factor * penalty_parameter[b];
    }
    return;
  }


  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number>     CellIntegratorScalar;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number> FaceIntegratorScalar;

  unsigned int const lanes     = scalar::size();
  unsigned int const n_batches = mf.n_cell_batches() + mf.n_ghost_cell_batches();

  // any scalar DoF index of matrix_free_en: the evaluators below only provide JxW
  unsigned int const dof_y = function_enrichment->get_dof_index_en_scalar();
  unsigned int const quad  = function_enrichment->get_quad_index();

  // per cell (batch * lanes + lane): sum_F w_F int_F |grad psi|^2  and  int_K |grad psi|^2,
  // with the cached grad psi (incl. the grad tau_w contribution)
  std::vector<double> numerator(n_batches * lanes, 0.0), denominator(n_batches * lanes, 0.0);
  // ---- volume: int_K |grad psi|^2 ------------------------------------------------
  {
    CellIntegratorScalar JxW_eval(mf, dof_y, quad, 0, 1);

    for(unsigned int cell = 0; cell < n_batches; ++cell)
    {
      if(mf.get_cell_iterator(cell, 0)->active_fe_index() != 1)
        continue;

      JxW_eval.reinit(cell);

      scalar integral = dealii::make_vectorized_array<Number>(0.0);
      for(unsigned int q = 0; q < JxW_eval.n_q_points; ++q)
      {
        integral += function_enrichment->get_grad_psi_cell(cell, q).norm_square() * JxW_eval.JxW(q);
      }

      for(unsigned int l = 0; l < lanes; ++l)
        denominator[cell * lanes + l] = integral[l];
    }
  }

  // ---- faces: w_F int_F |grad psi|^2 on the enriched side(s) ------------------------
  auto const add_face_side = [&](unsigned int const face, bool const interior, double const weight) {
    FaceIntegratorScalar JxW_eval(mf, interior, dof_y, quad, 0, 1);
    JxW_eval.reinit(face);

    // \int_F |\nabla\psi|^2 on this side of the face
    scalar integral = dealii::make_vectorized_array<Number>(0.0);
    for(unsigned int q = 0; q < JxW_eval.n_q_points; ++q)
    {
      integral +=
        function_enrichment->get_grad_psi_face(face, interior, q).norm_square() * JxW_eval.JxW(q);
    }
    auto const & info  = mf.get_face_info(face);
    auto const & cells = interior ? info.cells_interior : info.cells_exterior;
    for(unsigned int l = 0; l < lanes; ++l)
    {

      if(cells[l] != dealii::numbers::invalid_unsigned_int)
        numerator[cells[l]] += weight * integral[l];
    }
  };

  for(unsigned int face = 0; face < mf.n_inner_face_batches(); ++face)
  {
    auto const range = std::make_pair(face, face + 1);
    if(mf.get_face_active_fe_index(range, true) == 1)
      add_face_side(face, true, 0.5);
    if(mf.get_face_active_fe_index(range, false) == 1)
      add_face_side(face, false, 0.5);
  }

  unsigned int const b_begin = mf.n_inner_face_batches();
  unsigned int const b_end   = b_begin + mf.n_boundary_face_batches();
  for(unsigned int face = b_begin; face < b_end; ++face)
    if(mf.get_face_active_fe_index(std::make_pair(face, face + 1), true) == 1)
      add_face_side(face, true, 1.0);

  // ---- tau^en = IP (l+1)^2 numerator / denominator ----------------------------------
  unsigned int const degree_en =
    function_enrichment->get_dof_handler_en_vector().get_fe(1).degree;
  double const factor = momentum_operator->get_viscous_kernel_data().IP_factor *
                        (degree_en + 1.0) * (degree_en + 1.0);

  penalty_enrichment.resize(n_batches);
  for(unsigned int b = 0; b < n_batches; ++b)
    for(unsigned int l = 0; l < lanes; ++l)
    {
      double const den       = denominator[b * lanes + l];
      penalty_enrichment[b][l] = (den > 0.0) ? factor * numerator[b * lanes + l] / den : 0.0;
    }


  // the total tau of each cell, for the consistent wall flux in FunctionEnrichment
  function_enrichment->wall_penalty.resize(n_batches);
  for(unsigned int b = 0; b < n_batches; ++b)
    function_enrichment->wall_penalty[b] =
      wall_penalty_factor * std::max(penalty_factor * penalty_parameter[b], penalty_enrichment[b]);
}

template<int dim, typename Number>
typename EnrichedMomentumOperator<dim, Number>::scalar
EnrichedMomentumOperator<dim, Number>::read_penalty_polynomial(FaceIntegratorU & phi) const
{
  return phi.read_cell_data(penalty_parameter) * penalty_factor;
}

template<int dim, typename Number>
typename EnrichedMomentumOperator<dim, Number>::scalar
EnrichedMomentumOperator<dim, Number>::read_penalty_polynomial(EnrichedFaceIntegrator & phi) const
{
  return phi.phi_dg.read_cell_data(penalty_parameter) * penalty_factor;
}

template<int dim, typename Number>
typename EnrichedMomentumOperator<dim, Number>::scalar
EnrichedMomentumOperator<dim, Number>::read_penalty_total(FaceIntegratorU & phi) const
{
  if(not enrichment_aware_penalty)
    return read_penalty_polynomial(phi); // Krank et al. (2018), Eq. (35)-(36)

  return std::max(read_penalty_polynomial(phi), phi.read_cell_data(penalty_enrichment));
}

template<int dim, typename Number>
typename EnrichedMomentumOperator<dim, Number>::scalar
EnrichedMomentumOperator<dim, Number>::read_penalty_total(EnrichedFaceIntegrator & phi) const
{
  if(not enrichment_aware_penalty)
    return read_penalty_polynomial(phi); // Krank et al. (2018), Eq. (35)-(36)

  return std::max(read_penalty_polynomial(phi), phi.phi_dg.read_cell_data(penalty_enrichment));
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::penalty_correction_loop(VectorType const & u_bar_shadow,
                                                               VectorType &       dst_bar_shadow) const
{
  auto const & mf = *function_enrichment->get_matrix_free();
  auto const & bc = momentum_operator->get_data().bc;

  unsigned int const dof_shadow = function_enrichment->get_dof_index_shadow_vector();
  unsigned int const quad       = function_enrichment->get_quad_index();
  scalar const       nu = dealii::make_vectorized_array<Number>(momentum_operator->get_viscous_kernel_data().viscosity);

  // interior faces with at least one enriched side
  for(unsigned int face = 0; enrichment_aware_penalty and face < mf.n_inner_face_batches(); ++face)
  {
    auto const         range = std::make_pair(face, face + 1);
    unsigned int const fe_m  = mf.get_face_active_fe_index(range, true);
    unsigned int const fe_p  = mf.get_face_active_fe_index(range, false);
    if(fe_m != 1 and fe_p != 1)
      continue;

    FaceIntegratorU phi_m(mf, true, dof_shadow, quad, 0, fe_m);
    FaceIntegratorU phi_p(mf, false, dof_shadow, quad, 0, fe_p);
    phi_m.reinit(face);
    phi_p.reinit(face);
    phi_m.gather_evaluate(u_bar_shadow, dealii::EvaluationFlags::values);
    phi_p.gather_evaluate(u_bar_shadow, dealii::EvaluationFlags::values);

    scalar const tau_total = std::max(read_penalty_total(phi_m), read_penalty_total(phi_p));
    scalar const tau_poly  = std::max(read_penalty_polynomial(phi_m), read_penalty_polynomial(phi_p));
    scalar const delta_tau = tau_total - tau_poly;

    for(unsigned int q = 0; q < phi_m.n_q_points; ++q)
    {
      vector const flux = nu * delta_tau * (phi_m.get_value(q) - phi_p.get_value(q));
      phi_m.submit_value(flux, q);
      phi_p.submit_value(-flux, q);
    }
    phi_m.integrate_scatter(dealii::EvaluationFlags::values, dst_bar_shadow);
    phi_p.integrate_scatter(dealii::EvaluationFlags::values, dst_bar_shadow);
  }

  // boundary faces of enriched cells (Dirichlet / wall: jump = 2 u_m, Neumann: 0)
  unsigned int const b_begin = mf.n_inner_face_batches();
  unsigned int const b_end   = b_begin + mf.n_boundary_face_batches();
  for(unsigned int face = b_begin; face < b_end; ++face)
  {
    if(mf.get_face_active_fe_index(std::make_pair(face, face + 1), true) != 1)
      continue;

    BoundaryTypeU const type = bc->get_boundary_type(mf.get_boundary_id(face));
    if(type != BoundaryTypeU::Dirichlet and type != BoundaryTypeU::WallEnrichment)
      continue;

    FaceIntegratorU phi_m(mf, true, dof_shadow, quad, 0, 1);
    phi_m.reinit(face);
    phi_m.gather_evaluate(u_bar_shadow, dealii::EvaluationFlags::values);

    Number const factor    = (type == BoundaryTypeU::WallEnrichment) ? wall_penalty_factor : 1.0;
    scalar const delta_tau = factor * read_penalty_total(phi_m) - read_penalty_polynomial(phi_m);

    for(unsigned int q = 0; q < phi_m.n_q_points; ++q)
      phi_m.submit_value(nu * delta_tau * 2.0 * phi_m.get_value(q), q);
    phi_m.integrate_scatter(dealii::EvaluationFlags::values, dst_bar_shadow);
  }
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::vmult(BlockVectorType & dst, BlockVectorType const & src) const
{
  dst.block(0).zero_out_ghost_values();
  dst.block(1).zero_out_ghost_values();
  // (1) K_bar_bar * U_bar on the whole domain (homogeneous operator)
  momentum_operator->apply(dst.block(0), src.block(0));

  auto const & mf_en = *function_enrichment->get_matrix_free();

  // (2) U_bar in shadow layout (enriched + halo cells)
  VectorType & u_bar = function_enrichment->shadow_velocity;
  function_enrichment->copy_global_dg_to_wall_layout(src.block(0), u_bar);
  u_bar.update_ghost_values();
  src.block(1).update_ghost_values();

  VectorType correction_bar;
  mf_en.initialize_dof_vector(correction_bar, function_enrichment->get_dof_index_shadow_vector());

  VectorType zero_bar;
  mf_en.initialize_dof_vector(zero_bar, function_enrichment->get_dof_index_shadow_vector());

  VectorType zero_tilde;
  mf_en.initialize_dof_vector(zero_tilde, function_enrichment->get_dof_index_en());

  zero_bar.update_ghost_values();
  zero_tilde.update_ghost_values();

  dst.block(1) = 0.0;

  bool const viscous = momentum_operator->get_data().viscous_problem;

  // (3) Pass A, U_tilde only: K_bar_tilde * U_tilde   -> correction_bar
  //                           K_tilde_tilde * U_tilde -> dst.block(1)
  cell_loop(zero_bar, src.block(1), &correction_bar, dst.block(1));
  if(viscous)
  {
    inner_face_loop(zero_bar, src.block(1), &correction_bar, dst.block(1));
    boundary_face_loop(zero_bar, src.block(1), &correction_bar, dst.block(1));
  }

  // (4) Pass B, U_bar only: K_tilde_bar * U_bar -> dst.block(1)
  //     (U_bar rows are not written: K_bar_bar * U_bar is already in step (1))
  cell_loop(u_bar, zero_tilde, nullptr, dst.block(1));
  if(viscous)
  {
    inner_face_loop(u_bar, zero_tilde, nullptr, dst.block(1));
    boundary_face_loop(u_bar, zero_tilde, nullptr, dst.block(1));
  }

  // (4b) Pass C, U_bar only: U_bar rows of the faces of enriched cells with the
  //      enrichment-aware tau (MomentumOperator used the polynomial tau there)
  if(viscous and (enrichment_aware_penalty or wall_penalty_factor != 1.0))
    penalty_correction_loop(u_bar, correction_bar);

  correction_bar.compress(dealii::VectorOperation::add);
  dst.block(1).compress(dealii::VectorOperation::add);

  // (5) add K_bar_tilde * U_tilde to block(0) on enriched and halo cells
  function_enrichment->add_wall_layout_to_global_dg(correction_bar, dst.block(0));

  u_bar.zero_out_ghost_values();
  src.block(1).zero_out_ghost_values();
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::rhs_add_boundary_data(VectorType & dst_bar,
                                                             VectorType & dst_tilde,
                                                             double const time) const
{
  if(not momentum_operator->get_data().viscous_problem)
    return;

  auto const & mf_en = *function_enrichment->get_matrix_free();
  auto const & bc    = momentum_operator->get_data().bc;

  auto const   vk_data = momentum_operator->get_viscous_kernel_data();
  scalar const nu      = dealii::make_vectorized_array<Number>(vk_data.viscosity);
  Number const sign    = (vk_data.IP_formulation == InteriorPenaltyFormulation::SIPG) ? -0.5 : 0.5;

  // zero vectors: only needed so that the evaluator computes y, u_tau (and psi)
  VectorType zero_bar, zero_tilde;
  mf_en.initialize_dof_vector(zero_bar, function_enrichment->get_dof_index_shadow_vector());
  mf_en.initialize_dof_vector(zero_tilde, function_enrichment->get_dof_index_en());

  // U_bar-row correction of the inflow penalty term (shadow layout)
  VectorType correction_bar;
  mf_en.initialize_dof_vector(correction_bar, function_enrichment->get_dof_index_shadow_vector());

  unsigned int const begin = mf_en.n_inner_face_batches();
  unsigned int const end   = begin + mf_en.n_boundary_face_batches();

  for(unsigned int face = begin; face < end; ++face)
  {
    if(mf_en.get_face_active_fe_index(std::make_pair(face, face + 1), true) != 1)
      continue;

    auto const          boundary_id = mf_en.get_boundary_id(face);
    BoundaryTypeU const type        = bc->get_boundary_type(boundary_id);

    // only true Dirichlet faces carry data (WallEnrichment: g = 0)
    if(type != BoundaryTypeU::Dirichlet)
      continue;

    auto const & g_function = bc->dirichlet_bc.find(boundary_id)->second;
    g_function->set_time(time);

    EnrichedFaceIntegrator phi_m(mf_en, *function_enrichment, true, 1);
    phi_m.reinit(face);
    phi_m.gather_evaluate(zero_bar, zero_tilde, dealii::EvaluationFlags::values);

    scalar const tau       = read_penalty_total(phi_m);
    scalar const delta_tau = tau - read_penalty_polynomial(phi_m);

    std::vector<vector> g_values(phi_m.n_q_points);

    for(unsigned int q = 0; q < phi_m.n_q_points; ++q)
    {
      vector const n  = phi_m.get_normal_vector(q);
      auto const   xq = phi_m.phi_dg.quadrature_point(q);

      // g at the quadrature point, lane by lane
      vector g;
      for(unsigned int l = 0; l < dealii::VectorizedArray<Number>::size(); ++l)
      {
        dealii::Point<dim> p;
        for(unsigned int d = 0; d < dim; ++d)
          p[d] = xq[d][l];
        for(unsigned int d = 0; d < dim; ++d)
          g[d][l] = g_function->value(p, d);
      }
      g_values[q] = g;

      // rhs = - (flux with u_m = 0, u_p = 2 g)
      vector const value_flux    = 2.0 * nu * tau * g;
      tensor const gradient_flux = 2.0 * sign * nu * dealii::outer_product(g, n);

      phi_m.submit_value_and_gradient(value_flux, gradient_flux, q);
    }

    phi_m.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
    phi_m.phi_en.distribute_local_to_global(dst_tilde); // U_tilde row only

    // U_bar row: the polynomial penalty part is in rhs_add_viscous_term; add the difference
    FaceIntegratorU phi_bar(mf_en,
                            true,
                            function_enrichment->get_dof_index_shadow_vector(),
                            function_enrichment->get_quad_index(),
                            0,
                            1);
    phi_bar.reinit(face);
    for(unsigned int q = 0; q < phi_bar.n_q_points; ++q)
      phi_bar.submit_value(2.0 * nu * delta_tau * g_values[q], q);
    phi_bar.integrate_scatter(dealii::EvaluationFlags::values, correction_bar);
  }

  dst_tilde.compress(dealii::VectorOperation::add);

  correction_bar.compress(dealii::VectorOperation::add);
  function_enrichment->add_wall_layout_to_global_dg(correction_bar, dst_bar);
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::cell_loop(VectorType const & src_bar,
                                                 VectorType const & src_tilde,
                                                 VectorType *       dst_bar,
                                                 VectorType &       dst_tilde) const
{
  auto const & mf_en   = *function_enrichment->get_matrix_free();
  auto const & op_data = momentum_operator->get_data();

  bool const   unsteady = op_data.unsteady_problem;
  bool const   viscous  = op_data.viscous_problem;
  scalar const mass_factor =
    dealii::make_vectorized_array<Number>(momentum_operator->get_scaling_factor_mass_operator());
  scalar const nu =
    dealii::make_vectorized_array<Number>(momentum_operator->get_viscous_kernel_data().viscosity);

  EnrichedCellIntegrator phi(mf_en, *function_enrichment);

  for(unsigned int cell = 0; cell < mf_en.n_cell_batches(); ++cell)
  {
    if(mf_en.get_cell_iterator(cell, 0)->active_fe_index() != 1)
      continue;

    phi.reinit(cell);
    phi.gather_evaluate(src_bar,
                        src_tilde,
                        dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    for(unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      vector value_flux;
      tensor gradient_flux;

      if(unsteady)
        value_flux += mass_factor * phi.get_value(q);
      if(viscous)
        gradient_flux += nu * phi.get_gradient(q);

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
EnrichedMomentumOperator<dim, Number>::inner_face_loop(VectorType const & src_bar,
                                                       VectorType const & src_tilde,
                                                       VectorType *       dst_bar,
                                                       VectorType &       dst_tilde) const
{
  auto const & mf_en = *function_enrichment->get_matrix_free();

  unsigned int const dof_shadow = function_enrichment->get_dof_index_shadow_vector();
  unsigned int const quad       = function_enrichment->get_quad_index();

  auto const process = [&](auto & phi_m, auto & phi_p, unsigned int const face) {
    phi_m.reinit(face);
    phi_p.reinit(face);

    gather(phi_m, src_bar, src_tilde);
    gather(phi_p, src_bar, src_tilde);

    scalar const tau = std::max(read_penalty_total(phi_m), read_penalty_total(phi_p));

    inner_face_flux(phi_m, phi_p, tau);

    scatter(phi_m, dst_bar, dst_tilde);
    scatter(phi_p, dst_bar, dst_tilde);
  };

  for(unsigned int face = 0; face < mf_en.n_inner_face_batches(); ++face)
  {
    auto const         range = std::make_pair(face, face + 1);
    unsigned int const fe_m  = mf_en.get_face_active_fe_index(range, true);
    unsigned int const fe_p  = mf_en.get_face_active_fe_index(range, false);

    // no enriched side: handled completely by MomentumOperator
    if(fe_m != 1 and fe_p != 1)
      continue;

    AssertThrow(fe_m != 0 and fe_p != 0,
                dealii::ExcMessage("Enriched cell has a neighbour outside the halo layer."));

    if(fe_m == 1 and fe_p == 1)
    {
      EnrichedFaceIntegrator phi_m(mf_en, *function_enrichment, true, 1);
      EnrichedFaceIntegrator phi_p(mf_en, *function_enrichment, false, 1);
      process(phi_m, phi_p, face);
    }
    else if(fe_m == 1)
    {
      EnrichedFaceIntegrator phi_m(mf_en, *function_enrichment, true, 1);
      FaceIntegratorU        phi_p(mf_en, false, dof_shadow, quad, 0, fe_p);
      process(phi_m, phi_p, face);
    }
    else
    {
      FaceIntegratorU        phi_m(mf_en, true, dof_shadow, quad, 0, fe_m);
      EnrichedFaceIntegrator phi_p(mf_en, *function_enrichment, false, 1);
      process(phi_m, phi_p, face);
    }
  }
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::boundary_face_loop(VectorType const & src_bar,
                                                          VectorType const & src_tilde,
                                                          VectorType *       dst_bar,
                                                          VectorType &       dst_tilde) const
{
  auto const & mf_en = *function_enrichment->get_matrix_free();
  auto const & bc    = momentum_operator->get_data().bc;

  auto const   vk_data = momentum_operator->get_viscous_kernel_data();
  scalar const nu      = dealii::make_vectorized_array<Number>(vk_data.viscosity);
  Number const sign    = (vk_data.IP_formulation == InteriorPenaltyFormulation::SIPG) ? -0.5 : 0.5;

  unsigned int const begin = mf_en.n_inner_face_batches();
  unsigned int const end   = begin + mf_en.n_boundary_face_batches();

  for(unsigned int face = begin; face < end; ++face)
  {
    if(mf_en.get_face_active_fe_index(std::make_pair(face, face + 1), true) != 1)
      continue;

    BoundaryTypeU const type = bc->get_boundary_type(mf_en.get_boundary_id(face));

    EnrichedFaceIntegrator phi_m(mf_en, *function_enrichment, true, 1);
    phi_m.reinit(face);
    gather(phi_m, src_bar, src_tilde);

    // weak no-slip on enrichment walls: tau_IP times wall_penalty_factor (Krank thesis, Ch. 10)
    scalar tau = read_penalty_total(phi_m);
    if(type == BoundaryTypeU::WallEnrichment)
      tau = tau * wall_penalty_factor;

    for(unsigned int q = 0; q < phi_m.n_q_points; ++q)
    {
      vector const u_m  = phi_m.get_value(q);
      vector const n    = phi_m.get_normal_vector(q);
      vector const dn_m = phi_m.get_gradient(q) * n;

      // homogeneous exterior state (inhomogeneous parts belong to the rhs)
      vector u_p, dn_p;
      if(type == BoundaryTypeU::Dirichlet or type == BoundaryTypeU::WallEnrichment)
      {
        u_p  = -u_m;
        dn_p = dn_m;
      }
      else if(type == BoundaryTypeU::Neumann)
      {
        u_p  = u_m;
        dn_p = -dn_m;
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage("EnrichedMomentumOperator: boundary type not supported "
                                       "on enriched cells."));
      }

      vector const jump  = u_m - u_p;
      tensor const gflux = sign * nu * dealii::outer_product(jump, n);
      vector const vflux = nu * 0.5 * (dn_m + dn_p) - nu * tau * jump;

      vector const value_flux = -vflux;
      submit(phi_m, value_flux, gflux, q);
    }

    scatter(phi_m, dst_bar, dst_tilde);
  }
}

template<int dim, typename Number>
template<typename IntegratorM, typename IntegratorP>
void
EnrichedMomentumOperator<dim, Number>::inner_face_flux(IntegratorM &  phi_m,
                                                       IntegratorP &  phi_p,
                                                       scalar const & tau) const
{
  auto const   vk_data = momentum_operator->get_viscous_kernel_data();
  scalar const nu      = dealii::make_vectorized_array<Number>(vk_data.viscosity);
  Number const sign    = (vk_data.IP_formulation == InteriorPenaltyFormulation::SIPG) ? -0.5 : 0.5;

  for(unsigned int q = 0; q < phi_m.n_q_points; ++q)
  {
    vector const u_m  = phi_m.get_value(q);
    vector const u_p  = phi_p.get_value(q);
    vector const n    = phi_m.get_normal_vector(q);
    vector const dn_m = phi_m.get_gradient(q) * n;
    vector const dn_p = phi_p.get_gradient(q) * n;

    vector const jump  = u_m - u_p;
    tensor const gflux = sign * nu * dealii::outer_product(jump, n);
    vector const vflux = nu * 0.5 * (dn_m + dn_p) - nu * tau * jump;

    vector const value_flux_m = -vflux;
    vector const value_flux_p = vflux; // + sign since n+ = -n-

    submit(phi_m, value_flux_m, gflux, q);
    submit(phi_p, value_flux_p, gflux, q);
  }
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::gather(FaceIntegratorU &  phi,
                                              VectorType const & src_bar,
                                              VectorType const & src_tilde) const
{
  (void)src_tilde; // halo side has no enrichment
  phi.gather_evaluate(src_bar, dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::gather(EnrichedFaceIntegrator & phi,
                                              VectorType const &       src_bar,
                                              VectorType const &       src_tilde) const
{
  phi.gather_evaluate(src_bar,
                      src_tilde,
                      dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::scatter(FaceIntegratorU & phi,
                                               VectorType *      dst_bar,
                                               VectorType &      dst_tilde) const
{
  (void)dst_tilde; // halo side has no enrichment row
  if(dst_bar == nullptr)
    return;

  phi.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
  phi.distribute_local_to_global(*dst_bar);
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::scatter(EnrichedFaceIntegrator & phi,
                                               VectorType *             dst_bar,
                                               VectorType &             dst_tilde) const
{
  phi.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

  if(dst_bar != nullptr)
    phi.phi_dg.distribute_local_to_global(*dst_bar);
  phi.phi_en.distribute_local_to_global(dst_tilde);
}

template<int dim, typename Number>
typename EnrichedMomentumOperator<dim, Number>::scalar
EnrichedMomentumOperator<dim, Number>::read_penalty(FaceIntegratorU & phi) const
{
  return phi.read_cell_data(penalty_parameter);
}

template<int dim, typename Number>
typename EnrichedMomentumOperator<dim, Number>::scalar
EnrichedMomentumOperator<dim, Number>::read_penalty(EnrichedFaceIntegrator & phi) const
{
  return phi.phi_dg.read_cell_data(penalty_parameter);
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::submit(FaceIntegratorU &  phi,
                                              vector const &     value_flux,
                                              tensor const &     gradient_flux,
                                              unsigned int const q) const
{
  // halo side: plain FEFaceEvaluation, one write per slot
  phi.submit_value(value_flux, q);
  phi.submit_gradient(gradient_flux, q);
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::submit(EnrichedFaceIntegrator & phi,
                                              vector const &           value_flux,
                                              tensor const &           gradient_flux,
                                              unsigned int const       q) const
{
  // enriched side: value flux and gradient flux in one call
  phi.submit_value_and_gradient(value_flux, gradient_flux, q);
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::set_enrichment_aware_penalty(bool const flag)
{
  enrichment_aware_penalty = flag;
}

template<int dim, typename Number>
void
EnrichedMomentumOperator<dim, Number>::set_wall_penalty_factor(double const factor)
{
  AssertThrow(factor > 0.0, dealii::ExcMessage("Wall penalty factor must be positive."));
  wall_penalty_factor = factor;
}

template class EnrichedMomentumOperator<2, float>;
template class EnrichedMomentumOperator<2, double>;
template class EnrichedMomentumOperator<3, float>;
template class EnrichedMomentumOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
