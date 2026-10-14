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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/convective_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/weak_boundary_conditions.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::set_velocity_copy(VectorType const & src) const
{
  kernel->set_velocity_copy(src);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::set_velocity_ptr(VectorType const & src) const
{
  kernel->set_velocity_ptr(src);
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
ConvectiveOperator<dim, Number>::get_velocity() const
{
  return kernel->get_velocity();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                   matrix_free,
  dealii::AffineConstraints<Number> const &                 affine_constraints,
  ConvectiveOperatorData<dim> const &                       data,
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel)
{
  operator_data = data;

  kernel = convective_kernel;

  Base::reinit(matrix_free, affine_constraints, data);

  this->integrator_flags = kernel->get_integrator_flags();

  if(operator_data.kernel_data.wall_enrichment_enabled)
  {
    AssertThrow(false,
                dealii::ExcMessage(
                "Wrong syntax for initialize function. Send function_enrichment as the last arugument when wall enrichment is enabled.")
                );
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                   matrix_free,
  dealii::AffineConstraints<Number> const &                 affine_constraints,
  ConvectiveOperatorData<dim> const &                       data,
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel,
  std::shared_ptr<FunctionEnrichment<dim, Number>>          function_enrichment)
{
  operator_data = data;

  kernel            = convective_kernel;

  this->function_enrichment = function_enrichment;

  Base::reinit(matrix_free, affine_constraints, data);

  this->integrator_flags = kernel->get_integrator_flags();
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_nonlinear_operator(VectorType &       dst,
                                                             VectorType const & src,
                                                             Number const       time) const
{
  this->time = time;

  this->matrix_free->loop(&This::cell_loop_nonlinear_operator,
                          &This::face_loop_nonlinear_operator,
                          &This::boundary_face_loop_nonlinear_operator,
                          this,
                          dst,
                          src,
                          true /*zero_dst_vector = true*/,
                          dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                          dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);

  // if(operator_data.kernel_data.wall_enrichment_enabled)
  // {
  //   std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_en = function_enrichment->get_matrix_free();
  //
  //   function_enrichment->copy_global_dg_to_wall_layout(src, function_enrichment->src);
  //
  //   matrix_free_en->loop(&This::cell_loop_nonlinear_operator_wall_enrichment,
  //                                             &This::face_loop_nonlinear_operator_wall_enrichment,
  //                                             &This::boundary_face_loop_nonlinear_operator_wall_enrichment,
  //                                             this,
  //                                             function_enrichment->dst,
  //                                             function_enrichment->src,
  //                                             true /*zero_dst_vector = true*/,
  //                                             dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
  //                                             dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
  //
  //   function_enrichment->copy_wall_layout_to_global_dg(function_enrichment->dst, dst);
  // }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_nonlinear_operator_enriched(
  VectorType &                            dst_bar_enriched,
  VectorType &                            dst_bar_halo,
  VectorType &                            dst_tilde,
  VectorType const &                      u_bar_shadow,
  VectorType const &                      u_tilde,
  FunctionEnrichment<dim, Number> const & function_enrichment,
  Number const                            evaluation_time) const
{
  AssertThrow(operator_data.kernel_data.ale == false,
              dealii::ExcMessage("Enriched convective term: ALE not implemented."));

  this->time = evaluation_time;

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number>          CellIntegratorU;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, dim, Number>      FaceIntegratorU;
  typedef FEEnrichedEvaluation<dim, Number, CellIntegratorU>     EnrichedCellIntegrator;
  typedef FEFaceEnrichedEvaluation<dim, Number, FaceIntegratorU> EnrichedFaceIntegrator;

  auto const &       mf      = *function_enrichment.get_matrix_free();
  unsigned int const dof_bar = function_enrichment.get_dof_index_shadow_vector();
  unsigned int const quad    = function_enrichment.get_quad_index();
  vector const       u_grid_zero;

  dst_bar_enriched = 0.0;
  dst_bar_halo     = 0.0;
  dst_tilde        = 0.0;

  bool const divergence_formulation = operator_data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation;
    {
      EnrichedCellIntegrator phi(mf, function_enrichment);
     
      for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
      {
        if(mf.get_cell_iterator(cell, 0)->active_fe_index() != 1)
          continue;
          
        phi.reinit(cell);
        
        if(divergence_formulation)
        {
          phi.gather_evaluate(u_bar_shadow, u_tilde,
  dealii::EvaluationFlags::values);
          for(unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            vector const u = phi.get_value(q);
            // minus sign due to integration by parts
            phi.submit_gradient(-outer_product(u, u), q);
          }
          phi.integrate(dealii::EvaluationFlags::gradients);
        } 
        else
        {
          phi.gather_evaluate(u_bar_shadow, u_tilde,
                              dealii::EvaluationFlags::values |
  dealii::EvaluationFlags::gradients);
          for(unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(phi.get_gradient(q) * phi.get_value(q), q);
          phi.integrate(dealii::EvaluationFlags::values);
        } 
       
        phi.phi_dg.distribute_local_to_global(dst_bar_enriched);
        phi.phi_en.distribute_local_to_global(dst_tilde);
    }
  }


  // values of u_h and U_bar on one side of a face (u_h = U_bar on halo cells)
  auto const values_on_side = [&](bool const            interior,
                                  unsigned int const    fe_index,
                                  unsigned int const    face,
                                  std::vector<vector> & values_h,
                                  std::vector<vector> & values_bar) {
    FaceIntegratorU phi_bar(mf, interior, dof_bar, quad, 0, fe_index);
    phi_bar.reinit(face);
    phi_bar.gather_evaluate(u_bar_shadow, dealii::EvaluationFlags::values);
    values_bar.resize(phi_bar.n_q_points);
    for(unsigned int q = 0; q < phi_bar.n_q_points; ++q)
      values_bar[q] = phi_bar.get_value(q);

    if(fe_index == 1)
    {
      EnrichedFaceIntegrator phi(mf, function_enrichment, interior, 1);
      phi.reinit(face);
      phi.gather_evaluate(u_bar_shadow, u_tilde, dealii::EvaluationFlags::values);
      values_h.resize(phi.n_q_points);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        values_h[q] = phi.get_value(q);
    }
    else
      values_h = values_bar;
  };

  // enriched side: complete flux (v_bar and psi v_tilde rows)
  // halo side:     correction flux_h - flux_bar (v_bar rows)
  auto const test_side = [&](bool const                  interior,
                             unsigned int const          fe_index,
                             unsigned int const          face,
                             std::vector<vector> const & flux_h,
                             std::vector<vector> const & flux_bar) {
    if(fe_index == 1)
    {
      EnrichedFaceIntegrator phi(mf, function_enrichment, interior, 1);
      phi.reinit(face);
      phi.gather_evaluate(u_bar_shadow, u_tilde, dealii::EvaluationFlags::values); // y, u_tau -> psi
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(flux_h[q], q);
      phi.integrate(dealii::EvaluationFlags::values);
      phi.phi_dg.distribute_local_to_global(dst_bar_enriched);
      phi.phi_en.distribute_local_to_global(dst_tilde);
    }
    else
    {
      FaceIntegratorU phi(mf, interior, dof_bar, quad, 0, fe_index);
      phi.reinit(face);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(flux_h[q] - flux_bar[q], q);
      phi.integrate_scatter(dealii::EvaluationFlags::values, dst_bar_halo);
    }
  };

  std::vector<vector> h_m, h_p, b_m, b_p, flux_h_m, flux_h_p, flux_b_m, flux_b_p;

  // ---- interior faces touching enriched cells -------------------------------------------
  for(unsigned int face = 0; face < mf.n_inner_face_batches(); ++face)
  {
    auto const         range = std::make_pair(face, face + 1);
    unsigned int const fe_m  = mf.get_face_active_fe_index(range, true);
    unsigned int const fe_p  = mf.get_face_active_fe_index(range, false);

    if(fe_m != 1 and fe_p != 1)
      continue;
    AssertThrow(fe_m != 0 and fe_p != 0,
                dealii::ExcMessage("Enriched cell has a neighbour outside the halo layer."));

    values_on_side(true, fe_m, face, h_m, b_m);
    values_on_side(false, fe_p, face, h_p, b_p);

    FaceIntegratorU normal_evaluator(mf, true, dof_bar, quad, 0, fe_m);
    normal_evaluator.reinit(face);

    unsigned int const n_q = h_m.size();
    flux_h_m.resize(n_q);
    flux_h_p.resize(n_q);
    flux_b_m.resize(n_q);
    flux_b_p.resize(n_q);
    for(unsigned int q = 0; q < n_q; ++q)
    {
      vector const n = normal_evaluator.get_normal_vector(q);
      auto const flux_h =
        kernel->calculate_flux_nonlinear_interior_and_neighbor(h_m[q], h_p[q], n, u_grid_zero);
      auto const flux_b =
        kernel->calculate_flux_nonlinear_interior_and_neighbor(b_m[q], b_p[q], n, u_grid_zero);
      flux_h_m[q] = std::get<0>(flux_h);
      flux_h_p[q] = std::get<1>(flux_h);
      flux_b_m[q] = std::get<0>(flux_b);
      flux_b_p[q] = std::get<1>(flux_b);
    }

    test_side(true, fe_m, face, flux_h_m, flux_b_m);
    test_side(false, fe_p, face, flux_h_p, flux_b_p);
  }

  // ---- ALL boundary faces of enriched cells (incl. walls: complete rows) ------------------
  unsigned int const b_begin = mf.n_inner_face_batches();
  unsigned int const b_end   = b_begin + mf.n_boundary_face_batches();

  for(unsigned int face = b_begin; face < b_end; ++face)
  {
    if(mf.get_face_active_fe_index(std::make_pair(face, face + 1), true) != 1)
      continue;

    dealii::types::boundary_id const boundary_id   = mf.get_boundary_id(face);
    BoundaryTypeU const              boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    values_on_side(true, 1, face, h_m, b_m);

    FaceIntegratorU phi_bar(mf, true, dof_bar, quad, 0, 1); // quadrature points, normals
    phi_bar.reinit(face);

    unsigned int const n_q = h_m.size();
    flux_h_m.resize(n_q);
    flux_b_m.resize(n_q);
    for(unsigned int q = 0; q < n_q; ++q)
    {
      vector const n     = phi_bar.get_normal_vector(q);
      vector const h_ext = calculate_exterior_value_nonlinear(h_m[q], q, phi_bar, boundary_type,
                                                              operator_data.kernel_data.type_dirichlet_bc,
                                                              boundary_id, operator_data.bc, this->time);
      flux_h_m[q] =
        kernel->calculate_flux_nonlinear_boundary(h_m[q], h_ext, n, u_grid_zero, boundary_type);
      flux_b_m[q] = flux_h_m[q]; // unused on enriched sides
    }

    test_side(true, 1, face, flux_h_m, flux_b_m);
  }

  dst_bar_enriched.compress(dealii::VectorOperation::add);
  dst_bar_halo.compress(dealii::VectorOperation::add);
  dst_tilde.compress(dealii::VectorOperation::add);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::apply_linearized_operator_enriched(
  VectorType &                            dst_bar_enriched,
  VectorType &                            dst_bar_halo,
  VectorType &                            dst_tilde,
  VectorType const &                      w_bar_shadow,
  VectorType const &                      w_tilde,
  VectorType const &                      delta_u_bar_shadow,
  VectorType const &                      delta_u_tilde,
  FunctionEnrichment<dim, Number> const & function_enrichment,
  Number const                            evaluation_time) const
{
  AssertThrow(operator_data.kernel_data.ale == false,
              dealii::ExcMessage("Enriched linearized convective term: ALE not implemented."));

  this->time = evaluation_time;

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number>          CellIntegratorU;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, dim, Number>      FaceIntegratorU;
  typedef FEEnrichedEvaluation<dim, Number, CellIntegratorU>     EnrichedCellIntegrator;
  typedef FEFaceEnrichedEvaluation<dim, Number, FaceIntegratorU> EnrichedFaceIntegrator;

  auto const &       mf      = *function_enrichment.get_matrix_free();
  unsigned int const dof_bar = function_enrichment.get_dof_index_shadow_vector();
  unsigned int const quad    = function_enrichment.get_quad_index();

  dst_bar_enriched = 0.0;
  dst_bar_halo     = 0.0;
  dst_tilde        = 0.0;

  bool const divergence_formulation =
    operator_data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation;

  AssertThrow(divergence_formulation or operator_data.kernel_data.formulation ==
                                          FormulationConvectiveTerm::ConvectiveFormulation,
              dealii::ExcMessage("Enriched linearized convective term: formulation not implemented."));

  // ---- cell integrals on enriched cells: complete rows (v_bar and psi v_tilde) -----------
  {
    EnrichedCellIntegrator phi_w(mf, function_enrichment);       // linearization point w_h
    EnrichedCellIntegrator phi_delta_u(mf, function_enrichment); // increment, also test function

    for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
    {
      if(mf.get_cell_iterator(cell, 0)->active_fe_index() != 1)
        continue;

      phi_w.reinit(cell);
      phi_delta_u.reinit(cell);

      if(divergence_formulation)
      {
        phi_w.gather_evaluate(w_bar_shadow, w_tilde, dealii::EvaluationFlags::values);
        phi_delta_u.gather_evaluate(delta_u_bar_shadow,
                                    delta_u_tilde,
                                    dealii::EvaluationFlags::values);

        for(unsigned int q = 0; q < phi_delta_u.n_q_points; ++q)
        {
          // F'(w) delta_u = w (x) delta_u + delta_u (x) w; minus sign due to integration by parts
          tensor const F = outer_product(phi_w.get_value(q), phi_delta_u.get_value(q));
          phi_delta_u.submit_gradient(-(F + transpose(F)), q);
        }
        phi_delta_u.integrate(dealii::EvaluationFlags::gradients);
      }
      else // convective formulation (strong form, integration by parts twice)
      {
        phi_w.gather_evaluate(w_bar_shadow,
                              w_tilde,
                              dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
        phi_delta_u.gather_evaluate(delta_u_bar_shadow,
                                    delta_u_tilde,
                                    dealii::EvaluationFlags::values |
                                      dealii::EvaluationFlags::gradients);

        for(unsigned int q = 0; q < phi_delta_u.n_q_points; ++q)
        {
          // grad(w) delta_u + grad(delta_u) w
          vector const flux = phi_w.get_gradient(q) * phi_delta_u.get_value(q) +
                              phi_delta_u.get_gradient(q) * phi_w.get_value(q);
          phi_delta_u.submit_value(flux, q);
        }
        phi_delta_u.integrate(dealii::EvaluationFlags::values);
      }

      phi_delta_u.phi_dg.distribute_local_to_global(dst_bar_enriched);
      phi_delta_u.phi_en.distribute_local_to_global(dst_tilde);
    }
  }

  // values of the enriched field (h) and of its polynomial part (bar) on one side of a face;
  // on halo cells the field is polynomial (h = bar)
  auto const values_on_side = [&](bool const            interior,
                                  unsigned int const    fe_index,
                                  unsigned int const    face,
                                  VectorType const &    src_bar_shadow,
                                  VectorType const &    src_tilde,
                                  std::vector<vector> & values_h,
                                  std::vector<vector> & values_bar) {
    FaceIntegratorU phi_bar(mf, interior, dof_bar, quad, 0, fe_index);
    phi_bar.reinit(face);
    phi_bar.gather_evaluate(src_bar_shadow, dealii::EvaluationFlags::values);
    values_bar.resize(phi_bar.n_q_points);
    for(unsigned int q = 0; q < phi_bar.n_q_points; ++q)
      values_bar[q] = phi_bar.get_value(q);

    if(fe_index == 1)
    {
      EnrichedFaceIntegrator phi(mf, function_enrichment, interior, 1);
      phi.reinit(face);
      phi.gather_evaluate(src_bar_shadow, src_tilde, dealii::EvaluationFlags::values);
      values_h.resize(phi.n_q_points);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        values_h[q] = phi.get_value(q);
    }
    else
      values_h = values_bar;
  };

  // enriched side: complete flux (v_bar and psi v_tilde rows)
  // halo side:     correction flux_h - flux_bar (v_bar rows)
  auto const test_side = [&](bool const                  interior,
                             unsigned int const          fe_index,
                             unsigned int const          face,
                             std::vector<vector> const & flux_h,
                             std::vector<vector> const & flux_bar) {
    if(fe_index == 1)
    {
      EnrichedFaceIntegrator phi(mf, function_enrichment, interior, 1);
      phi.reinit(face);
      phi.gather_evaluate(w_bar_shadow, w_tilde, dealii::EvaluationFlags::values); // y, u_tau -> psi
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(flux_h[q], q);
      phi.integrate(dealii::EvaluationFlags::values);
      phi.phi_dg.distribute_local_to_global(dst_bar_enriched);
      phi.phi_en.distribute_local_to_global(dst_tilde);
    }
    else
    {
      FaceIntegratorU phi(mf, interior, dof_bar, quad, 0, fe_index);
      phi.reinit(face);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(flux_h[q] - flux_bar[q], q);
      phi.integrate_scatter(dealii::EvaluationFlags::values, dst_bar_halo);
    }
  };

  std::vector<vector> w_h_m, w_h_p, w_b_m, w_b_p;             // linearization point
  std::vector<vector> du_h_m, du_h_p, du_b_m, du_b_p;         // increment
  std::vector<vector> flux_h_m, flux_h_p, flux_b_m, flux_b_p; // linearized fluxes

  // ---- interior faces touching enriched cells -------------------------------------------
  for(unsigned int face = 0; face < mf.n_inner_face_batches(); ++face)
  {
    auto const         range = std::make_pair(face, face + 1);
    unsigned int const fe_m  = mf.get_face_active_fe_index(range, true);
    unsigned int const fe_p  = mf.get_face_active_fe_index(range, false);

    if(fe_m != 1 and fe_p != 1)
      continue;
    AssertThrow(fe_m != 0 and fe_p != 0,
                dealii::ExcMessage("Enriched cell has a neighbour outside the halo layer."));

    values_on_side(true, fe_m, face, w_bar_shadow, w_tilde, w_h_m, w_b_m);
    values_on_side(false, fe_p, face, w_bar_shadow, w_tilde, w_h_p, w_b_p);
    values_on_side(true, fe_m, face, delta_u_bar_shadow, delta_u_tilde, du_h_m, du_b_m);
    values_on_side(false, fe_p, face, delta_u_bar_shadow, delta_u_tilde, du_h_p, du_b_p);

    FaceIntegratorU normal_evaluator(mf, true, dof_bar, quad, 0, fe_m);
    normal_evaluator.reinit(face);

    unsigned int const n_q = w_h_m.size();
    flux_h_m.resize(n_q);
    flux_h_p.resize(n_q);
    flux_b_m.resize(n_q);
    flux_b_p.resize(n_q);
    for(unsigned int q = 0; q < n_q; ++q)
    {
      vector const n      = normal_evaluator.get_normal_vector(q);
      auto const   flux_h = kernel->calculate_flux_linearized_interior_and_neighbor(
        w_h_m[q], w_h_p[q], du_h_m[q], du_h_p[q], n, q);
      auto const flux_b = kernel->calculate_flux_linearized_interior_and_neighbor(
        w_b_m[q], w_b_p[q], du_b_m[q], du_b_p[q], n, q);
      flux_h_m[q] = std::get<0>(flux_h);
      flux_h_p[q] = std::get<1>(flux_h);
      flux_b_m[q] = std::get<0>(flux_b);
      flux_b_p[q] = std::get<1>(flux_b);
    }

    test_side(true, fe_m, face, flux_h_m, flux_b_m);
    test_side(false, fe_p, face, flux_h_p, flux_b_p);
  }

  // ---- ALL boundary faces of enriched cells (incl. walls: complete rows) ------------------
  unsigned int const b_begin = mf.n_inner_face_batches();
  unsigned int const b_end   = b_begin + mf.n_boundary_face_batches();

  for(unsigned int face = b_begin; face < b_end; ++face)
  {
    if(mf.get_face_active_fe_index(std::make_pair(face, face + 1), true) != 1)
      continue;

    dealii::types::boundary_id const boundary_id   = mf.get_boundary_id(face);
    BoundaryTypeU const              boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    values_on_side(true, 1, face, w_bar_shadow, w_tilde, w_h_m, w_b_m);
    values_on_side(true, 1, face, delta_u_bar_shadow, delta_u_tilde, du_h_m, du_b_m);

    FaceIntegratorU phi_bar(mf, true, dof_bar, quad, 0, 1); // quadrature points, normals
    phi_bar.reinit(face);

    unsigned int const n_q = w_h_m.size();
    flux_h_m.resize(n_q);
    flux_b_m.resize(n_q);
    for(unsigned int q = 0; q < n_q; ++q)
    {
      vector const n = phi_bar.get_normal_vector(q);

      // exterior state of the linearization point: boundary data at the current time
      vector const w_ext = calculate_exterior_value_nonlinear(w_h_m[q],
                                                              q,
                                                              phi_bar,
                                                              boundary_type,
                                                              operator_data.kernel_data.type_dirichlet_bc,
                                                              boundary_id,
                                                              operator_data.bc,
                                                              this->time);
      // exterior state of the increment: homogeneous boundary data
      vector delta_u_m = du_h_m[q];
      vector const delta_u_ext =
        kernel->calculate_exterior_value_linearized(delta_u_m, q, phi_bar, boundary_type);

      flux_h_m[q] = kernel->calculate_flux_linearized_boundary(
        w_h_m[q], w_ext, delta_u_m, delta_u_ext, n, boundary_type, q);
      flux_b_m[q] = flux_h_m[q]; // unused on enriched sides
    }

    test_side(true, 1, face, flux_h_m, flux_b_m);
  }

  dst_bar_enriched.compress(dealii::VectorOperation::add);
  dst_bar_halo.compress(dealii::VectorOperation::add);
  dst_tilde.compress(dealii::VectorOperation::add);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_nonlinear_operator_add(VectorType &       dst,
                                                                 VectorType const & src,
                                                                 Number const       time) const
{
  this->time = time;

  this->matrix_free->loop(&This::cell_loop_nonlinear_operator,
                          &This::face_loop_nonlinear_operator,
                          &This::boundary_face_loop_nonlinear_operator,
                          this,
                          dst,
                          src,
                          false /*zero_dst_vector = false*/,
                          dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                          dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::rhs(VectorType & dst) const
{
  (void)dst;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function rhs() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::rhs_add(VectorType & dst) const
{
  (void)dst;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function rhs_add() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function evaluate() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::evaluate_add(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function evaluate_add() does not make sense for the convective operator."));
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::cell_loop_nonlinear_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range) const
{
  IntegratorCell integrator(matrix_free,
                            operator_data.dof_index,
                            operator_data.quad_index_nonlinear);
  IntegratorCell integrator_grid_velocity(matrix_free,
                                          operator_data.dof_index,
                                          operator_data.quad_index_nonlinear);


  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator.gather_evaluate(src, this->integrator_flags.cell_evaluate);

    if(operator_data.kernel_data.ale)
    {
      integrator_grid_velocity.reinit(cell);
      integrator_grid_velocity.gather_evaluate(kernel->get_grid_velocity(),
                                               dealii::EvaluationFlags::values);
    }

    do_cell_integral_nonlinear_operator(integrator, integrator_grid_velocity);

    integrator.integrate_scatter(this->integrator_flags.cell_integrate, dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::face_loop_nonlinear_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  IntegratorFace integrator_m(matrix_free,
                              true,
                              operator_data.dof_index,
                              operator_data.quad_index_nonlinear);
  IntegratorFace integrator_p(matrix_free,
                              false,
                              operator_data.dof_index,
                              operator_data.quad_index_nonlinear);

  IntegratorFace integrator_grid_velocity(matrix_free,
                                          true,
                                          operator_data.dof_index,
                                          operator_data.quad_index_nonlinear);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator_m.gather_evaluate(src, this->integrator_flags.face_evaluate);

    integrator_p.gather_evaluate(src, this->integrator_flags.face_evaluate);

    if(operator_data.kernel_data.ale)
    {
      integrator_grid_velocity.reinit(face);
      integrator_grid_velocity.gather_evaluate(kernel->get_grid_velocity(),
                                               dealii::EvaluationFlags::values);
    }

    do_face_integral_nonlinear_operator(integrator_m, integrator_p, integrator_grid_velocity);

    integrator_m.integrate_scatter(this->integrator_flags.face_integrate, dst);

    integrator_p.integrate_scatter(this->integrator_flags.face_integrate, dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::boundary_face_loop_nonlinear_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  IntegratorFace integrator_m(matrix_free,
                              true,
                              operator_data.dof_index,
                              operator_data.quad_index_nonlinear);

  IntegratorFace integrator_grid_velocity(matrix_free,
                                          true,
                                          operator_data.dof_index,
                                          operator_data.quad_index_nonlinear);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator_m.gather_evaluate(src, this->integrator_flags.face_evaluate);

    if(operator_data.kernel_data.ale)
    {
      integrator_grid_velocity.reinit(face);
      integrator_grid_velocity.gather_evaluate(kernel->get_grid_velocity(),
                                               dealii::EvaluationFlags::values);
    }

    do_boundary_integral_nonlinear_operator(integrator_m,
                                            integrator_grid_velocity,
                                            matrix_free.get_boundary_id(face));

    integrator_m.integrate_scatter(this->integrator_flags.face_integrate, dst);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral_nonlinear_operator(
  IntegratorCell & integrator,
  IntegratorCell & integrator_u_grid) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u = integrator.get_value(q);

    if(operator_data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      // nonlinear convective flux F(u) = uu
      tensor F = outer_product(u, u);
      // minus sign due to integration by parts
      integrator.submit_gradient(-F, q);
    }
    else if(operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      // convective formulation: (u * grad) u = grad(u) * u
      tensor gradient_u = integrator.get_gradient(q);
      if(operator_data.kernel_data.ale == true)
        u -= integrator_u_grid.get_value(q);

      vector F = gradient_u * u;

      // plus sign since the strong formulation is used, i.e.
      // integration by parts is performed twice
      integrator.submit_value(F, q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_integral_nonlinear_operator(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p,
  IntegratorFace & integrator_grid_velocity) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m      = integrator_m.get_value(q);
    vector u_p      = integrator_p.get_value(q);
    vector normal_m = integrator_m.get_normal_vector(q);

    vector u_grid;
    if(operator_data.kernel_data.ale == true)
      u_grid = integrator_grid_velocity.get_value(q);

    std::tuple<vector, vector> flux =
      kernel->calculate_flux_nonlinear_interior_and_neighbor(u_m, u_p, normal_m, u_grid);

    integrator_m.submit_value(std::get<0>(flux), q);
    integrator_p.submit_value(std::get<1>(flux), q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral_nonlinear_operator(
  IntegratorFace &                   integrator,
  IntegratorFace &                   integrator_grid_velocity,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = integrator.get_value(q);
    vector u_p = calculate_exterior_value_nonlinear(u_m,
                                                    q,
                                                    integrator,
                                                    boundary_type,
                                                    operator_data.kernel_data.type_dirichlet_bc,
                                                    boundary_id,
                                                    operator_data.bc,
                                                    this->time);

    vector normal_m = integrator.get_normal_vector(q);

    vector u_grid;
    if(operator_data.kernel_data.ale == true)
      u_grid = integrator_grid_velocity.get_value(q);

    vector flux =
      kernel->calculate_flux_nonlinear_boundary(u_m, u_p, normal_m, u_grid, boundary_type);

    integrator.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::cell_loop_nonlinear_operator_wall_enrichment(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range) const
{
  if (matrix_free.get_cell_range_category(cell_range) != 1)
  {
    return;
  }
  FEEnrichedEvaluation<dim, Number, IntegratorCell> integrator(matrix_free,
                                                               *(this->function_enrichment));

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator.gather_evaluate(src,
                               function_enrichment->enrichment_velocity,
                               this->integrator_flags.cell_evaluate);

    if(operator_data.kernel_data.ale)
    {
      AssertThrow(false,
                  dealii::ExcMessage("Wall enrichment with ALE not yet implemented"));
    }

    do_cell_integral_nonlinear_operator(integrator);

    integrator.integrate_scatter(this->integrator_flags.cell_integrate,
                                 dst,
                                 function_enrichment->enrichment_velocity);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::face_loop_nonlinear_operator_wall_enrichment(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  if (matrix_free.get_cell_range_category(face_range) == 0)
  {
    return;
  }
  unsigned int const fe_m = matrix_free.get_face_active_fe_index(face_range, true);
  unsigned int const fe_p = matrix_free.get_face_active_fe_index(face_range, false);
  FEFaceEnrichedEvaluation<dim, Number, IntegratorFace> integrator_m(matrix_free,
                                                                 *(this->function_enrichment),
                                                                 true,
                                                                 fe_m);
  FEFaceEnrichedEvaluation<dim, Number, IntegratorFace> integrator_p(matrix_free,
                                                                 *(this->function_enrichment),
                                                                 false,
                                                                 fe_p);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator_m.gather_evaluate(src,
                                 function_enrichment->enrichment_velocity,
                                 this->integrator_flags.face_evaluate);

    integrator_p.gather_evaluate(src,
                                 function_enrichment->enrichment_velocity,
                                 this->integrator_flags.face_evaluate);

    if(operator_data.kernel_data.ale)
    {
      AssertThrow(false,
                  dealii::ExcMessage("Wall enrichment with ALE not yet implemented"));
    }

    do_face_integral_nonlinear_operator(integrator_m, integrator_p);

    integrator_m.integrate_scatter(this->integrator_flags.face_integrate,
                                   dst,
                                   function_enrichment->enrichment_residual);

    integrator_p.integrate_scatter(this->integrator_flags.face_integrate,
                                   dst,
                                   function_enrichment->enrichment_residual);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::boundary_face_loop_nonlinear_operator_wall_enrichment(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  if (matrix_free.get_cell_range_category(face_range) == 0)
  {
    return;
  }

  unsigned int const fe_m = matrix_free.get_face_active_fe_index(face_range, true);
  FEFaceEnrichedEvaluation<dim, Number, IntegratorFace> integrator_m(matrix_free,
                                                                 *(this->function_enrichment),
                                                                 true,
                                                                 fe_m);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_m.reinit(face);

    // Strictly speaking, the variable integrator_flags refers to the linearized operator, but
    // integrator_flags is also valid for the nonlinear operator.
    integrator_m.gather_evaluate(src,
                                 function_enrichment->enrichment_velocity,
                                 this->integrator_flags.face_evaluate);

    if(operator_data.kernel_data.ale)
    {
      AssertThrow(false,
                  dealii::ExcMessage("Wall enrichment with ALE not yet implemented"));
    }

    do_boundary_integral_nonlinear_operator(integrator_m,
                                            matrix_free.get_boundary_id(face));

    integrator_m.integrate_scatter(this->integrator_flags.face_integrate,
                                   dst,
                                   function_enrichment->enrichment_residual);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral_nonlinear_operator(
  FEEnrichedEvaluation<dim, Number, IntegratorCell> & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u = integrator.get_value(q);

    if(operator_data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      // nonlinear convective flux F(u) = uu
      tensor F = outer_product(u, u);
      // minus sign due to integration by parts
      integrator.submit_gradient(-F, q);
    }
    else if(operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      // convective formulation: (u * grad) u = grad(u) * u
      tensor gradient_u = integrator.get_gradient(q);

      vector F = gradient_u * u;

      // plus sign since the strong formulation is used, i.e.
      // integration by parts is performed twice
      integrator.submit_value(F, q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_integral_nonlinear_operator(
  FEFaceEnrichedEvaluation<dim, Number, IntegratorFace> & integrator_m,
  FEFaceEnrichedEvaluation<dim, Number, IntegratorFace> & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m      = integrator_m.get_value(q);
    vector u_p      = integrator_p.get_value(q);
    vector normal_m = integrator_m.get_normal_vector(q);

    vector u_grid;
    u_grid = 0.0;

    std::tuple<vector, vector> flux =
      kernel->calculate_flux_nonlinear_interior_and_neighbor(u_m, u_p, normal_m, u_grid);

    integrator_m.submit_value(std::get<0>(flux), q);
    integrator_p.submit_value(std::get<1>(flux), q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral_nonlinear_operator(
  FEFaceEnrichedEvaluation<dim, Number, IntegratorFace> &                   integrator,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = integrator.get_value(q);
    vector u_p = calculate_exterior_value_nonlinear(u_m,
                                                    q,
                                                    integrator.phi_dg,
                                                    boundary_type,
                                                    operator_data.kernel_data.type_dirichlet_bc,
                                                    boundary_id,
                                                    operator_data.bc,
                                                    this->time);

    vector normal_m = integrator.get_normal_vector(q);

    vector u_grid;
    u_grid = 0.0;

    vector flux =
      kernel->calculate_flux_nonlinear_boundary(u_m, u_p, normal_m, u_grid, boundary_type);

    integrator.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_cell_derived(IntegratorCell &   integrator,
                                                     unsigned int const cell) const
{
  (void)integrator;

  kernel->reinit_cell(cell);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face_derived(IntegratorFace &   integrator_m,
                                                     IntegratorFace &   integrator_p,
                                                     unsigned int const face) const
{
  (void)integrator_m;
  (void)integrator_p;

  kernel->reinit_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_boundary_face_derived(IntegratorFace &   integrator_m,
                                                              unsigned int const face) const
{
  (void)integrator_m;

  kernel->reinit_boundary_face(face);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::reinit_face_cell_based_derived(
  IntegratorFace &                 integrator_m,
  IntegratorFace &                 integrator_p,
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  (void)integrator_m;
  (void)integrator_p;

  kernel->reinit_face_cell_based(cell, face, boundary_id);
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector delta_u = integrator.get_value(q);

    if(operator_data.kernel_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      tensor flux = kernel->get_volume_flux_linearized_divergence_formulation(delta_u, q);

      integrator.submit_gradient(flux, q);
    }
    else if(operator_data.kernel_data.formulation ==
            FormulationConvectiveTerm::ConvectiveFormulation)
    {
      tensor grad_delta_u = integrator.get_gradient(q);

      vector flux =
        kernel->get_volume_flux_linearized_convective_formulation(delta_u, grad_delta_u, q);

      integrator.submit_value(flux, q);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = kernel->get_velocity_p(q);

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p = integrator_p.get_value(q);

    vector normal_m = integrator_m.get_normal_vector(q);

    std::tuple<vector, vector> flux = kernel->calculate_flux_linearized_interior_and_neighbor(
      u_m, u_p, delta_u_m, delta_u_p, normal_m, q);

    integrator_m.submit_value(std::get<0>(flux) /* flux_m */, q);
    integrator_p.submit_value(std::get<1>(flux) /* flux_p */, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                      IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = kernel->get_velocity_p(q);

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p; // set exterior value to zero

    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux =
      kernel->calculate_flux_linearized_interior(u_m, u_p, delta_u_m, delta_u_p, normal_m, q);

    integrator_m.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_int_integral_cell_based(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    // TODO
    // Accessing exterior data is currently not available in deal.II/matrixfree.
    // Hence, we simply use the interior value, but note that the diagonal and block-diagonal
    // are not calculated exactly.
    vector u_p = u_m;

    vector delta_u_m = integrator_m.get_value(q);
    vector delta_u_p; // set exterior value to zero

    vector normal_m = integrator_m.get_normal_vector(q);

    vector flux =
      kernel->calculate_flux_linearized_interior(u_m, u_p, delta_u_m, delta_u_p, normal_m, q);

    integrator_m.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
                                                      IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = kernel->get_velocity_p(q);

    vector delta_u_m; // set exterior value to zero
    vector delta_u_p = integrator_p.get_value(q);

    vector normal_p = -integrator_p.get_normal_vector(q);

    vector flux =
      kernel->calculate_flux_linearized_interior(u_p, u_m, delta_u_p, delta_u_m, normal_p, q);

    integrator_p.submit_value(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectiveOperator<dim, Number>::do_boundary_integral(
  IntegratorFace &                   integrator,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  // make sure that this function is only accessed for OperatorType::homogeneous
  AssertThrow(
    operator_type == OperatorType::homogeneous,
    dealii::ExcMessage(
      "For the linearized convective operator, only OperatorType::homogeneous makes sense."));

  BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector u_m = kernel->get_velocity_m(q);
    vector u_p = calculate_exterior_value_nonlinear(u_m,
                                                    q,
                                                    integrator,
                                                    boundary_type,
                                                    operator_data.kernel_data.type_dirichlet_bc,
                                                    boundary_id,
                                                    operator_data.bc,
                                                    this->time);

    vector delta_u_m = integrator.get_value(q);
    vector delta_u_p =
      kernel->calculate_exterior_value_linearized(delta_u_m, q, integrator, boundary_type);

    vector normal_m = integrator.get_normal_vector(q);

    vector flux = kernel->calculate_flux_linearized_boundary(
      u_m, u_p, delta_u_m, delta_u_p, normal_m, boundary_type, q);

    integrator.submit_value(flux, q);
  }
}

template class ConvectiveOperator<2, float>;
template class ConvectiveOperator<2, double>;

template class ConvectiveOperator<3, float>;
template class ConvectiveOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
