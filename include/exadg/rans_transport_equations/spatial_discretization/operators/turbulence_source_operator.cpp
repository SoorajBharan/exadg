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

#include <exadg/rans_transport_equations/spatial_discretization/operators/turbulence_source_operator.h>

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <unordered_map>
#include "exadg/operators/wall_modelling/wall_law_evaluator.h"

namespace ExaDG
{
namespace RANS
{
template<int dim, typename Number>
TurbulenceSourceOperator<dim, Number>::TurbulenceSourceOperator() : matrix_free(nullptr)
{
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                         matrix_free_in,
  TurbulenceSourceOperatorData<dim> const &                       data_in,
  std::shared_ptr<Operators::TurbulenceSourceKernel<dim, Number>> kernel_in)
{
  matrix_free = &matrix_free_in;
  data        = data_in;
  kernel      = kernel_in;
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::evaluate_add_explicit(BlockVectorType &       dst,
                                                             BlockVectorType const & solution_n,
                                                             double const            time) const
{
  kernel->set_solution_ptr(solution_n);
  kernel->set_evaluation_time(time);

  terms = Terms::explicit_terms;

  // the solution of the kernel is no longer the linearization point
  linearization_stored = false;

  BlockVectorType src;
  // enrichment blocks: the loops of the RANS MatrixFree see the RANS variables only
  unsigned int const n_rans          = kernel->get_n_variables();
  bool const         enriched_blocks = dst.n_blocks() > n_rans;
  BlockVectorType    dst_rans, src_rans;
  if(enriched_blocks)
  {
    dst_rans = copy_leading_blocks(dst, n_rans);
  }
  BlockVectorType &       dst_loop = enriched_blocks ? dst_rans : dst;
  BlockVectorType const & src_loop = src;

  if(diffusion_in_terms())
    matrix_free->loop(&This::cell_loop_terms,
                      &This::face_loop_terms,
                      &This::boundary_face_loop_terms,
                      this,
                      dst_loop,
                      src_loop);
  else
    matrix_free->cell_loop(&This::cell_loop_terms, this, dst_loop, src_loop);

  if(enriched_blocks)
    copy_back_leading_blocks(dst_rans, dst);

  // enrichment of RANS variables: source terms of the enriched wall cells
  evaluate_add_enriched(dst);
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::evaluate_add_newton_explicit(BlockVectorType &       dst,
                                                                    BlockVectorType const & solution,
                                                                    double const time) const
{
  kernel->set_solution_ptr(solution);
  kernel->set_evaluation_time(time);

  terms = Terms::newton_explicit_terms;

  BlockVectorType src;
  // enrichment blocks: the loops of the RANS MatrixFree see the RANS variables only
  unsigned int const n_rans          = kernel->get_n_variables();
  bool const         enriched_blocks = dst.n_blocks() > n_rans;
  BlockVectorType    dst_rans, src_rans;
  if(enriched_blocks)
  {
    dst_rans = copy_leading_blocks(dst, n_rans);
  }
  BlockVectorType &       dst_loop = enriched_blocks ? dst_rans : dst;
  BlockVectorType const & src_loop = src;

  if(diffusion_in_terms())
    matrix_free->loop(&This::cell_loop_terms,
                      &This::face_loop_terms,
                      &This::boundary_face_loop_terms,
                      this,
                      dst_loop,
                      src_loop);
  else
    matrix_free->cell_loop(&This::cell_loop_terms, this, dst_loop, src_loop);

  if(enriched_blocks)
    copy_back_leading_blocks(dst_rans, dst);

  // enrichment of RANS variables: source terms of the enriched wall cells
  evaluate_add_enriched(dst);

  // the loops above stored the linearization point of apply_add_newton_implicit()
  linearization_stored = true;
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::apply_add_newton_implicit(BlockVectorType &       dst,
                                                                 BlockVectorType const & src) const
{
  if(not has_implicit_terms())
    return;

  AssertThrow(linearization_stored,
              dealii::ExcMessage("Call evaluate_add_newton_explicit() before "
                                 "apply_add_newton_implicit(): the linearization point is stored "
                                 "by the Newton_Explicit loops."));

  // enrichment blocks: the loops of the RANS MatrixFree see the RANS variables only
  unsigned int const n_rans          = kernel->get_n_variables();
  bool const         enriched_blocks = dst.n_blocks() > n_rans;
  BlockVectorType    dst_rans, src_rans;
  if(enriched_blocks)
  {
    dst_rans = copy_leading_blocks(dst, n_rans);
    src_rans = copy_leading_blocks(src, n_rans);
  }
  BlockVectorType &       dst_loop = enriched_blocks ? dst_rans : dst;
  BlockVectorType const & src_loop = enriched_blocks ? src_rans : src;

  if(kernel->get_data().viscosity_implicit)
    matrix_free->loop(&This::cell_loop_newton_implicit,
                      &This::face_loop_newton_implicit,
                      &This::boundary_face_loop_newton_implicit,
                      this,
                      dst_loop,
                      src_loop);
  else
    matrix_free->cell_loop(&This::cell_loop_newton_implicit, this, dst_loop, src_loop);

  if(enriched_blocks)
    copy_back_leading_blocks(dst_rans, dst);

  // enrichment of RANS variables: Jacobian of the enriched wall cells and their faces
  apply_add_newton_implicit_enriched(dst, src);
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::apply_add_block_diagonal_elementwise(
  unsigned int const   block,
  unsigned int const   cell,
  scalar * const       dst,
  scalar const * const src) const
{
  if(not has_implicit_terms())
    return;

  AssertThrow(linearization_stored,
              dealii::ExcMessage("Call evaluate_add_newton_explicit() before apply_add_block_diagonal_elementwise(): the linearization point is stored by the Newton_Explicit loops."));

  AssertThrow(block < kernel->get_n_variables(), dealii::ExcMessage("block out of range"));

  unsigned int const n_variables = kernel->get_n_variables();
  bool const         diffusion   = kernel->get_data().viscosity_implicit;

  IntegratorCell integrator(*matrix_free, data.dof_index, data.quad_index);
  integrator.reinit(cell);

  // linearization point (stored by the Newton_Explicit loop)
  kernel->reinit_cell_stored(cell);

  // element vector of variable "block"
  for(unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
    integrator.begin_dof_values()[i] = src[i];

  integrator.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

  dealii::EvaluationFlags::EvaluationFlags integration_flags = dealii::EvaluationFlags::values;
  if(diffusion)
    integration_flags = integration_flags | dealii::EvaluationFlags::gradients;

  // increment: only variable "block" is non-zero, so the kernel gives d(...)/dQ_block only
  dealii::AlignedVector<scalar> values(n_variables, dealii::make_vectorized_array<Number>(0.0));
  dealii::AlignedVector<vector> gradients(n_variables);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    values[block]    = integrator.get_value(q);
    gradients[block] = integrator.get_gradient(q);

    integrator.submit_value(kernel->get_source_newton_implicit(dealii::make_array_view(values),
                                                               dealii::make_array_view(gradients),
                                                               q,
                                                               block), q);

    if(diffusion)
      integrator.submit_gradient(kernel->get_diffusion_cell_flux_newton_implicit(
        dealii::make_array_view(values),
        dealii::make_array_view(gradients),
        q,
        block), q);
  }
  integrator.integrate(integration_flags);

  for(unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
    dst[i] += integrator.begin_dof_values()[i];
}

template<int dim, typename Number>
bool
TurbulenceSourceOperator<dim, Number>::has_implicit_terms() const
{
  Operators::TurbulenceSourceKernelData<dim> const & kernel_data = kernel->get_data();

  return kernel_data.production_implicit or kernel_data.dissipation_implicit or
  kernel_data.square_gradient_implicit or kernel_data.cross_diffusion_implicit or
  kernel_data.viscosity_implicit;
}

template<int dim, typename Number>
bool
TurbulenceSourceOperator<dim, Number>::diffusion_in_terms() const
{
  bool const viscosity_implicit = kernel->get_data().viscosity_implicit;

  return (terms == Terms::explicit_terms) ? not viscosity_implicit : viscosity_implicit;
}

/*
 * Right-hand side loops: Explicit or Newton_Explicit terms. The fluxes of the kernel have the
 * left-hand side sign, so the turbulent diffusion is submitted with the opposite sign here.
 */
template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::cell_loop_terms(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockVectorType &                       dst,
  BlockVectorType const &                 src,
  Range const &                           cell_range) const
{
  (void)src;

  unsigned int const n_variables = kernel->get_n_variables();
  bool const         diffusion   = diffusion_in_terms();

  std::vector<std::shared_ptr<IntegratorCell>> integrator(n_variables);
  for(auto & integrator_block : integrator)
    integrator_block =
      std::make_shared<IntegratorCell>(matrix_free, data.dof_index, data.quad_index);

  dealii::EvaluationFlags::EvaluationFlags integration_flags = dealii::EvaluationFlags::values;
  if(diffusion)
    integration_flags = integration_flags | dealii::EvaluationFlags::gradients;

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    // P, D, SG, nu_t at the solution set by set_solution_ptr()
    kernel->reinit_cell(cell);

    // linearization point of the Newton_Implicit loops
    if(terms == Terms::newton_explicit_terms)
      kernel->store_cell(cell);

    // enrichment of RANS variables: source terms and turbulent diffusion (cell part) of the
    // enriched wall cells are evaluated by evaluate_add_enriched(); here only on the other cells
    // (lane mask 1)
    scalar source_mask = dealii::make_vectorized_array<Number>(1.0);
    if(function_enrichment_rans)
      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        if(function_enrichment_rans->is_enriched_cell(
             matrix_free.get_cell_iterator(cell, v)->global_active_cell_index()))
          source_mask[v] = 0.0;

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator[i]->reinit(cell);

      for(unsigned int q = 0; q < integrator[i]->n_q_points; ++q)
      {
        scalar const source = (terms == Terms::explicit_terms) ?
                                kernel->get_source_explicit_value(q, i) :
                                kernel->get_source_newton_explicit_value(q, i);

        integrator[i]->submit_value(source_mask * source, q);

        if(diffusion)
          integrator[i]->submit_gradient(-source_mask * kernel->get_diffusion_cell_flux(q, i), q);
      }

      integrator[i]->integrate_scatter(integration_flags, dst.block(i));
    }
  }
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::face_loop_terms(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockVectorType &                       dst,
  BlockVectorType const &                 src,
  Range const &                           face_range) const
{
  (void)src;

  unsigned int const n_variables = kernel->get_n_variables();

  std::vector<std::shared_ptr<IntegratorFace>> integrator_m(n_variables), integrator_p(n_variables);
  for(unsigned int i = 0; i < n_variables; ++i)
  {
    integrator_m[i] =
      std::make_shared<IntegratorFace>(matrix_free, true, data.dof_index, data.quad_index);
    integrator_p[i] =
      std::make_shared<IntegratorFace>(matrix_free, false, data.dof_index, data.quad_index);
  }

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    // solution, nu_t on both sides at the solution set by set_solution_ptr()
    kernel->reinit_face(face);

    // faces of enriched wall cells: evaluate_add_enriched()
    scalar const mask = face_mask_not_enriched(matrix_free, face, false);

    // linearization point of the Newton_Implicit loops
    if(terms == Terms::newton_explicit_terms)
      kernel->store_face(face);

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator_m[i]->reinit(face);
      integrator_p[i]->reinit(face);

      for(unsigned int q = 0; q < integrator_m[i]->n_q_points; ++q)
      {
        auto const flux = kernel->get_diffusion_face_flux(q, i);

        // right-hand side: opposite sign of the left-hand side (-value on e-, +value on e+)
        integrator_m[i]->submit_value(mask * flux.value, q);
        integrator_p[i]->submit_value(-mask * flux.value, q);

        integrator_m[i]->submit_normal_derivative(-mask * flux.gradient_m, q);
        integrator_p[i]->submit_normal_derivative(-mask * flux.gradient_p, q);
      }

      integrator_m[i]->integrate_scatter(dealii::EvaluationFlags::values |
                                           dealii::EvaluationFlags::gradients,
                                         dst.block(i));
      integrator_p[i]->integrate_scatter(dealii::EvaluationFlags::values |
                                           dealii::EvaluationFlags::gradients,
                                         dst.block(i));
    }
  }
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::boundary_face_loop_terms(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockVectorType &                       dst,
  BlockVectorType const &                 src,
  Range const &                           face_range) const
{
  (void)src;

  unsigned int const n_variables = kernel->get_n_variables();

  std::vector<std::shared_ptr<IntegratorFace>> integrator_m(n_variables);
  for(auto & integrator_block : integrator_m)
    integrator_block =
      std::make_shared<IntegratorFace>(matrix_free, true, data.dof_index, data.quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    // interior solution, exterior values from the boundary conditions at evaluation_time
    kernel->reinit_boundary_face(face);

    // boundary faces of enriched wall cells: evaluate_add_enriched()
    scalar const mask = face_mask_not_enriched(matrix_free, face, true);

    // linearization point of the Newton_Implicit loops
    if(terms == Terms::newton_explicit_terms)
      kernel->store_face(face);

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator_m[i]->reinit(face);

      for(unsigned int q = 0; q < integrator_m[i]->n_q_points; ++q)
      {
        auto const flux = kernel->get_diffusion_face_flux(q, i);

        integrator_m[i]->submit_value(mask * flux.value, q);
        integrator_m[i]->submit_normal_derivative(-mask * flux.gradient_m, q);
      }

      integrator_m[i]->integrate_scatter(dealii::EvaluationFlags::values |
                                           dealii::EvaluationFlags::gradients,
                                         dst.block(i));
    }
  }
}

/*
 * Left-hand side loops: Newton_Implicit terms applied to the increment src.
 */
template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::cell_loop_newton_implicit(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockVectorType &                       dst,
  BlockVectorType const &                 src,
  Range const &                           cell_range) const
{
  unsigned int const n_variables = kernel->get_n_variables();
  bool const         diffusion   = kernel->get_data().viscosity_implicit;

  std::vector<std::shared_ptr<IntegratorCell>> integrator(n_variables);
  for(auto & integrator_block : integrator)
    integrator_block =
      std::make_shared<IntegratorCell>(matrix_free, data.dof_index, data.quad_index);

  dealii::EvaluationFlags::EvaluationFlags integration_flags = dealii::EvaluationFlags::values;
  if(diffusion)
    integration_flags = integration_flags | dealii::EvaluationFlags::gradients;

  // increment dQ_j and its gradient at the current quadrature point, for all variables j
  dealii::AlignedVector<scalar> values(n_variables);
  dealii::AlignedVector<vector> gradients(n_variables);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    // linearization point: P, D, SG, nu_t and their derivatives (stored by the Newton_Explicit loop)
    kernel->reinit_cell_stored(cell);

    // enriched wall cells: apply_add_newton_implicit_enriched() (lane mask 0)
    scalar enriched_mask = dealii::make_vectorized_array<Number>(1.0);
    if(function_enrichment_rans)
      for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
        if(function_enrichment_rans->is_enriched_cell(
             matrix_free.get_cell_iterator(cell, v)->global_active_cell_index()))
          enriched_mask[v] = 0.0;

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator[i]->reinit(cell);
      integrator[i]->gather_evaluate(src.block(i),
                                     dealii::EvaluationFlags::values |
                                       dealii::EvaluationFlags::gradients);
    }

    for(unsigned int q = 0; q < integrator[0]->n_q_points; ++q)
    {
      for(unsigned int j = 0; j < n_variables; ++j)
      {
        values[j]    = integrator[j]->get_value(q);
        gradients[j] = integrator[j]->get_gradient(q);
      }

      for(unsigned int i = 0; i < n_variables; ++i)
      {
        integrator[i]->submit_value(enriched_mask * kernel->get_source_newton_implicit(
                                      dealii::make_array_view(values),
                                      dealii::make_array_view(gradients),
                                      q,
                                      i),
                                    q);

        if(diffusion)
          integrator[i]->submit_gradient(enriched_mask * kernel->get_diffusion_cell_flux_newton_implicit(
                                           dealii::make_array_view(values),
                                           dealii::make_array_view(gradients),
                                           q,
                                           i),
                                         q);
      }
    }

    for(unsigned int i = 0; i < n_variables; ++i)
      integrator[i]->integrate_scatter(integration_flags, dst.block(i));
  }
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::face_loop_newton_implicit(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockVectorType &                       dst,
  BlockVectorType const &                 src,
  Range const &                           face_range) const
{
  unsigned int const n_variables = kernel->get_n_variables();

  std::vector<std::shared_ptr<IntegratorFace>> integrator_m(n_variables), integrator_p(n_variables);
  for(unsigned int i = 0; i < n_variables; ++i)
  {
    integrator_m[i] =
      std::make_shared<IntegratorFace>(matrix_free, true, data.dof_index, data.quad_index);
    integrator_p[i] =
      std::make_shared<IntegratorFace>(matrix_free, false, data.dof_index, data.quad_index);
  }

  // increment dQ_j and its normal gradient on both sides at the current quadrature point
  dealii::AlignedVector<scalar> values_m(n_variables), values_p(n_variables);
  dealii::AlignedVector<scalar> normal_gradients_m(n_variables), normal_gradients_p(n_variables);

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    // linearization point on both sides (stored by the Newton_Explicit loop)
    kernel->reinit_face_stored(face);

    // faces of enriched wall cells: apply_add_newton_implicit_enriched()
    scalar const mask = face_mask_not_enriched(matrix_free, face, false);

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator_m[i]->reinit(face);
      integrator_p[i]->reinit(face);
      integrator_m[i]->gather_evaluate(src.block(i),
                                       dealii::EvaluationFlags::values |
                                       dealii::EvaluationFlags::gradients);
      integrator_p[i]->gather_evaluate(src.block(i),
                                       dealii::EvaluationFlags::values |
                                       dealii::EvaluationFlags::gradients);
    }

    for(unsigned int q = 0; q < integrator_m[0]->n_q_points; ++q)
    {
      for(unsigned int j = 0; j < n_variables; ++j)
      {
        values_m[j]           = integrator_m[j]->get_value(q);
        values_p[j]           = integrator_p[j]->get_value(q);
        normal_gradients_m[j] = integrator_m[j]->get_normal_derivative(q);
        normal_gradients_p[j] = integrator_p[j]->get_normal_derivative(q);
      }

      for(unsigned int i = 0; i < n_variables; ++i)
      {
        auto const flux =
          kernel->get_diffusion_face_flux_newton_implicit(dealii::make_array_view(values_m),
                                                          dealii::make_array_view(values_p),
                                                          dealii::make_array_view(normal_gradients_m),
                                                          dealii::make_array_view(normal_gradients_p),
                                                          q,
                                                          i);

        // left-hand side, as in DiffusiveOperator::do_face_integral()
        integrator_m[i]->submit_value(-mask * flux.value, q);
        integrator_p[i]->submit_value(mask * flux.value, q);

        integrator_m[i]->submit_normal_derivative(mask * flux.gradient_m, q);
        integrator_p[i]->submit_normal_derivative(mask * flux.gradient_p, q);
      }
    }

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator_m[i]->integrate_scatter(dealii::EvaluationFlags::values |
                                         dealii::EvaluationFlags::gradients,
                                         dst.block(i));
      integrator_p[i]->integrate_scatter(dealii::EvaluationFlags::values |
                                         dealii::EvaluationFlags::gradients,
                                         dst.block(i));
    }
  }
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::boundary_face_loop_newton_implicit(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockVectorType &                       dst,
  BlockVectorType const &                 src,
  Range const &                           face_range) const
{
  unsigned int const n_variables = kernel->get_n_variables();

  std::vector<std::shared_ptr<IntegratorFace>> integrator_m(n_variables);
  for(auto & integrator_block : integrator_m)
  integrator_block =
    std::make_shared<IntegratorFace>(matrix_free, true, data.dof_index, data.quad_index);

  // increment dQ_j and its normal gradient on the interior side at the current quadrature point
  dealii::AlignedVector<scalar> values_m(n_variables);
  dealii::AlignedVector<scalar> normal_gradients_m(n_variables);

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    // linearization point (stored by the Newton_Explicit loop); exterior values from the boundary conditions
    kernel->reinit_boundary_face_stored(face);

    // boundary faces of enriched wall cells: apply_add_newton_implicit_enriched()
    scalar const mask = face_mask_not_enriched(matrix_free, face, true);

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator_m[i]->reinit(face);
      integrator_m[i]->gather_evaluate(src.block(i),
                                       dealii::EvaluationFlags::values |
                                         dealii::EvaluationFlags::gradients);
    }

    for(unsigned int q = 0; q < integrator_m[0]->n_q_points; ++q)
    {
      for(unsigned int j = 0; j < n_variables; ++j)
      {
        values_m[j]           = integrator_m[j]->get_value(q);
        normal_gradients_m[j] = integrator_m[j]->get_normal_derivative(q);
      }

      for(unsigned int i = 0; i < n_variables; ++i)
      {
        // on boundary faces the kernel uses only the interior arrays (homogeneous boundary
        // conditions for the increment)
        auto const flux =
          kernel->get_diffusion_face_flux_newton_implicit(dealii::make_array_view(values_m),
                                                          dealii::make_array_view(values_m),
                                                          dealii::make_array_view(normal_gradients_m),
                                                          dealii::make_array_view(normal_gradients_m),
                                                          q,
                                                          i);

        integrator_m[i]->submit_value(-mask * flux.value, q);
        integrator_m[i]->submit_normal_derivative(mask * flux.gradient_m, q);
      }
    }

    for(unsigned int i = 0; i < n_variables; ++i)
      integrator_m[i]->integrate_scatter(dealii::EvaluationFlags::values |
                                           dealii::EvaluationFlags::gradients,
                                         dst.block(i));
  }
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::set_velocity_ptr(VectorType const & velocity_in) const
{
  kernel->set_velocity_ptr(velocity_in);
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::set_time_step_size(double const dt,
                                                          double const time,
                                                          double const end_time,
                                                          double const time_step_number) const
{
  kernel->set_time_step_size(dt, time, end_time, time_step_number);
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::set_function_enrichment_rans(
  std::shared_ptr<FunctionEnrichmentRANS<dim, Number>> function_enrichment_rans_in) const
{
  function_enrichment_rans = function_enrichment_rans_in;
}

/*
 * Source terms on the enriched wall cells, enrichment quadrature of FunctionEnrichmentRANS, with
 * W_h = W_bar + psi W_tilde of the enriched variables and the exact velocity gradient:
 *   dst.block(b)            += int v_bar S_b(Q_h)         (enriched cells of the RANS variables)
 *   dst.block(n_var + i)    += int psi v_tilde S_r(Q_h)   (enriched variable i, RANS block r)
 * Explicit or Newton_Explicit part according to "terms". The kernel is evaluated at the points of
 * the enrichment quadrature (reinit_points) and reset afterwards.
 */
template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::evaluate_add_enriched(BlockVectorType & dst) const
{
  if(not function_enrichment_rans)
    return;

  FunctionEnrichmentRANS<dim, Number> const & fer   = *function_enrichment_rans;
  dealii::MatrixFree<dim, Number> const &     mf_en = *fer.get_matrix_free();

  unsigned int const n_variables = kernel->get_n_variables();
  unsigned int const n_enriched  = fer.n_variables();
  unsigned int const w0          = n_variables + n_enriched; // first wall-law field

  BlockVectorType const & solution = kernel->get_solution();

  // input: bar parts of all RANS variables (shadow layout), tilde parts, wall-law fields
  // (nu_t bound, P_k, wall-cell indicator; zero if not set)
  BlockVectorType src(w0 + 3);
  for(unsigned int b = 0; b < n_variables; ++b)
  {
    mf_en.initialize_dof_vector(src.block(b), fer.get_dof_index_dg());
    fer.copy_global_to_wall_layout(solution.block(b), src.block(b));
  }
  for(unsigned int i = 0; i < n_enriched; ++i)
  {
    mf_en.initialize_dof_vector(src.block(n_variables + i), fer.get_dof_index_en());
    src.block(n_variables + i).copy_locally_owned_data_from(solution.block(n_variables + i));
  }
  std::array<VectorType const *, 3> const wall_law_fields = {
    {kernel->get_eddy_viscosity_wall_law_ptr(),
     kernel->get_tke_production_from_wall_law_ptr(),
     kernel->get_wall_cell_indicator_ptr()}};
  for(unsigned int w = 0; w < 3; ++w)
  {
    mf_en.initialize_dof_vector(src.block(w0 + w), fer.get_dof_index_dg());
    if(wall_law_fields[w] != nullptr)
      fer.copy_eddy_viscosity_to_wall_layout(*wall_law_fields[w], src.block(w0 + w));
  }
  src.collect_sizes();

  // output: rows tested with v_bar (shadow layout) per RANS variable, rows tested with psi v_tilde
  std::vector<VectorType> dst_enriched(n_variables + n_enriched);
  for(unsigned int b = 0; b < n_variables; ++b)
    mf_en.initialize_dof_vector(dst_enriched[b], fer.get_dof_index_dg());
  for(unsigned int i = 0; i < n_enriched; ++i)
    mf_en.initialize_dof_vector(dst_enriched[n_variables + i], fer.get_dof_index_en());

  // kernel at the points of the enrichment quadrature, reset to its own cells / faces afterwards
  unsigned int const n_q_points_kernel      = kernel->get_n_q_points_cell();
  unsigned int const n_q_points_face_kernel = kernel->get_n_q_points_face();
  kernel->reinit_points(mf_en.get_n_q_points(fer.get_quad_index()));
  kernel->reinit_face_points(mf_en.get_n_q_points_face(fer.get_quad_index()));

  // Newton_Explicit terms: this solution is the linearization point of the enriched Jacobian
  if(terms == Terms::newton_explicit_terms)
    kernel->resize_storage(linearization_enriched,
                           mf_en.n_cell_batches(),
                           mf_en.n_inner_face_batches() + mf_en.n_boundary_face_batches());

  // turbulent diffusion part of these terms: face terms of the enriched cells as well
  if(diffusion_in_terms())
  {
    update_penalty_parameter_enriched();
    mf_en.loop(&This::cell_loop_terms_enriched,
               &This::face_loop_terms_enriched,
               &This::boundary_face_loop_terms_enriched,
               this,
               dst_enriched,
               src,
               true /*zero_dst*/);
  }
  else
    mf_en.cell_loop(&This::cell_loop_terms_enriched, this, dst_enriched, src, true /*zero_dst*/);

#ifdef EXADG_DEBUG
  if(balance_requested)
    print_omega_balance(mf_en.get_task_info().communicator);
#endif

  kernel->reinit_points(n_q_points_kernel);
  kernel->reinit_face_points(n_q_points_face_kernel);

  linearization_enriched_stored = (terms == Terms::newton_explicit_terms);

  // enriched cells -> global layout (halo cells of the shadow layout are zero)
  for(unsigned int b = 0; b < n_variables; ++b)
    fer.add_wall_layout_to_global(dst_enriched[b], dst.block(b));
  for(unsigned int i = 0; i < n_enriched; ++i)
    dst.block(n_variables + i) += dst_enriched[n_variables + i];
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::cell_loop_terms_enriched(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  std::vector<VectorType> &                     dst,
  BlockVectorType const &                       src,
  Range const &                                 cell_range) const
{
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number>                    CellIntegratorScalar;
  typedef FEOmegaEnrichedEvaluation<dim, Number, CellIntegratorScalar> CellIntegratorEnriched;

  FunctionEnrichmentRANS<dim, Number> const & fer = *function_enrichment_rans;

  // only enriched cells (hp: one active_fe_index per range)
  if(matrix_free.get_cell_range_category(cell_range) != fer.get_active_fe_index())
    return;

  unsigned int const n_variables = kernel->get_n_variables();
  unsigned int const n_enriched  = fer.n_variables();
  unsigned int const w0          = n_variables + n_enriched;

  // enriched variable index of RANS block b (-1: not enriched)
  std::vector<int> enriched_index(n_variables, -1);
  for(unsigned int i = 0; i < n_enriched; ++i)
    enriched_index[fer.get_rans_index(i)] = static_cast<int>(i);

  // integrators: W_bar + psi W_tilde for enriched variables, W_bar otherwise
  std::vector<std::shared_ptr<CellIntegratorScalar>>   phi(n_variables);
  std::vector<std::shared_ptr<CellIntegratorEnriched>> phi_enriched(n_variables);
  for(unsigned int b = 0; b < n_variables; ++b)
  {
    if(enriched_index[b] >= 0)
      phi_enriched[b] = std::make_shared<CellIntegratorEnriched>(matrix_free, fer, enriched_index[b]);
    else
      phi[b] = std::make_shared<CellIntegratorScalar>(
        matrix_free, fer.get_dof_index_dg(), fer.get_quad_index(), 0, fer.get_active_fe_index());
  }

  // wall-law fields (shadow layout)
  std::array<std::shared_ptr<CellIntegratorScalar>, 3> phi_wall_law;
  for(auto & integrator : phi_wall_law)
    integrator = std::make_shared<CellIntegratorScalar>(
      matrix_free, fer.get_dof_index_dg(), fer.get_quad_index(), 0, fer.get_active_fe_index());

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  scalar const zero = dealii::make_vectorized_array<Number>(0.0);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    for(unsigned int b = 0; b < n_variables; ++b)
    {
      if(phi_enriched[b])
      {
        phi_enriched[b]->reinit(cell);
        phi_enriched[b]->gather_evaluate(src.block(b),
                                         src.block(n_variables + enriched_index[b]),
                                         flags);
      }
      else
      {
        phi[b]->reinit(cell);
        phi[b]->gather_evaluate(src.block(b), flags);
      }
    }

    for(unsigned int w = 0; w < 3; ++w)
    {
      phi_wall_law[w]->reinit(cell);
      phi_wall_law[w]->gather_evaluate(src.block(w0 + w), dealii::EvaluationFlags::values);
    }

    // all terms of the RANS model at the points of the enrichment quadrature
    unsigned int const n_q_points = matrix_free.get_n_q_points(fer.get_quad_index());
    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      for(unsigned int b = 0; b < n_variables; ++b)
      {
        if(phi_enriched[b])
          kernel->set_point_solution(q,
                                     b,
                                     phi_enriched[b]->get_value(q),
                                     phi_enriched[b]->get_gradient(q));
        else
          kernel->set_point_solution(q, b, phi[b]->get_value(q), phi[b]->get_gradient(q));
      }

      kernel->evaluate_point(q,
                             fer.get_velocity_gradient_cell(cell, q),
                             kernel->get_eddy_viscosity_wall_law_ptr() != nullptr ?
                               phi_wall_law[0]->get_value(q) :
                               zero,
                             kernel->get_tke_production_from_wall_law_ptr() != nullptr ?
                               phi_wall_law[1]->get_value(q) :
                               zero,
                             kernel->get_wall_cell_indicator_ptr() != nullptr ?
                               phi_wall_law[2]->get_value(q) :
                               zero);
    }

#ifdef EXADG_DEBUG
    if(balance_requested)
      record_omega_balance(*phi_wall_law[0],
                           cell,
                           n_q_points,
                           matrix_free.n_active_entries_per_cell_batch(cell),
                           fer.get_rans_index(0));
#endif

    // linearization point of the enriched Jacobian
    if(terms == Terms::newton_explicit_terms)
      kernel->save_cell_data(linearization_enriched, cell);

    // right-hand side rows: v_bar (all variables) and psi v_tilde (enriched variables); source
    // terms and, if part of these terms, the cell flux of the turbulent diffusion (right-hand
    // side sign: -flux, tested with grad v_bar and grad(psi v_tilde))
    bool const diffusion = diffusion_in_terms();

    // molecular diffusivity (enriched variables)
    scalar const diffusivity = dealii::make_vectorized_array<Number>(kernel->get_data().diffusivity);

    dealii::EvaluationFlags::EvaluationFlags integration_flags = dealii::EvaluationFlags::values;
    if(diffusion)
      integration_flags = integration_flags | dealii::EvaluationFlags::gradients;

    for(unsigned int b = 0; b < n_variables; ++b)
    {
      for(unsigned int q = 0; q < n_q_points; ++q)
      {
        scalar const source = (terms == Terms::explicit_terms) ?
                                kernel->get_source_explicit_value(q, b) :
                                kernel->get_source_newton_explicit_value(q, b);

        if(phi_enriched[b])
        {
          // enriched variable: turbulent and molecular diffusion (the molecular diffusion of
          // the enriched cells is switched off in CombinedOperator)
          if(diffusion)
            phi_enriched[b]->submit_value_and_gradient(
              source,
              -(kernel->get_diffusion_cell_flux(q, b) +
                diffusivity * phi_enriched[b]->get_gradient(q)),
              q);
          else
            phi_enriched[b]->submit_value(source, q);
        }
        else
        {
          phi[b]->submit_value(source, q);
          if(diffusion)
            phi[b]->submit_gradient(-kernel->get_diffusion_cell_flux(q, b), q);
        }
      }

      if(phi_enriched[b])
        phi_enriched[b]->integrate_scatter(integration_flags,
                                           dst[b],
                                           dst[n_variables + enriched_index[b]]);
      else
        phi[b]->integrate_scatter(integration_flags, dst[b]);
    }
  }
}

/*
 * Interior penalty parameter per cell of the MatrixFree of FunctionEnrichmentRANS, copied from the
 * kernel's array (same cells, same finite element) by global active cell index: tau of the
 * enriched face terms is the tau of the face terms of the kernel. The geometry does not change,
 * computed once.
 */
template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::update_penalty_parameter_enriched() const
{
  FunctionEnrichmentRANS<dim, Number> const & fer   = *function_enrichment_rans;
  dealii::MatrixFree<dim, Number> const &     mf_en = *fer.get_matrix_free();

  unsigned int const n_batches_en = mf_en.n_cell_batches() + mf_en.n_ghost_cell_batches();
  if(penalty_parameter_enriched.size() == n_batches_en)
    return;

  // kernel's penalty parameter by global active cell index (locally owned and ghost cells)
  dealii::AlignedVector<scalar> const & array = kernel->get_penalty_parameter_array();
  std::unordered_map<dealii::types::global_cell_index, Number> penalty;
  for(unsigned int cell = 0; cell < array.size(); ++cell)
    for(unsigned int v = 0; v < matrix_free->n_active_entries_per_cell_batch(cell); ++v)
      penalty[matrix_free->get_cell_iterator(cell, v)->global_active_cell_index()] = array[cell][v];

  penalty_parameter_enriched.resize(n_batches_en);
  for(unsigned int cell = 0; cell < n_batches_en; ++cell)
  {
    penalty_parameter_enriched[cell] = dealii::make_vectorized_array<Number>(0.0);
    for(unsigned int v = 0; v < mf_en.n_active_entries_per_cell_batch(cell); ++v)
    {
      auto const it = penalty.find(mf_en.get_cell_iterator(cell, v)->global_active_cell_index());
      if(it != penalty.end())
        penalty_parameter_enriched[cell][v] = it->second;
    }
  }
}

/*
 * Turbulent diffusion on the faces of the enriched wall cells (SIPG, as face_loop_terms), on the
 * MatrixFree of FunctionEnrichmentRANS: W_h = W_bar + psi W_tilde on enriched sides, W_bar on halo
 * sides. Right-hand side sign as in face_loop_terms.
 */
template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::face_loop_terms_enriched(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<VectorType> &               dst,
  BlockVectorType const &                 src,
  Range const &                           face_range) const
{
  integrate_face_terms_enriched(matrix_free, dst, src, face_range, false);
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::boundary_face_loop_terms_enriched(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<VectorType> &               dst,
  BlockVectorType const &                 src,
  Range const &                           face_range) const
{
  integrate_face_terms_enriched(matrix_free, dst, src, face_range, true);
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::integrate_face_terms_enriched(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<VectorType> &               dst,
  BlockVectorType const &                 src,
  Range const &                           face_range,
  bool const                              boundary) const
{
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number>                        FaceIntegratorScalar;
  typedef FEFaceOmegaEnrichedEvaluation<dim, Number, FaceIntegratorScalar> FaceIntegratorEnriched;

  FunctionEnrichmentRANS<dim, Number> const & fer = *function_enrichment_rans;

  unsigned int const n_variables = kernel->get_n_variables();
  unsigned int const n_enriched  = fer.n_variables();
  unsigned int const n_sides     = boundary ? 1 : 2;
  unsigned int const lanes       = scalar::size();

  // faces with an enriched cell on one side (halo cells surround the enriched cells); boundary
  // faces of enriched cells. All other faces belong to face_loop_terms / boundary_face_loop_terms.
  std::array<unsigned int, 2> fe_index = {{matrix_free.get_face_active_fe_index(face_range, true),
                                           boundary ?
                                             0u :
                                             matrix_free.get_face_active_fe_index(face_range, false)}};
  unsigned int const enriched_fe_index = fer.get_active_fe_index();
  if(boundary and fe_index[0] != enriched_fe_index)
    return;
  if(not boundary and fe_index[0] != enriched_fe_index and fe_index[1] != enriched_fe_index)
    return;

  // enriched variable index of RANS block b (-1: not enriched)
  std::vector<int> enriched_index(n_variables, -1);
  for(unsigned int i = 0; i < n_enriched; ++i)
    enriched_index[fer.get_rans_index(i)] = static_cast<int>(i);

  // integrators per side and variable: W_bar + psi W_tilde on enriched sides of enriched
  // variables, W_bar otherwise
  std::array<std::vector<std::shared_ptr<FaceIntegratorScalar>>, 2>   phi;
  std::array<std::vector<std::shared_ptr<FaceIntegratorEnriched>>, 2> phi_enriched;
  for(unsigned int side = 0; side < n_sides; ++side)
  {
    bool const interior = (side == 0);
    phi[side].resize(n_variables);
    phi_enriched[side].resize(n_variables);
    for(unsigned int b = 0; b < n_variables; ++b)
    {
      if(fe_index[side] == enriched_fe_index and enriched_index[b] >= 0)
        phi_enriched[side][b] = std::make_shared<FaceIntegratorEnriched>(
          matrix_free, fer, interior, fe_index[side], enriched_index[b]);
      else
        phi[side][b] = std::make_shared<FaceIntegratorScalar>(
          matrix_free, interior, fer.get_dof_index_dg(), fer.get_quad_index(), 0, fe_index[side]);
    }
  }

  // the integrator of the bar part of variable 0 on a side (cell data, quadrature points)
  auto const bar_integrator = [&](unsigned int const side) -> FaceIntegratorScalar & {
    return phi_enriched[side][0] ? phi_enriched[side][0]->phi_dg : *phi[side][0];
  };

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  unsigned int const n_q_points = matrix_free.get_n_q_points_face(fer.get_quad_index());

  kernel->set_boundary_face(boundary);

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    for(unsigned int side = 0; side < n_sides; ++side)
      for(unsigned int b = 0; b < n_variables; ++b)
      {
        if(phi_enriched[side][b])
        {
          phi_enriched[side][b]->reinit(face);
          phi_enriched[side][b]->gather_evaluate(src.block(b),
                                                 src.block(n_variables + enriched_index[b]),
                                                 flags);
        }
        else
        {
          phi[side][b]->reinit(face);
          phi[side][b]->gather_evaluate(src.block(b), flags);
        }
      }

    // penalty: tau = max over both sides (boundary: interior side) * IP factor of the kernel
    scalar tau = bar_integrator(0).read_cell_data(penalty_parameter_enriched);
    if(not boundary)
      tau = std::max(tau, bar_integrator(1).read_cell_data(penalty_parameter_enriched));
    kernel->set_face_penalty(tau * kernel->get_penalty_factor());

    // boundary face: wall of the wall model and lanes with an active enrichment
    dealii::types::boundary_id const boundary_id =
      boundary ? matrix_free.get_boundary_id(face) : dealii::numbers::invalid_boundary_id;
    bool const wall = boundary and fer.is_wall_boundary(boundary_id);

    scalar enrichment_active = dealii::make_vectorized_array<Number>(0.0);
    if(wall)
    {
      auto const & cells = matrix_free.get_face_info(face).cells_interior;
      for(unsigned int v = 0; v < lanes; ++v)
        if(cells[v] != dealii::numbers::invalid_unsigned_int and
           fer.is_enrichment_active_lane(cells[v] / lanes, cells[v] % lanes))
          enrichment_active[v] = 1.0;
    }

    // lanes switched off by the k criterion (velocity enrichment active): omega Dirichlet with the
    // wall-function value (OmegaEnrichmentOffWallCondition::WallFunction), as U enriched only
    scalar k_off = dealii::make_vectorized_array<Number>(0.0);
    if(wall and fer.get_off_wall_condition() == OmegaEnrichmentOffWallCondition::WallFunction)
    {
      auto const & cells = matrix_free.get_face_info(face).cells_interior;
      for(unsigned int v = 0; v < lanes; ++v)
        if(cells[v] != dealii::numbers::invalid_unsigned_int and
           fer.is_k_switched_off_lane(cells[v] / lanes, cells[v] % lanes))
          k_off[v] = 1.0;
    }

    // Dirichlet mask per variable of a boundary face (Newton_Implicit face flux): wall condition
    // of the enriched variables on active lanes, condition of the application otherwise
    if(boundary)
      for(unsigned int b = 0; b < n_variables; ++b)
      {
        if(wall and enriched_index[b] >= 0)
          kernel->set_face_dirichlet_mask(
            b,
            (fer.get_wall_condition(enriched_index[b]) == OmegaEnrichmentWallCondition::WeakDirichlet ?
               enrichment_active :
               dealii::make_vectorized_array<Number>(0.0)) +
              k_off);
        else
          kernel->set_face_dirichlet_mask(
            b, dealii::make_vectorized_array<Number>(kernel->is_dirichlet(b, boundary_id) ? 1.0 : 0.0));
      }

    // face data of the kernel at the points of the enrichment quadrature
    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      for(unsigned int side = 0; side < n_sides; ++side)
      {
        for(unsigned int b = 0; b < n_variables; ++b)
        {
          if(phi_enriched[side][b])
            kernel->set_face_point_solution(side,
                                            q,
                                            b,
                                            phi_enriched[side][b]->get_value(q),
                                            phi_enriched[side][b]->get_normal_derivative(q));
          else
            kernel->set_face_point_solution(
              side, q, b, phi[side][b]->get_value(q), phi[side][b]->get_normal_derivative(q));
        }

        kernel->evaluate_face_point(side, q, fer.get_velocity_gradient_face(face, side == 0, q));
      }

      if(boundary)
      {
        for(unsigned int b = 0; b < n_variables; ++b)
        {
          scalar const zero = dealii::make_vectorized_array<Number>(0.0);

          if(wall and enriched_index[b] >= 0)
          {
            // enriched variable at the wall: chosen condition, on lanes with active enrichment;
            // switched-off lanes: zero flux
            // active lanes: the condition of the enrichment; lanes switched off by the k
            // criterion: wall-function value (k_off); all other lanes (y+ switched off, empty):
            // zero flux. The two masks do not overlap.
            bool const   weak = fer.get_wall_condition(enriched_index[b]) ==
                              OmegaEnrichmentWallCondition::WeakDirichlet;
            scalar const active_mask = weak ? enrichment_active : zero;
            scalar const mask        = active_mask + k_off;
            scalar const g = active_mask * fer.get_wall_value_face(enriched_index[b], face, q) +
                             k_off * fer.get_wall_value_off_face(enriched_index[b], face, q);
            kernel->set_boundary_exterior_point_lanes(q, b, mask, g);
          }
          else
          {
            // all other cases: boundary condition of the application
            bool const dirichlet = kernel->is_dirichlet(b, boundary_id);
            scalar const g =
              dirichlet ?
                kernel->get_dirichlet_value(b, boundary_id, bar_integrator(0).quadrature_point(q)) :
                zero;
            kernel->set_boundary_exterior_point(q, b, dirichlet, g);
          }
        }

        kernel->copy_boundary_viscosity(q);
      }
    }

    // linearization point of the enriched Jacobian
    if(terms == Terms::newton_explicit_terms)
      kernel->save_face_data(linearization_enriched, face);

    // SIPG fluxes, right-hand side sign: -value / normal derivative on e-, +value on e+
    scalar const diffusivity = dealii::make_vectorized_array<Number>(kernel->get_data().diffusivity);

    for(unsigned int b = 0; b < n_variables; ++b)
    {
      for(unsigned int q = 0; q < n_q_points; ++q)
      {
        auto flux = kernel->get_diffusion_face_flux(q, b);

        // enriched variable: + molecular diffusion (switched off in CombinedOperator on these
        // faces), SIPG with the same penalty and exterior values
        if(enriched_index[b] >= 0)
        {
          scalar const jump = kernel->get_face_point_value(0, q, b) - kernel->get_face_point_value(1, q, b);
          flux.value += 0.5 * diffusivity *
                          (kernel->get_face_point_normal_gradient(0, q, b) +
                           kernel->get_face_point_normal_gradient(1, q, b)) -
                        kernel->get_face_penalty() * diffusivity * jump;
          flux.gradient_m += -0.5 * diffusivity * jump;
          flux.gradient_p += -0.5 * diffusivity * jump;

          if(fer.get_face_formulation(enriched_index[b]) == OmegaEnrichmentFaceFormulation::NIPG)
          {
            flux.gradient_m = -flux.gradient_m;
            flux.gradient_p = -flux.gradient_p;
          }
        }

        for(unsigned int side = 0; side < n_sides; ++side)
        {
          scalar const value             = (side == 0) ? flux.value : -flux.value;
          scalar const normal_derivative = (side == 0) ? -flux.gradient_m : -flux.gradient_p;

          if(phi_enriched[side][b])
            phi_enriched[side][b]->submit_value_and_normal_derivative(value, normal_derivative, q);
          else
          {
            phi[side][b]->submit_value(value, q);
            phi[side][b]->submit_normal_derivative(normal_derivative, q);
          }
        }
      }

      for(unsigned int side = 0; side < n_sides; ++side)
      {
        if(phi_enriched[side][b])
          phi_enriched[side][b]->integrate_scatter(flags,
                                                   dst[b],
                                                   dst[n_variables + enriched_index[b]]);
        else
          phi[side][b]->integrate_scatter(flags, dst[b]);
      }
    }
  }
}

/*
 * Lane mask of a face batch of the kernel's MatrixFree: 0 if the face touches an enriched wall cell
 * (its turbulent diffusion is evaluated by evaluate_add_enriched()), 1 otherwise.
 */
template<int dim, typename Number>
typename TurbulenceSourceOperator<dim, Number>::scalar
TurbulenceSourceOperator<dim, Number>::face_mask_not_enriched(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      face,
  bool const                              boundary) const
{
  scalar mask = dealii::make_vectorized_array<Number>(1.0);
  if(not function_enrichment_rans)
    return mask;

  unsigned int const lanes = scalar::size();
  auto const &       info  = matrix_free.get_face_info(face);

  auto const is_enriched = [&](unsigned int const cell_index) {
    return cell_index != dealii::numbers::invalid_unsigned_int and
           function_enrichment_rans->is_enriched_cell(
             matrix_free.get_cell_iterator(cell_index / lanes, cell_index % lanes)
               ->global_active_cell_index());
  };

  for(unsigned int v = 0; v < lanes; ++v)
    if(is_enriched(info.cells_interior[v]) or (not boundary and is_enriched(info.cells_exterior[v])))
      mask[v] = 0.0;

  return mask;
}

/*
 * Newton_Implicit part of the enriched wall cells and their faces applied to the increment
 * dQ_h = dQ_bar + psi dQ_tilde (enriched variables, enriched sides), linearized at the solution of
 * the last Newton_Explicit evaluate_add_enriched() call (linearization_enriched), MatrixFree of
 * FunctionEnrichmentRANS. Left-hand side signs as cell_loop_newton_implicit / face loops.
 *   dst.block(b)         += rows tested with v_bar      (enriched and halo cells)
 *   dst.block(n_var + i) += rows tested with psi v_tilde
 */
template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::apply_add_newton_implicit_enriched(
  BlockVectorType &       dst,
  BlockVectorType const & src) const
{
  if(not function_enrichment_rans)
    return;

  AssertThrow(linearization_enriched_stored,
              dealii::ExcMessage("Enriched Newton_Implicit terms: call evaluate_add_newton_explicit() "
                                 "first (linearization point of the enriched wall cells)."));

  FunctionEnrichmentRANS<dim, Number> const & fer   = *function_enrichment_rans;
  dealii::MatrixFree<dim, Number> const &     mf_en = *fer.get_matrix_free();

  unsigned int const n_variables = kernel->get_n_variables();
  unsigned int const n_enriched  = fer.n_variables();

  // increment: bar parts of all variables (shadow layout), tilde parts
  BlockVectorType src_enriched(n_variables + n_enriched);
  for(unsigned int b = 0; b < n_variables; ++b)
  {
    mf_en.initialize_dof_vector(src_enriched.block(b), fer.get_dof_index_dg());
    fer.copy_global_to_wall_layout(src.block(b), src_enriched.block(b));
  }
  for(unsigned int i = 0; i < n_enriched; ++i)
  {
    mf_en.initialize_dof_vector(src_enriched.block(n_variables + i), fer.get_dof_index_en());
    src_enriched.block(n_variables + i).copy_locally_owned_data_from(src.block(n_variables + i));
  }
  src_enriched.collect_sizes();

  std::vector<VectorType> dst_enriched(n_variables + n_enriched);
  for(unsigned int b = 0; b < n_variables; ++b)
    mf_en.initialize_dof_vector(dst_enriched[b], fer.get_dof_index_dg());
  for(unsigned int i = 0; i < n_enriched; ++i)
    mf_en.initialize_dof_vector(dst_enriched[n_variables + i], fer.get_dof_index_en());

  // kernel at the points of the enrichment quadrature, reset afterwards
  unsigned int const n_q_points_kernel      = kernel->get_n_q_points_cell();
  unsigned int const n_q_points_face_kernel = kernel->get_n_q_points_face();
  kernel->reinit_points(mf_en.get_n_q_points(fer.get_quad_index()));
  kernel->reinit_face_points(mf_en.get_n_q_points_face(fer.get_quad_index()));

  if(kernel->get_data().viscosity_implicit)
    mf_en.loop(&This::cell_loop_newton_implicit_enriched,
               &This::face_loop_newton_implicit_enriched,
               &This::boundary_face_loop_newton_implicit_enriched,
               this,
               dst_enriched,
               src_enriched,
               true /*zero_dst*/);
  else
    mf_en.cell_loop(
      &This::cell_loop_newton_implicit_enriched, this, dst_enriched, src_enriched, true /*zero_dst*/);

  kernel->reinit_points(n_q_points_kernel);
  kernel->reinit_face_points(n_q_points_face_kernel);

  for(unsigned int b = 0; b < n_variables; ++b)
    fer.add_wall_layout_to_global(dst_enriched[b], dst.block(b));
  for(unsigned int i = 0; i < n_enriched; ++i)
    dst.block(n_variables + i) += dst_enriched[n_variables + i];
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::cell_loop_newton_implicit_enriched(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<VectorType> &               dst,
  BlockVectorType const &                 src,
  Range const &                           cell_range) const
{
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number>                    CellIntegratorScalar;
  typedef FEOmegaEnrichedEvaluation<dim, Number, CellIntegratorScalar> CellIntegratorEnriched;

  FunctionEnrichmentRANS<dim, Number> const & fer = *function_enrichment_rans;

  // only enriched cells (hp: one active_fe_index per range)
  if(matrix_free.get_cell_range_category(cell_range) != fer.get_active_fe_index())
    return;

  unsigned int const n_variables = kernel->get_n_variables();
  unsigned int const n_enriched  = fer.n_variables();
  bool const         diffusion   = kernel->get_data().viscosity_implicit;
  scalar const diffusivity = dealii::make_vectorized_array<Number>(kernel->get_data().diffusivity);

  std::vector<int> enriched_index(n_variables, -1);
  for(unsigned int i = 0; i < n_enriched; ++i)
    enriched_index[fer.get_rans_index(i)] = static_cast<int>(i);

  std::vector<std::shared_ptr<CellIntegratorScalar>>   phi(n_variables);
  std::vector<std::shared_ptr<CellIntegratorEnriched>> phi_enriched(n_variables);
  for(unsigned int b = 0; b < n_variables; ++b)
  {
    if(enriched_index[b] >= 0)
      phi_enriched[b] = std::make_shared<CellIntegratorEnriched>(matrix_free, fer, enriched_index[b]);
    else
      phi[b] = std::make_shared<CellIntegratorScalar>(
        matrix_free, fer.get_dof_index_dg(), fer.get_quad_index(), 0, fer.get_active_fe_index());
  }

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
  dealii::EvaluationFlags::EvaluationFlags integration_flags = dealii::EvaluationFlags::values;
  if(diffusion)
    integration_flags = flags;

  // increment dQ_j and its gradient at the current point, all variables j
  dealii::AlignedVector<scalar> values(n_variables);
  dealii::AlignedVector<vector> gradients(n_variables);

  unsigned int const n_q_points = matrix_free.get_n_q_points(fer.get_quad_index());

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    // linearization point of the enriched cell (stored by evaluate_add_enriched())
    kernel->load_cell_data(linearization_enriched, cell);

    for(unsigned int b = 0; b < n_variables; ++b)
    {
      if(phi_enriched[b])
      {
        phi_enriched[b]->reinit(cell);
        phi_enriched[b]->gather_evaluate(src.block(b),
                                         src.block(n_variables + enriched_index[b]),
                                         flags);
      }
      else
      {
        phi[b]->reinit(cell);
        phi[b]->gather_evaluate(src.block(b), flags);
      }
    }

    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      for(unsigned int j = 0; j < n_variables; ++j)
      {
        values[j]    = phi_enriched[j] ? phi_enriched[j]->get_value(q) : phi[j]->get_value(q);
        gradients[j] = phi_enriched[j] ? phi_enriched[j]->get_gradient(q) : phi[j]->get_gradient(q);
      }

      for(unsigned int b = 0; b < n_variables; ++b)
      {
        scalar const source = kernel->get_source_newton_implicit(dealii::make_array_view(values),
                                                                 dealii::make_array_view(gradients),
                                                                 q,
                                                                 b);

        vector flux;
        if(diffusion)
        {
          flux = kernel->get_diffusion_cell_flux_newton_implicit(dealii::make_array_view(values),
                                                                 dealii::make_array_view(gradients),
                                                                 q,
                                                                 b);
          // enriched variable: + molecular diffusion (switched off in CombinedOperator)
          if(phi_enriched[b])
            flux += diffusivity * gradients[b];
        }

        if(phi_enriched[b])
        {
          if(diffusion)
            phi_enriched[b]->submit_value_and_gradient(source, flux, q);
          else
            phi_enriched[b]->submit_value(source, q);
        }
        else
        {
          phi[b]->submit_value(source, q);
          if(diffusion)
            phi[b]->submit_gradient(flux, q);
        }
      }
    }

    for(unsigned int b = 0; b < n_variables; ++b)
    {
      if(phi_enriched[b])
        phi_enriched[b]->integrate_scatter(integration_flags,
                                           dst[b],
                                           dst[n_variables + enriched_index[b]]);
      else
        phi[b]->integrate_scatter(integration_flags, dst[b]);
    }
  }
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::face_loop_newton_implicit_enriched(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<VectorType> &               dst,
  BlockVectorType const &                 src,
  Range const &                           face_range) const
{
  integrate_face_newton_implicit_enriched(matrix_free, dst, src, face_range, false);
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::boundary_face_loop_newton_implicit_enriched(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<VectorType> &               dst,
  BlockVectorType const &                 src,
  Range const &                           face_range) const
{
  integrate_face_newton_implicit_enriched(matrix_free, dst, src, face_range, true);
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::integrate_face_newton_implicit_enriched(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  std::vector<VectorType> &               dst,
  BlockVectorType const &                 src,
  Range const &                           face_range,
  bool const                              boundary) const
{
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number>                    FaceIntegratorScalar;
  typedef FEFaceOmegaEnrichedEvaluation<dim, Number, FaceIntegratorScalar> FaceIntegratorEnriched;

  FunctionEnrichmentRANS<dim, Number> const & fer = *function_enrichment_rans;

  unsigned int const n_variables = kernel->get_n_variables();
  unsigned int const n_enriched  = fer.n_variables();
  unsigned int const n_sides     = boundary ? 1 : 2;

  // the faces of integrate_face_terms_enriched()
  std::array<unsigned int, 2> fe_index = {{matrix_free.get_face_active_fe_index(face_range, true),
                                           boundary ?
                                             0u :
                                             matrix_free.get_face_active_fe_index(face_range, false)}};
  unsigned int const enriched_fe_index = fer.get_active_fe_index();
  if(boundary and fe_index[0] != enriched_fe_index)
    return;
  if(not boundary and fe_index[0] != enriched_fe_index and fe_index[1] != enriched_fe_index)
    return;

  std::vector<int> enriched_index(n_variables, -1);
  for(unsigned int i = 0; i < n_enriched; ++i)
    enriched_index[fer.get_rans_index(i)] = static_cast<int>(i);

  std::array<std::vector<std::shared_ptr<FaceIntegratorScalar>>, 2>   phi;
  std::array<std::vector<std::shared_ptr<FaceIntegratorEnriched>>, 2> phi_enriched;
  for(unsigned int side = 0; side < n_sides; ++side)
  {
    bool const interior = (side == 0);
    phi[side].resize(n_variables);
    phi_enriched[side].resize(n_variables);
    for(unsigned int b = 0; b < n_variables; ++b)
    {
      if(fe_index[side] == enriched_fe_index and enriched_index[b] >= 0)
        phi_enriched[side][b] = std::make_shared<FaceIntegratorEnriched>(
          matrix_free, fer, interior, fe_index[side], enriched_index[b]);
      else
        phi[side][b] = std::make_shared<FaceIntegratorScalar>(
          matrix_free, interior, fer.get_dof_index_dg(), fer.get_quad_index(), 0, fe_index[side]);
    }
  }

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  scalar const diffusivity = dealii::make_vectorized_array<Number>(kernel->get_data().diffusivity);
  scalar const one         = dealii::make_vectorized_array<Number>(1.0);

  // increment dQ_j and its normal gradient on both sides at the current point
  dealii::AlignedVector<scalar> values_m(n_variables), values_p(n_variables);
  dealii::AlignedVector<scalar> normal_gradients_m(n_variables), normal_gradients_p(n_variables);

  unsigned int const n_q_points = matrix_free.get_n_q_points_face(fer.get_quad_index());

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    // linearization point and Dirichlet masks (stored by evaluate_add_enriched())
    kernel->load_face_data(linearization_enriched, face, boundary);

    for(unsigned int side = 0; side < n_sides; ++side)
      for(unsigned int b = 0; b < n_variables; ++b)
      {
        if(phi_enriched[side][b])
        {
          phi_enriched[side][b]->reinit(face);
          phi_enriched[side][b]->gather_evaluate(src.block(b),
                                                 src.block(n_variables + enriched_index[b]),
                                                 flags);
        }
        else
        {
          phi[side][b]->reinit(face);
          phi[side][b]->gather_evaluate(src.block(b), flags);
        }
      }

    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      for(unsigned int j = 0; j < n_variables; ++j)
      {
        values_m[j] =
          phi_enriched[0][j] ? phi_enriched[0][j]->get_value(q) : phi[0][j]->get_value(q);
        normal_gradients_m[j] = phi_enriched[0][j] ? phi_enriched[0][j]->get_normal_derivative(q) :
                                                     phi[0][j]->get_normal_derivative(q);
        if(boundary)
        {
          // on boundary faces the kernel uses only the interior arrays
          values_p[j]           = values_m[j];
          normal_gradients_p[j] = normal_gradients_m[j];
        }
        else
        {
          values_p[j] =
            phi_enriched[1][j] ? phi_enriched[1][j]->get_value(q) : phi[1][j]->get_value(q);
          normal_gradients_p[j] = phi_enriched[1][j] ?
                                    phi_enriched[1][j]->get_normal_derivative(q) :
                                    phi[1][j]->get_normal_derivative(q);
        }
      }

      for(unsigned int b = 0; b < n_variables; ++b)
      {
        auto flux =
          kernel->get_diffusion_face_flux_newton_implicit(dealii::make_array_view(values_m),
                                                          dealii::make_array_view(values_p),
                                                          dealii::make_array_view(normal_gradients_m),
                                                          dealii::make_array_view(normal_gradients_p),
                                                          q,
                                                          b);

        // enriched variable: + molecular diffusion (linear), exterior increment of a boundary face
        // from the Dirichlet mask (Dirichlet: dQ+ = -dQ-, zero flux: dQ+ = dQ-)
        if(enriched_index[b] >= 0)
        {
          scalar increment_p = values_p[b], normal_gradient_p = normal_gradients_p[b];
          if(boundary)
          {
            scalar const mask = kernel->get_face_dirichlet_mask(b);
            increment_p       = (one - 2.0 * mask) * values_m[b];
            normal_gradient_p = (2.0 * mask - one) * normal_gradients_m[b];
          }
          scalar const jump = values_m[b] - increment_p;
          flux.value += 0.5 * diffusivity * (normal_gradients_m[b] + normal_gradient_p) -
                        kernel->get_face_penalty() * diffusivity * jump;
          flux.gradient_m += -0.5 * diffusivity * jump;
          flux.gradient_p += -0.5 * diffusivity * jump;

          if(fer.get_face_formulation(enriched_index[b]) == OmegaEnrichmentFaceFormulation::NIPG)
          {
            flux.gradient_m = -flux.gradient_m;
            flux.gradient_p = -flux.gradient_p;
          }
        }

        // left-hand side, as in face_loop_newton_implicit(): -value / gradient_m on e-,
        // +value / gradient_p on e+
        for(unsigned int side = 0; side < n_sides; ++side)
        {
          scalar const value             = (side == 0) ? -flux.value : flux.value;
          scalar const normal_derivative = (side == 0) ? flux.gradient_m : flux.gradient_p;

          if(phi_enriched[side][b])
            phi_enriched[side][b]->submit_value_and_normal_derivative(value, normal_derivative, q);
          else
          {
            phi[side][b]->submit_value(value, q);
            phi[side][b]->submit_normal_derivative(normal_derivative, q);
          }
        }
      }
    }

    for(unsigned int side = 0; side < n_sides; ++side)
      for(unsigned int b = 0; b < n_variables; ++b)
      {
        if(phi_enriched[side][b])
          phi_enriched[side][b]->integrate_scatter(flags,
                                                   dst[b],
                                                   dst[n_variables + enriched_index[b]]);
        else
          phi[side][b]->integrate_scatter(flags, dst[b]);
      }
  }
}

#ifdef EXADG_DEBUG
template<int dim, typename Number>
template<typename Integrator>
void
TurbulenceSourceOperator<dim, Number>::record_omega_balance(Integrator const & integrator_points,
                                                            unsigned int const cell,
                                                            unsigned int const n_q_points,
                                                            unsigned int const n_active_lanes,
                                                            unsigned int const block_omega) const
{
  (void)cell;

  for(unsigned int v = 0; v < n_active_lanes; ++v)
  {
    // x range of this cell (lane)
    double x_min = std::numeric_limits<double>::max(), x_max = -std::numeric_limits<double>::max();
    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      double const x = integrator_points.quadrature_point(q)[0][v];
      x_min          = std::min(x_min, x);
      x_max          = std::max(x_max, x);
    }

    for(unsigned int s = 0; s < balance_x_stations.size(); ++s)
    {
      double const x_s = balance_x_stations[s];
      if(x_s < x_min or x_s > x_max)
        continue;

      // column of points closest to x_s
      double x_column = x_min;
      for(unsigned int q = 0; q < n_q_points; ++q)
      {
        double const x = integrator_points.quadrature_point(q)[0][v];
        if(std::abs(x - x_s) < std::abs(x_column - x_s))
          x_column = x;
      }

      for(unsigned int q = 0; q < n_q_points; ++q)
      {
        double const x = integrator_points.quadrature_point(q)[0][v];
        if(std::abs(x - x_column) > 1.e-12 * std::max(1.0, std::abs(x_column)))
          continue;

        auto const k_terms = kernel->get_point_terms(q, 0);
        auto const w_terms = kernel->get_point_terms(q, block_omega);
        std::array<double, n_balance_entries> const record = {
          {double(s),
           x,
           double(integrator_points.quadrature_point(q)[1][v]),
           double(k_terms[0][v]),
           double(w_terms[0][v]),
           double(k_terms[1][v]),
           double(k_terms[2][v]),
           double(w_terms[1][v]),
           double(w_terms[2][v]),
           double(w_terms[3][v]),
           double(w_terms[4][v]),
           double(w_terms[5][v]),
           double(w_terms[6][v])}};
        balance_records.insert(balance_records.end(), record.begin(), record.end());
      }
    }
  }
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::print_omega_balance(MPI_Comm const comm) const
{
  balance_requested = false;

  std::vector<std::vector<double>> const all = dealii::Utilities::MPI::all_gather(comm, balance_records);
  balance_records.clear();
  if(dealii::Utilities::MPI::this_mpi_process(comm) != 0)
    return;

  // records of all processes, per station sorted by y
  std::vector<std::array<double, n_balance_entries>> records;
  for(auto const & part : all)
    for(unsigned int i = 0; i + n_balance_entries <= part.size(); i += n_balance_entries)
    {
      std::array<double, n_balance_entries> r;
      std::copy(part.begin() + i, part.begin() + i + n_balance_entries, r.begin());
      records.push_back(r);
    }
  std::sort(records.begin(), records.end(), [](auto const & a, auto const & b) {
    return a[0] < b[0] or (a[0] == b[0] and a[2] < b[2]);
  });

  std::cout << "  [omega balance] time step " << balance_time_step
            << ": enriched wall cell, column of points closest to x; k equation: P_k, D_k; "
               "W equation (transported form): P_W, D_W, SG_W, source P - D + SG (= -diffusion if "
               "steady); nu_D diffusion viscosity (after the wall-law bound), nu_D,RANS before it, "
               "nu_t^psi the bound (unclamped)"
            << std::endl;
  int station = -1;
  for(auto const & r : records)
  {
    if(int(r[0]) != station)
    {
      station = int(r[0]);
      std::cout << "    x = " << balance_x_stations[station] << " (column at x = " << r[1] << ")"
                << std::endl
                << "      " << std::setw(11) << "y" << std::setw(12) << "k" << std::setw(12) << "W"
                << std::setw(12) << "P_k" << std::setw(12) << "D_k" << std::setw(12) << "P_W"
                << std::setw(12) << "D_W" << std::setw(12) << "SG_W" << std::setw(12)
                << "source_W" << std::setw(12) << "nu_D" << std::setw(12) << "nu_D,RANS"
                << std::setw(12) << "nu_t^psi" << std::endl;
    }
    std::cout << "      " << std::scientific << std::setprecision(3) << std::setw(11) << r[2]
              << std::setw(12) << r[3] << std::setw(12) << r[4] << std::setw(12) << r[5]
              << std::setw(12) << r[6] << std::setw(12) << r[7] << std::setw(12) << r[8]
              << std::setw(12) << r[9] << std::setw(12) << (r[7] - r[8] + r[9]) << std::setw(12)
              << r[10] << std::setw(12) << r[11] << std::setw(12) << r[12] << std::endl;
  }
  std::cout << std::defaultfloat;
}
#endif

template class TurbulenceSourceOperator<2, float>;
template class TurbulenceSourceOperator<2, double>;
template class TurbulenceSourceOperator<3, float>;
template class TurbulenceSourceOperator<3, double>;

} // namespace RANS
} // namespace ExaDG
