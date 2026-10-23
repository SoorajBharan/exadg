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

#include <exadg/convection_diffusion/spatial_discretization/operators/turbulence_source_operator.h>

namespace ExaDG
{
namespace ConvDiff
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

  BlockVectorType src;
  if(diffusion_in_terms())
    matrix_free->loop(&This::cell_loop_terms,
                      &This::face_loop_terms,
                      &This::boundary_face_loop_terms,
                      this,
                      dst,
                      src);
  else
    matrix_free->cell_loop(&This::cell_loop_terms, this, dst, src);
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
  if(diffusion_in_terms())
    matrix_free->loop(&This::cell_loop_terms,
                      &This::face_loop_terms,
                      &This::boundary_face_loop_terms,
                      this,
                      dst,
                      src);
  else
    matrix_free->cell_loop(&This::cell_loop_terms, this, dst, src);
}

template<int dim, typename Number>
void
TurbulenceSourceOperator<dim, Number>::apply_add_newton_implicit(BlockVectorType &       dst,
                                                                 BlockVectorType const & src) const
{
  if(not has_implicit_terms())
    return;

  if(kernel->get_data().viscosity_implicit)
    matrix_free->loop(&This::cell_loop_newton_implicit,
                      &This::face_loop_newton_implicit,
                      &This::boundary_face_loop_newton_implicit,
                      this,
                      dst,
                      src);
  else
    matrix_free->cell_loop(&This::cell_loop_newton_implicit, this, dst, src);
}

template<int dim, typename Number>
bool
TurbulenceSourceOperator<dim, Number>::has_implicit_terms() const
{
  Operators::TurbulenceSourceKernelData<dim> const & kernel_data = kernel->get_data();

  return kernel_data.production_implicit or kernel_data.dissipation_implicit or
         kernel_data.square_gradient_implicit or kernel_data.viscosity_implicit;
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

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator[i]->reinit(cell);

      for(unsigned int q = 0; q < integrator[i]->n_q_points; ++q)
      {
        scalar const source = (terms == Terms::explicit_terms) ?
                                kernel->get_source_explicit_value(q, i) :
                                kernel->get_source_newton_explicit_value(q, i);

        integrator[i]->submit_value(source, q);

        if(diffusion)
          integrator[i]->submit_gradient(-kernel->get_diffusion_cell_flux(q, i), q);
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

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator_m[i]->reinit(face);
      integrator_p[i]->reinit(face);

      for(unsigned int q = 0; q < integrator_m[i]->n_q_points; ++q)
      {
        auto const flux = kernel->get_diffusion_face_flux(q, i);

        // right-hand side: opposite sign of the left-hand side (-value on e-, +value on e+)
        integrator_m[i]->submit_value(flux.value, q);
        integrator_p[i]->submit_value(-flux.value, q);

        integrator_m[i]->submit_normal_derivative(-flux.gradient_m, q);
        integrator_p[i]->submit_normal_derivative(-flux.gradient_p, q);
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

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator_m[i]->reinit(face);

      for(unsigned int q = 0; q < integrator_m[i]->n_q_points; ++q)
      {
        auto const flux = kernel->get_diffusion_face_flux(q, i);

        integrator_m[i]->submit_value(flux.value, q);
        integrator_m[i]->submit_normal_derivative(-flux.gradient_m, q);
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
    // linearization point: P, D, SG, nu_t and their derivatives
    kernel->reinit_cell(cell);

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
        integrator[i]->submit_value(kernel->get_source_newton_implicit(
                                      dealii::make_array_view(values),
                                      dealii::make_array_view(gradients),
                                      q,
                                      i),
                                    q);

        if(diffusion)
          integrator[i]->submit_gradient(kernel->get_diffusion_cell_flux_newton_implicit(
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
    // linearization point on both sides
    kernel->reinit_face(face);

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
        integrator_m[i]->submit_value(-flux.value, q);
        integrator_p[i]->submit_value(flux.value, q);

        integrator_m[i]->submit_normal_derivative(flux.gradient_m, q);
        integrator_p[i]->submit_normal_derivative(flux.gradient_p, q);
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
    // linearization point; exterior values from the boundary conditions
    kernel->reinit_boundary_face(face);

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

        integrator_m[i]->submit_value(-flux.value, q);
        integrator_m[i]->submit_normal_derivative(flux.gradient_m, q);
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

template class TurbulenceSourceOperator<2, float>;
template class TurbulenceSourceOperator<2, double>;
template class TurbulenceSourceOperator<3, float>;
template class TurbulenceSourceOperator<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
