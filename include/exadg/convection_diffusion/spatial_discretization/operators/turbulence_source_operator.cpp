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
template<int dim, typename Number, int n_components>
TurbulenceSourceOperator<dim, Number, n_components>::TurbulenceSourceOperator() : matrix_free(nullptr), time(0.0)
{
}

template<int dim, typename Number, int n_components>
void
TurbulenceSourceOperator<dim, Number, n_components>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  dealii::AffineConstraints<Number> const & affine_constraints_in,
  TurbulenceSourceOperatorData<dim> const &            data_in,
  std::shared_ptr<Operators::TurbulenceSourceKernel<dim, Number, n_components>> kernel_in)
{
  this->matrix_free = &matrix_free_in;
  this->data        = data_in;

  this->kernel = kernel_in;

  Base::reinit(matrix_free_in, affine_constraints_in, data_in);

  this->integrator_flags.cell_evaluate  = dealii::EvaluationFlags::values;
  this->integrator_flags.cell_integrate = dealii::EvaluationFlags::values;
}

template<int dim, typename Number, int n_components>
void
TurbulenceSourceOperator<dim, Number, n_components>::evaluate_source(VectorType & dst,
                                                                     double const evaluation_time) const
{
  dst = 0;
  evaluate_add_source(dst, evaluation_time);
}

template<int dim, typename Number, int n_components>
void
TurbulenceSourceOperator<dim, Number, n_components>::evaluate_add_source(VectorType & dst,
                                                                         double const evaluation_time) const
{
  this->time = evaluation_time;

  VectorType src;
  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number, int n_components>
void
TurbulenceSourceOperator<dim, Number, n_components>::cell_loop(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range) const
{
  (void)src;

  IntegratorCell integrator(matrix_free, data.dof_index, data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    kernel->reinit_cell(cell);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      integrator.submit_value(kernel->get_volume_flux(q), q);
    }

    integrator.integrate(dealii::EvaluationFlags::values);

    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number, int n_components>
void
TurbulenceSourceOperator<dim, Number, n_components>::reinit_cell_derived(
  IntegratorCell &   integrator,
  unsigned int const cell) const
{
  (void)integrator;
  kernel->reinit_cell(cell);
}

template<int dim, typename Number, int n_components>
void
TurbulenceSourceOperator<dim, Number, n_components>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    AssertThrow(false, dealii::ExcMessage("DEBUG"));
    integrator.submit_value(kernel->get_volume_flux(q), q);
  }
}

template<int dim, typename Number, int n_components>
void
TurbulenceSourceOperator<dim, Number, n_components>::set_velocity_ptr(VectorType const & velocity_in) const
{
  kernel->set_velocity_ptr(velocity_in);
}

template<int dim, typename Number, int n_components>
void
TurbulenceSourceOperator<dim, Number, n_components>::set_eddy_viscosity_ptr(VectorType const & src) const
{
  kernel->set_eddy_viscosity_ptr(src);
}

template<int dim, typename Number, int n_components>
void
TurbulenceSourceOperator<dim, Number, n_components>::set_solution_ptr(VectorType const & src) const
{
  kernel->set_solution_ptr(src);
}

template<int dim, typename Number, int n_components>
void
TurbulenceSourceOperator<dim, Number, n_components>::set_time_step_size(double const dt,
                                                           double const time,
                                                           double const end_time,
                                                           double const time_step_number) const
{
  kernel->set_time_step_size(dt, time, end_time, time_step_number);
}

template class TurbulenceSourceOperator<2, float, 1>;
template class TurbulenceSourceOperator<2, double, 1>;
template class TurbulenceSourceOperator<3, float, 1>;
template class TurbulenceSourceOperator<3, double, 1>;

template class TurbulenceSourceOperator<2, float, 2>;
template class TurbulenceSourceOperator<2, double, 2>;
template class TurbulenceSourceOperator<3, float, 2>;
template class TurbulenceSourceOperator<3, double, 2>;

} // namespace ConvDiff
} // namespace ExaDG
