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

#include <exadg/convection_diffusion/spatial_discretization/operators/rhs_operator.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
RHSOperator<dim, Number>::RHSOperator() : matrix_free(nullptr), time(0.0)
{
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  RHSOperatorData<dim> const &            data_in,
  std::shared_ptr<Operators::RHSKernel<dim, Number>> kernel_in)
{
  this->matrix_free = &matrix_free_in;
  this->data        = data_in;

  this->kernel = kernel_in;
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::evaluate(VectorType & dst,
                                                 double const evaluation_time) const
{
  dst = 0;
  evaluate_add(dst, evaluation_time);
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::evaluate_add(VectorType & dst,
                                                     double const evaluation_time) const
{
  this->time = evaluation_time;

  VectorType src;
  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_value(kernel->get_volume_flux(integrator, q, time), q);
  }

  integrator.integrate(dealii::EvaluationFlags::values);
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::cell_loop(
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

    do_cell_integral(integrator);

    integrator.distribute_local_to_global(dst);
  }
}

template class RHSOperator<2, float>;
template class RHSOperator<2, double>;
template class RHSOperator<3, float>;
template class RHSOperator<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
