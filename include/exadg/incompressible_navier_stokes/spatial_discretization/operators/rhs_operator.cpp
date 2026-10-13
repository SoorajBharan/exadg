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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/rhs_operator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
RHSOperator<dim, Number>::RHSOperator() : matrix_free(nullptr), time(0.0), temperature(nullptr)
{
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                     RHSOperatorData<dim> const &            data_in)
{
  this->matrix_free = &matrix_free_in;
  this->data        = data_in;

  kernel.reinit(data.kernel_data);
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::evaluate(VectorType & dst, Number const evaluation_time) const
{
  time = evaluation_time;

  VectorType src;
  matrix_free->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::evaluate_add(VectorType & dst, Number const evaluation_time) const
{
  time = evaluation_time;

  VectorType src;
  matrix_free->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::set_temperature(VectorType const & T)
{
  this->temperature = &T;
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::do_cell_integral(Integrator &       integrator,
                                           IntegratorScalar & integrator_temperature) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_value(kernel.get_volume_flux(integrator, integrator_temperature, q, time), q);
  }
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
                                    VectorType &                            dst,
                                    VectorType const &                      src,
                                    Range const &                           cell_range) const
{
  (void)src;

  Integrator integrator(matrix_free, data.dof_index, data.quad_index);

  IntegratorScalar integrator_temperature(matrix_free, data.dof_index_scalar, data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    if(data.kernel_data.boussinesq_term)
    {
      integrator_temperature.reinit(cell);
      integrator_temperature.gather_evaluate(*temperature, dealii::EvaluationFlags::values);
    }

    do_cell_integral(integrator, integrator_temperature);

    integrator.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::evaluate_add_enriched(
  VectorType &                            dst_bar_enriched,
  VectorType &                            dst_tilde,
  FunctionEnrichment<dim, Number> const & function_enrichment,
  Number const                            evaluation_time) const
{
  AssertThrow(not data.kernel_data.boussinesq_term,
              dealii::ExcMessage("RHSOperator: Boussinesq term is not implemented for wall "
                                 "enrichment (temperature is not available on matrix_free_en)."));

  time = evaluation_time;

  typedef FEEnrichedEvaluation<dim, Number, Integrator> EnrichedIntegrator;

  auto const &       mf              = *function_enrichment.get_matrix_free();
  unsigned int const active_fe_index = function_enrichment.get_active_fe_index();

  EnrichedIntegrator integrator(mf, function_enrichment);

  // required by the kernel interface; never evaluated since there is no Boussinesq term
  IntegratorScalar integrator_temperature(mf,
                                          function_enrichment.get_dof_index_en_scalar(),
                                          function_enrichment.get_quad_index(),
                                          0,
                                          active_fe_index);

  for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
  {
    if(mf.get_cell_iterator(cell, 0)->active_fe_index() != active_fe_index)
      continue;

    integrator.reinit(cell);
    integrator.evaluate_enrichment_function(); // y, u_tau -> psi

    // (v_bar, f) and (psi v_tilde, f)
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_value(
        kernel.get_volume_flux(integrator.phi_dg, integrator_temperature, q, time), q);

    integrator.integrate(dealii::EvaluationFlags::values);
    integrator.distribute_local_to_global(dst_bar_enriched, dst_tilde);
  }

  dst_bar_enriched.compress(dealii::VectorOperation::add);
  dst_tilde.compress(dealii::VectorOperation::add);
}

template class RHSOperator<2, float>;
template class RHSOperator<2, double>;

template class RHSOperator<3, float>;
template class RHSOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
