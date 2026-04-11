/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#include <exadg/convection_diffusion/spatial_discretization/turbulence_model.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, int n_components, typename Number>
TurbulenceModel<dim, n_components, Number>::TurbulenceModel()
  : dealii::Subscriptor(),
  quad_index(0),
  dof_index(0),
  matrix_free(nullptr)
{
}

template<int dim, int n_components, typename Number>
void
TurbulenceModel<dim, n_components, Number>::initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                            TurbulenceModelData const &                 turbulence_model_data_in,
                                            unsigned int const                      dof_index_in,
                                            unsigned int const                      dof_index_viscosity_in,
                                            unsigned int const                      quad_index_in)
{
  matrix_free = &matrix_free_in;

  dof_index  = dof_index_in;
  dof_index_viscosity  = dof_index_viscosity_in;
  quad_index = quad_index_in;

  turbulence_model_data = turbulence_model_data_in;

  model_coefficients = turbulence_model_data_in.turbulence_data_base->get_all_coefficients();

  matrix_free->initialize_dof_vector(eddy_viscosity, dof_index_viscosity_in);
}

template<int dim, int n_components, typename Number>
void
TurbulenceModel<dim, n_components, Number>::set_viscosity(VectorType const & solution)
{
  matrix_free->cell_loop(&TurbulenceModel<dim, n_components, Number>::cell_loop,
                         this,
                         eddy_viscosity,
                         solution);
}

template<int dim, int n_components, typename Number>
void
TurbulenceModel<dim, n_components, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType & dst,
  VectorType const & src,
  Range const &      cell_range)
{
  if(n_components != 2)
  {
    AssertThrow(false, dealii::ExcMessage("Current implemetation is only for 2 components turbulence models"));
  }

  IntegratorCell integrator(matrix_free, dof_index, quad_index);
  IntegratorCellScalar integrator_viscosity(matrix_free, dof_index_viscosity, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);

    integrator_viscosity.reinit(cell);

    for(unsigned int dof = 0; dof < integrator_viscosity.dofs_per_cell; ++dof)
    {
      auto solution_values = integrator.get_dof_value(dof);
      
      scalar eddy_viscosity_dof;

      evaluate_eddy_viscosity(solution_values, eddy_viscosity_dof);

      integrator_viscosity.submit_dof_value(eddy_viscosity_dof, dof);
    }
    integrator_viscosity.set_dof_values(dst);
  }
}

template<int dim, int n_components, typename Number>
template<typename DataType>
void
TurbulenceModel<dim, n_components, Number>::evaluate_eddy_viscosity(DataType const & solution_values,
                                                                    scalar & viscosity) const
{
  if constexpr (n_components == 1)
  {
    AssertThrow(false, dealii::ExcMessage("Turbulence models with 1 components not yet implemented."));
  }
  else if constexpr (n_components == 2)
  {
    // === 2-EQUATION MODELS (e.g., k-epsilon, k-omega) ===
    // Here, 'solution_values' is guaranteed to be a Tensor.

    switch(turbulence_model_data.turbulence_model)
    {
      case TurbulenceEddyViscosityModel::StandardKEpsilon:
        standard_k_epsilon_model(solution_values, viscosity);
        break;
      case TurbulenceEddyViscosityModel::StandardKOmega1988:
        standard_k_omega_1988_model(solution_values, viscosity);
        break;
      case TurbulenceEddyViscosityModel::Undefined:
        AssertThrow(false, dealii::ExcMessage("RANS::TurbulenceEddyViscosityModel must be specified"));
        break;
    }
  }
  else
{
    AssertThrow(false, dealii::ExcMessage("Turbulence models with >2 components not yet implemented."));
  }
}

template<int dim, int n_components, typename Number>
void
TurbulenceModel<dim, n_components, Number>::standard_k_epsilon_model(dealii::Tensor<1, n_components, scalar> const & solution_values,
                                                                    scalar & viscosity) const
{
  double C_mu = model_coefficients[3];

  scalar tke = solution_values[0];
  scalar epsilon = solution_values[1];

  if(turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::LogarithmicTransportVariable)
  {
    scalar log_terms = (dealii::make_vectorized_array<Number>(2.0) * tke) - epsilon;
    viscosity = C_mu * std::exp(log_terms);
  }
  else if(turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::Clipper)
  {
    viscosity = C_mu * tke * tke / std::max(epsilon, dealii::make_vectorized_array<Number>(1.e-6));
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "PositivityPreservingLimiter needs to be specified for  calculating viscosity"));
  }
}

template<int dim, int n_components, typename Number>
void
TurbulenceModel<dim, n_components, Number>::standard_k_omega_1988_model(dealii::Tensor<1, n_components, scalar> const & solution_values,
                                                                        scalar & viscosity) const
{
  scalar tke = solution_values[0];
  scalar omega = solution_values[1];

  if(turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::LogarithmicTransportVariable)
  {
    scalar log_terms = tke - omega;
    viscosity = std::exp(log_terms);
  }
  else if(turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::Clipper)
  {
    viscosity = tke / std::max(omega, dealii::make_vectorized_array<Number>(1.e-6));
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "PositivityPreservingLimiter needs to be specified for  calculating viscosity"));
  }
}

template<int dim, int n_components, typename Number>
void
TurbulenceModel<dim, n_components, Number>::get_eddy_viscosity(VectorType & dst) const
{
  dst.equ(1.0, eddy_viscosity);
}

template<int dim, int n_components, typename Number>
typename TurbulenceModel<dim, n_components, Number>::VectorType const &
TurbulenceModel<dim, n_components, Number>::get_eddy_viscosity_ref() const
{
  return eddy_viscosity;
}

template<int dim, int n_components, typename Number>
void
TurbulenceModel<dim, n_components, Number>::get_turbulent_kinetic_energy(VectorType & dst,
                                                                         VectorType const & solution) const
{
  matrix_free->cell_loop(&TurbulenceModel<dim, n_components, Number>::cell_loop_extract_tke,
                         this,
                         dst,
                         solution);
}

template<int dim, int n_components, typename Number>
void
TurbulenceModel<dim, n_components, Number>::cell_loop_extract_tke(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType & dst,
  VectorType const & src,
  Range const &      cell_range) const
{
  if constexpr (n_components < 2)
  {
    AssertThrow(false, dealii::ExcMessage("Turbulence Models with 1 component not yet implemented."));
  }

  IntegratorCell integrator(matrix_free, dof_index, quad_index);
  IntegratorCellScalar integrator_tke(matrix_free, dof_index_viscosity, quad_index);

  if(turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon)
  {
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src);

      integrator_tke.reinit(cell);
      for(unsigned int dof = 0; dof < integrator_tke.dofs_per_cell; ++dof)
      {
        auto solution_values = integrator.get_dof_value(dof);
        scalar tke = solution_values[0];

        if(turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          tke = std::exp(tke);
        }

        integrator_tke.submit_dof_value(tke, dof);
      }
      integrator_tke.set_dof_values(dst);
    }
  }
}

template class TurbulenceModel<2, 1, float>;
template class TurbulenceModel<2, 1, double>;
template class TurbulenceModel<3, 1, float>;
template class TurbulenceModel<3, 1, double>;

template class TurbulenceModel<2, 2, float>;
template class TurbulenceModel<2, 2, double>;
template class TurbulenceModel<3, 2, float>;
template class TurbulenceModel<3, 2, double>;
} // namespace RANS
} // namespace ExaDG
