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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_EDDY_VISCOSITY_EVALUATOR_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_EDDY_VISCOSITY_EVALUATOR_H_

// deal.II
#include <deal.II/base/array_view.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

// C++
#include <memory>

// ExaDG
#include <exadg/convection_diffusion/spatial_discretization/rans_models/create_rans_model.h>
#include <exadg/convection_diffusion/user_interface/turbulence_model_data.h>

namespace ExaDG
{
namespace ConvDiff
{
/*
 * Pointwise eddy viscosity of the RANS model: nu_t(Q, grad u) and d nu_t / d solution at one
 * (quadrature) point, from the transported values Q (e.g. k and ln(omega)). The only definition
 * of nu_t: used by the RANS equations (TurbulenceModel, TurbulenceSourceKernel) and by the
 * momentum equation (IncNS::TurbulenceModel), so that both see the same nu_t.
 */
template<int dim, typename Number>
class EddyViscosityEvaluator
{
public:
  typedef dealii::VectorizedArray<Number> scalar;

  EddyViscosityEvaluator() : n_variables(0)
  {
  }

  void
  initialize(TurbulenceModelData const & turbulence_model_data_in)
  {
    turbulence_model_data = turbulence_model_data_in;

    // the RANS model; create_rans_model() also checks one positivity preserving limiter per
    // transport variable, and stops for k-epsilon (not implemented yet)
    model       = create_rans_model<dim, Number>(turbulence_model_data);
    n_variables = model->get_n_variables();

    point_input.reinit(n_variables);
    point_eddy_viscosity.reinit(n_variables);
    point_diffusion_viscosity.reinit(n_variables);
  }

  unsigned int
  get_n_variables() const
  {
    return n_variables;
  }

  // transported values Q -> physical variables (exp for logarithmic variables) and d solution / d Q
  void
  get_physical_variables(dealii::ArrayView<scalar const> const & transported_values,
                         dealii::ArrayView<scalar> const &       solution,
                         dealii::ArrayView<scalar> const &       solution_derivative) const
  {
    AssertDimension(transported_values.size(), n_variables);
    AssertDimension(solution.size(), n_variables);

    bool const compute_solution_derivative = solution_derivative.size() > 0;

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      if(turbulence_model_data.positivity_preserving_limiter[i] ==
        PositivityPreservingLimiter::LogarithmicTransportVariable)
      {
        // transported value = log(solution)
        solution[i] = std::exp(transported_values[i]);

        if(compute_solution_derivative)
          solution_derivative[i] = solution[i];
      }
      else
    {
        // transported value = solution
        solution[i] = transported_values[i];

        if(compute_solution_derivative)
          solution_derivative[i] = dealii::make_vectorized_array<Number>(1.0);
      }
    }
  }

  // nu_t and d nu_t / d solution (physical variables) at one point, from the RANS model
  void
  evaluate_eddy_viscosity(dealii::ArrayView<scalar const> const & solution,
                          dealii::Tensor<2, dim, scalar> const &  velocity_gradient,
                          scalar &                                viscosity,
                          dealii::ArrayView<scalar> const &       viscosity_derivative) const
  {
    AssertDimension(solution.size(), n_variables);

    for(unsigned int j = 0; j < n_variables; ++j)
      point_input.values[j] = solution[j];
    point_input.velocity_gradient   = velocity_gradient;
    point_input.compute_derivatives = (viscosity_derivative.size() > 0);

    model->evaluate_viscosities(point_input, point_eddy_viscosity, point_diffusion_viscosity);

    viscosity = point_eddy_viscosity.value;

    if(point_input.compute_derivatives)
    {
      AssertDimension(viscosity_derivative.size(), n_variables);
      for(unsigned int j = 0; j < n_variables; ++j)
        viscosity_derivative[j] = point_eddy_viscosity.derivative[j];
    }
  }

  // the RANS model (used directly by TurbulenceSourceKernel from step 5 on)
  RANSModelBase<dim, Number> const &
  get_model() const
  {
    AssertThrow(model != nullptr,
                dealii::ExcMessage("EddyViscosityEvaluator: call initialize() first."));
    return *model;
  }

private:
  TurbulenceModelData turbulence_model_data;

  unsigned int n_variables;

  // the RANS model (create_rans_model()): the only definition of nu_t
  std::shared_ptr<RANSModelBase<dim, Number> const> model;

  // work arrays of evaluate_eddy_viscosity(), one point at a time (matrix_free runs without task
  // parallelism, as the mutable work arrays of TurbulenceSourceKernel)
  mutable RANSPointInput<dim, Number> point_input;
  mutable RANSViscosity<Number>       point_eddy_viscosity;
  mutable RANSViscosity<Number>       point_diffusion_viscosity;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_EDDY_VISCOSITY_EVALUATOR_H_ */
