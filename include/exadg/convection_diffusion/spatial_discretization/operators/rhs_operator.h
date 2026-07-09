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

#ifndef INCLUDE_OPERATORS_RHS_OPERATOR
#define INCLUDE_OPERATORS_RHS_OPERATOR

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>
#include "exadg/utilities/lazy_ptr.h"

#include <exadg/convection_diffusion/spatial_discretization/turbulence_model.h>

namespace ExaDG
{
namespace ConvDiff
{
namespace Operators
{
template<int dim>
struct RHSKernelData
{
  RHSKernelData() : rans_model(false), positivity_preserving_limiter(PositivityPreservingLimiter::Undefined)
  {
  }
  std::shared_ptr<dealii::Function<dim>> f;

  bool                        rans_model;
  unsigned int                dof_index_eddy_viscosity;
  unsigned int                dof_index_velocity;
  unsigned int                dof_index;
  double                      diffusivity;
  double                      time_step_size;
  TurbulenceModelData         turbulence_model_data;
  PositivityPreservingLimiter positivity_preserving_limiter;
};

template<int dim, typename Number, int n_components = 1>
class RHSKernel
{
private:
  typedef CellIntegrator<dim, n_components, Number> IntegratorCell;
  typedef CellIntegrator<dim, dim, Number>                   CellIntegratorVelocity;
  typedef CellIntegrator<dim, 1, Number>                     CellIntegratorScalar;

  using value_type = typename IntegratorCell::value_type;
  using gradient_type = typename IntegratorCell::gradient_type;

  typedef dealii::VectorizedArray<Number>   scalar;

  typedef dealii::Tensor<2, dim, scalar> tensor;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  void
  reinit(RHSKernelData<dim> const & data_in) const
  {
    data = data_in;
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free_in,
         RHSKernelData<dim> const &              data_in,
         unsigned int const                      quad_index)
  {
    data = data_in;
    if(data.rans_model)
    {
      integrator_velocity =
        std::make_shared<CellIntegratorVelocity>(matrix_free_in, data.dof_index_velocity, quad_index);
      integrator_solution =
        std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index, quad_index);
      integrator_eddy_viscosity =
        std::make_shared<CellIntegratorScalar>(matrix_free_in, data.dof_index_eddy_viscosity, quad_index);

    }
  }

  void
  reinit_cell(unsigned int const cell) const
  {
    if(data.rans_model)
    {
      integrator_velocity->reinit(cell);
      integrator_velocity->gather_evaluate(*velocity, dealii::EvaluationFlags::gradients);

      integrator_solution->reinit(cell);
      integrator_solution->gather_evaluate(*solution, dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      integrator_eddy_viscosity->reinit(cell);
      integrator_eddy_viscosity->gather_evaluate(*eddy_viscosity, dealii::EvaluationFlags::values);
    }
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values |
                  dealii::update_quadrature_points |
                  dealii::update_values |
                  dealii::update_gradients;

    // no face integrals

    return flags;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    value_type
    get_volume_flux(IntegratorCell const & integrator,
                    unsigned int const     q,
                    Number const &         time) const
  {
    value_type volume_flux;
    dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

    if constexpr(n_components == 1)
    {
      volume_flux = FunctionEvaluator<0, dim, Number>::value(*(data.f), q_points, time);
    }
    else if constexpr(n_components == dim)
    {
      volume_flux = FunctionEvaluator<1, dim, Number>::value(*(data.f), q_points, time);
    }
    else
    {
      volume_flux = MultiComponentFunctionEvaluator<n_components, dim,  Number>::value(*(data.f), q_points, time);
    }

    if(data.rans_model)
    {
      volume_flux += get_production_term(q);
      volume_flux -= get_dissipation_term(q);
      if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::LogarithmicTransportVariable)
      {
        volume_flux += get_square_gradient_term(q);
      }
    }

    return volume_flux;
  }

  value_type
  get_square_gradient_term(unsigned int const q) const
  {
    if constexpr (n_components >= 2)
    {
      scalar viscosity = integrator_eddy_viscosity->get_value(q);

      gradient_type solution_gradient = integrator_solution->get_gradient(q);

      value_type square_gradient_term;

      if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon)
      {
        scalar sigma_k = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[0]);
        scalar sigma_E = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[4]);

        if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          square_gradient_term[0] = (data.diffusivity + viscosity /sigma_k) * scalar_product(solution_gradient[0], solution_gradient[0]);
          square_gradient_term[1] = (data.diffusivity + viscosity /sigma_E) * scalar_product(solution_gradient[1], solution_gradient[1]);
        }
      }
      else if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
      {
        scalar sigma_star = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[4]);
        scalar sigma = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[3]);

        if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          square_gradient_term[0] = (data.diffusivity + viscosity /sigma_star) * scalar_product(solution_gradient[0], solution_gradient[0]);
          square_gradient_term[1] = (data.diffusivity + viscosity /sigma) * scalar_product(solution_gradient[1], solution_gradient[1]);
        }
      }

      return square_gradient_term;
    }
    else {
      value_type zero_flux = dealii::make_vectorized_array<Number>(0.0);
      AssertThrow(false, dealii::ExcMessage("Square gradient term for turbulence models with 1 component not yet implemented."));
      return zero_flux;
    }
  }

  value_type
  get_production_term(unsigned int const q) const
  {
    if constexpr (n_components >= 2)
    {
      scalar viscosity = integrator_eddy_viscosity->get_value(q);

      tensor velocity_gradient = integrator_velocity->get_gradient(q);

      tensor symmetric_velocity_gradient = (velocity_gradient + transpose(velocity_gradient));

      scalar gradient_product = scalar_product(symmetric_velocity_gradient, velocity_gradient);

      value_type solution = integrator_solution->get_value(q);

      value_type production_term;

      for(unsigned int c = 0; c < n_components; ++c)
      {
        production_term[c] = viscosity * gradient_product;
      }
      if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon)
      {
        scalar C_e1 = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[1]);

        if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          production_term[0] /= std::exp(solution[0]);
          production_term[1] *= C_e1 / std::exp(solution[0]);
        }
        else if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::Clipper)
        {
          production_term[1] *= C_e1 * solution[1] / std::max(solution[0], dealii::make_vectorized_array<Number>(1.e-6));
        }
      }
      else if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
      {
        scalar alpha = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[0]);

        if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          production_term[0] /= std::exp(solution[0]);
          production_term[1] *= alpha / std::exp(solution[0]);
        }
        else if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::Clipper)
        {
          production_term[1] *= alpha * solution[1] / std::max(solution[0], dealii::make_vectorized_array<Number>(1.e-6));
        }
      }

      return production_term;
    }
    else {
      value_type zero_flux = dealii::make_vectorized_array<Number>(0.0);
      AssertThrow(false, dealii::ExcMessage("Production term for turbulence models with 1 component not yet implemented."));
      return zero_flux;
    }
  }

  value_type
  get_dissipation_term(unsigned int const q) const
  {
    if constexpr (n_components >= 2)
    {
      value_type solution = integrator_solution->get_value(q);

      value_type dissipation_term;

      if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon)
      {
        scalar C_e2 = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[2]);
        scalar C_mu = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[3]);

        if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          dissipation_term[0] = solution[1];
          dissipation_term[1] = C_e2 * std::exp(solution[1] - solution[0]);
        }
        else if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::Clipper)
        {
          dissipation_term[0] = solution[1];
          dissipation_term[1] = C_e2 * C_mu * solution[1] * solution[1] / std::max(solution[0], dealii::make_vectorized_array<Number>(1.e-6));
        }
      }
      else if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
      {
        scalar beta = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[1]);
        scalar beta_star = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[2]);

        if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          dissipation_term[0] = beta * std::exp(solution[1]);
          dissipation_term[1] = beta_star * std::exp(solution[1]);
        }
        else if(data.turbulence_model_data.positivity_preserving_limiter == PositivityPreservingLimiter::Clipper)
        {
          dissipation_term[0] = beta * solution[0] * solution[1];
          dissipation_term[1] = beta_star * solution[1] * solution[1];
        }
      }
      return dissipation_term;
    }
    else{
      value_type zero_flux = dealii::make_vectorized_array<Number>(0.0);
      AssertThrow(false, dealii::ExcMessage("Dissipation term for turbulence models other than StandardKEpsilon not yet implemented."));
      return zero_flux;
    }
  }

  /*
   * Function for taking value of velocity from NS solver
   */
  void
  set_velocity_ptr(VectorType const & velocity_in)
  {
    velocity.own() = velocity_in;
    velocity->update_ghost_values();
  }

  /*
   * Function for taking value of solution from pde_operator
   */
  void
  set_solution_ptr(VectorType const & sol)
  {
    // solution = &sol;
    solution.own() = sol;
    solution->update_ghost_values();
  }

  void
  set_eddy_viscosity_ptr(VectorType const & eddy_viscosity_in)
  {
    eddy_viscosity.own() = eddy_viscosity_in;
    eddy_viscosity->update_ghost_values();
  }

  void
  set_time_step_size(double const dt) const
  {
    data.time_step_size = dt;
  }

  std::shared_ptr<TurbulenceModel<dim, n_components, Number>> turbulence_model_ptr;

private:
  mutable RHSKernelData<dim> data;

  mutable lazy_ptr<VectorType> velocity;

  std::shared_ptr<CellIntegratorVelocity> integrator_velocity;

  mutable lazy_ptr<VectorType> solution;

  mutable lazy_ptr<VectorType> eddy_viscosity;

  std::shared_ptr<IntegratorCell> integrator_solution;

  std::shared_ptr<CellIntegratorScalar> integrator_eddy_viscosity;
};

} // namespace Operators


template<int dim>
struct RHSOperatorData
{
  RHSOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  Operators::RHSKernelData<dim> kernel_data;
};

template<int dim, typename Number, int n_components = 1>
class RHSOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef RHSOperator<dim, Number, n_components> This;

  typedef CellIntegrator<dim, n_components, Number> IntegratorCell;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  /*
   * Constructor.
   */
  RHSOperator();

  /*
   * Initialization.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             RHSOperatorData<dim> const &            data,
             std::shared_ptr<Operators::RHSKernel<dim, Number, n_components>> kernel_in);

  /*
   * Evaluate operator and overwrite dst-vector.
   */
  void
  evaluate(VectorType & dst, double const evaluation_time) const;

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add(VectorType & dst, double const evaluation_time) const;

  void
  set_velocity_ptr(VectorType const & velocity_in) const;

  void
  set_solution_ptr(VectorType const & src) const;

  void
  set_eddy_viscosity_ptr(VectorType const & eddy_viscosity_in) const;

  void
  set_time_step_size(double const dt) const;
private:
  void
  do_cell_integral(IntegratorCell & integrator) const;

  /*
   * The right-hand side operator involves only cell integrals so we only need a function looping
   * over all cells and computing the cell integrals.
   */
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  RHSOperatorData<dim> data;

  mutable double time;

  // Operators::RHSKernel<dim, Number, n_components> kernel;
  std::shared_ptr<Operators::RHSKernel<dim, Number, n_components>> kernel;
};

} // namespace ConvDiff

} // namespace ExaDG

#endif
