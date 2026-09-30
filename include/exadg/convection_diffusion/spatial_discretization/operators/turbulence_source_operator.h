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

#ifndef INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_TURBULENCE_SOURCE_OPERATOR
#define INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_TURBULENCE_SOURCE_OPERATOR

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>
#include "exadg/utilities/lazy_ptr.h"

#include <exadg/convection_diffusion/spatial_discretization/turbulence_model.h>
#include <exadg/convection_diffusion/user_interface/enum_types.h>

namespace ExaDG
{
namespace ConvDiff
{
namespace Operators
{
template<int dim>
struct TurbulenceSourceKernelData
{
  TurbulenceSourceKernelData() : rans_model(false)
  {
  }
  std::shared_ptr<dealii::Function<dim>> f;

  bool                        rans_model;
  unsigned int                dof_index_eddy_viscosity;
  unsigned int                dof_index_velocity;
  unsigned int                dof_index;
  double                      diffusivity;
  double                      time_step_size;
  double                      time_step_number;
  double                      current_time;
  double                      end_time;
  bool                        dissipation_implicit;
  bool                        production_implicit;
  bool                        square_gradient_implicit;
  TreatmentOfVariableViscosity treatment_of_variable_viscosity = TreatmentOfVariableViscosity::Explicit;
  bool                        k_log;
  bool                        omega_log;
  
  TurbulenceModelData         turbulence_model_data;
};

template<int dim, typename Number, int n_components = 1>
class TurbulenceSourceKernel
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
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free_in,
         TurbulenceSourceKernelData<dim> const & data_in,
         unsigned int const                      quad_index)
  {
    data = data_in;

    integrator_velocity =
      std::make_shared<CellIntegratorVelocity>(matrix_free_in, data.dof_index_velocity, quad_index);
    integrator_solution =
      std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index, quad_index);
    integrator_eddy_viscosity =
      std::make_shared<CellIntegratorScalar>(matrix_free_in, data.dof_index_eddy_viscosity, quad_index);
  }

  void
  reinit_cell(unsigned int const cell) const
  {
    integrator_velocity->reinit(cell);
    integrator_velocity->gather_evaluate(*velocity, dealii::EvaluationFlags::gradients);

    integrator_solution->reinit(cell);
    integrator_solution->gather_evaluate(*solution, dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    integrator_eddy_viscosity->reinit(cell);
    integrator_eddy_viscosity->gather_evaluate(*eddy_viscosity, dealii::EvaluationFlags::values);
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
  set_time_step_size(double const dt,
                     double const time,
                     double const end_time,
                     double const time_step_size) const
  {
    data.time_step_size = dt;
    data.current_time = time;
    data.end_time = end_time;
    data.time_step_number = time_step_size;
  }

  value_type
  get_volume_flux(unsigned int const q) const
  {
    if constexpr (n_components == 2)
    {
      value_type production  = get_production_term(q);
      check_nan(production, q, "production");
      value_type dissipation = get_dissipation_term(q);
      check_nan(dissipation, q, "dissipation");

      value_type limited_production;
      for(unsigned int c = 0; c < n_components; ++c)
      {
        limited_production[c] = data.production_implicit ? production[c] : std::min(production[c], dealii::make_vectorized_array<Number>(10.0) * dissipation[c]);
      }

      value_type volume_flux = limited_production;
      volume_flux -= dissipation;
      value_type square_gradient = get_square_gradient_term(q);
      check_nan(square_gradient, q, "square_gradient");
      volume_flux += square_gradient;

      return volume_flux;
    }
    else
  {
      AssertThrow(false, dealii::ExcMessage("get_volume_flux() is only valid for 2-equation turbulence models."));
      return value_type();
    }
  }

  void
  check_nan(value_type const & value, unsigned int const q, char const * name) const
  {
    for(unsigned int c = 0; c < n_components; ++c)
      for(unsigned int lane = 0; lane < dealii::VectorizedArray<Number>::size(); ++lane)
        AssertThrow(dealii::numbers::is_finite(value[c][lane]),
                    dealii::ExcMessage(std::string("Non-finite value in ") + name +
                                       " at q=" + std::to_string(q) +
                                       ", component=" + std::to_string(c) +
                                       ", lane=" + std::to_string(lane)));
  }

  value_type
  get_square_gradient_term(unsigned int const q) const
  {
    if constexpr (n_components >= 2)
    {
      if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon)
      {
        AssertThrow(!data.square_gradient_implicit,
                    dealii::ExcMessage("Implicit production term for StandardKEpsilon not yet implemented."));
      }
      value_type square_gradient_term = data.square_gradient_implicit ? get_square_gradient_implicit(q) : get_square_gradient_explicit(q);

      return square_gradient_term;
    }
    else {
      value_type zero_flux = dealii::make_vectorized_array<Number>(0.0);
      AssertThrow(false, dealii::ExcMessage("Square gradient term for turbulence models with 1 component not yet implemented."));
      return zero_flux;
    }
  }

  struct SquareGradientCoefficients
  {
    scalar Sk_k_coefficient = dealii::make_vectorized_array<Number>(0.0);
    scalar Sk_w_coefficient = dealii::make_vectorized_array<Number>(0.0);
    dealii::Tensor<1, dim, scalar> grad_k_coefficient;

    scalar Sw_k_coefficient = dealii::make_vectorized_array<Number>(0.0);
    scalar Sw_w_coefficient = dealii::make_vectorized_array<Number>(0.0);
    dealii::Tensor<1, dim, scalar> grad_w_coefficient;
  };

  SquareGradientCoefficients
  get_square_gradient_coefficients(unsigned int const q) const
  {
    value_type solution = integrator_solution->get_value(q);
    gradient_type solution_gradient = integrator_solution->get_gradient(q);
    scalar nu_t = integrator_eddy_viscosity->get_value(q);

    SquareGradientCoefficients coefficients;

    if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
    {
      scalar tke = data.k_log ? std::exp(solution[0]) : solution[0];
      scalar omega = data.omega_log ? std::exp(solution[1]) : solution[1];

      scalar safe_tke = std::max(tke, dealii::make_vectorized_array<Number>(1.0e-6));
      scalar safe_omega = std::max(omega, dealii::make_vectorized_array<Number>(1.0e-6));

      scalar sigma_star = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[4]);
      scalar sigma = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[3]);
      value_type nu_e;
      nu_e[0] = data.diffusivity + nu_t * sigma_star;
      nu_e[1] = data.diffusivity + nu_t * sigma;

      scalar G_k = scalar_product(solution_gradient[0], solution_gradient[0]);
      scalar G_w = scalar_product(solution_gradient[1], solution_gradient[1]);

      if(data.k_log && data.omega_log)              // Case 1
      {
        coefficients.Sk_k_coefficient   =  sigma_star * nu_t * G_k;
        coefficients.Sk_w_coefficient   = -sigma_star * nu_t * G_k;
        coefficients.grad_k_coefficient =  dealii::make_vectorized_array<Number>(2.0) * nu_e[0] * solution_gradient[0];

        coefficients.Sw_k_coefficient   =  sigma * nu_t * G_w;
        coefficients.Sw_w_coefficient   = -sigma * nu_t * G_w;
        coefficients.grad_w_coefficient =  dealii::make_vectorized_array<Number>(2.0) * nu_e[1] * solution_gradient[1];
      }
      else if(data.k_log && !data.omega_log)        // Case 2: only the K row exists
      {
        coefficients.Sk_k_coefficient   =  sigma_star * nu_t * G_k;
        coefficients.Sk_w_coefficient   = -sigma_star * nu_t * G_k / safe_omega;
        coefficients.grad_k_coefficient =  dealii::make_vectorized_array<Number>(2.0) * nu_e[0] * solution_gradient[0];
      }
      else if(!data.k_log && data.omega_log)        // Case 3: only the W row exists
      {
        coefficients.Sw_k_coefficient   =  sigma * nu_t * G_w / safe_tke;
        coefficients.Sw_w_coefficient   = -sigma * nu_t * G_w;
        coefficients.grad_w_coefficient =  dealii::make_vectorized_array<Number>(2.0) * nu_e[1] * solution_gradient[1];
      }
    }
    else{
      AssertThrow(false, dealii::ExcMessage("Square gradient coefficients for turbulence models other than StandardKOmega1988 not yet implemented."));
    }

    return coefficients;
  }

  value_type
  get_square_gradient_implicit(unsigned int const q) const
  {
    value_type solution = integrator_solution->get_value(q);

    gradient_type solution_gradient = integrator_solution->get_gradient(q);

    scalar nu_t = integrator_eddy_viscosity->get_value(q);

    value_type square_gradient_term;
    for(unsigned int i = 0; i < n_components; ++i)
    {
      square_gradient_term[i] = dealii::make_vectorized_array<Number>(0.0);
    }

    if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
    {
      scalar one = dealii::make_vectorized_array<Number>(1.0);

      scalar sigma_star = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[4]);
      scalar sigma = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[3]);
      value_type nu_e;
      nu_e[0] = data.diffusivity + nu_t * sigma_star;
      nu_e[1] = data.diffusivity + nu_t * sigma;

      scalar G_k = scalar_product(solution_gradient[0], solution_gradient[0]);
      scalar G_w = scalar_product(solution_gradient[1], solution_gradient[1]);

      if(data.k_log && data.omega_log)                    // Case 1
      {
        scalar log_ratio = solution[1] - solution[0];     // W^n - K^n

        square_gradient_term[0] = sigma_star * nu_t * G_k * log_ratio - nu_e[0] * G_k;
        square_gradient_term[1] = sigma      * nu_t * G_w * log_ratio - nu_e[1] * G_w;
      }
      else if(data.k_log && !data.omega_log)              // Case 2: only the K row exists
      {
        square_gradient_term[0] = sigma_star * nu_t * G_k * (one - solution[0]) - nu_e[0] * G_k;
      }
      else if(!data.k_log && data.omega_log)              // Case 3: only the W row exists
      {
        square_gradient_term[1] = sigma * nu_t * G_w * (solution[1] - one) - nu_e[1] * G_w;
      }
    }
    else{
      AssertThrow(false, dealii::ExcMessage("Implicit production term for turbulence models other than StandardKOmega1988 not yet implemented."));
    }

    return square_gradient_term;
  }


  value_type
  get_square_gradient_explicit(unsigned int const q) const
  {
    gradient_type solution_gradient = integrator_solution->get_gradient(q);

    scalar nu_t = integrator_eddy_viscosity->get_value(q);

    value_type square_gradient_term;

    if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon)
    {
      scalar sigma_k = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[0]);
      scalar sigma_E = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[4]);

      if(data.turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::LogarithmicTransportVariable)
      {
        square_gradient_term[0] = (data.diffusivity + nu_t /sigma_k) * scalar_product(solution_gradient[0], solution_gradient[0]);
      }
      else if(data.turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::Clipper)
      {
        square_gradient_term[0] = dealii::make_vectorized_array<Number>(0.0);
      }
      else
    {
        AssertThrow(false, dealii::ExcMessage("Wrong argument for PositivityPreservingLimiter"));
      }

      if(data.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::LogarithmicTransportVariable)
      {
        square_gradient_term[1] = (data.diffusivity + nu_t /sigma_E) * scalar_product(solution_gradient[1], solution_gradient[1]);
      }
      else if(data.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::Clipper)
      {
        square_gradient_term[1] = dealii::make_vectorized_array<Number>(0.0);
      }
      else
    {
        AssertThrow(false, dealii::ExcMessage("Wrong argument for PositivityPreservingLimiter"));
      }
    }
    else if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
    {
      scalar sigma_star = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[4]);
      scalar sigma = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[3]);

      if(data.turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::LogarithmicTransportVariable)
      {
        square_gradient_term[0] = (data.diffusivity + nu_t * sigma_star) * scalar_product(solution_gradient[0], solution_gradient[0]);
      }
      else if(data.turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::Clipper)
      {
        square_gradient_term[0] = dealii::make_vectorized_array<Number>(0.0);
      }
      else
    {
        AssertThrow(false, dealii::ExcMessage("Wrong argument for PositivityPreservingLimiter"));
      }

      if(data.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::LogarithmicTransportVariable)
      {
        square_gradient_term[1] = (data.diffusivity + nu_t * sigma) * scalar_product(solution_gradient[1], solution_gradient[1]);
      }
      else if(data.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::Clipper)
      {
        square_gradient_term[1] = dealii::make_vectorized_array<Number>(0.0);
      }
      else
    {
        AssertThrow(false, dealii::ExcMessage("Wrong argument for PositivityPreservingLimiter"));
      }
    }

    // Number C_max = 2.0e-2;
    //
    // value_type safe_limit;
    // safe_limit[0] = dealii::make_vectorized_array<Number>(C_max / data.time_step_size);
    // safe_limit[1] = dealii::make_vectorized_array<Number>(C_max / data.time_step_size);
    //
    // square_gradient_term[0] = std::min(square_gradient_term[0], safe_limit[0]);
    // square_gradient_term[1] = std::min(square_gradient_term[1], safe_limit[1]);

    // bool time_ramping = true;

    // if(time_ramping)
    // {
    //   Number ramp_factor = -5.0;
    //   Number final_time_step = 100.0;
    //   Number ramp_value = 1.0 - std::exp(ramp_factor * data.time_step_number / final_time_step);
    //   scalar ramp_vec = dealii::make_vectorized_array<Number>(ramp_value);
    //
    //   square_gradient_term[0] *= ramp_vec;
    //   square_gradient_term[1] *= ramp_vec;
    // }

    // if(data.current_time < (0.1 * data.end_time))
    // {
    //   square_gradient_term[0] *=  dealii::make_vectorized_array<Number>(0.01);
    //   square_gradient_term[1] *=  dealii::make_vectorized_array<Number>(0.01);
    // }
    // else if(data.current_time < 0.2 * data.end_time)
    // {
    //   square_gradient_term[0] *=  dealii::make_vectorized_array<Number>(0.1);
    //   square_gradient_term[1] *=  dealii::make_vectorized_array<Number>(0.1);
    // }

    return square_gradient_term;
  }

  value_type
  get_production_term(unsigned int const q) const
  {
    if constexpr (n_components >= 2)
    {
      value_type production_term;

      for(unsigned int c = 0; c < n_components; ++c)
      {
        production_term[c] = dealii::make_vectorized_array<Number>(0.0);
      }
      if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon)
      {
        AssertThrow(!data.production_implicit,
                    dealii::ExcMessage("Implicit production term for StandardKEpsilon not yet implemented."));
        production_term = get_production_explicit(q);
      }
      else if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
      {
        production_term = data.production_implicit ? get_production_implicit(q) : get_production_explicit(q);
      }

      return production_term;
    }
    else {
      value_type zero_flux = dealii::make_vectorized_array<Number>(0.0);
      AssertThrow(false, dealii::ExcMessage("TurbulenceSource term for turbulence models with 1 component not yet implemented."));
      return zero_flux;
    }
  }

  value_type
  get_production_explicit(unsigned int const q) const
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

      if(data.turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::LogarithmicTransportVariable)
      {
        production_term[0] /= std::exp(solution[0]);
      }

      if(data.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::LogarithmicTransportVariable)
      {
        production_term[0] /= std::exp(solution[0]);
        production_term[1] *= C_e1 / std::exp(solution[0]);
      }
      else if(data.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::Clipper)
      {
        production_term[1] *= C_e1 * solution[1] / std::max(solution[0], dealii::make_vectorized_array<Number>(1.e-6));
      }
    }
    else if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
    {
      scalar alpha = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[0]);

      scalar tke_factor = data.k_log ? std::exp(-solution[0]) : dealii::make_vectorized_array<Number>(1.0) / std::max(solution[0], dealii::make_vectorized_array<Number>(1.0e-6));

      production_term[0] = viscosity * gradient_product;
      production_term[1] = alpha * viscosity * gradient_product * tke_factor;

      if(data.k_log)
      {
        production_term[0] *= tke_factor;
      }

      if(!data.omega_log)
      {
        production_term[1] *= solution[1];
      }
    }

    return production_term;
  }

  value_type
  get_production_implicit(unsigned int const q) const
  {
    tensor velocity_gradient = integrator_velocity->get_gradient(q);

    tensor symmetric_velocity_gradient = (velocity_gradient + transpose(velocity_gradient));

    scalar gradient_product = scalar_product(symmetric_velocity_gradient, velocity_gradient);

    value_type solution = integrator_solution->get_value(q);

    value_type production_term;

    if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
    {
      scalar alpha = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[0]);

      scalar one = dealii::make_vectorized_array<Number>(1.0);

      if(data.k_log && data.omega_log)
      {
        scalar omega_previous = solution[1];
        scalar remainder = gradient_product * std::exp(-omega_previous) * (one + omega_previous);
        production_term[0] = remainder;
        production_term[1] = alpha * remainder;
      }
      else if(!data.k_log && !data.omega_log)
      {
        scalar k_previous = solution[0];
        scalar omega_previous = std::max(solution[1], dealii::make_vectorized_array<Number>(1.0e-6));
        production_term[0] = gradient_product * k_previous / omega_previous;
        production_term[1] = alpha * gradient_product;
      }
      else if(data.k_log && !data.omega_log)
      {
        scalar omega_previous = solution[1];
        production_term[0] = dealii::make_vectorized_array<Number>(2.0) * gradient_product / omega_previous;
        production_term[1] = alpha * gradient_product;
      }
      else {
        scalar k_previous = solution[0];
        scalar omega_previous = solution[1];
        production_term[0] = k_previous * omega_previous * gradient_product * std::exp(-omega_previous);
        production_term[1] = alpha * gradient_product * std::exp(-omega_previous) * (one + omega_previous);
      }
    }
    else{
      AssertThrow(false, dealii::ExcMessage("Implicit production term for turbulence models other than StandardKOmega1988 not yet implemented."));
    }

    return production_term;
  }

  dealii::Tensor<2, n_components, scalar>
  get_production_implicit_coefficient(unsigned int const q) const
  {
    dealii::Tensor<2, n_components, scalar> implicit_coefficients;
    for(unsigned int i = 0; i < n_components; ++i)
      for(unsigned int j = 0; j < n_components; ++j)
        implicit_coefficients[i][j] = dealii::make_vectorized_array<Number>(0.0);

    if constexpr (n_components == 2)
    {
      if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
      {
        tensor velocity_gradient = integrator_velocity->get_gradient(q);
        tensor symmetric_velocity_gradient = (velocity_gradient + transpose(velocity_gradient));
        scalar gradient_product = scalar_product(symmetric_velocity_gradient, velocity_gradient);

        value_type solution = integrator_solution->get_value(q);
        scalar alpha = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[0]);

        if(data.k_log && data.omega_log)
        {
          scalar omega_previous = solution[1];
          scalar e = std::exp(-omega_previous);

          implicit_coefficients[0][1] = -gradient_product * e;
          implicit_coefficients[1][1] = -alpha * gradient_product * e;
        }
        else if(!data.k_log && !data.omega_log)
        {
          scalar k_previous     = solution[0];
          scalar omega_previous = std::max(solution[1], dealii::make_vectorized_array<Number>(1.0e-6));

          implicit_coefficients[0][0] =  gradient_product / omega_previous;
          implicit_coefficients[0][1] = -gradient_product * k_previous / (omega_previous * omega_previous);
        }
        else if(data.k_log && !data.omega_log)
        {
          scalar omega_previous = solution[1];

          implicit_coefficients[0][1] = -gradient_product / (omega_previous * omega_previous);
        }
        else
      {
          scalar k_previous     = solution[0];
          scalar omega_previous = solution[1];
          scalar e = std::exp(-omega_previous);

          implicit_coefficients[0][0] =  gradient_product * e;
          implicit_coefficients[0][1] = -gradient_product * k_previous * e;
          implicit_coefficients[1][1] = -alpha * gradient_product * e;
          // implicit_coefficients[1][0] stays zero: P_omega never depended on k in this case (it cancelled).
        }
      }
      else
    {
        AssertThrow(false, dealii::ExcMessage(
          "Implicit production coefficient for turbulence models other than StandardKOmega1988 not yet implemented."));
      }
    }
    else
  {
      AssertThrow(false, dealii::ExcMessage(
        "get_production_implicit_coefficient() is only valid for 2-equation turbulence models."));
    }

    return implicit_coefficients;
  }

  value_type
  get_dissipation_term(unsigned int const q) const
  {
    if constexpr (n_components >= 2)
    {
      value_type dissipation_term;

      for(unsigned int c = 0; c < n_components; ++c)
      {
        dissipation_term[c] = dealii::make_vectorized_array<Number>(0.0);
      }
      if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon)
      {
        AssertThrow(!data.dissipation_implicit,
                    dealii::ExcMessage("Implicit production term for StandardKEpsilon not yet implemented."));
        dissipation_term = get_dissipation_explicit(q);
      }
      else if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
      {
        dissipation_term = data.dissipation_implicit ? get_dissipation_implicit(q) : get_dissipation_explicit(q);
      }

      return dissipation_term;
    }
    else {
      value_type zero_flux = dealii::make_vectorized_array<Number>(0.0);
      AssertThrow(false, dealii::ExcMessage("TurbulenceSource term for turbulence models with 1 component not yet implemented."));
      return zero_flux;
    }
  }

  struct DissipationCoefficients
  {
    scalar Dk_k_coefficient = dealii::make_vectorized_array<Number>(0.0);
    scalar Dk_w_coefficient = dealii::make_vectorized_array<Number>(0.0);

    scalar Dw_k_coefficient = dealii::make_vectorized_array<Number>(0.0);
    scalar Dw_w_coefficient = dealii::make_vectorized_array<Number>(0.0);
  };

  DissipationCoefficients
  get_dissipation_implicit_coefficients(unsigned int const q) const
  {
    DissipationCoefficients coefficients;
    if constexpr (n_components == 2)
    {
      if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
      {
        value_type solution = integrator_solution->get_value(q);

        scalar one = dealii::make_vectorized_array<Number>(1.0);

        scalar beta = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[1]);
        scalar beta_star = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[2]);

        if(data.k_log && data.omega_log)
        {
          coefficients.Dk_w_coefficient = beta_star * std::exp(solution[1]);
          coefficients.Dw_w_coefficient = beta * std::exp(solution[1]);
        }
        else if(!data.k_log && !data.omega_log)
        {
          coefficients.Dk_k_coefficient = beta_star * solution[1];
          coefficients.Dk_w_coefficient = beta_star * solution[0];
          coefficients.Dw_w_coefficient = dealii::make_vectorized_array<Number>(2.0) * beta * solution[1];
        }
        else if(data.k_log && !data.omega_log)
        {
          coefficients.Dk_w_coefficient = beta_star;

          coefficients.Dw_w_coefficient = dealii::make_vectorized_array<Number>(2.0) * beta * solution[1];
        }
        else if(!data.k_log && data.omega_log) 
        {
          coefficients.Dk_k_coefficient = beta_star * std::exp(solution[1]);
          coefficients.Dk_w_coefficient = beta_star * solution[0] * std::exp(solution[1]);
          coefficients.Dw_w_coefficient = beta * std::exp(solution[1]);
        }
        else {
          AssertThrow(false, dealii::ExcMessage("Transport variables are not specified properly"));
        }
      }
      else{
        AssertThrow(false, dealii::ExcMessage(
          "Implicit production coefficient for turbulence models other than StandardKOmega1988 not yet implemented."));
      }
    }
    else {
      AssertThrow(false, dealii::ExcMessage(
        "get_production_implicit_coefficient() is only valid for 2-equation turbulence models."));
    }

    return coefficients;
  }

  value_type
  get_dissipation_implicit(unsigned int const q) const
  {
    value_type solution = integrator_solution->get_value(q);

    value_type dissipation_term;

    if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
    {
      scalar one = dealii::make_vectorized_array<Number>(1.0);

      scalar beta = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[1]);
      scalar beta_star = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[2]);

      if(data.k_log && data.omega_log)
      {
        scalar compute_value = std::exp(solution[1]) * (one - solution[1]);
        dissipation_term[0] = beta_star * compute_value;
        dissipation_term[1] = beta * compute_value;
      }
      else if(!data.k_log && !data.omega_log)
      {
        dissipation_term[0] = -beta_star * solution[0] * solution[1];
        dissipation_term[1] = -beta * solution[1] * solution[1];
      }
      else if(data.k_log && !data.omega_log)
      {
        dissipation_term[0] = dealii::make_vectorized_array<Number>(0.0);
        dissipation_term[1] = -beta * solution[1] * solution[1];
      }
      else if(!data.k_log && data.omega_log)
      {
        dissipation_term[0] = -beta_star * solution[0] * std::exp(solution[1]) * solution[1];
        dissipation_term[1] = beta * std::exp(solution[1]) * (one - solution[1]);
      }
      else {
        AssertThrow(false, dealii::ExcMessage("Transport variables are not set properly"));
      }
    }
    else{
      AssertThrow(false, dealii::ExcMessage("Implicit production term for turbulence models other than StandardKOmega1988 not yet implemented."));
    }

    return dissipation_term;
  }

  value_type
  get_dissipation_explicit(unsigned int const q) const
  {
    if constexpr (n_components >= 2)
    {
      value_type solution = integrator_solution->get_value(q);

      value_type dissipation_term;

      if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon)
      {
        scalar C_e2 = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[2]);
        scalar C_mu = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[3]);

        if(data.turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          dissipation_term[0] = solution[1];
        }
        else if(data.turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::Clipper)
        {
          dissipation_term[0] = solution[1];
        }

        if(data.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          dissipation_term[1] = C_e2 * std::exp(solution[1] - solution[0]);
        }
        else if(data.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::Clipper)
        {
          dissipation_term[1] = C_e2 * C_mu * solution[1] * solution[1] / std::max(solution[0], dealii::make_vectorized_array<Number>(1.e-6));
        }
      }
      else if(data.turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
      {
        scalar beta = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[1]);
        scalar beta_star = dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[2]);

        scalar omega = data.omega_log ? std::exp(solution[1]) : solution[1];

        if(data.turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          dissipation_term[0] = beta_star * omega;

        }
        else if(data.turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::Clipper)
        {
          dissipation_term[0] = beta_star * solution[0] * omega;
        }

        if(data.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          dissipation_term[1] = beta * std::exp(solution[1]);
        }
        else if(data.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::Clipper)
        {
          dissipation_term[1] = beta * solution[1] * solution[1];
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

  std::shared_ptr<TurbulenceModel<dim, n_components, Number>> turbulence_model_ptr;

private:
  mutable TurbulenceSourceKernelData<dim> data;

  mutable lazy_ptr<VectorType> velocity;

  std::shared_ptr<CellIntegratorVelocity> integrator_velocity;

  mutable lazy_ptr<VectorType> solution;

  mutable lazy_ptr<VectorType> eddy_viscosity;

  std::shared_ptr<IntegratorCell> integrator_solution;

  std::shared_ptr<CellIntegratorScalar> integrator_eddy_viscosity;
};

} // namespace Operators


template<int dim>
struct TurbulenceSourceOperatorData : public OperatorBaseData
{
  TurbulenceSourceOperatorData() : OperatorBaseData()
  {
  }

  Operators::TurbulenceSourceKernelData<dim> kernel_data;
};

template<int dim, typename Number, int n_components = 1>
class TurbulenceSourceOperator : public OperatorBase<dim, Number, n_components>
{
private:
  typedef OperatorBase<dim, Number, n_components> Base;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef TurbulenceSourceOperator<dim, Number, n_components> This;

  typedef CellIntegrator<dim, n_components, Number> IntegratorCell;

  typedef std::pair<unsigned int, unsigned int> Range;

  using value_type = typename IntegratorCell::value_type;

  typedef dealii::VectorizedArray<Number>   scalar;

public:
  /*
   * Constructor.
   */
  TurbulenceSourceOperator();

  /*
   * Initialization.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             TurbulenceSourceOperatorData<dim> const &            data,
             std::shared_ptr<Operators::TurbulenceSourceKernel<dim, Number, n_components>> kernel_in);

  /*
   * Evaluate operator and overwrite dst-vector.
   */
  void
  evaluate_source(VectorType & dst, double const evaluation_time) const;

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add_source(VectorType & dst, double const evaluation_time) const;

  void
  set_velocity_ptr(VectorType const & velocity_in) const;

  void
  set_solution_ptr(VectorType const & src) const;

  void
  set_eddy_viscosity_ptr(VectorType const & eddy_viscosity_in) const;

  void
  set_time_step_size(double const dt,
                     double const time,
                     double const end_time,
                     double const time_step_number) const;
private:
  /*
   * The right-hand side operator involves only cell integrals so we only need a function looping
   * over all cells and computing the cell integrals.
   */
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            VectorType const &                      src,
            Range const &                           cell_range) const;

  /*
   * OperatorBase overrides for the implicit dissipation path (apply()/vmult(), used only when
   * has_implicit_dissipation() == true -- the caller is responsible for only summing this
   * operator into the LHS in that case).
   */
  void
  reinit_cell_derived(IntegratorCell & integrator, unsigned int const cell) const final;

  void
  do_cell_integral(IntegratorCell & integrator) const final;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  TurbulenceSourceOperatorData<dim> data;

  mutable double time;

  // Operators::TurbulenceSourceKernel<dim, Number, n_components> kernel;
  std::shared_ptr<Operators::TurbulenceSourceKernel<dim, Number, n_components>> kernel;
};

} // namespace ConvDiff

} // namespace ExaDG

#endif
