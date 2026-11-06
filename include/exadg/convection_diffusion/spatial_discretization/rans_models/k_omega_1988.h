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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_K_OMEGA_1988_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_K_OMEGA_1988_H_

// ExaDG
#include <exadg/convection_diffusion/spatial_discretization/rans_models/rans_model_base.h>

namespace ExaDG
{
namespace ConvDiff
{
/*
   * Wilcox (1988) k-omega model as in Bassi et al. (2014), eqs. (3)-(9). The realizability limiter
   * k_bar = max(k, 0), omega_r = max(omega, omega_r0) is used in nu_t, production and dissipation.
   *
   * Variables: 0 = k, 1 = omega (physical).
   * Coefficients: [0] alpha, [1] beta, [2] beta*, [3] sigma (omega), [4] sigma* (k).
   */
template<int dim, typename Number>
class KOmega1988Model : public RANSModelBase<dim, Number>
{
private:
  typedef RANSModelBase<dim, Number>     Base;
  typedef typename Base::scalar          scalar;
  typedef typename Base::vector          vector;
  typedef dealii::Tensor<2, dim, scalar> tensor;

  bool const k_log;
  bool const omega_log;

public:
  KOmega1988Model(TurbulenceModelData const & turbulence_model_data_in)
    : Base(turbulence_model_data_in),
    k_log(turbulence_model_data_in.positivity_preserving_limiter.at(0) ==
          PositivityPreservingLimiter::LogarithmicTransportVariable),
    omega_log(turbulence_model_data_in.positivity_preserving_limiter.at(1) ==
              PositivityPreservingLimiter::LogarithmicTransportVariable)
  {
  }

  unsigned int
  get_n_variables() const override
  {
    return 2;
  }

  // k: sigma*, omega: sigma (as in operator.cpp, eddy_viscosity_factor)
  double
  get_diffusion_factor(unsigned int const i) const override
  {
    AssertIndexRange(i, 2);
    return (i == 0) ? this->coefficients[4] : this->coefficients[3];
  }

  bool
  sources_in_transported_form() const override
  {
    return true;
  }

  /*
     * nu_t = k_bar / omega_r with the realizability limiter, and d nu_t / d (k, omega); copied from
     * EddyViscosityEvaluator::standard_k_omega_1988_model(). Wilcox 1988 (Bassi et al. 2014) uses the
     * same nu_t in the turbulent diffusion: nu_D = nu_t.
     */
  void
  evaluate_viscosities(RANSPointInput<dim, Number> const & input,
                       RANSViscosity<Number> &             eddy_viscosity,
                       RANSViscosity<Number> &             diffusion_viscosity) const override
  {
    scalar const zero = dealii::make_vectorized_array<Number>(0.0);
    scalar const one  = dealii::make_vectorized_array<Number>(1.0);

    scalar const tke   = input.values[0];
    scalar const omega = input.values[1];

    // realizability limiter (Bassi et al. 2014, eqs. (5)-(8)): k_bar = max(0, k),
    // omega_r = max(omega_r0, omega)
    scalar const omega_r0      = compute_realizability_alpha(input.velocity_gradient);
    scalar const tke_limited   = std::max(tke, zero);
    scalar const omega_r = std::max(omega, omega_r0);

    // nu_t = k_bar / omega_r (alpha* = 1)
    eddy_viscosity.value = tke_limited / omega_r;

    if(input.compute_derivatives)
    {
      // derivatives vanish where a limiter is active
      scalar const tke_not_limited =
        dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(tke, zero, one, zero);
      scalar const omega_not_limited =
        dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(omega,
                                                                             omega_r0,
                                                                             one,
                                                                             zero);

      eddy_viscosity.derivative[0] = tke_not_limited / omega_r;
      eddy_viscosity.derivative[1] =
        -omega_not_limited * tke_limited / (omega_r * omega_r);
    }

    // nu_D = nu_t
    diffusion_viscosity.value = eddy_viscosity.value;
    if(input.compute_derivatives)
      for(unsigned int j = 0; j < 2; ++j)
        diffusion_viscosity.derivative[j] = eddy_viscosity.derivative[j];
  }

  /*
     * Source terms of Wilcox 1988, all in the transported form (one branch per variable choice, see
     * compute_production(), compute_dissipation(), evaluate_square_gradient()). No cross-diffusion in
     * any case: it stays zero.
     */
  void
  evaluate_sources(RANSPointInput<dim, Number> const & input,
                   RANSViscosity<Number> const &       eddy_viscosity,
                   RANSSourceTerm<dim, Number> &       production,
                   RANSSourceTerm<dim, Number> &       dissipation,
                   RANSSourceTerm<dim, Number> &       cross_diffusion) const override
  {
    (void)eddy_viscosity;

    compute_production(input, production);
    compute_dissipation(input, dissipation);
    cross_diffusion.set_zero();
  }

  /*
       * Square-gradient term of a log equation (it comes from dividing the equation by phi):
       * SG_i = (nu + sigma_i nu_D) |grad Q_i|^2 with grad Q_i = grad phi_i / phi_i. A linear variable has
       * no square-gradient term.
       *
       *   Case 1 (k, omega)       : none
       *   Case 2 (ln k, ln omega) : SG_K = (nu + sigma* nu_D) |grad K|^2,  SG_W = (nu + sigma nu_D) |grad W|^2
       *   Case 3 (k, ln omega)    : SG_W = (nu + sigma nu_D) |grad W|^2
       *   Case 4 (ln k, omega)    : SG_K = (nu + sigma* nu_D) |grad K|^2
       *
       * Jacobian: d SG_i / d Q_j = sigma_i |grad Q_i|^2 d nu_D / d Q_j, and
       * d SG_i / d grad Q_i = 2 (nu + sigma_i nu_D) grad Q_i, stored at gradient_jacobian(i, i).
       * diffusion_viscosity: nu_D and d nu_D / d Q_j as used by the kernel (after the wall-law bound).
       */
  void
  evaluate_square_gradient(RANSPointInput<dim, Number> const & input,
                           scalar const &                      molecular_viscosity,
                           RANSViscosity<Number> const &       diffusion_viscosity,
                           RANSSourceTerm<dim, Number> &       square_gradient) const override
  {
    scalar const one        = dealii::make_vectorized_array<Number>(1.0);
    scalar const two        = dealii::make_vectorized_array<Number>(2.0);
    scalar const sigma      = dealii::make_vectorized_array<Number>(this->coefficients[3]);
    scalar const sigma_star = dealii::make_vectorized_array<Number>(this->coefficients[4]);

    scalar const nu   = molecular_viscosity;
    scalar const nu_D = diffusion_viscosity.value;

    square_gradient.set_zero();

    if(not k_log and not omega_log)
    {
      // Case 1: k, omega: no square-gradient term, everything stays zero
    }
    else if(k_log and omega_log)
    {
      // Case 2: K = ln k, W = ln omega
      vector const grad_K           = (one / input.values[0]) * input.gradients[0];
      vector const grad_W           = (one / input.values[1]) * input.gradients[1];
      scalar const grad_K_square    = grad_K * grad_K;
      scalar const grad_W_square    = grad_W * grad_W;
      scalar const coefficient_K    = nu + sigma_star * nu_D;
      scalar const coefficient_W    = nu + sigma * nu_D;

      square_gradient.value[0] = coefficient_K * grad_K_square;
      square_gradient.value[1] = coefficient_W * grad_W_square;

      if(input.compute_derivatives)
      {
        square_gradient.jacobian[square_gradient.index(0, 0)] =
          sigma_star * grad_K_square * diffusion_viscosity.derivative[0];
        square_gradient.jacobian[square_gradient.index(0, 1)] =
          sigma_star * grad_K_square * diffusion_viscosity.derivative[1];
        square_gradient.jacobian[square_gradient.index(1, 0)] =
          sigma * grad_W_square * diffusion_viscosity.derivative[0];
        square_gradient.jacobian[square_gradient.index(1, 1)] =
          sigma * grad_W_square * diffusion_viscosity.derivative[1];

        square_gradient.gradient_jacobian[square_gradient.index(0, 0)] =
          two * coefficient_K * grad_K;
        square_gradient.gradient_jacobian[square_gradient.index(1, 1)] =
          two * coefficient_W * grad_W;
      }
    }
    else if(not k_log and omega_log)
    {
      // Case 3: k, W = ln omega: only the omega equation
      vector const grad_W        = (one / input.values[1]) * input.gradients[1];
      scalar const grad_W_square = grad_W * grad_W;
      scalar const coefficient_W = nu + sigma * nu_D;

      square_gradient.value[1] = coefficient_W * grad_W_square;

      if(input.compute_derivatives)
      {
        square_gradient.jacobian[square_gradient.index(1, 0)] =
          sigma * grad_W_square * diffusion_viscosity.derivative[0];
        square_gradient.jacobian[square_gradient.index(1, 1)] =
          sigma * grad_W_square * diffusion_viscosity.derivative[1];

        square_gradient.gradient_jacobian[square_gradient.index(1, 1)] =
          two * coefficient_W * grad_W;
      }
    }
    else
  {
      // Case 4: K = ln k, omega: only the k equation
      vector const grad_K        = (one / input.values[0]) * input.gradients[0];
      scalar const grad_K_square = grad_K * grad_K;
      scalar const coefficient_K = nu + sigma_star * nu_D;

      square_gradient.value[0] = coefficient_K * grad_K_square;

      if(input.compute_derivatives)
      {
        square_gradient.jacobian[square_gradient.index(0, 0)] =
          sigma_star * grad_K_square * diffusion_viscosity.derivative[0];
        square_gradient.jacobian[square_gradient.index(0, 1)] =
          sigma_star * grad_K_square * diffusion_viscosity.derivative[1];

        square_gradient.gradient_jacobian[square_gradient.index(0, 0)] =
          two * coefficient_K * grad_K;
      }
    }
  }

private:
  /*
     * Lower bound omega_r0 of omega (Bassi et al. 2014, eqs. (5)-(8)): 3 times the largest
     * eigenvalue of the 2x2 blocks of S, at least 1e-6; with use_stress_limiter also at least
     * C_lim sqrt(2 S:S / beta*). Copied unchanged from
     * EddyViscosityEvaluator::compute_realizability_alpha().
     */
  scalar
  compute_realizability_alpha(tensor const & grad_u) const
  {
    scalar const zero  = dealii::make_vectorized_array<Number>(0.0);
    scalar const three = dealii::make_vectorized_array<Number>(3.0);

    // strain rate S = 0.5 (grad u + grad u^T), embedded in 3x3 (zero in 2D for the third row)
    dealii::Tensor<2, 3, scalar> S;
    for(unsigned int i = 0; i < 3; ++i)
      for(unsigned int j = 0; j < 3; ++j)
        S[i][j] = zero;

    for(unsigned int i = 0; i < dim; ++i)
      for(unsigned int j = 0; j < dim; ++j)
        S[i][j] = dealii::make_vectorized_array<Number>(0.5) * (grad_u[i][j] + grad_u[j][i]);

    scalar alpha = zero;

    // realizability limiter switched off: alpha = 0, only the positivity floor below remains
    for(unsigned int i = 0; i < 3 and this->turbulence_model_data.use_realizability_limiter; ++i)
    {
      for(unsigned int j = i + 1; j < 3; ++j)
      {
        scalar const B    = three * (S[i][i] + S[j][j]);
        scalar const diff = S[i][i] - S[j][j];

        // delta = 9 (S_ii - S_jj)^2 + 36 S_ij^2
        scalar delta = dealii::make_vectorized_array<Number>(9.0) * diff * diff +
          dealii::make_vectorized_array<Number>(36.0) * S[i][j] * S[i][j];
        delta = std::max(delta, zero);

        // the largest root bounds alpha
        scalar const root_1 = dealii::make_vectorized_array<Number>(0.5) * (B + std::sqrt(delta));
        scalar const root_2 = dealii::make_vectorized_array<Number>(0.5) * (B - std::sqrt(delta));
        alpha = std::max(alpha, std::max(root_1, root_2));
      }
    }

    scalar alpha_safe = std::max(alpha, dealii::make_vectorized_array<Number>(1.0e-6));

    if(this->turbulence_model_data.use_stress_limiter)
    {
      scalar S_S = zero;
      for(unsigned int i = 0; i < dim; ++i)
        for(unsigned int j = 0; j < dim; ++j)
          S_S += S[i][j] * S[i][j];

      scalar const C_lim     = dealii::make_vectorized_array<Number>(7.0 / 8.0);
      scalar const beta_star = dealii::make_vectorized_array<Number>(this->coefficients[2]);

      alpha_safe =
        std::max(alpha_safe,
                 C_lim * std::sqrt(dealii::make_vectorized_array<Number>(2.0) * S_S / beta_star));
    }

    return alpha_safe;
  }

  /*
       * Production (Bassi et al. 2014, eqs. (3)-(4)) in the transported form, one branch per variable
       * choice. G = (grad u + grad u^T) : grad u, k_bar = max(k, 0), omega_r = max(omega, omega_r0).
       * input.values are always the physical k and omega.
       *
       *   Case 1 (k, omega)       : P_k = G k_bar / omega_r,   P_w = alpha G omega_bar / omega_r
       *   Case 2 (ln k, ln omega) : P_K = G / omega_r,         P_W = alpha G / omega_r
       *   Case 3 (k, ln omega)    : P_k = G k_bar / omega_r,   P_W = alpha G / omega_r
       *   Case 4 (ln k, omega)    : P_K = G / omega_r,         P_w = alpha G omega_bar / omega_r
       *
       * Jacobian with respect to the transported variables: d / d(ln phi) = phi d / d phi.
       * Entries that are not assigned stay zero from set_zero().
       */
  void
  compute_production(RANSPointInput<dim, Number> const & input,
                     RANSSourceTerm<dim, Number> &       production) const
  {
    scalar const zero  = dealii::make_vectorized_array<Number>(0.0);
    scalar const one   = dealii::make_vectorized_array<Number>(1.0);
    scalar const alpha = dealii::make_vectorized_array<Number>(this->coefficients[0]);

    scalar const tke   = input.values[0];
    scalar const omega = input.values[1];

    tensor const & grad_u = input.velocity_gradient;
    scalar const   G      = scalar_product(grad_u + transpose(grad_u), grad_u);

    // realizability limiter: k_bar = max(k, 0), omega_r = max(omega, omega_r0)
    scalar const omega_r0      = compute_realizability_alpha(grad_u);
    scalar const tke_positive   = std::max(tke, zero);
    scalar const omega_r = std::max(omega, omega_r0);
    scalar const omega_limited_square = omega_r * omega_r;

    production.set_zero();

    scalar const tke_derivative_sign =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(tke, zero, one, zero);
    scalar const omega_derivative_sign =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(omega,
                                                                           omega_r0,
                                                                           one,
                                                                             zero);

    // linear omega: omega_bar = max(omega, 0) in P_w (no negative production where omega
    // undershoots below zero), with d omega_bar / d omega
    scalar const omega_positive = std::max(omega, zero);
    scalar const omega_positive_derivative_sign =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(omega, zero, one, zero);

    if(not k_log and not omega_log)
    {
      // Case 1: k, omega
      production.value[0] = G * tke_positive / omega_r;
      production.value[1] = alpha * G * omega_positive / omega_r;

      if(input.compute_derivatives)
      {
        production.jacobian[production.index(0, 0)] = G * tke_derivative_sign / omega_r;
        production.jacobian[production.index(0, 1)] =
          -G * tke_positive * omega_derivative_sign / omega_limited_square;
        production.jacobian[production.index(1, 1)] =
          alpha * G *
          (omega_positive_derivative_sign / omega_r -
           omega_positive * omega_derivative_sign / omega_limited_square);
      }
    }
    else if(k_log and omega_log)
    {
      // Case 2: K = ln k, W = ln omega (k > 0, omega > 0)
      production.value[0] = G / omega_r;
      production.value[1] = alpha * G / omega_r;

      if(input.compute_derivatives)
      {
        production.jacobian[production.index(0, 1)] =
          -G * omega_derivative_sign * omega / omega_limited_square;
        production.jacobian[production.index(1, 1)] =
          -alpha * G * omega_derivative_sign * omega / omega_limited_square;
      }
    }
    else if(not k_log and omega_log)
    {
      // Case 3: k, W = ln omega (omega > 0)
      production.value[0] = G * tke_positive / omega_r;
      production.value[1] = alpha * G / omega_r;

      if(input.compute_derivatives)
      {
        production.jacobian[production.index(0, 0)] = G * tke_derivative_sign / omega_r;
        production.jacobian[production.index(0, 1)] =
          -G * tke_positive * omega_derivative_sign * omega / omega_limited_square;
        production.jacobian[production.index(1, 1)] =
          -alpha * G * omega_derivative_sign * omega / omega_limited_square;
      }
    }
    else
    {
      // Case 4: K = ln k (k > 0), omega
      production.value[0] = G / omega_r;
      production.value[1] = alpha * G * omega_positive / omega_r;

      if(input.compute_derivatives)
      {
        production.jacobian[production.index(0, 1)] =
          -G * omega_derivative_sign / omega_limited_square;
        production.jacobian[production.index(1, 1)] =
          alpha * G *
          (omega_positive_derivative_sign / omega_r -
           omega_positive * omega_derivative_sign / omega_limited_square);
      }
    }
  }

  /*
       * Dissipation (Bassi et al. 2014, eqs. (3)-(4)) in the transported form, one branch per variable
       * choice. k_bar = max(k, 0), omega_r = max(omega, omega_r0). input.values are always the
       * physical k and omega.
       *
       *   Case 1 (k, omega)       : D_k = beta* k_bar omega_r,   D_w = beta omega omega_r
       *   Case 2 (ln k, ln omega) : D_K = beta* omega_r,         D_W = beta omega_r
       *   Case 3 (k, ln omega)    : D_k = beta* k_bar omega_r,   D_W = beta omega_r
       *   Case 4 (ln k, omega)    : D_K = beta* omega_r,         D_w = beta omega omega_r
       *
       * Jacobian with respect to the transported variables: d / d(ln phi) = phi d / d phi.
       * Entries that are not assigned stay zero from set_zero().
       */
  void
  compute_dissipation(RANSPointInput<dim, Number> const & input,
                      RANSSourceTerm<dim, Number> &       dissipation) const
  {
    scalar const zero      = dealii::make_vectorized_array<Number>(0.0);
    scalar const one       = dealii::make_vectorized_array<Number>(1.0);
    scalar const beta      = dealii::make_vectorized_array<Number>(this->coefficients[1]);
    scalar const beta_star = dealii::make_vectorized_array<Number>(this->coefficients[2]);

    scalar const tke   = input.values[0];
    scalar const omega = input.values[1];

    // realizability limiter: k_bar = max(k, 0), omega_r = max(omega, omega_r0)
    scalar const omega_r0      = compute_realizability_alpha(input.velocity_gradient);
    scalar const tke_positive   = std::max(tke, zero);
    scalar const omega_r = std::max(omega, omega_r0);

    scalar const tke_derivative_sign = 
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(tke, zero, one, zero);
    scalar const omega_derivative_sign =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(omega,
                                                                           omega_r0,
                                                                           one,
                                                                           zero);
    dissipation.set_zero();
    if(not k_log and not omega_log)
    {
      // Case 1: k, omega
      dissipation.value[0] = beta_star * tke_positive * omega_r;
      dissipation.value[1] = beta * omega * omega_r;

      if(input.compute_derivatives)
      {
        dissipation.jacobian[dissipation.index(0, 0)] =
          beta_star * tke_derivative_sign * omega_r;
        dissipation.jacobian[dissipation.index(0, 1)] =
          beta_star * tke_positive * omega_derivative_sign;
        dissipation.jacobian[dissipation.index(1, 1)] =
          beta * (omega_r + omega * omega_derivative_sign);
      }
    }
    else if(k_log and omega_log)
    {
      // Case 2: K = ln k, W = ln omega (k > 0, omega > 0)
      dissipation.value[0] = beta_star * omega_r;
      dissipation.value[1] = beta * omega_r;

      if(input.compute_derivatives)
      {
        dissipation.jacobian[dissipation.index(0, 1)] = beta_star * omega_derivative_sign * omega;
        dissipation.jacobian[dissipation.index(1, 1)] = beta * omega_derivative_sign * omega;
      }
    }
    else if(not k_log and omega_log)
    {
      // Case 3: k, W = ln omega (omega > 0)
      dissipation.value[0] = beta_star * tke_positive * omega_r;
      dissipation.value[1] = beta * omega_r;

      if(input.compute_derivatives)
      {
        dissipation.jacobian[dissipation.index(0, 0)] =
          beta_star * tke_derivative_sign * omega_r;
        dissipation.jacobian[dissipation.index(0, 1)] =
          beta_star * tke_positive * omega_derivative_sign * omega;
        dissipation.jacobian[dissipation.index(1, 1)] = beta * omega_derivative_sign * omega;
      }
    }
    else
  {
      // Case 4: K = ln k (k > 0), omega
      dissipation.value[0] = beta_star * omega_r;
      dissipation.value[1] = beta * omega * omega_r;

      if(input.compute_derivatives)
      {
        dissipation.jacobian[dissipation.index(0, 1)] = beta_star * omega_derivative_sign;
        dissipation.jacobian[dissipation.index(1, 1)] =
          beta * (omega_r + omega * omega_derivative_sign);
      }
    }
  }
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_K_OMEGA_1988_H_ */
