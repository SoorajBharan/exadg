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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_K_OMEGA_2006_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_K_OMEGA_2006_H_

// ExaDG
#include <exadg/convection_diffusion/spatial_discretization/rans_models/rans_model_base.h>

namespace ExaDG
{
namespace ConvDiff
{
/*
   * Wilcox (2006) k-omega model, incompressible (NASA Turbulence Modeling Resource,
   * tmbwg.github.io/turbmodels/wilcox.html):
   *
   *   k     : P_k = nu_t G,                        D_k = beta* k omega
   *   omega : P_w = alpha (omega / omega_hat) G,   D_w = beta omega^2,
   *           cross-diffusion CD = (sigma_d / omega) grad k . grad omega
   *
   *   nu_t      = k / omega_hat, omega_hat = max(omega, C_lim sqrt(2 S_bar : S_bar / beta*))
   *               (the stress limiter acts on nu_t only)
   *   nu_D      = k / omega (turbulent diffusion)
   *   beta      = beta_0 f_beta (f_beta = 1 in 2D)
   *   sigma_d   = sigma_d0 if grad k . grad omega > 0, otherwise 0
   *
   * The switches use_realizability_limiter / use_stress_limiter of TurbulenceModelData are not used.
   * Positivity for linear k / linear omega, as in Wilcox 1988: k_bar = max(k, 0), and
   * omega_safe = max(omega, omega_min) wherever omega divides.
   *
   * Variables: 0 = k, 1 = omega (physical).
   * Coefficients: [0] alpha, [1] beta_0, [2] beta*, [3] sigma (omega), [4] sigma* (k), [5] sigma_d0,
   * [6] C_lim.
   */
template<int dim, typename Number>
class KOmega2006Model : public RANSModelBase<dim, Number>
{
private:
  typedef RANSModelBase<dim, Number>     Base;
  typedef typename Base::scalar          scalar;
  typedef dealii::Tensor<2, dim, scalar> tensor;
  typedef typename Base::vector          vector;

  // true: the variable is transported as its logarithm (ln k, ln omega)
  bool const k_log;
  bool const omega_log;

public:
  KOmega2006Model(TurbulenceModelData const & turbulence_model_data_in)
    : Base(turbulence_model_data_in),
      k_log(turbulence_model_data_in.positivity_preserving_limiter.at(0) ==
            PositivityPreservingLimiter::LogarithmicTransportVariable),
      omega_log(turbulence_model_data_in.positivity_preserving_limiter.at(1) ==
                PositivityPreservingLimiter::LogarithmicTransportVariable)
  {
    AssertThrow(this->coefficients.size() == 7,
                dealii::ExcMessage("Wilcox 2006: 7 coefficients expected."));
  }

  unsigned int
  get_n_variables() const override
  {
    return 2;
  }

  // k: sigma*, omega: sigma
  double
  get_diffusion_factor(unsigned int const i) const override
  {
    AssertIndexRange(i, 2);
    return (i == 0) ? this->coefficients[4] : this->coefficients[3];
  }

  // cross-diffusion in the omega equation
  bool
  has_cross_diffusion() const override
  {
    return true;
  }

  // all source terms (production, dissipation, cross-diffusion, square gradient) are returned in
  // the transported form
  bool
  sources_in_transported_form() const override
  {
    return true;
  }

  /*
     * nu_t = k_bar / omega_hat (stress limiter) and nu_D = k_bar / omega_safe, with their derivatives
     * with respect to the physical k and omega:
     *
     *   k_bar      = max(k, 0)
     *   omega_safe = max(omega, omega_min)
     *   omega_hat  = max(omega_safe, C_lim sqrt(2 S_bar : S_bar / beta*))
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

    scalar const tke_limited     = std::max(tke, zero);
    scalar const omega_safe      = compute_safe_omega(omega);
    scalar const omega_hat_bound = compute_stress_limiter_bound(input.velocity_gradient);
    scalar const omega_hat       = std::max(omega_safe, omega_hat_bound);

    eddy_viscosity.value      = tke_limited / omega_hat;
    diffusion_viscosity.value = tke_limited / omega_safe;

    if(not input.compute_derivatives)
      return;

    // 1 where the clipping / floor / stress limiter is inactive, 0 where it is active
    scalar const tke_not_limited =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(tke, zero, one, zero);
    scalar const omega_above_floor =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
        omega, dealii::make_vectorized_array<Number>(omega_min), one, zero);
    scalar const stress_limiter_inactive =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(omega_safe,
                                                                           omega_hat_bound,
                                                                           one,
                                                                           zero);
    // nu_t = k_bar / omega_hat
    eddy_viscosity.derivative[0] = tke_not_limited / omega_hat;
    eddy_viscosity.derivative[1] = -stress_limiter_inactive * omega_above_floor * tke_limited / (omega_hat * omega_hat);

    // nu_D = k_bar / omega_safe
    diffusion_viscosity.derivative[0] = tke_not_limited / omega_safe;
    diffusion_viscosity.derivative[1] = -omega_above_floor * tke_limited / (omega_safe * omega_safe);
  }

  /*
     * Source terms of Wilcox 2006, all in the transported form (one branch per variable choice):
     * compute_production(), compute_dissipation(), compute_cross_diffusion(),
     * evaluate_square_gradient().
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
    compute_cross_diffusion(input, cross_diffusion);
  }

  /*
     * Square-gradient term of a log equation: SG_i = (nu + sigma_i nu_D) |grad Q_i|^2,
     * grad Q_i = grad phi_i / phi_i, nu_D = k_bar / omega_safe. A linear variable has none.
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
      vector const grad_K        = (one / input.values[0]) * input.gradients[0];
      vector const grad_W        = (one / input.values[1]) * input.gradients[1];
      scalar const grad_K_square = grad_K * grad_K;
      scalar const grad_W_square = grad_W * grad_W;
      scalar const coefficient_K = nu + sigma_star * nu_D;
      scalar const coefficient_W = nu + sigma * nu_D;

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
  // lower bound of omega wherever omega divides (linear omega can undershoot below zero)
  static constexpr double omega_min = 1.0e-6;

  scalar
  compute_safe_omega(scalar const & omega) const
  {
    return std::max(omega, dealii::make_vectorized_array<Number>(omega_min));
  }

  /*
     * Stress-limiter bound C_lim sqrt(2 S_bar : S_bar / beta*), S = 0.5 (grad u + grad u^T),
     * S_bar = S - 1/3 div(u) I, so that S_bar : S_bar = S : S - (div u)^2 / 3.
     */
  scalar
  compute_stress_limiter_bound(tensor const & grad_u) const
  {
    scalar const zero      = dealii::make_vectorized_array<Number>(0.0);
    scalar const half      = dealii::make_vectorized_array<Number>(0.5);
    scalar const C_lim     = dealii::make_vectorized_array<Number>(this->coefficients[6]);
    scalar const beta_star = dealii::make_vectorized_array<Number>(this->coefficients[2]);

    tensor const S     = half * (grad_u + transpose(grad_u));
    scalar const div_u = trace(grad_u);

    scalar const S_bar_S_bar =
      std::max(scalar_product(S, S) - div_u * div_u / dealii::make_vectorized_array<Number>(3.0),
               zero);

    return C_lim *
    std::sqrt(dealii::make_vectorized_array<Number>(2.0) * S_bar_S_bar / beta_star);
  }

  /*
     * f_beta = (1 + 85 chi) / (1 + 100 chi), chi = |Omega_ij Omega_jk S_hat_ki| / (beta* omega)^3,
     * Omega = 0.5 (grad u - grad u^T), S_hat = S - 1/2 div(u) I. In 2D, Omega_ij Omega_jk S_hat_ki = 0,
     * so chi = 0 and f_beta = 1.
     *
     * f_beta_derivative = d f_beta / d omega = (-15 / (1 + 100 chi)^2) (-3 chi / omega).
     */
  void
  compute_f_beta(tensor const & grad_u,
                 scalar const & omega_safe,
                 scalar &       f_beta,
                 scalar &       f_beta_derivative) const
  {
    scalar const one       = dealii::make_vectorized_array<Number>(1.0);
    scalar const half      = dealii::make_vectorized_array<Number>(0.5);
    scalar const beta_star = dealii::make_vectorized_array<Number>(this->coefficients[2]);

    tensor const S = half * (grad_u + transpose(grad_u));
    tensor const W = half * (grad_u - transpose(grad_u));

    tensor       S_hat = S;
    scalar const div_u = trace(grad_u);
    for(unsigned int d = 0; d < dim; ++d)
      S_hat[d][d] -= half * div_u;

    // Omega_ij Omega_jk S_hat_ki = trace(Omega Omega S_hat)
    scalar const vortex_stretching = trace(W * W * S_hat);

    scalar const beta_star_omega = beta_star * omega_safe;
    scalar const chi =
      std::abs(vortex_stretching) / (beta_star_omega * beta_star_omega * beta_star_omega);

    scalar const denominator = one + dealii::make_vectorized_array<Number>(100.0) * chi;

    f_beta = (one + dealii::make_vectorized_array<Number>(85.0) * chi) / denominator;
    f_beta_derivative =
      dealii::make_vectorized_array<Number>(45.0) * chi / (denominator * denominator * omega_safe);
  }

  /*
     * Production in the transported form, one branch per variable choice. G = (grad u + grad u^T) :
     * grad u, k_bar = max(k, 0), omega_bar = max(omega, 0),
     * omega_hat = max(omega_safe, C_lim sqrt(2 S_bar : S_bar / beta*)).
     *
     *   Case 1 (k, omega)       : P_k = G k_bar / omega_hat,   P_w = alpha G omega_bar / omega_hat
     *   Case 2 (ln k, ln omega) : P_K = G / omega_hat,         P_W = alpha G / omega_hat
     *   Case 3 (k, ln omega)    : P_k = G k_bar / omega_hat,   P_W = alpha G / omega_hat
     *   Case 4 (ln k, omega)    : P_K = G / omega_hat,         P_w = alpha G omega_bar / omega_hat
     *
     * P_w = alpha (omega / k) nu_t G with the stress-limited nu_t = k / omega_hat (Wilcox 2006);
     * where the stress limiter is inactive, omega_hat = omega and P_w = alpha G.
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

    scalar const tke_positive     = std::max(tke, zero);
    scalar const omega_positive   = std::max(omega, zero);
    scalar const omega_safe       = compute_safe_omega(omega);
    scalar const omega_hat_bound  = compute_stress_limiter_bound(grad_u);
    scalar const omega_hat        = std::max(omega_safe, omega_hat_bound);
    scalar const omega_hat_square = omega_hat * omega_hat;

    // d k_bar / d k, d omega_bar / d omega: 1 where the clipping is inactive
    scalar const tke_derivative_sign =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(tke, zero, one, zero);
    scalar const omega_positive_derivative_sign =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(omega, zero, one, zero);

    // d omega_hat / d omega: 1 where neither the floor nor the stress limiter is active
    scalar const omega_hat_derivative_sign =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
        omega, dealii::make_vectorized_array<Number>(omega_min), one, zero) *
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(omega_safe,
                                                                           omega_hat_bound,
                                                                           one,
                                                                           zero);

    production.set_zero();

    if(not k_log and not omega_log)
    {
      // Case 1: k, omega
      production.value[0] = G * tke_positive / omega_hat;
      production.value[1] = alpha * G * omega_positive / omega_hat;

      if(input.compute_derivatives)
      {
        production.jacobian[production.index(0, 0)] = G * tke_derivative_sign / omega_hat;
        production.jacobian[production.index(0, 1)] =
          -G * tke_positive * omega_hat_derivative_sign / omega_hat_square;
        production.jacobian[production.index(1, 1)] =
          alpha * G *
          (omega_positive_derivative_sign / omega_hat -
           omega_positive * omega_hat_derivative_sign / omega_hat_square);
      }
    }
    else if(k_log and omega_log)
    {
      // Case 2: K = ln k, W = ln omega (k > 0, omega > 0)
      production.value[0] = G / omega_hat;
      production.value[1] = alpha * G / omega_hat;

      if(input.compute_derivatives)
      {
        production.jacobian[production.index(0, 1)] =
          -G * omega_hat_derivative_sign * omega / omega_hat_square;
        production.jacobian[production.index(1, 1)] =
          -alpha * G * omega_hat_derivative_sign * omega / omega_hat_square;
      }
    }
    else if(not k_log and omega_log)
    {
      // Case 3: k, W = ln omega (omega > 0)
      production.value[0] = G * tke_positive / omega_hat;
      production.value[1] = alpha * G / omega_hat;

      if(input.compute_derivatives)
      {
        production.jacobian[production.index(0, 0)] = G * tke_derivative_sign / omega_hat;
        production.jacobian[production.index(0, 1)] =
          -G * tke_positive * omega_hat_derivative_sign * omega / omega_hat_square;
        production.jacobian[production.index(1, 1)] =
          -alpha * G * omega_hat_derivative_sign * omega / omega_hat_square;
      }
    }
    else
    {
      // Case 4: K = ln k (k > 0), omega
      production.value[0] = G / omega_hat;
      production.value[1] = alpha * G * omega_positive / omega_hat;

      if(input.compute_derivatives)
      {
        production.jacobian[production.index(0, 1)] =
          -G * omega_hat_derivative_sign / omega_hat_square;
        production.jacobian[production.index(1, 1)] =
          alpha * G *
          (omega_positive_derivative_sign / omega_hat -
           omega_positive * omega_hat_derivative_sign / omega_hat_square);
      }
    }
  }

  /*
     * Dissipation in the transported form, one branch per variable choice. k_bar = max(k, 0),
     * omega_safe = max(omega, omega_min), beta = beta_0 f_beta(omega_safe) (f_beta = 1 in 2D).
     *
     *   Case 1 (k, omega)       : D_k = beta* k_bar omega_safe,   D_w = beta omega omega_safe
     *   Case 2 (ln k, ln omega) : D_K = beta* omega_safe,         D_W = beta omega_safe
     *   Case 3 (k, ln omega)    : D_k = beta* k_bar omega_safe,   D_W = beta omega_safe
     *   Case 4 (ln k, omega)    : D_K = beta* omega_safe,         D_w = beta omega omega_safe
     *
     * d beta / d omega = beta_0 f_beta' d omega_safe / d omega.
     * Jacobian with respect to the transported variables: d / d(ln phi) = phi d / d phi.
     * Entries that are not assigned stay zero from set_zero().
     */
  void
  compute_dissipation(RANSPointInput<dim, Number> const & input,
                      RANSSourceTerm<dim, Number> &       dissipation) const
  {
    scalar const zero      = dealii::make_vectorized_array<Number>(0.0);
    scalar const one       = dealii::make_vectorized_array<Number>(1.0);
    scalar const beta_0    = dealii::make_vectorized_array<Number>(this->coefficients[1]);
    scalar const beta_star = dealii::make_vectorized_array<Number>(this->coefficients[2]);

    scalar const tke   = input.values[0];
    scalar const omega = input.values[1];

    scalar const tke_positive = std::max(tke, zero);
    scalar const omega_safe   = compute_safe_omega(omega);

    scalar f_beta, f_beta_derivative;
    compute_f_beta(input.velocity_gradient, omega_safe, f_beta, f_beta_derivative);

    scalar const beta = beta_0 * f_beta;

    // d k_bar / d k and d omega_safe / d omega: 1 where the clipping / floor is inactive
    scalar const tke_derivative_sign =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(tke, zero, one, zero);
    scalar const omega_safe_derivative_sign =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
        omega, dealii::make_vectorized_array<Number>(omega_min), one, zero);

    // d beta / d omega
    scalar const beta_derivative = beta_0 * f_beta_derivative * omega_safe_derivative_sign;

    dissipation.set_zero();

    if(not k_log and not omega_log)
    {
      // Case 1: k, omega
      dissipation.value[0] = beta_star * tke_positive * omega_safe;
      dissipation.value[1] = beta * omega * omega_safe;

      if(input.compute_derivatives)
      {
        dissipation.jacobian[dissipation.index(0, 0)] =
          beta_star * tke_derivative_sign * omega_safe;
        dissipation.jacobian[dissipation.index(0, 1)] =
          beta_star * tke_positive * omega_safe_derivative_sign;
        dissipation.jacobian[dissipation.index(1, 1)] =
          beta * (omega_safe + omega * omega_safe_derivative_sign) +
          beta_derivative * omega * omega_safe;
      }
    }
    else if(k_log and omega_log)
    {
      // Case 2: K = ln k, W = ln omega (k > 0, omega > 0)
      dissipation.value[0] = beta_star * omega_safe;
      dissipation.value[1] = beta * omega_safe;

      if(input.compute_derivatives)
      {
        dissipation.jacobian[dissipation.index(0, 1)] =
          beta_star * omega_safe_derivative_sign * omega;
        dissipation.jacobian[dissipation.index(1, 1)] =
          (beta * omega_safe_derivative_sign + beta_derivative * omega_safe) * omega;
      }
    }
    else if(not k_log and omega_log)
    {
      // Case 3: k, W = ln omega (omega > 0)
      dissipation.value[0] = beta_star * tke_positive * omega_safe;
      dissipation.value[1] = beta * omega_safe;

      if(input.compute_derivatives)
      {
        dissipation.jacobian[dissipation.index(0, 0)] =
          beta_star * tke_derivative_sign * omega_safe;
        dissipation.jacobian[dissipation.index(0, 1)] =
          beta_star * tke_positive * omega_safe_derivative_sign * omega;
        dissipation.jacobian[dissipation.index(1, 1)] =
          (beta * omega_safe_derivative_sign + beta_derivative * omega_safe) * omega;
      }
    }
    else
    {
      // Case 4: K = ln k (k > 0), omega
      dissipation.value[0] = beta_star * omega_safe;
      dissipation.value[1] = beta * omega * omega_safe;

      if(input.compute_derivatives)
      {
        dissipation.jacobian[dissipation.index(0, 1)] = beta_star * omega_safe_derivative_sign;
        dissipation.jacobian[dissipation.index(1, 1)] =
          beta * (omega_safe + omega * omega_safe_derivative_sign) +
          beta_derivative * omega * omega_safe;
      }
    }
  }

  /*
     * Cross-diffusion of the omega equation in the transported form, one branch per variable choice.
     * Physical form: CD = sigma_d / omega grad k . grad omega, sigma_d = sigma_d0 where
     * grad k . grad omega > 0, otherwise 0 (kept fixed in the Jacobians). With grad k = k grad K and
     * grad omega = omega grad W:
     *
     *   Case 1 (k, omega)       : CD   = sigma_d grad k . grad omega / omega_safe
     *   Case 2 (ln k, ln omega) : CD_W = sigma_d (k / omega) grad K . grad W
     *   Case 3 (k, ln omega)    : CD_W = sigma_d grad k . grad W / omega
     *   Case 4 (ln k, omega)    : CD   = sigma_d k grad K . grad omega / omega_safe
     *
     * omega_safe = max(omega, omega_min) only for linear omega (ln omega: omega > 0). The k row is
     * zero. Jacobians with respect to the transported values and gradients.
     */
  void
  compute_cross_diffusion(RANSPointInput<dim, Number> const & input,
                          RANSSourceTerm<dim, Number> &       cross_diffusion) const
  {
    scalar const zero     = dealii::make_vectorized_array<Number>(0.0);
    scalar const one      = dealii::make_vectorized_array<Number>(1.0);
    scalar const sigma_d0 = dealii::make_vectorized_array<Number>(this->coefficients[5]);

    scalar const tke   = input.values[0];
    scalar const omega = input.values[1];

    scalar const omega_safe = compute_safe_omega(omega);

    // d omega_safe / d omega: 1 above the floor, 0 below
    scalar const omega_safe_derivative_sign =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
        omega, dealii::make_vectorized_array<Number>(omega_min), one, zero);

    cross_diffusion.set_zero();

    if(not k_log and not omega_log)
    {
      // Case 1: k, omega
      vector const & grad_k     = input.gradients[0];
      vector const & grad_omega = input.gradients[1];

      scalar const grad_k_grad_omega = grad_k * grad_omega;
      scalar const sigma_d =
        sigma_d0 * dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                     grad_k_grad_omega, zero, one, zero);

      cross_diffusion.value[1] = sigma_d * grad_k_grad_omega / omega_safe;

      if(input.compute_derivatives)
      {
        cross_diffusion.jacobian[cross_diffusion.index(1, 1)] =
          -omega_safe_derivative_sign * sigma_d * grad_k_grad_omega / (omega_safe * omega_safe);

        cross_diffusion.gradient_jacobian[cross_diffusion.index(1, 0)] =
          (sigma_d / omega_safe) * grad_omega;
        cross_diffusion.gradient_jacobian[cross_diffusion.index(1, 1)] =
          (sigma_d / omega_safe) * grad_k;
      }
    }
    else if(k_log and omega_log)
    {
      // Case 2: K = ln k, W = ln omega (k > 0, omega > 0)
      vector const grad_K = (one / tke) * input.gradients[0];
      vector const grad_W = (one / omega) * input.gradients[1];

      scalar const grad_K_grad_W = grad_K * grad_W;
      scalar const sigma_d =
        sigma_d0 * dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                     grad_K_grad_W, zero, one, zero);

      scalar const factor = sigma_d * tke / omega;

      cross_diffusion.value[1] = factor * grad_K_grad_W;

      if(input.compute_derivatives)
      {
        cross_diffusion.jacobian[cross_diffusion.index(1, 0)] = cross_diffusion.value[1];
        cross_diffusion.jacobian[cross_diffusion.index(1, 1)] = -cross_diffusion.value[1];

        cross_diffusion.gradient_jacobian[cross_diffusion.index(1, 0)] = factor * grad_W;
        cross_diffusion.gradient_jacobian[cross_diffusion.index(1, 1)] = factor * grad_K;
      }
    }
    else if(not k_log and omega_log)
    {
      // Case 3: k, W = ln omega (omega > 0)
      vector const & grad_k = input.gradients[0];
      vector const   grad_W = (one / omega) * input.gradients[1];

      scalar const grad_k_grad_W = grad_k * grad_W;
      scalar const sigma_d =
        sigma_d0 * dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                     grad_k_grad_W, zero, one, zero);

      scalar const factor = sigma_d / omega;

      cross_diffusion.value[1] = factor * grad_k_grad_W;

      if(input.compute_derivatives)
      {
        cross_diffusion.jacobian[cross_diffusion.index(1, 1)] = -cross_diffusion.value[1];

        cross_diffusion.gradient_jacobian[cross_diffusion.index(1, 0)] = factor * grad_W;
        cross_diffusion.gradient_jacobian[cross_diffusion.index(1, 1)] = factor * grad_k;
      }
    }
    else
    {
      // Case 4: K = ln k (k > 0), omega
      vector const   grad_K     = (one / tke) * input.gradients[0];
      vector const & grad_omega = input.gradients[1];

      scalar const grad_K_grad_omega = grad_K * grad_omega;
      scalar const sigma_d =
        sigma_d0 * dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                     grad_K_grad_omega, zero, one, zero);

      scalar const factor = sigma_d * tke / omega_safe;

      cross_diffusion.value[1] = factor * grad_K_grad_omega;

      if(input.compute_derivatives)
      {
        cross_diffusion.jacobian[cross_diffusion.index(1, 0)] = cross_diffusion.value[1];
        cross_diffusion.jacobian[cross_diffusion.index(1, 1)] =
          -omega_safe_derivative_sign * cross_diffusion.value[1] / omega_safe;

        cross_diffusion.gradient_jacobian[cross_diffusion.index(1, 0)] = factor * grad_omega;
        cross_diffusion.gradient_jacobian[cross_diffusion.index(1, 1)] = factor * grad_K;
      }
    }
  }
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_K_OMEGA_2006_H_ */
