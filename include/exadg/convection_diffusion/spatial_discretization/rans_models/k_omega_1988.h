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
  typedef dealii::Tensor<2, dim, scalar> tensor;

public:
  KOmega1988Model(TurbulenceModelData const & turbulence_model_data_in)
  : Base(turbulence_model_data_in)
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
    scalar const omega_limited = std::max(omega, omega_r0);

    // nu_t = k_bar / omega_r (alpha* = 1)
    eddy_viscosity.value = tke_limited / omega_limited;

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

      eddy_viscosity.derivative[0] = tke_not_limited / omega_limited;
      eddy_viscosity.derivative[1] =
        -omega_not_limited * tke_limited / (omega_limited * omega_limited);
    }

    // nu_D = nu_t
    diffusion_viscosity.value = eddy_viscosity.value;
    if(input.compute_derivatives)
      for(unsigned int j = 0; j < 2; ++j)
        diffusion_viscosity.derivative[j] = eddy_viscosity.derivative[j];
  }

  /*
     * All source terms of Wilcox 1988 in physical form. The model has no gradient source
     * (no cross-diffusion); the |grad ln omega|^2 term of the ln omega equation comes from the
     * variable transformation and is added by the kernel. nu_t is not needed: P_k is computed as
     * G k_bar / omega_r, as in the previous implementation.
     */
  void
  evaluate_sources(RANSPointInput<dim, Number> const & input,
                   RANSViscosity<Number> const &       eddy_viscosity,
                   RANSSourceTerm<dim, Number> &       production,
                   RANSSourceTerm<dim, Number> &       dissipation,
                   RANSSourceTerm<dim, Number> &       gradient_source) const override
  {
    (void)eddy_viscosity;

    compute_production(input, production);
    compute_dissipation(input, dissipation);
    gradient_source.set_zero();
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
     * Production in physical form (Bassi et al. 2014, eqs. (3)-(4)) and its Jacobian with respect to
     * the physical k and omega; the linear-k / linear-omega branches of
     * TurbulenceSourceKernel::k_omega_1988_production(), unchanged:
     *
     *   k     : P_k = G k_bar / omega_r                 (= nu_t G)
     *   omega : P_w = alpha G max(omega, 0) / omega_r
     *
     * with G = (grad u + grad u^T) : grad u. The ln k / ln omega forms follow from
     * convert_to_transported_form() in the kernel.
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
    scalar const tke_limited   = std::max(tke, zero);
    scalar const omega_limited = std::max(omega, omega_r0);
    scalar const inverse_omega = one / omega_limited;

    // max(omega, 0): no (negative) production where omega undershoots below zero, otherwise
    // alpha G omega / omega_r drives omega further down (only possible for linear omega)
    scalar const omega_bar = std::max(omega, zero);

    production.set_zero();
    production.value[0] = G * tke_limited * inverse_omega;
    production.value[1] = alpha * G * omega_bar * inverse_omega;

    if(not input.compute_derivatives)
      return;

    // 1 where the limiter / clipping is inactive, 0 where it is active
    scalar const tke_not_limited =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(tke, zero, one, zero);
    scalar const omega_not_limited =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(omega,
                                                                           omega_r0,
                                                                           one,
                                                                           zero);
    scalar const omega_positive =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(omega, zero, one, zero);

    // d (1 / omega_r) / d omega
    scalar const inverse_omega_derivative = -omega_not_limited * inverse_omega * inverse_omega;

    // row of k
    production.jacobian[production.index(0, 0)] = G * tke_not_limited * inverse_omega;
    production.jacobian[production.index(0, 1)] = G * tke_limited * inverse_omega_derivative;

    // row of omega (does not depend on k: index(1, 0) stays zero from set_zero())
    production.jacobian[production.index(1, 1)] =
      alpha * G * (omega_positive * inverse_omega + omega_bar * inverse_omega_derivative);
  }

  /*
     * Dissipation in physical form (Bassi et al. 2014, eqs. (3)-(4)) and its Jacobian with respect
     * to the physical k and omega; the linear-k / linear-omega branches of
     * TurbulenceSourceKernel::k_omega_1988_dissipation(), unchanged:
     *
     *   k     : D_k = beta* k_bar omega_r
     *   omega : D_w = beta omega omega_r
     *
     * The ln k / ln omega forms (beta* omega_r, beta omega_r) follow from convert_to_transported_form()
     * in the kernel.
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
    scalar const tke_limited   = std::max(tke, zero);
    scalar const omega_limited = std::max(omega, omega_r0);

    dissipation.set_zero();
    dissipation.value[0] = beta_star * tke_limited * omega_limited;
    dissipation.value[1] = beta * omega * omega_limited;


    if(not input.compute_derivatives)
      return;

    // 1 where the limiter / clipping is inactive, 0 where it is active
    scalar const tke_not_limited =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(tke, zero, one, zero);
    scalar const omega_not_limited =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(omega,
                                                                           omega_r0,
                                                                           one,
                                                                           zero);

    // row of k
    dissipation.jacobian[dissipation.index(0, 0)] = beta_star * tke_not_limited * omega_limited;
    dissipation.jacobian[dissipation.index(0, 1)] = beta_star * tke_limited * omega_not_limited;

    // row of omega (does not depend on k: index(1, 0) stays zero from set_zero())
    dissipation.jacobian[dissipation.index(1, 1)] =
      beta * (omega_limited + omega * omega_not_limited);
  }
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_K_OMEGA_1988_H_ */
