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
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_OPERATORS_WALL_MODELLING_WALL_LAW_EVALUATOR_H_
#define INCLUDE_OPERATORS_WALL_MODELLING_WALL_LAW_EVALUATOR_H_

#include <deal.II/base/vectorization.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/exceptions.h>
#include <cmath>
#include <limits>

namespace ExaDG
{

enum class WallLawType
{
  Spalding,
  Reichardt,
  VanDriest,
  PowerLaw,
  Undefined
};

template<typename Number>
dealii::VectorizedArray<Number>
initial_guess_u_plus(dealii::VectorizedArray<Number> const y_plus,
                     double kappa = 0.41,
                     double beta = 5.17)
{
  using scalar = dealii::VectorizedArray<Number>;

  scalar safe_y_plus = std::max(y_plus, dealii::make_vectorized_array<Number>(1e-6));

  scalar const linear_guess = safe_y_plus;
  scalar const log_guess = std::log(safe_y_plus) / kappa + dealii::make_vectorized_array<Number>(beta);

  scalar const threshold = dealii::make_vectorized_array<Number>(5.0);

  return dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(safe_y_plus, threshold, log_guess, linear_guess);
}

template<typename Number, typename Functor>
dealii::VectorizedArray<Number>
solve_newton_raphson(
  dealii::VectorizedArray<Number> const y_plus,
  dealii::VectorizedArray<Number> const initial_guess,
  Functor const & function,
  unsigned int const max_iterations = 100)
{
  using scalar = dealii::VectorizedArray<Number>;
  Number const tol = 100 * std::numeric_limits<Number>::epsilon();

  scalar u_plus = initial_guess;

  for (unsigned int iter = 0; iter < max_iterations; ++iter)
  {
    std::pair<scalar, scalar> values = function(u_plus);

    scalar const residual = values.first - y_plus;

    u_plus -= residual / values.second;

    bool converged = true;
    for(unsigned int v = 0; v < scalar::size(); ++v)
      if(std::abs(residual[v]) > tol * std::max(Number(1), y_plus[v]))
        converged = false;
    if(converged)
      return u_plus;
  }

  AssertThrow(false, dealii::ExcMessage("Spalding's law: Newton iteration did not converge."));
  return u_plus;
}

struct SpaldingLaw
{
  double kappa = 0.41;
  double beta  = 5.17;
  double expKB = std::exp(-1.0 * beta * kappa);

  template<typename Number>
  std::pair<dealii::VectorizedArray<Number>, dealii::VectorizedArray<Number>>
  operator()(
    dealii::VectorizedArray<Number> const u_plus) const
  {
    using scalar = dealii::VectorizedArray<Number>;

    scalar k_u = kappa * u_plus;

    scalar term_1 = std::exp(k_u) - 1.0 - k_u - (k_u * k_u) / 2.0 - (k_u * k_u * k_u) / 6.0 - (k_u * k_u * k_u * k_u) / 24.0;

    scalar term_2 = kappa * (std::exp(k_u) - 1.0 - k_u - (k_u * k_u) / 2.0 - (k_u * k_u * k_u) / 6.0);

    // Spalding's implicit law: y^+ = u^+ + e^{-\beta \kappa} \left( e^{u^+ \kappa} - 1 - u^+ \kappa - \frac{(u^+ \kappa)^2}{2!} - \frac{(u^+ \kappa)^3}{3!} - \frac{(u^+ \kappa)^4}{4!} \right)
    scalar y_plus = u_plus + expKB * term_1;

    // \frac{dy^+}{du^+} = 1 + e^{-\beta \kappa} \kappa \left( e^{u^+ \kappa} - 1 - u^+ \kappa - \frac{(u^+ \kappa)^2}{2!} - \frac{(u^+ \kappa)^3}{3!} \right)
    scalar dy_plus = 1.0 + expKB * term_2;

    return std::make_pair(y_plus, dy_plus);
  }
};

// condition of the enriched omega at the wall face
enum class OmegaEnrichmentWallCondition
{
  ZeroFlux,     // no condition at the wall face; the equations set the amplitude omega_tilde
  WeakDirichlet // omega_h = (u_tau^2 / nu) psi_omega(0) (ln of it for ln omega), SIPG
};

enum class OmegaEnrichmentFaceFormulation
{
  NIPG,
  SIPG
};

// omega wall condition of wall cells whose omega enrichment is switched off by the k criterion
enum class OmegaEnrichmentOffWallCondition
{
  ZeroFlux,    // no wall value for omega (as the lanes switched off by the y+ criterion)
  WallFunction // omega Dirichlet: omega_wf = (u_tau^2 / nu) psi_omega at y = a y_c (as U enriched only)
};

// cap of psi_omega at the wall (psi_omega -> infinity for y+ -> 0)
enum class OmegaWallCap
{
  ShiftedWallDistance, // psi_omega(y+ + y+_0)
  MenterWallValue      // min(psi_omega(y+), F 6 / (beta y+_cap^2))
};

// blended omega wall law in wall units (omega+ = omega nu / u_tau^2):
//   psi_omega = sqrt(omega_vis+^2 + omega_log+^2)
//   omega_vis+ = 6 / (beta y+^2),  omega_log+ = 1 / (sqrt(beta*) kappa y+)
// returns psi_omega and d psi_omega / d y+
struct OmegaWallLaw
{
  double kappa     = 0.41;
  double beta      = 0.075; // k-omega 1988: 3/40
  double beta_star = 0.09;

  OmegaWallCap cap           = OmegaWallCap::ShiftedWallDistance;
  double       y_plus_shift  = 0.5;  // y+_0
  double       menter_factor = 10.0; // F
  double       y_plus_cap    = 1.0;  // y+_cap

  template<typename Number>
  std::pair<dealii::VectorizedArray<Number>, dealii::VectorizedArray<Number>>
  operator()(dealii::VectorizedArray<Number> const y_plus) const
  {
    using scalar = dealii::VectorizedArray<Number>;

    Number const a = 6.0 / beta;
    Number const b = 1.0 / (std::sqrt(beta_star) * kappa);

    // y+ used in the blend; floor: no division by zero at the wall
    scalar Y = y_plus;
    if(cap == OmegaWallCap::ShiftedWallDistance)
      Y = Y + Number(y_plus_shift);
    Y = std::max(Y, dealii::make_vectorized_array<Number>(1.0e-6));

    scalar const Y2 = Y * Y;

    // psi_omega = sqrt(a^2 + b^2 Y^2) / Y^2
    scalar psi = std::sqrt(a * a + b * b * Y2) / Y2;

    // d psi_omega / dY = -(2 a^2 / Y^5 + b^2 / Y^3) / psi_omega
    scalar dpsi = -(2.0 * a * a / (Y2 * Y2 * Y) + b * b / (Y2 * Y)) / psi;

    if(cap == OmegaWallCap::MenterWallValue)
    {
      scalar const psi_cap =
        dealii::make_vectorized_array<Number>(menter_factor * a / (y_plus_cap * y_plus_cap));
      scalar const zero = dealii::make_vectorized_array<Number>(0.0);

      // capped lanes: psi = psi_cap,
      dpsi = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(psi, psi_cap, zero, dpsi);
      psi  = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(psi, psi_cap, psi_cap, psi);
    }

    return std::make_pair(psi, dpsi);
  }
};

template<int dim, typename Number>
class WallLawEvaluator
{
public:
  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, scalar>;

  WallLawEvaluator();
  
  scalar 
  get_value(scalar const y_plus) const;

  scalar
  get_gradient(scalar const y_plus) const;

  std::pair<scalar, scalar>
  get_value_and_gradient(scalar const y_plus) const;

private:

  WallLawType wall_law_type;
};

} // namespace ExaDG

#endif
