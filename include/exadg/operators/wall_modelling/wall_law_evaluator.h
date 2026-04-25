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

template<typename Number, typename Functor>
dealii::VectorizedArray<Number>
solve_newton_raphson(
  dealii::VectorizedArray<Number> const y_plus,
  dealii::VectorizedArray<Number> const initial_guess,
  Functor const & function,
  unsigned int const max_iterations = 100)
{
  using scalar = dealii::VectorizedArray<Number>;
  scalar u_plus = initial_guess;

  for (unsigned int iter = 0; iter < max_iterations; ++iter)
  {
    std::pair<scalar, scalar> values = function(u_plus);

    u_plus -= (values.first - y_plus) / values.second;
  }
  return u_plus;
}

struct SpaldingLaw
{
  double kappa = 0.41;
  double beta  = 5.2;
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

    // \frac{dy^+}{du^+} = \left[ 1 + e^{-\beta \kappa} \kappa \left( e^{u^+ \kappa} - 1 - \frac{(u^+ \kappa)^2}{2!} - \frac{(u^+ \kappa)^3}{3!} - \frac{(u^+ \kappa)^4}{4!} \right) \right]^{-1}
    scalar dy_plus = 1.0 + expKB * term_2;

    return std::make_pair(y_plus, dy_plus);
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

private:

  WallLawType wall_law_type;
};

} // namespace ExaDG

#endif
