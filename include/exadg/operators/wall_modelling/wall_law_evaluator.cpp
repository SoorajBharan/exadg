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

#include "wall_law_evaluator.h"
#include <exadg/operators/wall_modelling/wall_law_evaluator.h>

namespace ExaDG
{
template<int dim, typename Number>
WallLawEvaluator<dim, Number>::WallLawEvaluator()
: wall_law_type(WallLawType::Spalding)
{}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
WallLawEvaluator<dim, Number>::get_value(
  scalar const y_plus) const
{
  if(wall_law_type == WallLawType::Spalding)
  {
    SpaldingLaw spalding_law;
    return solve_newton_raphson(y_plus, y_plus, spalding_law, 10);
  }
  else
{
    AssertThrow(false,
                dealii::ExcMessage("Not Implemented"));
    return scalar();
  }
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
WallLawEvaluator<dim, Number>::get_gradient(
  scalar const y_plus) const
{
  if(wall_law_type == WallLawType::Spalding)
  {
    SpaldingLaw spalding_law;

    scalar u_plus = solve_newton_raphson(y_plus, y_plus, spalding_law, 10);

    scalar dy_du_plus = spalding_law(u_plus).second;

    scalar du_dy_plus = 1.0 / dy_du_plus;

    return du_dy_plus;
  }
  else
{
    AssertThrow(false,
                dealii::ExcMessage("Not Implemented"));
    return scalar();
  }
}

template class WallLawEvaluator<2, double>;
template class WallLawEvaluator<3, double>;
template class WallLawEvaluator<2, float>;
template class WallLawEvaluator<3, float>;

} // namespace ExaDG
