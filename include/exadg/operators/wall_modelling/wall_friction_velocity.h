
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

#ifndef INCLUDE_EXADG_OPERATORS_WALL_MODELLING_WALL_FRICTION_VELOCITY_H_
#define INCLUDE_EXADG_OPERATORS_WALL_MODELLING_WALL_FRICTION_VELOCITY_H_

// C/C++
#include <limits>
#include <vector>

// deal.II
#include <deal.II/base/point.h>

namespace ExaDG
{
/*
   * Friction velocity u_tau at the wall nodes of the wall enrichment, on every process (all wall
   * nodes of the domain). Written by FunctionEnrichment after every u_tau update, read by other
   * solvers, e.g. the wall boundary condition of the RANS equations: u_tau of the nearest wall node,
   * the same rule as for the enrichment function psi.
   */
template<int dim>
class WallFrictionVelocity
{
public:
  void
  set(std::vector<dealii::Point<dim>> const & points_in, std::vector<double> const & values_in)
  {
    points = points_in;
    values = values_in;
  }

  // false before the first u_tau update
  bool
  is_set() const
  {
    return not values.empty();
  }

  // u_tau of the nearest wall node, 0 if not set
  double
  value(dealii::Point<dim> const & p) const
  {
    double min_distance = std::numeric_limits<double>::max();
    double u_tau        = 0.0;
    for(unsigned int i = 0; i < points.size(); ++i)
    {
      double const distance = p.distance_square(points[i]);
      if(distance < min_distance)
      {
        min_distance = distance;
        u_tau        = values[i];
      }
    }
    return u_tau;
  }

private:
  std::vector<dealii::Point<dim>> points;
  std::vector<double>             values;
};

  } // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_WALL_MODELLING_WALL_FRICTION_VELOCITY_H_ */
