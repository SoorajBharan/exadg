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

#include <exadg/convection_diffusion/spatial_discretization/turbulence_model.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
TurbulenceModel<dim, Number>::TurbulenceModel()
  : dealii::Subscriptor(),
  quad_index(0),
  dof_index(0),
  matrix_free(nullptr)
{
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                            unsigned int const                      dof_index_in,
                                            unsigned int const                      quad_index_in)
{
  matrix_free = &matrix_free_in;

  dof_index  = dof_index_in;
  quad_index = quad_index_in;
}

template class TurbulenceModel<2, float>;
template class TurbulenceModel<2, double>;
template class TurbulenceModel<3, float>;
template class TurbulenceModel<3, double>;

} // namespace RANS
} // namespace ExaDG
