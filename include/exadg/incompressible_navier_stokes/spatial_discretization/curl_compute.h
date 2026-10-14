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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CURL_COMPUTE_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CURL_COMPUTE_H_

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename FEEval>
struct CurlCompute
{
  static dealii::Tensor<1, dim, dealii::VectorizedArray<typename FEEval::number_type>>
  compute(FEEval & fe_eval, unsigned int const q_point)
  {
    return fe_eval.get_curl(q_point);
  }
};

// use partial specialization of templates to handle the 2-dimensional case
template<typename FEEval>
struct CurlCompute<2, FEEval>
{
  static dealii::Tensor<1, 2, dealii::VectorizedArray<typename FEEval::number_type>>
  compute(FEEval & fe_eval, unsigned int const q_point)
  {
    dealii::Tensor<1, 2, dealii::VectorizedArray<typename FEEval::number_type>> curl;
    /*
     * fe_eval = / phi \   _____\   grad(fe_eval) = / d(phi)/dx1   d(phi)/dx2 \
     *           \  0  /        /                   \     0             0     /
     */
    dealii::Tensor<2, 2, dealii::VectorizedArray<typename FEEval::number_type>> temp =
      fe_eval.get_gradient(q_point);
    /*
     *         __    /  0  \     /   d(phi)/dx2 \
     *  curl = \/ X  |  0  |  =  | - d(phi)/dx1 |
     *               \ phi /     \      0       /
     */
    curl[0] = temp[0][1];  //   d(phi)/dx2
    curl[1] = -temp[0][0]; // - d(phi)/dx1
    return curl;
  }
};

template<int dim, typename Number>
dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
vorticity_from_gradient(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>
                        const & G)
{
  dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> omega;
  if constexpr(dim == 2)
  {
    omega[0] = G[1][0] - G[0][1];
  }
  else
{
    omega[0] = G[2][1] - G[1][2];
    omega[1] = G[0][2] - G[2][0];
    omega[2] = G[1][0] - G[0][1];
  }
  return omega;
}

template<int dim, typename Number>
dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>
curl_of_vorticity_from_gradient(dealii::Tensor<2, dim,
                                dealii::VectorizedArray<Number>> const & G)
{
  dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> curl;
  if constexpr(dim == 2)
  {
    curl[0] = G[0][1];
    curl[1] = -G[0][0];
  }
  else
{
    curl = vorticity_from_gradient<dim, Number>(G);
  }
  return curl;
}

} // namespace IncNS
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_CURL_COMPUTE_H_ */
