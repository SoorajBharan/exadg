
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

#ifndef INCLUDE_OPERATORS_VANDERMONDE_MATRIX_H
#define INCLUDE_OPERATORS_VANDERMONDE_MATRIX_H

#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <memory>
#include <deal.II/fe/mapping_q_generic.h>

namespace ExaDG
{

template<int dim, int n_components, typename Number>
class VanderMondeMatrixKernel
{
public:
  typedef dealii::VectorizedArray<Number> scalar;

  VanderMondeMatrixKernel()
  : fe_degree(0), n_1D(0)
  {}

  void initialize(unsigned int const fe_degree_in)
  {
    fe_degree = fe_degree_in;
    n_1D = fe_degree + 1;
    V_1D.reinit(n_1D, n_1D);
    V_1D_inverse.reinit(n_1D, n_1D);

    compute_1D_VDM_matrix();
  }

  void transform_local_to_modal(std::vector<scalar> & cell_data) const
  {
    std::vector<scalar> temp = cell_data;
    apply_1D_matrix_to_tensor(V_1D_inverse,
                              temp,
                              cell_data,
                              0);
    if constexpr (dim >= 2)
    {
      temp = cell_data;
      apply_1D_matrix_to_tensor(V_1D_inverse,
                                temp,
                                cell_data,
                                1);
    }
    if constexpr (dim == 3)
    {
      temp = cell_data;
      apply_1D_matrix_to_tensor(V_1D_inverse,
                                temp,
                                cell_data,
                                2);
    }
  }

  void transform_local_to_nodal(std::vector<scalar> & cell_data) const
  {
    std::vector<scalar> temp = cell_data;
    apply_1D_matrix_to_tensor(V_1D,
                              temp,
                              cell_data,
                              0);

    if constexpr (dim >= 2)
    {
      temp = cell_data;
      apply_1D_matrix_to_tensor(V_1D,
                                temp,
                                cell_data,
                                1);
    }
    if constexpr (dim == 3)
    {
      temp = cell_data;
      apply_1D_matrix_to_tensor(V_1D,
                                temp,
                                cell_data,
                                2);
    }
  }

private:
  void compute_1D_VDM_matrix()
  {
    std::vector<dealii::Polynomials::Polynomial<double>> legendre_polys;
    for (unsigned int p = 0; p <= fe_degree; ++p)
    {
      legendre_polys.push_back(dealii::Polynomials::Legendre(p));
    }

    dealii::QGaussLobatto<1> quad_1d(n_1D);

    for (unsigned int i = 0; i < n_1D; ++i)
    {
      double const x = quad_1d.point(i)[0];
      for (unsigned int j = 0; j < n_1D; ++j)
      {
        V_1D(i, j) = legendre_polys[j].value(x);
      }
    }

    V_1D_inverse.invert(V_1D);
  }

  void apply_1D_matrix_to_tensor(
    dealii::FullMatrix<Number> const & matrix_1D,
    std::vector<scalar> const & src,
    std::vector<scalar> & dst,
    unsigned int const direction) const
  {
    for (auto & val : dst)
    val = dealii::make_vectorized_array<Number>(0.0);

    if constexpr (dim == 1)
    {
      for (unsigned int i = 0; i < n_1D; ++i)
      {
        for (unsigned int m = 0; m < n_1D; ++m)
        {
          dst[i] += matrix_1D(i, m) * src[m];
        }
      }
    }
    else if constexpr (dim == 2)
    {
      for (unsigned int j = 0; j < n_1D; ++j)
      {
        for (unsigned int i = 0; i < n_1D; ++i)
        {
          unsigned int const idx = i + j * n_1D;
          for (unsigned int m = 0; m < n_1D; ++m)
          {
            if (direction == 0)      
            {
              dst[idx] += matrix_1D(i, m) * src[m + j * n_1D];
            }
            else if (direction == 1) 
            {
              dst[idx] += matrix_1D(j, m) * src[i + m * n_1D];
            }
          }
        }
      }
    }
    else if constexpr (dim == 3)
    {
      for (unsigned int k = 0; k < n_1D; ++k)
      {
        for (unsigned int j = 0; j < n_1D; ++j)
        {
          for (unsigned int i = 0; i < n_1D; ++i)
          {
            unsigned int const idx = i + j * n_1D + k * n_1D * n_1D;
            for (unsigned int m = 0; m < n_1D; ++m)
            {
              if (direction == 0)
              {
                dst[idx] += matrix_1D(i, m) * src[m + j * n_1D + k * n_1D * n_1D];
              }
              else if (direction == 1) 
              {
                dst[idx] += matrix_1D(j, m) * src[i + m * n_1D + k * n_1D * n_1D];
              }
              else if (direction == 2) 
              {
                dst[idx] += matrix_1D(k, m) * src[i + j * n_1D + m * n_1D * n_1D];
              }
            }
          }
        }
      }
    }
  }

  unsigned int fe_degree;
  unsigned int n_1D;

  dealii::FullMatrix<Number> V_1D;
  dealii::FullMatrix<Number> V_1D_inverse;
};

template<int dim, int n_components, typename Number>
class VanderMondeMatrixOperator
{
public:
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef dealii::FEEvaluation<dim, -1, 0, n_components, Number> Integrator;

  VanderMondeMatrixOperator();

  void initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                  unsigned int const dof_index,
                  unsigned int const quad_index);

  void transform_to_modal(VectorType const & src,
                          VectorType & dst,
                          unsigned int const cell_id) const;

  void transform_to_nodal(VectorType const & src,
                          VectorType & dst,
                          unsigned int const cell_id) const;

private:
  dealii::MatrixFree<dim, Number> const * matrix_free;
  unsigned int dof_index;
  unsigned int quad_index;

  VanderMondeMatrixKernel<dim, n_components, Number> kernel;
};

} // namespace ExaDG

#endif
