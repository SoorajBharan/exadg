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

#include <exadg/operators/vandermonde_matrix.h>
#include <deal.II/base/quadrature_lib.h>

namespace ExaDG
{
template<int dim, int n_components, typename Number>
VanderMondeMatrixOperator<dim, n_components, Number>::VanderMondeMatrixOperator() 
  : matrix_free(nullptr), dof_index(0), quad_index(0)
{}

template<int dim, int n_components, typename Number>
void VanderMondeMatrixOperator<dim, n_components, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const dof_index_in,
  unsigned int const quad_index_in)
{
  matrix_free = &matrix_free_in;
  dof_index = dof_index_in;
  quad_index = quad_index_in; 

  unsigned int const fe_degree = matrix_free->get_dof_handler(dof_index).get_fe().degree;
  kernel.initialize(fe_degree);
}

template<int dim, int n_components, typename Number>
void VanderMondeMatrixOperator<dim, n_components, Number>::transform_to_modal(
  VectorType const & src, 
  VectorType & dst, 
  unsigned int const cell_id) const
{
  Integrator integrator(*matrix_free, dof_index, quad_index);
  std::vector<scalar> local_data(integrator.tensor_dofs_per_cell);

  integrator.reinit(cell_id);
  integrator.read_dof_values(src);

  if constexpr (n_components == 1)
  {
    std::vector<scalar> local_data(integrator.tensor_dofs_per_cell);
    for(unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
    {
      local_data[i] = integrator.get_dof_value(i);
    }

    kernel.transform_local_to_modal(local_data);

    for(unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
    {
      integrator.submit_dof_value(local_data[i], i);
    }
  }
  else
{
    std::vector<dealii::Tensor<1, n_components, scalar>> all_components(integrator.tensor_dofs_per_cell);
    for (unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
    {
      all_components[i] = integrator.get_dof_value(i);
    }

    std::vector<scalar> local_data(integrator.tensor_dofs_per_cell);
    for (unsigned int c = 0; c < n_components; ++c)
    {
      for(unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
      {
        local_data[i] = all_components[i][c];
      }

      kernel.transform_local_to_modal(local_data);

      for (unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
      {
        all_components[i][c] = local_data[i];
      }

      for(unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
      {
        integrator.submit_dof_value(all_components[i], i);
      }
    }
  }

  integrator.set_dof_values(dst, 0);
}

template<int dim, int n_components, typename Number>
void VanderMondeMatrixOperator<dim, n_components, Number>::transform_to_nodal(
  VectorType const & src, 
  VectorType & dst, 
  unsigned int const cell_id) const
{
  Integrator integrator(*matrix_free, dof_index, quad_index);
  std::vector<scalar> local_data(integrator.tensor_dofs_per_cell);

  integrator.reinit(cell_id);
  integrator.read_dof_values(src);

  if constexpr (n_components == 1)
  {
    std::vector<scalar> local_data(integrator.tensor_dofs_per_cell);
    for(unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
    {
      local_data[i] = integrator.get_dof_value(i);
    }

    kernel.transform_local_to_nodal(local_data);

    for(unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
    {
      integrator.submit_dof_value(local_data[i], i);
    }
  }
  else
{
    std::vector<dealii::Tensor<1, n_components, scalar>> all_components(integrator.tensor_dofs_per_cell);
    for (unsigned int i = 0; i < n_components; ++i)
    {
      all_components[i] = integrator.get_dof_value(i);
    }

    std::vector<scalar> local_data(integrator.tensor_dofs_per_cell);
    for (unsigned int c = 0; c < n_components; ++c)
    {
      for(unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
      {
        local_data[i] = all_components[i][c];
      }

      kernel.transform_local_to_nodal(local_data);

      for (unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
      {
        all_components[i][c] = local_data[i];
      }

      for(unsigned int i = 0; i < integrator.tensor_dofs_per_cell; ++i)
      {
        integrator.submit_dof_value(all_components[i], i);
      }
    }
  }

  integrator.set_dof_values(dst, 0);
}

template class VanderMondeMatrixOperator<2, 1, float>;
template class VanderMondeMatrixOperator<2, 1, double>;
template class VanderMondeMatrixOperator<3, 1, float>;
template class VanderMondeMatrixOperator<3, 1, double>;

template class VanderMondeMatrixOperator<2, 2, float>;
template class VanderMondeMatrixOperator<2, 2, double>;
template class VanderMondeMatrixOperator<3, 2, float>;
template class VanderMondeMatrixOperator<3, 2, double>;

template class VanderMondeMatrixOperator<2, 3, float>;
template class VanderMondeMatrixOperator<2, 3, double>;
template class VanderMondeMatrixOperator<3, 3, float>;
template class VanderMondeMatrixOperator<3, 3, double>;
} // namespace ExaDG
