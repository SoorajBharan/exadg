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

#include <exadg/operators/modal_filter.h>

namespace ExaDG
{
template<int dim, int n_components, typename Number>
ModalFilter<dim, n_components, Number>::ModalFilter() 
  : matrix_free(nullptr), dof_index(0), quad_index(0), cut_off_degree(0), n_1D(0)
{}

template<int dim, int n_components, typename Number>
void ModalFilter<dim, n_components, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const dof_index_in,
  unsigned int const quad_index_in,
  unsigned int const cut_off_degree_in,
  Number const gradient_threshold_in)
{
  matrix_free = &matrix_free_in;
  dof_index = dof_index_in;
  quad_index = quad_index_in;
  cut_off_degree = cut_off_degree_in;
  gradient_threshold = gradient_threshold_in;

  unsigned int const fe_degree = matrix_free->get_dof_handler(dof_index).get_fe().degree;
  n_1D = fe_degree + 1;

  VDM_operator.initialize(matrix_free_in, dof_index, quad_index);
}

template<int dim, int n_components, typename Number>
void ModalFilter<dim, n_components, Number>::apply_filter(VectorType & solution) const
{
  requires_ghost_update = false;

  VectorType src;
  matrix_free->cell_loop(&This::cell_loop_filter, this, solution, src, false);

  if (requires_ghost_update)
  {
    solution.update_ghost_values();
  }
}

template<int dim, int n_components, typename Number>
void ModalFilter<dim, n_components, Number>::cell_loop_filter(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  VectorType & dst,
  VectorType const & src,
  Range const & cell_range) const
{
  (void)src;

  Integrator integrator(matrix_free_in, dof_index, quad_index);

  typename Integrator::value_type filter_mask;

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    if (is_selected(cell, dst, filter_mask))
    {
      requires_ghost_update = true;

      VDM_operator.transform_to_modal(dst, modal_vector, cell);

      integrator.reinit(cell);
      integrator.read_dof_values(modal_vector);

      if constexpr (dim == 2)
      {
        for (unsigned int j = 0; j < n_1D; ++j)
        {
          for (unsigned int i = 0; i < n_1D; ++i)
          {
            if (i > cut_off_degree || j > cut_off_degree)
            {
              unsigned int const idx = i + j * n_1D;
              if constexpr (n_components == 1)
              {
                scalar modal_val = integrator.get_dof_value(idx);
                modal_val *= filter_mask;
                integrator.submit_dof_value(modal_val, idx);
              }
              else
            {
                dealii::Tensor<1, n_components, scalar> modal_val = integrator.get_dof_value(idx); 

                for (unsigned int c = 0; c < n_components; ++c)
                {
                  modal_val[c] *= filter_mask[c];
                }
                integrator.submit_dof_value(modal_val, idx);
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
              if (i > cut_off_degree || j > cut_off_degree || k > cut_off_degree)
              {
                unsigned int const idx = i + j * n_1D + k * n_1D * n_1D;
                if constexpr (n_components == 1)
                {
                  scalar modal_val = integrator.get_dof_value(idx);
                  modal_val *= filter_mask;
                  integrator.submit_dof_value(modal_val, idx);
                }
                else
              {
                  dealii::Tensor<1, n_components, scalar> modal_val = integrator.get_dof_value(idx); 

                  for (unsigned int c = 0; c < n_components; ++c)
                  {
                    modal_val[c] *= filter_mask[c];
                  }
                  integrator.submit_dof_value(modal_val, idx);
                }
              }
            }
          }
        }
      }

      integrator.set_dof_values(modal_vector, 0);

      VDM_operator.transform_to_nodal(modal_vector, dst, cell);
    }
  }
}

template<int dim, int n_components, typename Number>
bool ModalFilter<dim, n_components, Number>::is_selected(
  unsigned int const cell_batch_id, 
  VectorType const & solution,
  typename Integrator::value_type & filter_mask) const
{
  Integrator integrator(*matrix_free, dof_index, quad_index);
  integrator.reinit(cell_batch_id);
  integrator.read_dof_values(solution);
  integrator.evaluate(dealii::EvaluationFlags::gradients);

  std::vector<std::vector<double>> max_grad_norm(n_components, std::vector<double>(scalar::size(), 0.0));

  for (unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    auto gradient = integrator.get_gradient(q);
    if constexpr (n_components == 1)
    {
      scalar grad_norm_sq = 0.0;
      for (unsigned int d = 0; d < dim; ++d)
      {
        grad_norm_sq += gradient[d] * gradient[d]; // Only one bracket needed!
      }

      scalar grad_norm = std::sqrt(grad_norm_sq);

      for (unsigned int v = 0; v < scalar::size(); ++v)
      {
        if (grad_norm[v] > max_grad_norm[0][v])
        {
          max_grad_norm[0][v] = grad_norm[v];
        }
      }
    }
    else
  {
      for(unsigned int c = 0; c < n_components; ++c)
      {
        scalar grad_norm_sq = 0.0;
        for (unsigned int d = 0; d < dim; ++d)
        {
          grad_norm_sq += gradient[c][d] * gradient[c][d];
        }

        scalar grad_norm = std::sqrt(grad_norm_sq);

        for (unsigned int v = 0; v < scalar::size(); ++v)
        {
          if (grad_norm[v] > max_grad_norm[c][v])
          {
            max_grad_norm[c][v] = grad_norm[v];
          }
        }
      }
    }
  }

  bool any_cells_selected = false;

  if constexpr (n_components == 1)
  {
    for (unsigned int v = 0; v < scalar::size(); ++v)
    {
      if (max_grad_norm[0][v] > gradient_threshold)
      {
        filter_mask[v] = 0.0; // Only one bracket!
        any_cells_selected = true;
      }
      else
    {
        filter_mask[v] = 1.0; 
      }
    }
  }
  else
{
    for (unsigned int c = 0; c < n_components; ++c)
    {
      for (unsigned int v = 0; v < scalar::size(); ++v)
      {
        if (max_grad_norm[c][v] > gradient_threshold)
        {
          filter_mask[c][v] = 0.0; // Gradient too high -> zero out the mode
          any_cells_selected = true;
        }
        else
      {
          filter_mask[c][v] = 1.0; // Gradient is fine -> multiply by 1 to keep original mode
        }
      }
    }
  }

  return any_cells_selected;
}

template class ModalFilter<2, 1, float>;
template class ModalFilter<2, 1, double>;
template class ModalFilter<3, 1, float>;
template class ModalFilter<3, 1, double>;

template class ModalFilter<2, 2, float>;
template class ModalFilter<2, 2, double>;
template class ModalFilter<3, 2, float>;
template class ModalFilter<3, 2, double>;

template class ModalFilter<2, 3, float>;
template class ModalFilter<2, 3, double>;
template class ModalFilter<3, 3, float>;
template class ModalFilter<3, 3, double>;

} // namespace ExaDG
