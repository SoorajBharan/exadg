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

#ifndef INCLUDE_OPERATORS_MODAL_FILTER_H
#define INCLUDE_OPERATORS_MODAL_FILTER_H

#include <exadg/operators/vandermonde_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <vector>

namespace ExaDG
{
template<int dim, int n_components, typename Number>
class ModalFilter
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::FEEvaluation<dim, -1, 0, n_components, Number> Integrator;

  typedef ModalFilter<dim, n_components, Number> This;
  typedef std::pair<unsigned int, unsigned int> Range;

  ModalFilter();

  void initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                  unsigned int const dof_index_in,
                  unsigned int const quad_index_in,
                  unsigned int const cut_off_degree_in,
                  Number const gradient_threshold_in);

  void apply_filter(VectorType & solution) const;

private:
  bool is_selected(unsigned int const cell_batch_id,
                   VectorType const & solution,
                   typename Integrator::value_type & filter_mask) const;

  void cell_loop_filter(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                        VectorType & dst,
                        VectorType const & src,
                        Range const & cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;
  unsigned int dof_index;
  unsigned int quad_index;
  
  unsigned int cut_off_degree;
  unsigned int n_1D;
  Number gradient_threshold;

  VanderMondeMatrixOperator<dim, n_components, Number> VDM_operator;

  mutable VectorType modal_vector;
  mutable std::atomic<bool> requires_ghost_update;
};

} // namepsace ExaDG

#endif
