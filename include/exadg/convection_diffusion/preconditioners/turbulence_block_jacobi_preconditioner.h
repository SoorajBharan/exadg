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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_PRECONDITIONERS_TURBULENCE_BLOCK_JACOBI_PRECONDITIONER_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_PRECONDITIONERS_TURBULENCE_BLOCK_JACOBI_PRECONDITIONER_H_

// ExaDG
#include <exadg/convection_diffusion/spatial_discretization/operators/combined_operator.h>
#include <exadg/convection_diffusion/spatial_discretization/operators/turbulence_source_operator.h>
#include <exadg/solvers_and_preconditioners/preconditioners/elementwise_preconditioners.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>
#include <exadg/solvers_and_preconditioners/solvers/wrapper_elementwise_solvers.h>

namespace ExaDG
{
namespace ConvDiff
{
/*
 * Elementwise operator of variable "block" for the matrix-free block Jacobi preconditioner:
 * element block of the combined operator (mass, convection, molecular diffusion: cell and own
 * faces) plus the diagonal block of the turbulence Newton_Implicit cell terms. Same interface as
 * Elementwise::OperatorBase, as required by Elementwise::IterativeSolver.
 */
template<int dim, typename Number>
class TurbulenceElementwiseOperator
{
public:
  typedef dealii::VectorizedArray<Number> scalar;

  TurbulenceElementwiseOperator(CombinedOperator<dim, Number> const &         combined_operator_in,
                                TurbulenceSourceOperator<dim, Number> const & turbulence_operator_in,
                                unsigned int const                            block_in)
    : combined_operator(combined_operator_in),
      turbulence_operator(turbulence_operator_in),
      block(block_in),
      current_cell(0),
      problem_size(1)
  {
  }

  dealii::MatrixFree<dim, Number> const &
  get_matrix_free() const
  {
    return combined_operator.get_matrix_free();
  }

  unsigned int
  get_dof_index() const
  {
    return combined_operator.get_dof_index();
  }

  unsigned int
  get_quad_index() const
  {
    return combined_operator.get_quad_index();
  }

  void
  setup(unsigned int const cell, unsigned int const size)
  {
    current_cell = cell;
    problem_size = size;
  }

  unsigned int
  get_problem_size() const
  {
    return problem_size;
  }

  // dst = (combined + turbulence) element block applied to src, on cell batch current_cell
  void
  vmult(scalar * dst, scalar const * src) const
  {
    Elementwise::vector_init(dst, problem_size);

    combined_operator.apply_add_block_diagonal_elementwise(current_cell, dst, src, problem_size);

    turbulence_operator.apply_add_block_diagonal_elementwise(block, current_cell, dst, src);
  }

private:
  CombinedOperator<dim, Number> const &         combined_operator;
  TurbulenceSourceOperator<dim, Number> const & turbulence_operator;

  // variable (block of the BlockVector) of this preconditioner
  unsigned int const block;

  unsigned int current_cell;
  unsigned int problem_size;
};

/*
 * Matrix-free block Jacobi preconditioner of variable "block" that includes the turbulence
 * Newton_Implicit cell terms. No matrices are stored: each element problem is solved
 * iteratively (Elementwise::IterativeSolver) with TurbulenceElementwiseOperator. The turbulence
 * terms always use the current linearization point (stored by evaluate_add_newton_explicit()).
 */
template<int dim, typename Number>
class TurbulenceBlockJacobiPreconditioner : public PreconditionerBase<Number>
{
public:
  typedef typename PreconditionerBase<Number>::VectorType VectorType;

  typedef TurbulenceElementwiseOperator<dim, Number>                        ElementwiseOperator;
  typedef Elementwise::PreconditionerBase<dealii::VectorizedArray<Number>> ElementwisePreconditioner;
  typedef Elementwise::IterativeSolver<dim, 1, Number, ElementwiseOperator, ElementwisePreconditioner>
    ElementwiseSolver;

  TurbulenceBlockJacobiPreconditioner(
    CombinedOperator<dim, Number> const &         combined_operator,
    TurbulenceSourceOperator<dim, Number> const & turbulence_operator,
    unsigned int const                            block,
    Elementwise::Preconditioner const             preconditioner_type,
    Elementwise::IterativeSolverData const &      solver_data)
  {
    elementwise_operator =
      std::make_shared<ElementwiseOperator>(combined_operator, turbulence_operator, block);

    dealii::MatrixFree<dim, Number> const & matrix_free = combined_operator.get_matrix_free();

    if(preconditioner_type == Elementwise::Preconditioner::None)
    {
      elementwise_preconditioner =
        std::make_shared<Elementwise::PreconditionerIdentity<dealii::VectorizedArray<Number>>>(
          matrix_free.get_dofs_per_cell(combined_operator.get_dof_index()));
    }
    else if(preconditioner_type == Elementwise::Preconditioner::InverseMassMatrix)
    {
      elementwise_preconditioner =
        std::make_shared<Elementwise::InverseMassPreconditioner<dim, 1, Number>>(
          matrix_free, combined_operator.get_dof_index(), combined_operator.get_quad_index());
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage("Turbulence block Jacobi: only None and InverseMassMatrix are "
                                     "implemented as elementwise preconditioners (PointJacobi would "
                                     "need the diagonal including the turbulence terms)."));
    }

    elementwise_solver =
      std::make_shared<ElementwiseSolver>(*elementwise_operator, *elementwise_preconditioner, solver_data);

    this->update_needed = false;
  }

  // nothing is stored: only the elementwise preconditioner (inverse mass) may need an update
  void
  update() final
  {
    elementwise_preconditioner->update();

    this->update_needed = false;
  }

  // dst = approximate inverse of the element blocks applied to src (one variable)
  void
  vmult(VectorType & dst, VectorType const & src) const final
  {
    elementwise_solver->solve(dst, src);
  }

private:
  std::shared_ptr<ElementwiseOperator>       elementwise_operator;
  std::shared_ptr<ElementwisePreconditioner> elementwise_preconditioner;
  std::shared_ptr<ElementwiseSolver>         elementwise_solver;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_PRECONDITIONERS_TURBULENCE_BLOCK_JACOBI_PRECONDITIONER_H_ */
