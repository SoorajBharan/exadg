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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_COMBINED_OPERATOR_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_COMBINED_OPERATOR_H_

#include <exadg/convection_diffusion/spatial_discretization/operators/convective_operator.h>
#include <exadg/convection_diffusion/spatial_discretization/operators/diffusive_operator.h>
#include <exadg/operators/mass_kernel.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim>
struct CombinedOperatorData : public OperatorBaseData
{
  CombinedOperatorData()
    : OperatorBaseData(),
      unsteady_problem(false),
      convective_problem(false),
      diffusive_problem(false)
  {
  }

  bool unsteady_problem;
  bool convective_problem;
  bool diffusive_problem;

  Operators::ConvectiveKernelData<dim> convective_kernel_data;
  Operators::DiffusiveKernelData       diffusive_kernel_data;

  std::shared_ptr<BoundaryDescriptor<dim> const> bc;
};

template<int dim, typename Number>
class CombinedOperator : public OperatorBase<dim, Number, 1>
{
public:
  typedef Number value_type;

private:
  typedef OperatorBase<dim, Number, 1> Base;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef typename Base::VectorType VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

public:
  CombinedOperator();

  /**
   * This function creates own kernels for the different terms of the combined PDE operator. This
   * function is typically called when using this operator as the PDE operator in multigrid.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             CombinedOperatorData<dim> const &         data);

  void
  initialize(dealii::MatrixFree<dim, Number> const &                   matrix_free,
             dealii::AffineConstraints<Number> const &                 affine_constraints,
             CombinedOperatorData<dim> const &                         data,
             std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel,
             std::shared_ptr<Operators::DiffusiveKernel<dim, Number>>  diffusive_kernel);

  CombinedOperatorData<dim> const &
  get_data() const;

  void
  update_after_grid_motion();

  dealii::LinearAlgebra::distributed::Vector<Number> const &
  get_velocity() const;

  void
  set_velocity_copy(VectorType const & velocity) const;

  void
  set_velocity_ptr(VectorType const & velocity) const;

  Number
  get_scaling_factor_mass_operator() const;

  void
  set_scaling_factor_mass_operator(Number const & scaling_factor);

  // CellDirichlet: penalty lambda = cell_dirichlet_penalty_factor / dt of the current time step
  void
  set_cell_dirichlet_penalty(double const penalty);

  // CellDirichlet: dst += lambda (g(t), v) on the CellDirichlet cells (right-hand side part of the
  // penalty; the part lambda (u, v) is in do_cell_integral()); t = time of the last set_time()
  void
  rhs_add_cell_dirichlet(VectorType & dst) const;

private:
  void
  reinit_cell_derived(IntegratorCell & integrator, unsigned int const cell) const final;

  void
  reinit_face_derived(IntegratorFace &   integrator_m,
                      IntegratorFace &   integrator_p,
                      unsigned int const face) const final;

  void
  reinit_boundary_face_derived(IntegratorFace & integrator_m, unsigned int const face) const final;

  void
  reinit_face_cell_based_derived(IntegratorFace &                 integrator_m,
                                 IntegratorFace &                 integrator_p,
                                 unsigned int const               cell,
                                 unsigned int const               face,
                                 dealii::types::boundary_id const boundary_id) const final;

  void
  do_cell_integral(IntegratorCell & integrator) const final;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_boundary_integral(IntegratorFace &                   integrator_m,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const final;

  // TODO can be removed later once matrix-free evaluation allows accessing neighboring data for
  // cell-based face loops
  void
  do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                  IntegratorFace & integrator_p) const final;

  // fills cell_dirichlet_mask(_per_id) from the CellDirichlet boundaries of operator_data.bc
  void
  setup_cell_dirichlet();

  // cell loop of rhs_add_cell_dirichlet(): + lambda (g, v) on the CellDirichlet cells
  void
  cell_loop_rhs_cell_dirichlet(dealii::MatrixFree<dim, Number> const &       matrix_free,
                               VectorType &                                  dst,
                               VectorType const &                            src,
                               std::pair<unsigned int, unsigned int> const & range) const;

  CombinedOperatorData<dim> operator_data;

  std::shared_ptr<MassKernel<dim, Number>>                  mass_kernel;
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel;
  std::shared_ptr<Operators::DiffusiveKernel<dim, Number>>  diffusive_kernel;

  double scaling_factor_mass;

  // CellDirichlet (penalty lambda (u - g) v on the cells with a face on a CellDirichlet boundary):
  // lambda = cell_dirichlet_penalty_factor / dt, set every time step by the PDE operator
  double cell_dirichlet_penalty;

  // per cell batch: 1 in the SIMD lanes of the CellDirichlet cells, 0 elsewhere
  // (empty: this operator has no CellDirichlet boundary)
  std::vector<scalar> cell_dirichlet_mask;

  // the same mask per CellDirichlet boundary ID (which function g is imposed in which cell)
  std::map<dealii::types::boundary_id, std::vector<scalar>> cell_dirichlet_mask_per_id;

  // mask of the cell batch of the last reinit_cell_derived(), read in do_cell_integral()
  mutable scalar current_cell_dirichlet_mask;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_COMBINED_OPERATOR_H_ \
        */
