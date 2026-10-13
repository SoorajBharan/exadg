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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_ENRICHED_PROJECTION_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_ENRICHED_PROJECTION_OPERATOR_H_

#include <deal.II/base/aligned_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/projection_operator.h>
#include <exadg/operators/wall_modelling/fe_enriched_evaluation.h>
#include <exadg/operators/wall_modelling/fe_face_enriched_evaluation.h>
#include <exadg/operators/wall_modelling/function_enrichment.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * Penalty step operator (mass + dt * (divergence penalty + continuity penalty)) for the
 * wall-enriched velocity u_h = U_bar + psi * U_tilde:
 *
 *   dst.block(0) = K_bar_bar   U_bar + K_bar_tilde   U_tilde   (global layout)
 *   dst.block(1) = K_tilde_bar U_bar + K_tilde_tilde U_tilde   (enrichment layout)
 *
 * K_bar_bar U_bar comes from ExaDG's ProjectionOperator; all coupling terms are computed on
 * matrix_free_en. Continuity penalty: normal components, no boundary data.
 */
template<int dim, typename Number>
class EnrichedProjectionOperator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;
  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, scalar>                          vector;
  typedef dealii::Tensor<2, dim, scalar>                          tensor;

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number>     CellIntegratorU;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, dim, Number> FaceIntegratorU;

  typedef FEEnrichedEvaluation<dim, Number, CellIntegratorU>     EnrichedCellIntegrator;
  typedef FEFaceEnrichedEvaluation<dim, Number, FaceIntegratorU> EnrichedFaceIntegrator;

  EnrichedProjectionOperator();

  void
  initialize(ProjectionOperator<dim, Number> const &          projection_operator_in,
             std::shared_ptr<FunctionEnrichment<dim, Number>> function_enrichment_in,
             unsigned int const                               degree_velocity_in,
             bool const                                       use_divergence_penalty_in,
             bool const                                       use_continuity_penalty_in,
             double const                                     divergence_penalty_factor_in,
             double const                                     continuity_penalty_factor_in);

  // penalty parameters from the full velocity (global layout) and the time step size
  void
  update(VectorType const & velocity_total, double const time_step_size_in);

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const;


  // divergence penalty parameter per cell batch, multiplied by the time step size
  // (for the block-Jacobi preconditioner)
  dealii::AlignedVector<scalar>
  get_tau_divergence_times_dt() const
  {
    dealii::AlignedVector<scalar> result(tau_divergence);
    for(unsigned int i = 0; i < result.size(); ++i)
      result[i] *= time_step_size;
    return result;
  }

  bool
  divergence_penalty_used() const
  {
    return use_divergence_penalty;
  }

private:
  // dst_bar == nullptr: U_bar rows are not written (they come from ProjectionOperator)
  void
  cell_loop(VectorType const & src_bar,
            VectorType const & src_tilde,
            VectorType *       dst_bar,
            VectorType &       dst_tilde) const;

  void
  inner_face_loop(VectorType const & src_bar,
                  VectorType const & src_tilde,
                  VectorType *       dst_bar,
                  VectorType &       dst_tilde) const;

  ProjectionOperator<dim, Number> const *          projection_operator;
  std::shared_ptr<FunctionEnrichment<dim, Number>> function_enrichment;

  unsigned int degree_velocity;
  bool         use_divergence_penalty;
  bool         use_continuity_penalty;
  Number       divergence_penalty_factor;
  Number       continuity_penalty_factor;
  Number       time_step_size;

  // per cell batch of matrix_free_en (incl. ghost batches)
  dealii::AlignedVector<scalar> tau_divergence;
  dealii::AlignedVector<scalar> tau_continuity;
};

} // namespace IncNS
} // namespace ExaDG

#endif
