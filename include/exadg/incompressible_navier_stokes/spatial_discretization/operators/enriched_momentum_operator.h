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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_ENRICHED_MOMENTUM_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_ENRICHED_MOMENTUM_OPERATOR_H_

#include <deal.II/base/aligned_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator.h>
#include <exadg/operators/wall_modelling/fe_enriched_evaluation.h>
#include <exadg/operators/wall_modelling/fe_face_enriched_evaluation.h>
#include <exadg/operators/wall_modelling/function_enrichment.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/turbulence_model.h>
#include <exadg/operators/variable_coefficients.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * Momentum operator (mass + viscous) for the wall-enriched velocity
 *   u_h = U_bar + psi * U_tilde
 *
 *   dst.block(0) = K_bar_bar   * U_bar + K_bar_tilde   * U_tilde   (global layout)
 *   dst.block(1) = K_tilde_bar * U_bar + K_tilde_tilde * U_tilde   (enrichment layout)
 *
 * K_bar_bar * U_bar comes from the existing MomentumOperator (whole domain).
 * All other terms are computed on matrix_free_en:
 *   - cell integrals on enriched cells (index 1)
 *   - face integrals on faces with at least one enriched side
 *     (other side enriched = index 1, or halo = index 2)
 *   - boundary face integrals on enriched cells
 *
 * Supported: Laplace formulation, constant viscosity, no convective term
 * (the case of solve_linear_momentum_equation).
 */
template<int dim, typename Number>
class EnrichedMomentumOperator
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

  EnrichedMomentumOperator();

  void
  initialize(MomentumOperator<dim, Number> const &            momentum_operator_in,
             std::shared_ptr<FunctionEnrichment<dim, Number>> function_enrichment_in);

  // Recompute the interior penalty parameter on matrix_free_en (call after mesh changes).
  void
  update_penalty_parameter();

  void
  update_enriched_penalty_parameter() const;

  // false (default): polynomial tau_IP on all faces, Krank et al. (2018) Eq. (35)-(36)
  void
  set_enrichment_aware_penalty(bool const flag);

  void
  set_wall_penalty_factor(double const factor);

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const;

  // inhomogeneous Dirichlet data (inflow): U_tilde row, and the penalty correction of the
  // U_bar row (enrichment-aware tau). dst_bar in GLOBAL layout.
  void
  rhs_add_boundary_data(VectorType & dst_bar, VectorType & dst_tilde, double const time) const;

  void
  initialize_variable_viscosity(dealii::Mapping<dim> const & mapping);

  void
  update_viscosity(VectorType const & velocity_total,
                   TurbulenceModel<dim, Number> const & turbulence_model) const;

  scalar
  get_viscosity_cell(unsigned int const cell,
                     unsigned int const q) const;

  scalar
  get_viscosity_boundary_face(unsigned int const face,
                              unsigned int const q) const;

  VectorType const &
  get_cell_viscosity() const
  {
    return cell_viscosity;
  }

private:
  // One pass with u_h = U_bar + psi U_tilde: dst_tilde gets the psi v_tilde rows  of u_h,
  // *dst_bar (not nullptr) the v_bar rows of psi U_tilde only (K_bar_tilde   U_tilde);
  // K_bar_bar U_bar is produced by MomentumOperator.
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

  void
  boundary_face_loop(VectorType const & src_bar,
                     VectorType const & src_tilde,
                     VectorType *       dst_bar,
                     VectorType &       dst_tilde) const;

  template<typename IntegratorM, typename IntegratorP>
  void
  inner_face_flux(IntegratorM & phi_m,
                  IntegratorP & phi_p,
                  scalar const & tau,
                  unsigned int const face) const;

  scalar
  get_viscosity_interior_face(unsigned int const face,
                              unsigned int const q) const;

  // overloads: plain side (halo) and enriched side
  void
  gather(FaceIntegratorU & phi, VectorType const & src_bar, VectorType const & src_tilde) const;
  void
  gather(EnrichedFaceIntegrator & phi, VectorType const & src_bar, VectorType const & src_tilde) const;

  void
  scatter(FaceIntegratorU & phi, VectorType * dst_bar, VectorType & dst_tilde) const;
  void
  scatter(EnrichedFaceIntegrator & phi, VectorType * dst_bar, VectorType & dst_tilde) const;

  scalar
  read_penalty(FaceIntegratorU & phi) const;

  // tau of one side: max(polynomial tau, enrichment tau)
  scalar
  read_penalty_total(FaceIntegratorU & phi) const;
  scalar
  read_penalty_total(EnrichedFaceIntegrator & phi) const;

  // polynomial tau of one side (as in ExaDG's viscous kernel)
  scalar
  read_penalty_polynomial(FaceIntegratorU & phi) const;
  scalar
  read_penalty_polynomial(EnrichedFaceIntegrator & phi) const;

  // Pass C: nu (tau_F - tau_F^poly) [u_bar].[v_bar] on all faces of enriched cells
  void
  penalty_correction_loop(VectorType const & u_bar_shadow, VectorType & dst_bar_shadow) const;

  scalar
  read_penalty(EnrichedFaceIntegrator & phi) const;

  // one submit of value flux and gradient flux per side and quadrature point
  void
  submit(FaceIntegratorU &  phi,
         vector const &     value_flux,
         tensor const &     gradient_flux,
         unsigned int const q) const;
  void
  submit(EnrichedFaceIntegrator & phi,
         vector const &           value_flux,
         tensor const &           gradient_flux,
         unsigned int const       q) const;

  MomentumOperator<dim, Number> const *            momentum_operator;
  std::shared_ptr<FunctionEnrichment<dim, Number>> function_enrichment;

  // surface/volume per cell batch of matrix_free_en (incl. ghost batches)
  dealii::AlignedVector<scalar> penalty_parameter;
  // enrichment tau per cell batch of matrix_free_en (incl. ghost batches); 0 on non-enriched cells
  mutable dealii::AlignedVector<scalar> penalty_enrichment;
  Number                        wall_penalty_factor;
  Number                        penalty_factor;

  bool enrichment_aware_penalty;
  bool viscosity_is_variable;
  double                               viscosity;
  mutable VariableCoefficients<scalar> viscosity_coefficients;

  dealii::AlignedVector<scalar> filter_width;

  mutable VectorType cell_viscosity;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_ENRICHED_MOMENTUM_OPERATOR_H_ */
