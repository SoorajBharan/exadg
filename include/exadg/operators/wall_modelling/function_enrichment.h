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
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_OPERATORS_WALL_MODELLING_FUNCTION_ENRICHMENT_H_
#define INCLUDE_OPERATORS_WALL_MODELLING_FUNCTION_ENRICHMENT_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/wall_modelling/wall_law_evaluator.h>
#include <exadg/operators/finite_element.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <map>
#include <vector>

namespace ExaDG
{

template<int dim, typename Number>
class FunctionEnrichment
{
public:
  typedef FunctionEnrichment<dim, Number> This;
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<1, dim, scalar> vector;
  typedef dealii::Tensor<2, dim, scalar> tensor;

  // ──  construct ─────────────────────────────────────────────────
  FunctionEnrichment(double kinematic_viscosity_in);

  // ──  build CG DoFHandler ──────────────────────────────────────
  /**
   * Create a Q1-continuous DoFHandler on @p triangulation.
   * Must be called before initialize().
   */
  void
  initialize_dofs(dealii::Triangulation<dim> const & triangulation,
                  unsigned int fe_degree_en);

  // ──  link to MatrixFree and allocate vectors ──────────────────
  /**
   * Store the MatrixFree object and pre-allocate wall_distance and
   * friction_velocity in the CG layout given by @p dof_index_en_in.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             std::shared_ptr<IncNS::BoundaryDescriptorU<dim> const> boundary_descriptor_in,
             unsigned int                            dof_index_in,
             unsigned int                            dof_index_en_in,
             unsigned int                            dof_index_cg_vector_in,
             unsigned int                            dof_index_cg_scalar_in,
             unsigned int                            quad_index_in);

  // ──  compute wall distances and node pairings ─────────────────
  /**
   * Computes the nodal wall distances y_B  for all DoFs within
   * enriched layers 
   *
   * Also builds the node_to_wall_node pairing (off-wall node → nearest wall
   * node) used every time step to propagate u_{\tau}
   *
   */
  void
  setup_wall_distance(dealii::Mapping<dim> const &                          mapping,
                      unsigned int                                          layers = 3);

  void
  evaluate_friction_velocity(VectorType const & velocity);

  // ── Accessors ──────────────────────────────────────────────────────────
  dealii::DoFHandler<dim> const &
  get_dof_handler_en_vector() const;

  dealii::DoFHandler<dim> const & 
  get_dof_handler_cg_scalar() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_cg_vector() const;

  unsigned int
  get_dof_index_dg() const;

  unsigned int
  get_dof_index_en() const;

  unsigned int
  get_dof_index_cg_vector() const;

  unsigned int 
  get_dof_index_cg_scalar() const;

  unsigned int
  get_quad_index() const;

  typename FunctionEnrichment<dim, Number>::scalar
  get_value(scalar const u_tau,
            scalar const y) const;

  typename FunctionEnrichment<dim, Number>::scalar
  get_gradient(scalar const u_tau,
               scalar const y) const;

  struct SchurData {
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_Vbar_Utilde; 
    std::vector<std::vector<dealii::VectorizedArray<Number>>> Schur_inverse;
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_Vbar_inverse; // Added explicit mass inverse
  };

  // Stored per macro-cell for fast matrix-free retrieval during the solver loop
  std::vector<SchurData> cell_schur_data;

  // New execution hooks for the Mass Operator
  void precompute_schur_matrices();
  
  void apply_schur_inverse_mass(VectorType & dst_bar,
                                VectorType & dst_tilde,
                                VectorType const & src_bar,
                                VectorType const & src_tilde) const;

  // ── Public output vectors (CG layout, index = dof_index_en) ───────────
  VectorType wall_distance;
  VectorType friction_velocity;

  VectorType wall_velocity;

  VectorType enrichment_velocity;
  VectorType enrichment_residual;

private:
  struct TractionVectors
  {
    std::array<VectorType*, dim> num;
    VectorType*                  den;
  };

  void
  project_velocity_to_wall(VectorType const & src);

  void
  loop_project_velocity_to_wall(dealii::MatrixFree<dim, Number> const & data,
                                VectorType &   dst,
                                VectorType const &                      src,
                                std::pair<unsigned int, unsigned int> const & range) const;

  void
  loop_lumped_mass(
    dealii::MatrixFree<dim, Number> const & data,
    VectorType &                            dst,
    VectorType const &                      src,
    std::pair<unsigned int, unsigned int> const & range) const;

  // ── Internal helpers ───────────────────────────────────────────────────

  /**
   * Inner face loop called by evaluate_friction_velocity.
   */
  void
  integrate_wall_traction(
    std::array<VectorType, dim> &             traction_num,
    VectorType &                              traction_den) const;

  void 
  local_integrate_wall_traction(
    dealii::MatrixFree<dim, Number> const & matrix_free_data,
    TractionVectors &                       dst,
    VectorType const &                      src_velocity,
    std::pair<unsigned int, unsigned int> const & face_range) const;

  void
  empty_loop(
    dealii::MatrixFree<dim, Number> const & matrix_free_data,
    TractionVectors &                       dst,
    VectorType const &                      src,
    std::pair<unsigned int, unsigned int> const & range) const;

  std::vector<std::vector<dealii::VectorizedArray<Number>>>
  invert_matrix_simd(std::vector<std::vector<dealii::VectorizedArray<Number>>> M, unsigned int n);

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index_dg; 
  unsigned int dof_index_en; 
  unsigned int dof_index_cg_scalar;
  unsigned int dof_index_cg_vector;
  unsigned int quad_index;

  double kinematic_viscosity; ///< \nu
  double kappa;               ///< von Karman constant ≈ 0.41
  double beta;                ///< log-law intercept ≈ 5.2

  // DG Enrichment velocity
  std::shared_ptr<dealii::FiniteElement<dim>>  fe_en_vector;
  dealii::DoFHandler<dim>                      dof_handler_en_vector;

  // CG Wall variables
  std::shared_ptr<dealii::FiniteElement<dim>>   fe_cg_scalar;
  dealii::DoFHandler<dim>                      dof_handler_cg_scalar;

  std::shared_ptr<dealii::FiniteElement<dim>>   fe_cg_vector;
  dealii::DoFHandler<dim>                      dof_handler_cg_vector;

  std::vector<dealii::types::boundary_id> wall_boundary_ids_;

  // Node to nearest wall node mapping (for quick lookup during distance computation)
  std::map<dealii::types::global_dof_index, dealii::types::global_dof_index> node_to_wall_node;

  WallLawEvaluator<dim, Number> wall_law_eval;

  std::shared_ptr<IncNS::BoundaryDescriptorU<dim> const> boundary_descriptor;
};

} // namespace ExaDG

#endif /* INCLUDE_OPERATORS_WALL_MODELLING_FUNCTION_ENRICHMENT_H_ */
