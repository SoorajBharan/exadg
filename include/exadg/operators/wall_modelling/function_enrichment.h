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
#include <exadg/operators/quadrature.h>
#include <exadg/incompressible_navier_stokes/user_interface/boundary_descriptor.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/hp/fe_collection.h>
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

void
  initialize(dealii::Triangulation<dim> const &                     triangulation,
             std::shared_ptr<dealii::Mapping<dim> const>            mapping_in,
             dealii::DoFHandler<dim> const &                        global_dof_handler_in,
             unsigned int                                           fe_degree_dg,
             unsigned int                                           fe_degree_en,
             std::shared_ptr<IncNS::BoundaryDescriptorU<dim> const> boundary_descriptor_in,
             unsigned int                                           quad_index_in,
             unsigned int                                           layers = 3);


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
  setup_wall_distance_cg(dealii::Mapping<dim> const &                          mapping,
                         unsigned int                                          layers = 3);

  void
  evaluate_friction_velocity(VectorType const & global_velocity);

  void 
  copy_global_dg_to_wall_layout(VectorType const & global_vec, VectorType & wall_vec) const;

  void 
  copy_wall_layout_to_global_dg(VectorType const & wall_vec, VectorType & global_vec) const;

  // ── Accessors ──────────────────────────────────────────────────────────
  dealii::DoFHandler<dim> const &
  get_dof_handler_en_vector() const;

  dealii::DoFHandler<dim> const & 
  get_dof_handler_en_scalar() const;

  dealii::DoFHandler<dim> const & 
  get_dof_handler_cg_scalar() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_cg_vector() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_shadow_vector() const;

  unsigned int
  get_dof_index_en() const;

  unsigned int
  get_dof_index_cg_vector() const;

  unsigned int 
  get_dof_index_cg_scalar() const;

  unsigned int
  get_dof_index_en_scalar() const;

  unsigned int
  get_quad_index() const;

  typename FunctionEnrichment<dim, Number>::scalar
  get_value(scalar const u_tau,
            scalar const y) const;

  typename FunctionEnrichment<dim, Number>::scalar
  get_gradient(scalar const u_tau,
               scalar const y) const;

  std::shared_ptr<dealii::MatrixFree<dim, Number>> 
  get_matrix_free() const;

  unsigned int 
  get_dof_index_shadow_vector() const;

  unsigned int
  get_active_fe_index() const;

  struct SchurData {
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_Vbar_Utilde; 
    std::vector<std::vector<dealii::VectorizedArray<Number>>> Schur_inverse;
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_Vbar_inverse;

    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_tilde_inv;

    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_Utilde_inverse;
  };

  // Add this pointer to store the global fluid layout
  dealii::DoFHandler<dim> const * global_dof_handler;

  VectorType wall_distance;
  VectorType friction_velocity;

  VectorType wall_velocity;

  VectorType enrichment_velocity;
  VectorType enrichment_residual;

  VectorType shadow_velocity;
  VectorType shadow_velocity_residual;

  std::map<dealii::types::global_cell_index, bool> is_cell_enriched;

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

  unsigned int dof_index_en; 
  unsigned int dof_index_en_scalar; 
  unsigned int dof_index_cg_scalar;
  unsigned int dof_index_cg_vector;
  unsigned int dof_index_shadow_vector;
  unsigned int quad_index;

  const unsigned int active_fe_index = 1;

  double kinematic_viscosity; ///< \nu
  double kappa;               ///< von Karman constant ≈ 0.41
  double beta;                ///< log-law intercept ≈ 5.2

  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_en;
  dealii::AffineConstraints<Number>                constraint_wall;

  // DG Enrichment velocity
  std::shared_ptr<dealii::FiniteElement<dim>> fe_en_vector;
  dealii::DoFHandler<dim>                     dof_handler_en_vector;
  dealii::hp::FECollection<dim>               fe_en_collection;

  std::shared_ptr<dealii::FiniteElement<dim>> fe_en_scalar;
  dealii::DoFHandler<dim>                     dof_handler_en_scalar;
  dealii::hp::FECollection<dim>               fe_en_collection_scalar;

  // DG shadow velocity 
  std::shared_ptr<dealii::FiniteElement<dim>> fe_shadow_vector;
  dealii::DoFHandler<dim>                     dof_handler_shadow_vector;
  dealii::hp::FECollection<dim>               fe_shadow_collection;

  // CG Wall variables
  std::shared_ptr<dealii::FiniteElement<dim>> fe_cg_scalar;
  dealii::DoFHandler<dim>                     dof_handler_cg_scalar;
  dealii::hp::FECollection<dim>               fe_cg_scalar_collection;

  std::shared_ptr<dealii::FiniteElement<dim>> fe_cg_vector;
  dealii::DoFHandler<dim>                     dof_handler_cg_vector;
  dealii::hp::FECollection<dim>               fe_cg_vector_collection;

  std::vector<dealii::types::boundary_id> wall_boundary_ids_;

  // Node to nearest wall node mapping (for quick lookup during distance computation)
  std::map<dealii::types::global_dof_index, dealii::types::global_dof_index> node_to_wall_node;

  WallLawEvaluator<dim, Number> wall_law_eval;

  std::shared_ptr<IncNS::BoundaryDescriptorU<dim> const> boundary_descriptor;
};

} // namespace ExaDG

#endif /* INCLUDE_OPERATORS_WALL_MODELLING_FUNCTION_ENRICHMENT_H_ */
