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
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools_mean_value.h>

#include <map>
#include <vector>
#include <array>

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
             std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>> const & periodic_faces,
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
  evaluate_friction_velocity(VectorType const & global_velocity);

  void
  evaluate_friction_velocity_from_residual(VectorType const & u_bar,
                                           VectorType const & u_tilde,
                                           VectorType const & u_bar_hat,
                                           VectorType const & u_tilde_hat,
                                           double const       mass_factor);
  
  // shared tail of both u_tau evaluations: tau_w = num/den at wall nodes -> u_tau, copy off-wall
  void
  finish_friction_velocity(std::array<VectorType, dim> & traction_num, VectorType & traction_den);

  // residual-based wall force, tested with the CG P1 scalar wall shape functions (per component)
  void
  integrate_wall_residual(std::array<VectorType, dim> & traction_num,
                          VectorType const &            u_tilde,
                          VectorType const &            u_tilde_hat,
                          double const                  mass_factor) const;

  void 
  copy_global_dg_to_wall_layout(VectorType const & global_vec, VectorType & wall_vec) const;

  // right-hand side of the projection of u_h^old = U_bar + psi_old * U_tilde onto the space
  // with psi_new (thesis Eq. 6.14), on enriched cells, in shadow / enrichment layout:
  //   dst_bar_shadow = int v_bar . u_h^old,   dst_tilde = int psi_new v_tilde . u_h^old
  void
  apply_mass_old_to_new(VectorType &       dst_bar_shadow,
                        VectorType &       dst_tilde,
                        VectorType const & u_bar_shadow,
                        VectorType const & u_tilde) const;

  void 
  copy_wall_layout_to_global_dg(VectorType const & wall_vec, VectorType & global_vec) const;

  void
  add_wall_layout_to_global_dg(VectorType const & wall_vec,VectorType & global_vec) const;

  void
  set_global_pressure_dof_handler(dealii::DoFHandler<dim> const & dof_handler_p_in);

  // global pressure -> shadow layout (enriched + halo cells), overwrites wall_p
  void
  copy_global_pressure_to_wall_layout(VectorType const & global_p, VectorType & wall_p) const;

  // shadow layout -> global pressure, ADDS on enriched + halo cells
  void
  add_wall_layout_to_global_pressure(VectorType const & wall_p, VectorType & global_p) const;

  // Largest y+ in the enriched layer: max(y) * max(u_tau) / nu.
  // Small values: psi is (nearly) linear, the enrichment is redundant -> do not use it.
  double
  get_max_y_plus() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_shadow_pressure() const;

  unsigned int
  get_dof_index_shadow_pressure() const;

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

  // psi and grad psi at the quadrature points of enriched cells and of the enriched sides of
  // faces, Krank et al. (2018) Eq. (12) with y+ = y_h sqrt(tau_w,h) / nu. Recomputed whenever
  // tau_w changes (finish_friction_velocity, setup_wall_distance).
  void
  update_enrichment_function();

  // cell: cell batch index of matrix_free_en; face: face batch index, interior / exterior side
  scalar const &
  get_psi_cell(unsigned int const cell, unsigned int const q) const;

  vector const &
  get_grad_psi_cell(unsigned int const cell, unsigned int const q) const;

  scalar const &
  get_psi_cell_old(unsigned int const cell, unsigned int const q) const;

  scalar const &
  get_psi_face(unsigned int const face, bool const interior, unsigned int const q) const;

  vector const &
  get_grad_psi_face(unsigned int const face, bool const interior, unsigned int const q) const;

  std::shared_ptr<dealii::MatrixFree<dim, Number>> 
  get_matrix_free() const;

  unsigned int 
  get_dof_index_shadow_vector() const;

  unsigned int
  get_active_fe_index() const;

  void
  set_minimum_wall_shear_stress_fraction(double const fraction);

  void
  compute_enrichment_component(VectorType &                           dst,
                               VectorType const &                     u_tilde,
                               dealii::DoFHandler<dim> const &        dof_handler_out,
                               dealii::Mapping<dim> const &           mapping) const;

  // Add this pointer to store the global fluid layout
  dealii::DoFHandler<dim> const * global_dof_handler;

  VectorType wall_distance;
  VectorType friction_velocity;
  VectorType wall_shear_stress;

  // interior-penalty parameter used by the viscous operator, per cell batch of matrix_free_en
  // (incl. ghost batches); set by EnrichedMomentumOperator. Used for the wall flux tau_w.
  dealii::AlignedVector<dealii::VectorizedArray<Number>> wall_penalty;
  
  VectorType friction_velocity_old;

  VectorType wall_velocity;

  VectorType enrichment_velocity;
  VectorType enrichment_residual;

  VectorType shadow_velocity;
  VectorType shadow_velocity_residual;

  VectorType shadow_pressure;

  VectorType src;
  VectorType dst;

  std::map<dealii::types::global_cell_index, bool> is_cell_enriched;

private:
  struct TractionVectors
  {
    std::array<VectorType*, dim> num;
    VectorType*                  den;
  };

  void
  project_velocity_to_wall();

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

  void
  setup_corner_nodes();

  // periodicity constraints for the hp CG DoFHandlers (parallel-safe, see .cpp)
  void
  make_periodicity_constraints_hp(
    dealii::DoFHandler<dim> const & dof_handler,
    std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::Triangulation<dim>::cell_iterator>> const & periodic_faces,
    dealii::AffineConstraints<Number> &                            constraints) const;

  void
  make_periodicity_constraints_hp(typename dealii::DoFHandler<dim>::face_iterator const & face_1,
                                  typename dealii::DoFHandler<dim>::face_iterator const & face_2,
                                  unsigned char const                                     orientation,
                                  dealii::AffineConstraints<Number> & constraints) const;

  // psi, grad psi from y_h, grad y_h, tau_w,h, grad tau_w,h at one quadrature point
  std::pair<scalar, vector>
  evaluate_enrichment_function(scalar const & y,
                               vector const & grad_y,
                               scalar const & tau_w,
                               vector const & grad_tau_w) const;

  unsigned int dof_index_en; 
  unsigned int dof_index_en_scalar; 
  unsigned int dof_index_cg_scalar;
  unsigned int dof_index_cg_vector;
  unsigned int dof_index_shadow_vector;
  unsigned int dof_index_shadow_pressure;
  unsigned int quad_index;

  const unsigned int active_fe_index = 1;

  double kinematic_viscosity; 

  double minimum_wall_shear_stress_fraction = 0.02;

  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_en;

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

  dealii::DoFHandler<dim> const * global_dof_handler_pressure = nullptr;
  dealii::DoFHandler<dim>         dof_handler_shadow_pressure;
  dealii::hp::FECollection<dim>   fe_shadow_pressure_collection;

  // CG Wall variables
  std::shared_ptr<dealii::FiniteElement<dim>> fe_cg_scalar;
  dealii::DoFHandler<dim>                     dof_handler_cg_scalar;
  dealii::hp::FECollection<dim>               fe_cg_scalar_collection;

  std::shared_ptr<dealii::FiniteElement<dim>> fe_cg_vector;
  dealii::DoFHandler<dim>                     dof_handler_cg_vector;
  dealii::hp::FECollection<dim>               fe_cg_vector_collection;

  // Node to nearest wall node mapping (for quick lookup during distance computation)
  std::map<dealii::types::global_dof_index, dealii::types::global_dof_index> node_to_wall_node;

  // wall nodes that also lie on a non-wall boundary (inflow/outflow corners) -> neighbouring wall
  // node on the same wall face. The residual-based tau_w is not a wall force there.
  std::map<dealii::types::global_dof_index, dealii::types::global_dof_index> corner_to_wall_node;

  WallLawEvaluator<dim, Number> wall_law_eval;

  std::shared_ptr<IncNS::BoundaryDescriptorU<dim> const> boundary_descriptor;

  dealii::AffineConstraints<Number> constraint_cg_scalar; // periodicity, dof_handler_cg_scalar
  dealii::AffineConstraints<Number> constraint_cg_vector; // periodicity, dof_handler_cg_vector
  dealii::AffineConstraints<Number> constraint_dg;

  // tau_w,h = 0 before the first wall shear stress evaluation: keeps y+ defined
  // (same as the former u_tau >= 1e-5); afterwards the 2% minimum of Step 2 applies
  static constexpr double tau_w_absolute_minimum = 1.e-10;

  unsigned int n_q_points_cell = 0;
  unsigned int n_q_points_face = 0;

  dealii::AlignedVector<scalar> psi_cell;      // [cell batch * n_q_points_cell + q]
  dealii::AlignedVector<vector> grad_psi_cell; // same layout
  dealii::AlignedVector<scalar> psi_cell_old;

  std::array<dealii::AlignedVector<scalar>, 2> psi_face;      // [0] interior, [1] exterior side
  std::array<dealii::AlignedVector<vector>, 2> grad_psi_face; // [face batch * n_q_points_face + q]
};

} // namespace ExaDG

#endif /* INCLUDE_OPERATORS_WALL_MODELLING_FUNCTION_ENRICHMENT_H_ */
