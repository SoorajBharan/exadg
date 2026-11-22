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

#ifndef INCLUDE_OPERATORS_WALL_MODELLING_FUNCTION_ENRICHMENT_RANS_H_
#define INCLUDE_OPERATORS_WALL_MODELLING_FUNCTION_ENRICHMENT_RANS_H_

#include <exadg/operators/wall_modelling/function_enrichment.h>
#include <exadg/operators/wall_modelling/wall_law_evaluator.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <limits>
#include <memory>
#include <unordered_set>
#include <vector>

namespace ExaDG
{
/*
 * Enrichment of RANS scalars in the wall cells of the velocity enrichment (same cells, layers,
 * active_fe_index, y and u_tau as FunctionEnrichment):
 *
 *   omega:     omega_h = omega_bar + psi_omega omega_tilde
 *   ln omega:  W_h     = W_bar     + ln(psi_omega) W_tilde
 *
 * One entry per enriched variable (omega now, k later).
 * Own MatrixFree on the same triangulation:
 *   dof index 0: shadow layout of the RANS scalars (bar part, RANS FE, enriched + halo cells)
 *   dof index 1: enrichment space (tilde part), DG of the degree of U_tilde, enriched cells only
 *   dof index 2: CG scalar of FunctionEnrichment (tau_w, u_tau)
 *   dof index 3: DG scalar of FunctionEnrichment (wall distance y)
 */
template<int dim, typename Number>
class FunctionEnrichmentRANS
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef dealii::VectorizedArray<Number>                    scalar;
  typedef dealii::Tensor<1, dim, scalar>                     vector;
  typedef dealii::Tensor<2, dim, scalar>                     tensor;
  typedef FunctionEnrichmentRANS<dim, Number>                This;
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  // shape of the enrichment function of one variable
  enum class Shape
  {
    Omega // psi_omega (OmegaWallLaw)
  };

  struct Variable
  {
    unsigned int rans_index; // block of the RANS solution (k = 0, omega = 1)
    Shape        shape;
    bool         log_form; // true: psi = ln(psi_omega), false: psi = psi_omega
    OmegaWallLaw omega_wall_law;
    // condition at the wall face (diffusion face terms)
    OmegaEnrichmentWallCondition wall_condition = OmegaEnrichmentWallCondition::WeakDirichlet;
    OmegaEnrichmentFaceFormulation face_formulation = OmegaEnrichmentFaceFormulation::NIPG;
  };

  FunctionEnrichmentRANS(FunctionEnrichment<dim, Number> const & velocity_enrichment_in,
                         double const                            kinematic_viscosity_in);

  // before initialize()
  void
  add_variable(Variable const & variable);

  // global_dof_handler_rans: scalar DG DoFHandler of the RANS variables
  void
  initialize(dealii::Triangulation<dim> const &          triangulation,
             std::shared_ptr<dealii::Mapping<dim> const> mapping,
             dealii::DoFHandler<dim> const &             global_dof_handler_rans,
             dealii::DoFHandler<dim> const &             global_dof_handler_eddy_viscosity);

  // velocity of the fluid at the new time, (U_bar, U_tilde) = (block 0 global layout, block 1):
  // copied into this MatrixFree; the exact velocity gradient
  //   grad u_h = grad U_bar + psi_u grad U_tilde + U_tilde (x) grad psi_u
  // is cached at the quadrature points of the enriched cells (get_velocity_gradient_cell)
  void
  set_velocity(BlockVectorType const & velocity_np_enriched);

  // fields on the RANS eddy viscosity DoFHandler (e.g. wall-law nu_t, P_k, wall-cell indicator):
  // global layout -> shadow layout (enriched + halo cells), DoF copy, overwrites shadow
  void
  copy_eddy_viscosity_to_wall_layout(VectorType const & global, VectorType & shadow) const;

  // after every u_tau update of FunctionEnrichment: psi -> psi_old, new psi (cells and faces)
  void
  update();

  // bar part: global RANS layout <-> shadow layout (enriched + halo cells), DoF copy
  void
  copy_global_to_wall_layout(VectorType const & global, VectorType & shadow) const;

  void
  add_wall_layout_to_global(VectorType const & shadow, VectorType & global) const;

  // bar part: shadow layout -> global RANS layout, OVERWRITES the enriched cells only (not the
  // halo cells); for cell-local operators (mass, inverse mass) computed on the enriched cells
  void
  copy_wall_layout_to_global(VectorType const & shadow, VectorType & global) const;

  // block mass on the enriched cells, variable `variable` (thesis 2x2 block mass):
  //   dst_bar = M_bb W_bar + M_bt W_tilde,   dst_tilde = M_tb W_bar + M_tt W_tilde
  // shadow / enrichment layout
  void
  apply_mass(unsigned int const variable,
             VectorType &       dst_bar_shadow,
             VectorType &       dst_tilde,
             VectorType const & src_bar_shadow,
             VectorType const & src_tilde) const;

  // right-hand side of the L2 projection of the wall law onto the enriched space, enriched cells:
  //   dst_bar = int v_bar W_wl,   dst_tilde = int psi v_tilde W_wl,
  //   W_wl = ln((u_tau^2 / nu) psi_omega(y+)) (ln omega) or (u_tau^2 / nu) psi_omega(y+) (omega)
  // (initial values of the enrichment, consistent with the wall law)
  void
  apply_mass_wall_law_target(unsigned int const variable,
                             VectorType &       dst_bar_shadow,
                             VectorType &       dst_tilde) const;

  // block inverse mass on the enriched cells (thesis Eqs. (6.16), (6.17)), variable `variable`:
  //   W_tilde = S^+ (R_tilde - M_bar_tilde^T M_bar_bar^{-1} R_bar)
  //   W_bar   = M_bar_bar^{-1} (R_bar - M_bar_tilde W_tilde)
  // R: right-hand side (shadow / enrichment layout); after precompute_inverse_mass()
  void
  apply_inverse_mass(unsigned int const variable,
                     VectorType &       dst_bar_shadow,
                     VectorType &       dst_tilde,
                     VectorType const & src_bar_shadow,
                     VectorType const & src_tilde) const;

  // right-hand side of the projection of the old solution onto the space with the new psi
  // (as FunctionEnrichment::apply_mass_old_to_new), enriched cells:
  //   dst_bar_shadow = int v_bar W_old,   dst_tilde = int psi_new v_tilde W_old,
  //   W_old = bar + psi_old tilde
  void
  apply_mass_old_to_new(unsigned int const variable,
                        VectorType &       dst_bar_shadow,
                        VectorType &       dst_tilde,
                        VectorType const & src_bar_shadow,
                        VectorType const & src_tilde) const;

  // modes of the Schur complement with lambda <= tolerance are removed (thesis Eqs. (6.16), (6.17));
  // before precompute_inverse_mass()
  void
  set_rank_tolerance(double const tolerance);

  // k criterion of the enrichment: on for k+ >= k_plus_on, off for k+ < k_plus_off (k_plus_on = 0:
  // no k criterion), and the omega wall condition of the wall cells switched off by it
  void
  set_k_criterion(double const                          k_plus_on,
                  double const                          k_plus_off,
                  OmegaEnrichmentOffWallCondition const off_wall_condition)
  {
    k_criterion_k_plus_on  = k_plus_on;
    k_criterion_k_plus_off = k_plus_off;
    off_wall_condition_    = off_wall_condition;
  }

  /*
   * k of the RANS solution (global layout of the RANS variables; ln k if k_logarithmic): k+ =
   * k_bar / u_tau^2 and the mean wall distance y_c of every wall cell, then the k criterion
   * (hysteresis). Call before update(): the new psi and wall values use it.
   */
  void
  set_turbulent_kinetic_energy(VectorType const & k_global, bool const k_logarithmic);

  // block mass of the enriched cells per variable (M_bar_bar^{-1}, M_bar_tilde, S^+, projector);
  // after every update() (psi changes)
  void
  precompute_inverse_mass();

  // projector P onto the well-posed enrichment modes (precompute_inverse_mass()), applied to the
  // W_tilde DoFs of each enriched cell: dst_tilde = P src_tilde (identity before the first
  // precompute_inverse_mass()); dst_tilde and src_tilde may be the same vector
  void
  apply_enrichment_projector(unsigned int const variable,
                             VectorType &       dst_tilde,
                             VectorType const & src_tilde) const;

  // number of removed enrichment modes of the last precompute_inverse_mass() (all processes)
  unsigned int
  get_n_modes_dropped() const;

  // enrichment part psi W_tilde of variable `variable` at the nodes of the RANS finite element
  // (Gauss-Lobatto) of the enriched cells, as DoF values in the global RANS layout (0 elsewhere):
  // W_node = W_bar_node + global_dst_node (nodal eddy viscosity of TurbulenceModel)
  void
  compute_enrichment_nodal(unsigned int const variable,
                           VectorType &       global_dst,
                           VectorType const & src_tilde) const;

  // ── accessors (FEOmegaEnrichedEvaluation) ──────────────────────────
  std::shared_ptr<dealii::MatrixFree<dim, Number>>
  get_matrix_free() const;

  unsigned int
  get_dof_index_dg() const;

  unsigned int
  get_dof_index_en() const;

  unsigned int
  get_quad_index() const;

  unsigned int
  get_active_fe_index() const;

  unsigned int
  n_variables() const;

  // enrichment switched on in lane of a cell batch of this MatrixFree (filled lane, y+ switch of
  // the velocity)
  bool
  is_enrichment_active_lane(unsigned int const cell, unsigned int const lane) const;

  // lane switched off by the k criterion only (velocity enrichment active): omega wall condition
  // get_off_wall_condition() there
  bool
  is_k_switched_off_lane(unsigned int const cell, unsigned int const lane) const;

  OmegaEnrichmentOffWallCondition
  get_off_wall_condition() const
  {
    return off_wall_condition_;
  }

  // boundary of the wall model (velocity boundary type WallEnrichment)
  bool
  is_wall_boundary(dealii::types::boundary_id const boundary_id) const;

  // condition of enriched variable `variable` at the wall face
  OmegaEnrichmentWallCondition
  get_wall_condition(unsigned int const variable) const;

  OmegaEnrichmentFaceFormulation
  get_face_formulation(unsigned int const variable) const;

  // wall cell of the enrichment (active_fe_index 1, also if switched off by the y+ criterion),
  // locally owned and ghost cells, by global active cell index
  bool
  is_enriched_cell(dealii::types::global_cell_index const cell_index) const;

  // enrichment switched on by the y+ criterion of the velocity enrichment (global cell index)
  bool
  is_enrichment_active_cell(dealii::types::global_cell_index const cell_index) const
  {
    return velocity_enrichment.is_enrichment_active_cell(cell_index);
  }

  // block of the RANS solution of an enriched variable (e.g. omega = 1)
  unsigned int
  get_rans_index(unsigned int const variable) const;

  // exact velocity gradient [i][j] = d u_i / d x_j at quadrature point q of an enriched cell batch
  // (set_velocity), quad_index; psi_u = 0 on lanes switched off by the y+ criterion
  tensor const &
  get_velocity_gradient_cell(unsigned int const cell, unsigned int const q) const;

  // velocity u_h = U_bar + psi_u U_tilde (enriched side) or U_bar (halo side) at quadrature point
  // q of a cell batch / face side (set_velocity)
  vector const &
  get_velocity_value_cell(unsigned int const cell, unsigned int const q) const;

  vector const &
  get_velocity_value_face(unsigned int const face,
                          bool const         interior,
                          unsigned int const q) const;

  // velocity gradient at quadrature point q of side (interior / exterior) of a face batch
  // (set_velocity): exact grad u_h on an enriched side, grad U_bar on a halo side
  tensor const &
  get_velocity_gradient_face(unsigned int const face,
                             bool const         interior,
                             unsigned int const q) const;

  // boundary face batch, interior side, point q: value of the transported variable at the wall
  // for the weak Dirichlet condition (update()): (u_tau^2 / nu) psi_omega(0) for omega,
  // ln((u_tau^2 / nu) psi_omega(0)) for ln omega (psi_omega capped)
  scalar const &
  get_wall_value_face(unsigned int const variable,
                      unsigned int const face,
                      unsigned int const q) const;

  // as get_wall_value_face(), for the wall cells switched off by the k criterion (WallFunction):
  // omega_wf = (u_tau^2 / nu) psi_omega(y+ = a y_c u_tau / nu), a = exp(-sum_{n=1}^p 1/n), y_c the
  // mean wall distance of the cell (ln omega_wf for ln omega)
  scalar const &
  get_wall_value_off_face(unsigned int const variable,
                          unsigned int const face,
                          unsigned int const q) const;

  // variable: index in the list of enriched variables (add_variable order)
  scalar const &
  get_psi_cell(unsigned int const variable, unsigned int const cell, unsigned int const q) const;

  vector const &
  get_grad_psi_cell(unsigned int const variable,
                    unsigned int const cell,
                    unsigned int const q) const;

  scalar const &
  get_psi_cell_old(unsigned int const variable,
                   unsigned int const cell,
                   unsigned int const q) const;

  scalar const &
  get_psi_face(unsigned int const variable,
               unsigned int const face,
               bool const         interior,
               unsigned int const q) const;

  vector const &
  get_grad_psi_face(unsigned int const variable,
                    unsigned int const face,
                    bool const         interior,
                    unsigned int const q) const;

#ifdef EXADG_DEBUG
  /*
   * Diagnostics of the wall layer of one enriched variable (printed on rank 0):
   *   - W_tilde at the quadrature points of the active wall cells: min, mean, max
   *   - quadrature point closest to the wall (per cell): W_h = W_bar + psi W_tilde against the
   *     wall law, ln(u_tau^2 / nu) + ln psi_omega (or (u_tau^2 / nu) psi_omega)
   *   - wall faces of active cells: W_h against the wall value g of the WeakDirichlet condition
   * global_bar: RANS block of the variable (global layout), src_tilde: its W_tilde block.
   */
  void
  print_wall_diagnostics(unsigned int const variable,
                         VectorType const & global_bar,
                         VectorType const & src_tilde,
                         unsigned int const time_step_number) const;
#endif

private:
  // psi, grad psi of one variable at one quadrature point from y, grad y, tau_w, grad tau_w
  std::pair<scalar, vector>
  evaluate_enrichment_function(Variable const & variable,
                               scalar const &   y,
                               vector const &   grad_y,
                               scalar const &   tau_w,
                               vector const &   grad_tau_w) const;

  // loop functions of update(): psi, grad psi in the enriched cells and on the enriched sides of
  // faces; src.block(0) = y, src.block(1) = tau_w; dst is not used
  void
  cell_loop_update(dealii::MatrixFree<dim, Number> const &       matrix_free,
                   VectorType &                                  dst,
                   BlockVectorType const &                       src,
                   std::pair<unsigned int, unsigned int> const & range);

  void
  face_loop_update(dealii::MatrixFree<dim, Number> const &       matrix_free,
                   VectorType &                                  dst,
                   BlockVectorType const &                       src,
                   std::pair<unsigned int, unsigned int> const & range);

  void
  boundary_face_loop_update(dealii::MatrixFree<dim, Number> const &       matrix_free,
                            VectorType &                                  dst,
                            BlockVectorType const &                       src,
                            std::pair<unsigned int, unsigned int> const & range);

  // psi, grad psi on one side of the faces of a face range (face loops of update())
  void
  fill_psi_face(dealii::MatrixFree<dim, Number> const &       matrix_free,
                BlockVectorType const &                       src,
                std::pair<unsigned int, unsigned int> const & range,
                bool const                                    interior);

  // cell loop of set_velocity(), enriched cells: exact velocity gradient at the quadrature points;
  // src: U_bar (0), U_tilde (1), y (2), tau_w (3); dst is not used
  void
  cell_loop_velocity_gradient(dealii::MatrixFree<dim, Number> const &       matrix_free,
                              VectorType &                                  dst,
                              BlockVectorType const &                       src,
                              std::pair<unsigned int, unsigned int> const & range);

  // face loops of set_velocity(): velocity gradient on the faces (enriched and halo sides)
  void
  face_loop_velocity_gradient(dealii::MatrixFree<dim, Number> const &       matrix_free,
                              VectorType &                                  dst,
                              BlockVectorType const &                       src,
                              std::pair<unsigned int, unsigned int> const & range);

  void
  boundary_face_loop_velocity_gradient(dealii::MatrixFree<dim, Number> const &       matrix_free,
                                       VectorType &                                  dst,
                                       BlockVectorType const &                       src,
                                       std::pair<unsigned int, unsigned int> const & range);

  // velocity gradient on one side of the faces of a face range
  void
  fill_velocity_gradient_face(dealii::MatrixFree<dim, Number> const &       matrix_free,
                              BlockVectorType const &                       src,
                              std::pair<unsigned int, unsigned int> const & range,
                              bool const                                    interior);

  // index map global layout of a DoFHandler with the RANS FE <-> shadow layout: locally owned
  // enriched cells (1) first, then halo cells (2); returns the number of enriched-cell pairs
  unsigned int
  build_index_map(dealii::DoFHandler<dim> const & dof_handler_global,
                  std::vector<unsigned int> &     index_global,
                  std::vector<unsigned int> &     index_shadow) const;

  // cell loop of apply_mass_old_to_new(), enriched cells, variable variable_mass_old_to_new:
  //   dst[0] = int v_bar W_old,   dst[1] = int psi_new v_tilde W_old,   W_old = bar + psi_old tilde
  // src.block(0): bar part (shadow layout), src.block(1): tilde part
  void
  cell_loop_mass_old_to_new(
    dealii::MatrixFree<dim, Number> const &                         matrix_free,
    std::vector<VectorType> &                                       dst,
    dealii::LinearAlgebra::distributed::BlockVector<Number> const & src,
    std::pair<unsigned int, unsigned int> const &                   range) const;

  // cell loop of apply_mass(), enriched cells, variable variable_mass:
  // dst[0] = rows tested with v_bar, dst[1] = rows tested with psi v_tilde
  void
  cell_loop_mass_apply(dealii::MatrixFree<dim, Number> const &       matrix_free,
                       std::vector<VectorType> &                     dst,
                       BlockVectorType const &                       src,
                       std::pair<unsigned int, unsigned int> const & range) const;

  // cell loop of compute_enrichment_nodal(), enriched cells, variable variable_enrichment_nodal:
  // dst = nodal values of psi W_tilde (shadow layout); src: W_tilde, y, tau_w
  void
  cell_loop_enrichment_nodal(dealii::MatrixFree<dim, Number> const &       matrix_free,
                             VectorType &                                  dst,
                             BlockVectorType const &                       src,
                             std::pair<unsigned int, unsigned int> const & range) const;

  // variable of the current compute_enrichment_nodal() call
  mutable unsigned int variable_enrichment_nodal = 0;

  // cell loop of apply_enrichment_projector(), enriched cells, variable variable_projector
  void
  cell_loop_projector(dealii::MatrixFree<dim, Number> const &       matrix_free,
                      VectorType &                                  dst,
                      VectorType const &                            src,
                      std::pair<unsigned int, unsigned int> const & range) const;

  mutable unsigned int variable_projector = 0;

  // cell loop of apply_mass_wall_law_target(), enriched cells; src: y (0), tau_w (1)
  void
  cell_loop_wall_law_target(dealii::MatrixFree<dim, Number> const &       matrix_free,
                            std::vector<VectorType> &                     dst,
                            BlockVectorType const &                       src,
                            std::pair<unsigned int, unsigned int> const & range) const;

  mutable unsigned int variable_wall_law_target = 0;

#ifdef EXADG_DEBUG
  // loops of print_wall_diagnostics(); src: W_bar (shadow, 0), W_tilde (1), y (2), tau_w (3)
  void
  cell_loop_wall_diagnostics(dealii::MatrixFree<dim, Number> const &       matrix_free,
                             VectorType &                                  dst,
                             BlockVectorType const &                       src,
                             std::pair<unsigned int, unsigned int> const & range) const;

  void
  face_loop_wall_diagnostics(dealii::MatrixFree<dim, Number> const &       matrix_free,
                             VectorType &                                  dst,
                             BlockVectorType const &                       src,
                             std::pair<unsigned int, unsigned int> const & range) const;

  void
  boundary_face_loop_wall_diagnostics(dealii::MatrixFree<dim, Number> const &       matrix_free,
                                      VectorType &                                  dst,
                                      BlockVectorType const &                       src,
                                      std::pair<unsigned int, unsigned int> const & range) const;

  // sums / extrema of print_wall_diagnostics() on this process
  struct WallDiagnostics
  {
    double tilde_min = std::numeric_limits<double>::max();
    double tilde_max = -std::numeric_limits<double>::max();
    double tilde_sum = 0.0, tilde_count = 0.0;

    double near_y_plus_sum = 0.0, near_value_sum = 0.0, near_wall_law_sum = 0.0;
    double near_difference_max = 0.0, near_count = 0.0;

    double face_value_sum = 0.0, face_target_sum = 0.0;
    double face_difference_max = 0.0, face_count = 0.0;

    // x coordinate of the point of each maximum (where along the wall)
    double tilde_max_x = 0.0, near_difference_max_x = 0.0, face_difference_max_x = 0.0;
  };

  mutable WallDiagnostics wall_diagnostics;
  mutable unsigned int    variable_wall_diagnostics = 0;
#endif

  // variable of the current apply_mass() call (read by cell_loop_mass_apply)
  mutable unsigned int variable_mass = 0;

  // cell loop of apply_inverse_mass(), enriched cells, variable variable_inverse_mass:
  // dst[0] = W_bar (shadow layout), dst[1] = W_tilde; src.block(0) = R_bar, src.block(1) = R_tilde
  void
  cell_loop_inverse_mass_apply(dealii::MatrixFree<dim, Number> const &       matrix_free,
                               std::vector<VectorType> &                     dst,
                               BlockVectorType const &                       src,
                               std::pair<unsigned int, unsigned int> const & range) const;

  // variable of the current apply_inverse_mass() call (read by cell_loop_inverse_mass_apply)
  mutable unsigned int variable_inverse_mass = 0;

  // variable of the current apply_mass_old_to_new() call (read by cell_loop_mass_old_to_new)
  mutable unsigned int variable_mass_old_to_new = 0;

  typedef std::vector<std::vector<scalar>> SimdMatrix;

  // psi-dependent part of the block mass of one enriched cell batch (one variable)
  struct InverseMassData
  {
    SimdMatrix M_bar_tilde;   // (N_j, psi N~_i)_K,  n_dg x n_en
    SimdMatrix schur_inverse; // S^+,  n_en x n_en
    SimdMatrix projector;     // P onto the well-posed enrichment modes,  n_en x n_en
  };

  // cell loop of precompute_inverse_mass(), enriched cells; dst and src are not used
  void
  cell_loop_inverse_mass(dealii::MatrixFree<dim, Number> const &       matrix_free,
                         VectorType &                                  dst,
                         VectorType const &                            src,
                         std::pair<unsigned int, unsigned int> const & range) const;

  // S^+ and projector P of one cell batch (scalar version of WallDGCoupler)
  SimdMatrix
  compute_schur_pseudo_inverse(SimdMatrix const & Schur,
                               SimdMatrix const & M_tilde_tilde,
                               SimdMatrix &       projector,
                               unsigned int const n_active_lanes,
                               unsigned int const cell) const;

  // inverse of an SPD matrix per SIMD lane (Gauss-Jordan)
  static SimdMatrix
  invert_matrix(SimdMatrix M);

  // [variable][cell batch], written by cell_loop_inverse_mass
  mutable std::vector<std::vector<InverseMassData>> inverse_mass_data;

  // [cell batch]: M_bar_bar^{-1} (geometry only, first call)
  mutable std::vector<SimdMatrix> M_bar_bar_inverse_cell;

  double rank_tolerance = 1.e-6;

  mutable unsigned int n_modes_total   = 0;
  mutable unsigned int n_modes_dropped = 0;

  // smallest kept eigenvalue of D^{-1/2} S D^{-1/2} of the last precompute_inverse_mass()
  mutable double lambda_min_kept = std::numeric_limits<double>::max();

  FunctionEnrichment<dim, Number> const & velocity_enrichment;

  double kinematic_viscosity;

  std::vector<Variable> variables;

  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  unsigned int const dof_index_dg    = 0;
  unsigned int const dof_index_en    = 1;
  unsigned int const dof_index_cg    = 2;
  unsigned int const dof_index_wall_distance = 3;
  // velocity of the fluid: U_bar (shadow layout of FunctionEnrichment), U_tilde
  unsigned int const dof_index_velocity_bar   = 4;
  unsigned int const dof_index_velocity_tilde = 5;
  unsigned int const quad_index      = 0;
  // nodes of the RANS finite element (Gauss-Lobatto, degree + 1 points)
  unsigned int const quad_index_nodal = 1;
  unsigned int const active_fe_index = 1;

  // shadow layout of the RANS scalars (bar part)
  dealii::DoFHandler<dim> const * global_dof_handler_rans = nullptr;
  dealii::DoFHandler<dim>         dof_handler_shadow_scalar;
  dealii::hp::FECollection<dim>   fe_shadow_scalar_collection;
  std::vector<unsigned int>       scalar_index_global;
  std::vector<unsigned int>       scalar_index_shadow;
  // [side][face batch * n_q_points + q]: velocity gradient on the faces (set_velocity)
  std::array<dealii::AlignedVector<tensor>, 2> velocity_gradient_face;
  std::array<dealii::AlignedVector<vector>, 2> velocity_value_face;
  unsigned int                                 n_q_points_face_velocity_gradient = 0;

  // per variable, [face batch * n_q_points_face + q]: wall value of the weak Dirichlet condition
  std::vector<dealii::AlignedVector<scalar>> wall_value_face;

  // omega wall value of the wall cells switched off by the k criterion, per variable and boundary
  // face point (update())
  std::vector<dealii::AlignedVector<scalar>> wall_value_off_face;

  // k criterion (set_k_criterion())
  double                          k_criterion_k_plus_on  = 0.0;
  double                          k_criterion_k_plus_off = 0.0;
  OmegaEnrichmentOffWallCondition off_wall_condition_ = OmegaEnrichmentOffWallCondition::WallFunction;

  // k transported as ln k (set_turbulent_kinetic_energy())
  bool k_log_form = false;

  // a = exp(-sum_{n=1}^p 1/n), p the RANS degree: point a y_c of the wall-function value
  double a_wall_function = 1.0;

  // cell-constant fields of the wall cells (shadow layout, dof_index_dg, ghost values set):
  // block 0: k+ = k_bar / u_tau^2, block 1: y_c = mean wall distance of the cell
  BlockVectorType wall_cell_fields;

  // wall cells (global cell index, locally owned and ghost) with the enrichment on by the k
  // criterion; kept from time step to time step (hysteresis)
  std::unordered_set<dealii::types::global_cell_index> k_active_cells;

  // cell loop of set_turbulent_kinetic_energy(): k+ (dst block 0) and y_c (block 1) in all DoFs of
  // each wall cell; src: k (shadow layout, 0), tau_w (1), y (2)
  void
  cell_loop_wall_cell_fields(dealii::MatrixFree<dim, Number> const &       matrix_free,
                             BlockVectorType &                             dst,
                             BlockVectorType const &                       src,
                             std::pair<unsigned int, unsigned int> const & range) const;

  // global active cell indices of the locally owned and ghost enriched cells
  std::unordered_set<dealii::types::global_cell_index> enriched_cells;

  // number of leading (enriched-cell) pairs of the index map
  unsigned int                    n_scalar_index_enriched = 0;

  // [cell batch * n_q_points + q]: exact velocity gradient (set_velocity)
  dealii::AlignedVector<tensor> velocity_gradient_cell;
  dealii::AlignedVector<vector> velocity_value_cell;
  unsigned int                  n_q_points_velocity_gradient = 0;

  // RANS eddy viscosity DoFHandler (same FE as the RANS variables) <-> shadow layout
  dealii::DoFHandler<dim> const * global_dof_handler_eddy_viscosity = nullptr;
  std::vector<unsigned int>       eddy_viscosity_index_global;
  std::vector<unsigned int>       eddy_viscosity_index_shadow;

  // velocity of the fluid (set_velocity): U_bar (dof_index_velocity_bar), U_tilde
  VectorType velocity_bar;
  VectorType velocity_tilde;

  // tilde part: DG scalar of the degree of U_tilde, enriched cells only
  std::shared_ptr<dealii::FiniteElement<dim>> fe_en;
  dealii::DoFHandler<dim>                     dof_handler_en;
  dealii::hp::FECollection<dim>               fe_en_collection;

  dealii::AffineConstraints<Number> constraint_dg;

  unsigned int n_q_points_cell = 0;
  unsigned int n_q_points_face = 0;

  // per variable: [cell batch * n_q_points_cell + q]
  std::vector<dealii::AlignedVector<scalar>> psi_cell;
  std::vector<dealii::AlignedVector<vector>> grad_psi_cell;
  std::vector<dealii::AlignedVector<scalar>> psi_cell_old;

  // per variable, [0] interior, [1] exterior side: [face batch * n_q_points_face + q]
  std::vector<std::array<dealii::AlignedVector<scalar>, 2>> psi_face;
  std::vector<std::array<dealii::AlignedVector<vector>, 2>> grad_psi_face;
};

} // namespace ExaDG

#endif /* INCLUDE_OPERATORS_WALL_MODELLING_FUNCTION_ENRICHMENT_RANS_H_ */
