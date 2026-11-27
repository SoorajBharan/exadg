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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_H_

// deal.II
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

// ExaDG
#include <exadg/convection_diffusion/spatial_discretization/interface.h>
#include <exadg/rans_transport_equations/spatial_discretization/turbulence_model.h>
#include <exadg/convection_diffusion/spatial_discretization/operators/combined_operator.h>
#include <exadg/convection_diffusion/spatial_discretization/operators/rhs_operator.h>
#include <exadg/rans_transport_equations/spatial_discretization/operators/turbulence_source_operator.h>
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/convection_diffusion/user_interface/field_functions.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/grid/grid.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/operators/wall_modelling/function_enrichment_rans.h>
#include <exadg/operators/solution_transfer.h>
#include <exadg/operators/modal_filter.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

#include <exadg/convection_diffusion/user_interface/component_function.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <exadg/utilities/timer_tree.h>
#include <exadg/convection_diffusion/preconditioners/block_preconditioner.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
class Operator : public dealii::Subscriptor, public Interface::Operator<Number>
{
private:
  typedef Operator<dim, Number> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<1, dim, scalar>  vector;
public:
  /*
   * Constructor.
   */
  Operator(std::shared_ptr<Grid<dim> const>                      grid,
           std::shared_ptr<dealii::Mapping<dim> const>           mapping,
           std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings,
           std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor,
           std::shared_ptr<FieldFunctions<dim> const>            field_functions,
           Parameters const &                                    param,
           std::string const &                                   field,
           MPI_Comm const &                                      mpi_comm);


  void
  fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const;

  bool
  uses_block_diagonal_with_ghost_faces() const;

  /**
   * Call this setup() function if the dealii::MatrixFree object can be set up by the present class.
   */
  void
  setup();

  /**
   * Call this setup() function if the dealii::MatrixFree object needs to be created outside this
   * class. The typical use case would be multiphysics-coupling with one MatrixFree object handed
   * over to several single-field solvers. Another typical use case is the use of an ALE
   * formulation. Note that you need to call the function fill_matrix_free_data() beforehand in
   * order to correctly initialize dealii::MatrixFree, which is then handed over to this setup()
   * function.
   */
  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
        std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data_in,
        std::string const &                                    dof_index_velocity_external_in = "");

  /*
   * Call this setup() function for setup after adaptive mesh refinement.
   */
  void
  setup_after_coarsening_and_refinement();

  /**
     * Adaptive mesh refinement, first part of setup_after_coarsening_and_refinement(): DoFs and
     * constraints on the new mesh, without a new MatrixFree object. Used by coupled solvers that set
     * up one MatrixFree object for several operators and then call setup(matrix_free, ...).
     */
  void
  distribute_dofs_after_coarsening_and_refinement();

  /*
   * Initialization of dof-vector.
   */
  virtual void
  initialize_dof_vector(BlockVectorType & src) const override;

  void
  initialize_dof_vector_eddy_viscosity(VectorType & src) const;

  /*
   * Initialization of velocity dof-vector (in case of numerical velocity field).
   */
  void
  initialize_dof_vector_velocity(VectorType & src) const final;

  /*
   * Obtain velocity dof-vector by interpolation of specified analytical velocity field.
   */
  void
  interpolate_velocity(VectorType & velocity, double const time) const;

  /*
   * Obtain velocity dof-vector by L2-projection using the specified analytical velocity field.
   */
  void
  project_velocity(VectorType & velocity, double const time) const final;

  /*
   * Prescribe initial conditions using a specified analytical function.
   */
  void
  prescribe_initial_conditions(BlockVectorType & src, double const evaluation_time) const final;

  /*
   * This function is used in case of explicit time integration:
   *
   * It evaluates the right-hand side operator, the convective and diffusive terms (subsequently
   * multiplied by -1.0 in order to shift these terms to the right-hand side of the equations) and
   * finally applies the inverse mass operator.
   */
  void
  evaluate_explicit_time_int(BlockVectorType &       dst,
                             BlockVectorType const & src,
                             double const       evaluation_time,
                             VectorType const * velocity = nullptr) const final;

  /*
   * This function evaluates the convective term which is needed when using an explicit formulation
   * for the convective term.
   */
  void
  evaluate_convective_term(BlockVectorType &       dst,
                           BlockVectorType const & src,
                           double const       evaluation_time,
                           VectorType const * velocity = nullptr) const;

  /*
   * This function calculates the inhomogeneous parts of all operators arising e.g. from
   * inhomogeneous boundary conditions or the solution at previous instants of time occurring in the
   * discrete time derivative term.
   *
   * Note that the convective operator only has a contribution to the right-hand side if it is
   * formulated implicitly in time. In case of an explicit treatment the whole convective operator
   * (call function evaluate_convective_term() instead of rhs()) has to be added to the right-hand
   * side of the equations.
   */
  void
  rhs(BlockVectorType &       dst,
      BlockVectorType const & src,
      double const       evaluation_time = 0.0,
      VectorType const * velocity        = nullptr) const final;

  /*
   * Turbulence terms of the linear system of the BDF time step for Q^{n+1} (linearly implicit,
   * linearized at the previous time step Q^n): dst += Explicit(Q^n) + Newton_Explicit(Q^n) +
   * J(Q^n) Q^n. time_n: time of the Dirichlet values of the explicit turbulent diffusion, time:
   * t^{n+1}. J(Q^n) Q^{n+1} is applied in vmult().
   */
  void
  rhs_add_turbulence(BlockVectorType &       dst,
                     BlockVectorType const & solution_n,
                     double const            time_n,
                     double const            time,
                     VectorType const *      velocity) const;

  /*
   * CellDirichlet boundaries: dst += lambda (g(t), v) on the cells adjacent to the boundary, with
   * lambda = cell_dirichlet_penalty_factor / dt and t of update_time_step_size() / rhs() of this
   * time step. Called once per time step at t_{n+1}, after rhs().
   */
  void
  rhs_add_cell_dirichlet(BlockVectorType & dst) const;

  /*
   * Newton solver: dst += right-hand side of the turbulence terms treated Explicit, at the previous
   * time step solution_n. Call once per time step, before evaluate_add_turbulence_newton_explicit().
   */
  void
  evaluate_add_turbulence_explicit(BlockVectorType &       dst,
                                   BlockVectorType const & solution_n,
                                   double const            time_n,
                                   VectorType const *      velocity) const;

  /*
   * Newton solver: dst += right-hand side of the Newton_Explicit part of the turbulence terms
   * treated Implicit, at solution (the Newton iterate). solution becomes the linearization point of
   * vmult().
   */
  void
  evaluate_add_turbulence_newton_explicit(BlockVectorType &       dst,
                                          BlockVectorType const & solution,
                                          double const            time,
                                          VectorType const *      velocity) const;

  /*
   * This function applies the mass operator to the src-vector and writes the result to the
   * dst-vector.
   */
  virtual void
  apply_mass_operator(BlockVectorType & dst,
                      BlockVectorType const & src) const;

  /*
   * This function applies the mass operator to the src-vector and adds the result to the
   * dst-vector.
   */
  virtual void
  apply_mass_operator_add(BlockVectorType & dst,
                          BlockVectorType const & src) const;

  /*
   * Inverse mass operator for all blocks; with enrichment of RANS variables the block inverse
   * (thesis Eqs. (6.16), (6.17)) on the enriched cells. dst and src may be the same vector.
   */
  virtual void
  apply_inverse_mass_operator(BlockVectorType & dst, BlockVectorType const & src) const;

  /*
   * Enrichment of RANS variables, conv-diff operator: mass term of the enriched variables with
   * the block mass on the enriched cells, dst.block(r) += f (M_block - M) src (enriched cells),
   * dst.block(n_blocks + i) = f M_block src (psi v_tilde rows), f = scaling factor of the mass
   */
  void
  apply_add_enriched_mass_correction(BlockVectorType & dst, BlockVectorType const & src) const;

  /*
   * Enrichment of RANS variables, preconditioner of the W_tilde blocks: dst = (1/f) S^+ src, the
   * W_tilde part of the block inverse mass for a zero bar part (f = scaling factor of the mass)
   */
  void
  apply_enrichment_preconditioner(BlockVectorType & dst, BlockVectorType const & src) const;

  /*
   * Enrichment of RANS variables: re-expresses a stored solution W_bar + psi_old W_tilde in the
   * space with the current psi (L2 projection on the enriched cells, after
   * FunctionEnrichmentRANS::update()); identity on all other cells.
   */
  void
  project_enrichment_history(BlockVectorType & solution) const;

  /*
   * Enrichment of RANS variables, initial values: on the enriched cells the enriched variables
   * become the L2 projection of the wall law (u_tau of the fluid) onto the enriched space
   * (W_bar, W_tilde); all other cells and variables unchanged.
   */
  void
  initialize_enrichment(BlockVectorType & solution) const;

  /*
   * Enrichment of RANS variables: W_tilde blocks -> projector P onto the well-posed enrichment
   * modes (FunctionEnrichmentRANS); the linear systems are solved on W_tilde in range(P). No-op
   * without enrichment.
   */
  void
  apply_enrichment_projector(BlockVectorType & vector) const;

  /*
   * Output (VTU, line plots): dst = src, with the enriched RANS block replaced by its total
   * field at the nodes, omega_bar + psi_omega omega_tilde (or W_bar + psi_omega W_tilde), on the
   * enriched cells. Without enrichment dst = src.
   */
  virtual void
  get_solution_for_output(BlockVectorType & dst, BlockVectorType const & src) const;

  // enrichment of RANS variables: k of the RANS solution (block 0) to FunctionEnrichmentRANS (k+
  // and wall distance of the wall cells, k criterion of the omega enrichment)
  void
  set_enrichment_turbulent_kinetic_energy(BlockVectorType const & solution) const
  {
    if(not function_enrichment_rans)
      return;
    bool const k_logarithmic = param.turbulence_model_data.positivity_preserving_limiter[0] ==
                               PositivityPreservingLimiter::LogarithmicTransportVariable;
    function_enrichment_rans->set_turbulent_kinetic_energy(solution.block(0), k_logarithmic);
  }

#ifdef EXADG_DEBUG
  // diagnostics of the wall layer of every enriched variable
  // (FunctionEnrichmentRANS::print_wall_diagnostics)
  void
  print_enrichment_wall_diagnostics(BlockVectorType const & solution,
                                    unsigned int const      time_step_number) const;
#endif

#ifdef EXADG_DEBUG
  // diagnostics: dst += Newton_Implicit part of one group of turbulence terms applied to src
  // (TurbulenceSourceOperator::apply_add_newton_implicit_part)
  void
  apply_add_turbulence_jacobian_part(BlockVectorType &       dst,
                                     BlockVectorType const & src,
                                     unsigned int const      part) const
  {
    turbulence_source_operator.apply_add_newton_implicit_part(dst, src, part);
  }

  // diagnostics: term balance of the enriched wall cells at the x stations, printed by the next
  // evaluation of the turbulence terms (TurbulenceSourceOperator::request_omega_balance)
  void
  request_omega_balance(std::vector<double> const & x_stations, unsigned int const time_step) const
  {
    turbulence_source_operator.request_omega_balance(x_stations, time_step);
  }
#endif

  /*
   * This function applies the convective operator to the src-vector and writes the result to the
   * dst-vector. It is needed for throughput measurements of the matrix-free implementation.
   */
  void
  apply_convective_term(BlockVectorType & dst,
                        BlockVectorType const & src) const;

  void
  update_convective_term(double const evaluation_time, VectorType const * velocity = nullptr) const;

  /*
   * This function applies the diffusive operator to the src-vector and writes the result to the
   * dst-vector. It is needed for throughput measurements of the matrix-free implementation.
   */
  void
  apply_diffusive_term(BlockVectorType & dst,
                       BlockVectorType const & src) const;

  /*
   * This function applies the combined mass-convection-diffusion operator to the src-vector
   * and writes the result to the dst-vector. It is needed for throughput measurements of the
   * matrix-free implementation.
   */
  void
  apply_conv_diff_operator(BlockVectorType & dst,
                           BlockVectorType const & src) const;

  void
  update_conv_diff_operator(double const       evaluation_time,
                            double const       scaling_factor,
                            VectorType const * velocity = nullptr);

  /*
   * Updates spatial operators after grid has been moved.
   */
  void
  update_after_grid_motion(bool const update_matrix_free);

  /*
   * Prepare and interpolation in adaptive mesh refinement.
   */
  void
  prepare_coarsening_and_refinement(std::vector<VectorType *> & vectors);

  void
  interpolate_after_coarsening_and_refinement(std::vector<VectorType *> & vectors);

  std::vector<dealii::Vector<float>>
  compute_error_indicators(BlockVectorType const & solution) const;

  /*
   * This function solves the linear system of equations in case of implicit time integration or
   * steady-state problems (potentially involving the mass, convective, and diffusive
   * operators).
   */
  unsigned int
  solve(BlockVectorType &       sol,
        BlockVectorType const & rhs,
        bool const         update_preconditioner,
        double const       scaling_factor = -1.0,
        double const       time           = -1.0,
        VectorType const * velocity       = nullptr) final;


  unsigned int
  get_n_blocks() const;

  TurbulenceModelData const &
  get_turbulence_model_data() const;

  virtual void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const;

  /*
   * Calculate time step size according to maximum efficiency criterion
   */
  double
  calculate_time_step_max_efficiency(unsigned int const order_time_integrator) const final;

  /*
   * Calculate time step size according to CFL criterion
   */

  // global CFL criterion
  double
  calculate_time_step_cfl_global(double const time) const final;

  // local CFL criterion: use numerical velocity field
  double
  calculate_time_step_cfl_numerical_velocity(VectorType const & velocity) const final;

  // local CFL criterion: use analytical velocity field
  double
  calculate_time_step_cfl_analytical_velocity(double const time) const final;

  /*
   * Calculate time step size according to diffusion term
   */
  double
  calculate_time_step_diffusion() const final;

  /*
   * Setters and getters.
   */

  dealii::MatrixFree<dim, Number> const &
  get_matrix_free() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_velocity() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler_eddy_viscosity() const;

  dealii::types::global_dof_index
  get_number_of_dofs() const;

  std::string
  get_dof_name() const;

  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const;

  dealii::AffineConstraints<Number> const &
  get_constraints() const;

  void
  update_eddy_viscosity(BlockVectorType const & src,
                        VectorType const * velocity = nullptr) const;

  void
  get_eddy_viscosity(VectorType & dst) const;

  // wall model: upper bound of the eddy viscosity in the turbulence source terms
  void
  set_eddy_viscosity_wall_law(VectorType const & eddy_viscosity_wall_law) const;

  void
  set_tke_production_from_wall_law(VectorType const & tke_production_from_wall_law,
                                   VectorType const & wall_cell_indicator) const;

  // wall model: exact velocity gradient of the enriched cells (dim*dim fields) and its indicator,
  // on the eddy viscosity DoFHandler; replaces the gradient of the velocity field in the cell terms
  // of the turbulence source terms where indicator = 1
  void
  set_velocity_gradient_enriched(std::vector<VectorType> const & velocity_gradient,
                                 VectorType const &              indicator) const;

  // wall model: enrichment of RANS variables in the wall cells (omega_h = omega_bar + psi_omega
  // omega_tilde); set by the driver after setup(), before the DoF vectors are initialized
  void
  set_function_enrichment_rans(
    std::shared_ptr<FunctionEnrichmentRANS<dim, Number>> function_enrichment_rans_in);

  // nullptr: no enrichment of RANS variables
  std::shared_ptr<FunctionEnrichmentRANS<dim, Number>> const &
  get_function_enrichment_rans() const;

  /*
   * Dof index for eddy viscosity (in case of turbulence model with eddy viscosity).
   */
  std::string
  get_dof_name_eddy_viscosity() const;

  unsigned int
  get_dof_index_eddy_viscosity() const;

  void
  get_turbulent_kinetic_energy(VectorType & dst, BlockVectorType const & solution) const;

  virtual void
  update_time_step_size(double const dt,
                        double const time,
                        double const end_time,
                        double const time_step_number) const;

  void
  apply_modal_filter(BlockVectorType & solution) const;

  void
  enforce_positivity(BlockVectorType & solution) const;

protected:
  /*
   * Hooks for derived operators (e.g. RANS::Operator), called at the places of the setup where a
   * derived operator adds its own DoFHandlers, mapping flags and quadrature rules; the order of
   * the MatrixFree entries is the same as before. Note: initialize_dof_handler_and_constraints()
   * is called by the constructor, where a virtual call runs this base version; a derived operator
   * distributes its DoFs in its own constructor, the hook is used after adaptive mesh refinement.
   */
  virtual void
  distribute_dofs_additional();

  virtual void
  append_mapping_flags_additional(MatrixFreeData<dim, Number> & matrix_free_data) const;

  virtual void
  insert_dof_handlers_additional(MatrixFreeData<dim, Number> & matrix_free_data) const;

  virtual void
  insert_quadratures_additional(MatrixFreeData<dim, Number> & matrix_free_data) const;

  virtual void
  setup_operators_additional();

  virtual std::shared_ptr<PreconditionerBase<Number>>
  create_block_jacobi_preconditioner_additional(unsigned int const block) const;

  virtual void
  rhs_additional(BlockVectorType & dst, double const time) const;

  virtual void
  apply_add_additional(BlockVectorType & dst, BlockVectorType const & src) const;

  virtual void
  set_velocity_additional(VectorType const & velocity);

  void
  vmult_transport_variables(BlockVectorType & dst, BlockVectorType const & src) const;

private:
  void
  do_setup();

  /**
   * Initializes dealii::DoFHandlers and dealii::AffineConstraints.
   */
  void
  initialize_dof_handler_and_constraints();

  /**
   * Performs setup of operators.
   */
  void
  setup_operators();

  /**
   * Calculates maximum velocity (required for global CFL criterion).
   */
  double
  calculate_maximum_velocity(double const time) const;

  /**
   * Calculates minimum element length.
   */
  double
  calculate_minimum_element_length() const;

  bool
  needs_own_dof_handler_velocity() const;

  std::string
  get_quad_name() const;

  std::string
  get_quad_name_overintegration() const;

  std::string
  get_dof_name_velocity() const;

  std::string
  get_quad_name_eddy_velocity() const;

  /*
   * Dof index for velocity (in case of numerical velocity field)
   */
  unsigned int
  get_dof_index_velocity() const;

  unsigned int
  get_quad_index_overintegration() const;

  unsigned int
  get_quad_index_eddy_velocity() const;

  /*
   * Initializes the preconditioner.
   */
  void
  setup_preconditioner();

  /*
   * Initializes the solver.
   */
  void
  setup_solver();

  /*
   * Grid
   */
  std::shared_ptr<Grid<dim> const> grid;

  /*
   * SolutionTransfer for adaptive mesh refinement.
   */
  std::shared_ptr<ExaDG::SolutionTransfer<dim, VectorType>> solution_transfer;

  /*
   * Grid motion for ALE formulations
   */
  std::shared_ptr<dealii::Mapping<dim> const> mapping;

  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim> const>     field_functions;

  std::vector<std::shared_ptr<BoundaryDescriptor<dim> const>> boundary_descriptor_block;

  /*
   * List of parameters.
   */
  Parameters const & param;

  std::string const field;

  unsigned int n_blocks;

  /*
   * Basic finite element ingredients.
   */
  std::shared_ptr<dealii::FiniteElement<dim>> fe;
  dealii::DoFHandler<dim>                     dof_handler;

  /*
   * Numerical velocity field.
   */
  std::shared_ptr<dealii::FiniteElement<dim>> fe_velocity;
  std::shared_ptr<dealii::DoFHandler<dim>>    dof_handler_velocity;

  /*
   * Eddy viscosity field.
   */
  std::shared_ptr<dealii::FiniteElement<dim>> fe_eddy_viscosity;
  std::shared_ptr<dealii::DoFHandler<dim>>    dof_handler_eddy_viscosity;

  /*
   * Enrichment of RANS variables in the wall cells (nullptr: none).
   */
  std::shared_ptr<FunctionEnrichmentRANS<dim, Number>> function_enrichment_rans;

  /*
   * Enrichment of RANS variables: explicit convective term of the enriched variables on the
   * enriched wall cells and their faces (evaluate_convective_term), MatrixFree of
   * FunctionEnrichmentRANS, exact velocity u_h. dst.block(r) += C(psi W_tilde) (enriched and halo
   * cells), dst.block(n_blocks + i) = psi v_tilde rows of C(W_h).
   */
  void
  evaluate_add_enriched_convective_term(BlockVectorType &       dst,
                                        BlockVectorType const & src,
                                        double const            time) const;

  void
  cell_loop_convective_enriched(dealii::MatrixFree<dim, Number> const &       matrix_free,
                                std::vector<VectorType> &                     dst,
                                BlockVectorType const &                       src,
                                std::pair<unsigned int, unsigned int> const & cell_range) const;

  void
  face_loop_convective_enriched(dealii::MatrixFree<dim, Number> const &       matrix_free,
                                std::vector<VectorType> &                     dst,
                                BlockVectorType const &                       src,
                                std::pair<unsigned int, unsigned int> const & face_range) const;

  void
  boundary_face_loop_convective_enriched(
    dealii::MatrixFree<dim, Number> const &       matrix_free,
    std::vector<VectorType> &                     dst,
    BlockVectorType const &                       src,
    std::pair<unsigned int, unsigned int> const & face_range) const;

  // numerical flux of the convective term (DivergenceFormulation), as ExaDG's ConvectiveKernel
  scalar
  convective_flux_enriched(scalar const & value_m,
                           scalar const & value_p,
                           scalar const & normal_velocity_m,
                           scalar const & normal_velocity_p) const;

  // enriched variable and time of the current evaluate_add_enriched_convective_term() call
  mutable unsigned int variable_convective_enriched = 0;
  mutable double       time_convective_enriched     = 0.0;

  // per enriched variable: nodal values psi W_tilde (global RANS layout) for the eddy viscosity
  mutable std::vector<VectorType> enrichment_nodal;

  /*
   * Constraints.
   */
  dealii::AffineConstraints<Number> affine_constraints;

  std::string const dof_index_std      = "conv_diff";
  std::string const dof_index_velocity = "conv_diff_velocity";
  std::string const dof_index_eddy_viscosity = "eddy_viscosity";

  std::string const quad_index_std             = "conv_diff";
  std::string const quad_index_overintegration = "conv_diff_overintegration";
  std::string const quad_index_eddy_velocity   = "conv_diff_eddy_velocity";

  std::string dof_index_velocity_external;

  /*
   * Matrix-free operator evaluation.
   */
  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data;

  // If we want to be able to update the mapping, we need a pointer to a non-const MatrixFree
  // object. In case this object is created, we let the above object called matrix_free point to
  // matrix_free_own_storage. This variable is needed for ALE formulations.
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free_own_storage;

  /*
   * Basic operators.
   */
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel;
  std::vector<std::shared_ptr<Operators::DiffusiveKernel<dim, Number>>>  diffusive_kernel;
  std::vector<std::shared_ptr<Operators::RHSKernel<dim, Number>>>  rhs_kernel;

  std::vector<std::shared_ptr<ConvectiveOperator<dim, Number>>>     convective_operator;
  std::vector<std::shared_ptr<DiffusiveOperator<dim, Number>>>      diffusive_operator;
  std::vector<std::shared_ptr<RHSOperator<dim, Number>>>            rhs_operator;

  MassOperator<dim, 1, Number>        mass_operator;
  InverseMassOperator<dim, 1, Number> inverse_mass_operator;

  std::shared_ptr<Operators::TurbulenceSourceKernel<dim, Number>>  turbulence_source_kernel;
  TurbulenceSourceOperator<dim, Number>            turbulence_source_operator;

  /*
   * Combined operator.
   */
  std::vector<std::shared_ptr<CombinedOperator<dim, Number>>> combined_operator;

  /*
   * Solvers and preconditioners
   */
  std::vector<std::shared_ptr<PreconditionerBase<Number>>>     preconditioner;

  BlockPreconditioner<Number> block_preconditioner;

  std::shared_ptr<Krylov::SolverBase<BlockVectorType>> iterative_solver;

  /*
   * MPI
   */
  MPI_Comm const mpi_comm;

  /*
   * Output to screen.
   */
  dealii::ConditionalOStream pcout;

  /*
   * Turbulence models.
   */
  std::shared_ptr<TurbulenceModel<dim, Number>> turbulence_model_ptr = std::make_shared<TurbulenceModel<dim, Number>>();

  /*
   * Modal Filter
   */
  ModalFilter<dim, 1, Number> modal_filter;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_H_ */
