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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_COUPLED_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_COUPLED_H_

#include <exadg/convection_diffusion/spatial_discretization/operators/combined_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>
#include <exadg/solvers_and_preconditioners/newton/newton_solver.h>

namespace ExaDG
{
namespace IncNS
{
// forward declaration
template<int dim, typename Number>
class OperatorCoupled;

template<int dim, typename Number>
class NonlinearOperatorCoupled
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef OperatorCoupled<dim, Number> PDEOperator;

public:
  NonlinearOperatorCoupled()
    : pde_operator(nullptr), rhs_vector(nullptr), time(0.0), scaling_factor_mass(1.0)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  void
  update(VectorType const & rhs_vector, double const & time, double const & scaling_factor)
  {
    this->rhs_vector          = &rhs_vector;
    this->time                = time;
    this->scaling_factor_mass = scaling_factor;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'evaluate_residual'.
   */
  void
  evaluate_residual(BlockVectorType & dst, BlockVectorType const & src) const
  {
    pde_operator->evaluate_nonlinear_residual(dst, src, rhs_vector, time, scaling_factor_mass);
  }

private:
  PDEOperator const * pde_operator;

  VectorType const * rhs_vector;
  double             time;
  double             scaling_factor_mass;
};

template<int dim, typename Number>
class LinearOperatorCoupled : public dealii::Subscriptor
{
private:
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef OperatorCoupled<dim, Number> PDEOperator;

public:
  LinearOperatorCoupled() : dealii::Subscriptor(), pde_operator(nullptr)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'set_solution_linearization'.
   */
  void
  set_solution_linearization(BlockVectorType const & solution_linearization) const
  {
    pde_operator->set_velocity_ptr(solution_linearization.block(0));
  }


  /*
   * The implementation of linear solvers in deal.ii requires that a function called 'vmult' is
   * provided.
   */
  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    pde_operator->apply_linearized_problem(dst, src);
  }

private:
  PDEOperator const * pde_operator;
};

template<int dim, typename Number>
class BlockPreconditioner
{
private:
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef OperatorCoupled<dim, Number> PDEOperator;

public:
  BlockPreconditioner() : update_needed(true), pde_operator(nullptr)
  {
  }

  void
  initialize(PDEOperator * pde_operator_in)
  {
    pde_operator = pde_operator_in;
  }

  void
  update()
  {
    pde_operator->update_block_preconditioner();

    this->update_needed = false;
  }

  bool
  needs_update() const
  {
    return update_needed;
  }

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    AssertThrow(this->update_needed == false,
                dealii::ExcMessage(
                  "BlockPreconditioner can not be applied because it is not up-to-date."));

    pde_operator->apply_block_preconditioner(dst, src);
  }

  std::shared_ptr<TimerTree>
  get_timings() const
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "Function get_timings() is not implemented for BlockPreconditioner."));

    return std::make_shared<TimerTree>();
  }

private:
  bool update_needed;

  PDEOperator * pde_operator;
};

/*
 * Wall enrichment (Krank et al. 2018) for the coupled solver. The unknowns are stored in a block
 * vector with three blocks
 *
 *   X = (U_bar, U_tilde, p^*),   u_h = \bar{u}_h + \psi \tilde{u}_h,   p = s_c p^*,
 *
 * block(0): U_bar (global DG layout), block(1): U_tilde (enrichment layout), block(2): scaled
 * pressure p^* (s_c = scaling_factor_continuity, as in the standard coupled solver). The momentum
 * rows are tested with \bar{v}_h and \psi \tilde{v}_h, the continuity rows with q_h. The classes
 * below mirror NonlinearOperatorCoupled, LinearOperatorCoupled and BlockPreconditioner for this
 * layout so that ExaDG's Newton::Solver can be used.
 */
template<int dim, typename Number>
class NonlinearOperatorCoupledEnriched
{
private:
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef OperatorCoupled<dim, Number> PDEOperator;

public:
  NonlinearOperatorCoupledEnriched()
    : pde_operator(nullptr), rhs_velocity(nullptr), time(0.0), scaling_factor_mass(1.0)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  // rhs_velocity: (U_bar, U_tilde) rows of the constant right-hand side
  void
  update(BlockVectorType const & rhs_velocity, double const & time, double const & scaling_factor)
  {
    this->rhs_velocity        = &rhs_velocity;
    this->time                = time;
    this->scaling_factor_mass = scaling_factor;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'evaluate_residual'.
   */
  void
  evaluate_residual(BlockVectorType & dst, BlockVectorType const & src) const
  {
    pde_operator->evaluate_nonlinear_residual_enriched(
      dst, src, *rhs_velocity, time, scaling_factor_mass);
  }

private:
  PDEOperator const * pde_operator;

  BlockVectorType const * rhs_velocity;
  double                  time;
  double                  scaling_factor_mass;
};

template<int dim, typename Number>
class LinearOperatorCoupledEnriched : public dealii::Subscriptor
{
private:
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef OperatorCoupled<dim, Number> PDEOperator;

public:
  LinearOperatorCoupledEnriched() : dealii::Subscriptor(), pde_operator(nullptr)
  {
  }

  void
  initialize(PDEOperator const & pde_operator)
  {
    this->pde_operator = &pde_operator;
  }

  /*
   * The implementation of the Newton solver requires a function called
   * 'set_solution_linearization'.
   */
  void
  set_solution_linearization(BlockVectorType const & solution_linearization) const
  {
    pde_operator->set_linearization_enriched(solution_linearization);
  }

  /*
   * The implementation of linear solvers in deal.ii requires that a function called 'vmult' is
   * provided.
   */
  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    pde_operator->apply_linearized_problem_enriched(dst, src);
  }

private:
  PDEOperator const * pde_operator;
};

template<int dim, typename Number>
class BlockPreconditionerCoupledEnriched
{
private:
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef OperatorCoupled<dim, Number> PDEOperator;

public:
  BlockPreconditionerCoupledEnriched() : pde_operator(nullptr)
  {
  }

  void
  initialize(PDEOperator * pde_operator_in)
  {
    pde_operator = pde_operator_in;
  }

  void
  update()
  {
    pde_operator->update_block_preconditioner_enriched();
  }

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    pde_operator->apply_block_preconditioner_enriched(dst, src);
  }

private:
  PDEOperator * pde_operator;
};

/*
 * Krylov solver (GMRES/FGMRES, right preconditioning as in ExaDG's Krylov::SolverGMRES) for the
 * linear(ized) enriched coupled system, with the interface required by Newton::Solver.
 */
template<int dim, typename Number>
class LinearSolverCoupledEnriched
{
private:
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

public:
  LinearSolverCoupledEnriched()
    : linear_operator(nullptr),
      preconditioner(nullptr),
      solver_type(SolverCoupled::GMRES),
      use_preconditioner(true)
  {
  }

  void
  initialize(LinearOperatorCoupledEnriched<dim, Number> const & linear_operator_in,
             BlockPreconditionerCoupledEnriched<dim, Number> &  preconditioner_in,
             SolverData const &                                 solver_data_in,
             SolverCoupled const                                solver_type_in,
             bool const                                         use_preconditioner_in)
  {
    linear_operator    = &linear_operator_in;
    preconditioner     = &preconditioner_in;
    solver_data        = solver_data_in;
    solver_type        = solver_type_in;
    use_preconditioner = use_preconditioner_in;
  }

  void
  update_preconditioner(bool const update) const
  {
    if(update and use_preconditioner)
      preconditioner->update();
  }

  unsigned int
  solve(BlockVectorType & dst, BlockVectorType const & rhs) const;

private:
  LinearOperatorCoupledEnriched<dim, Number> const * linear_operator;
  BlockPreconditionerCoupledEnriched<dim, Number> *  preconditioner;

  SolverData    solver_data;
  SolverCoupled solver_type;
  bool          use_preconditioner;
};

template<int dim, typename Number = double>
class OperatorCoupled : public SpatialOperatorBase<dim, Number>
{
private:
  typedef SpatialOperatorBase<dim, Number> Base;
  typedef OperatorCoupled<dim, Number>     This;

  typedef typename Base::MultigridPoisson MultigridPoisson;

  typedef typename Base::VectorType VectorType;

  typedef typename Base::BlockVectorType BlockVectorType;

public:
  /*
   * Constructor.
   */
  OperatorCoupled(std::shared_ptr<Grid<dim> const>                      grid,
                  std::shared_ptr<dealii::Mapping<dim> const>           mapping,
                  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings,
                  std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor,
                  std::shared_ptr<FieldFunctions<dim> const>            field_functions,
                  Parameters const &                                    parameters,
                  std::string const &                                   field,
                  MPI_Comm const &                                      mpi_comm);

  /*
   * Destructor.
   */
  virtual ~OperatorCoupled();

private:
  void
  setup_derived() final;

  void
  setup_preconditioners_and_solvers() final;

public:
  /*
   *  Update divergence penalty operator by recalculating the penalty parameter
   *  which depends on the current velocity field
   */
  void
  update_divergence_penalty_operator(VectorType const & velocity);

  /*
   *  Update continuity penalty operator by recalculating the penalty parameter
   *  which depends on the current velocity field
   */
  void
  update_continuity_penalty_operator(VectorType const & velocity);

  /*
   * Setters and getters.
   */

  /*
   *  This function sets the variable scaling_factor_continuity, and also the related scaling factor
   * for the pressure gradient operator.
   */
  void
  set_scaling_factor_continuity(double const scaling_factor);

  /*
   * Stokes equations or convective term treated explicitly: solve linear system of equations
   */

  /*
   * This function solves the linear Stokes problem (Stokes equations or Navier-Stokes
   * equations with an explicit treatment of the convective term). The parameter
   * scaling_factor_mass has to be specified for unsteady problem and
   * can be omitted for steady problems.
   */
  unsigned int
  solve_linear_stokes_problem(BlockVectorType &       dst,
                              BlockVectorType const & src,
                              bool const &            update_preconditioner,
                              double const &          scaling_factor_mass = 1.0);

  /*
   * Convective term treated implicitly: solve non-linear system of equations
   */

  /*
   * This function solves the nonlinear problem.
   */
  std::tuple<unsigned int, unsigned int>
  solve_nonlinear_problem(BlockVectorType &  dst,
                          VectorType const & rhs_vector,
                          bool const &       update_preconditioner,
                          double const &     time                = 0.0,
                          double const &     scaling_factor_mass = 1.0);


  /*
   * This function evaluates the nonlinear residual.
   */
  void
  evaluate_nonlinear_residual(BlockVectorType &       dst,
                              BlockVectorType const & src,
                              VectorType const *      rhs_vector,
                              double const &          time,
                              double const &          scaling_factor_mass) const;

  /*
   * This function evaluates the nonlinear residual of the steady Navier-Stokes equations.
   * This function has to be implemented separately (for example, the convective term will be
   * evaluated in case of the Navier-Stokes equations and the time-derivative term is never
   * evaluated).
   */
  void
  evaluate_nonlinear_residual_steady(BlockVectorType &       dst,
                                     BlockVectorType const & src,
                                     double const &          time) const;

  /*
   * This function calculates the matrix-vector product for the linear(ized) problem.
   */
  void
  apply_linearized_problem(BlockVectorType & dst, BlockVectorType const & src) const;

  /*
   * This function calculates the right-hand side of the steady Stokes problem, or unsteady Stokes
   * problem, or unsteady Navier-Stokes problem with explicit treatment of the convective term. The
   * parameter 'time' has to be specified for unsteady problems and can be omitted for steady
   * problems.
   */
  void
  rhs_stokes_problem(BlockVectorType & dst, double const & time = 0.0) const;

  /*
   * Block preconditioner
   */
  void
  update_block_preconditioner();

  void
  apply_block_preconditioner(BlockVectorType & dst, BlockVectorType const & src) const;

  /*
   * Wall enrichment (Krank et al. 2018): coupled system for X = (U_bar, U_tilde, p^*), see
   * NonlinearOperatorCoupledEnriched. Saddle point system (homogeneous part)
   *
   *   [ A_h   s_c G_h ] [ u_h ]   [ b_u ]
   *   [ -s_c D_h   0  ] [ p^* ] = [ b_p ],
   *
   * A_h: mass + viscous (+ linearized convective) operator tested with \bar{v}_h and
   * \psi \tilde{v}_h, G_h/D_h: pressure gradient/velocity divergence of u_h. The velocity rows
   * are restricted to the well-posed enrichment space and to the strong no-slip condition
   * (if selected) with E = diag(E_bar, P) as in EnrichmentConstrainedOperator:
   *   K_c = E K E + (I - E) on the velocity rows, right-hand side E b_u.
   */

  // X with three blocks (U_bar, U_tilde, p^*)
  void
  initialize_block_vector_velocity_pressure_enriched(BlockVectorType & dst) const;

  /*
   * Right-hand side of the Stokes problem (or explicit convective term): inhomogeneous boundary
   * data of the pressure gradient, viscous and divergence terms, and the body force, all rows.
   * The terms of the time derivative and the explicit convective term are added by the caller.
   */
  void
  rhs_stokes_problem_enriched(BlockVectorType & dst, double const time) const;

  unsigned int
  solve_linear_stokes_problem_enriched(BlockVectorType &       dst,
                                       BlockVectorType const & rhs,
                                       double const            scaling_factor_mass);

  /*
   * Implicit convective term: Newton's method for X; rhs_velocity contains the constant
   * (U_bar, U_tilde) rows of the right-hand side (time derivative history, body force).
   */
  std::tuple<unsigned int, unsigned int>
  solve_nonlinear_problem_enriched(BlockVectorType &       dst,
                                   BlockVectorType const & rhs_velocity,
                                   double const            time,
                                   double const            scaling_factor_mass);

  // constrained nonlinear residual R_c(X) = E R(X) + (I - E) X (velocity rows)
  void
  evaluate_nonlinear_residual_enriched(BlockVectorType &       dst,
                                       BlockVectorType const & src,
                                       BlockVectorType const & rhs_velocity,
                                       double const            time,
                                       double const            scaling_factor_mass) const;

  // linearization point w_h of the convective term (Newton)
  void
  set_linearization_enriched(BlockVectorType const & solution_linearization) const;

  // constrained linear(ized) operator K_c
  void
  apply_linearized_problem_enriched(BlockVectorType & dst, BlockVectorType const & src) const;

  void
  update_block_preconditioner_enriched();

  void
  apply_block_preconditioner_enriched(BlockVectorType & dst, BlockVectorType const & src) const;

private:
  void
  setup_solver_coupled();

  /*
   * Wall enrichment
   */
  void
  setup_wall_enrichment_coupled();

  // X <- E X on the velocity blocks (block(0), block(1)) of X = (U_bar, U_tilde, p^*)
  void
  constrain_velocity_enriched(BlockVectorType & dst) const;

  // (U_bar, U_tilde) of X = (U_bar, U_tilde, p^*) to/from a velocity block vector
  void
  copy_velocity_enriched(BlockVectorType & velocity, BlockVectorType const & src) const;

  void
  set_velocity_enriched(BlockVectorType & dst, BlockVectorType const & velocity) const;

  // A_h u_h: mass + viscous (+ linearized convective) terms, homogeneous
  void
  apply_velocity_block_enriched(BlockVectorType & dst, BlockVectorType const & velocity) const;

  // approximate inverse of the velocity block (block-Jacobi, includes the constraints E)
  void
  apply_preconditioner_velocity_block_enriched(BlockVectorType &       dst,
                                               BlockVectorType const & src) const;

  /*
   * Block preconditioner
   */
  void
  setup_block_preconditioner();

  void
  initialize_vectors();

  void
  initialize_preconditioner_velocity_block();

  void
  setup_multigrid_preconditioner_momentum();

  void
  setup_iterative_solver_momentum();

  void
  initialize_preconditioner_pressure_block();

  void
  setup_multigrid_preconditioner_schur_complement();

  void
  setup_iterative_solver_schur_complement();

  void
  setup_pressure_convection_diffusion_operator();

  void
  apply_preconditioner_velocity_block(VectorType & dst, VectorType const & src) const;

  void
  apply_preconditioner_pressure_block(VectorType & dst, VectorType const & src) const;

  void
  apply_inverse_negative_laplace_operator(VectorType & dst, VectorType const & src) const;

  /*
   * Newton-Krylov solver for (non-)linear problem
   */

  // temporary vector needed to evaluate both the nonlinear residual and the linearized operator
  VectorType mutable temp_vector;

  double scaling_factor_continuity;

  // Nonlinear operator
  NonlinearOperatorCoupled<dim, Number> nonlinear_operator;

  // Newton solver
  std::shared_ptr<Newton::Solver<BlockVectorType,
                                 NonlinearOperatorCoupled<dim, Number>,
                                 LinearOperatorCoupled<dim, Number>,
                                 Krylov::SolverBase<BlockVectorType>>>
    newton_solver;

  // Linear operator
  LinearOperatorCoupled<dim, Number> linear_operator;

  // Linear solver
  std::shared_ptr<Krylov::SolverBase<BlockVectorType>> linear_solver;

  /*
   * Block preconditioner for linear(ized) problem
   */
  typedef BlockPreconditioner<dim, Number> Preconditioner;
  Preconditioner                           block_preconditioner;

  // preconditioner velocity/momentum block
  std::shared_ptr<PreconditionerBase<Number>> preconditioner_momentum;

  std::shared_ptr<Krylov::SolverBase<VectorType>> solver_velocity_block;

  // preconditioner pressure/Schur-complement block
  std::shared_ptr<PreconditionerBase<Number>> multigrid_preconditioner_schur_complement;
  std::shared_ptr<PreconditionerBase<Number>> inverse_mass_preconditioner_schur_complement;

  std::shared_ptr<ConvDiff::CombinedOperator<dim, 1, Number>> pressure_conv_diff_operator;

  std::shared_ptr<Poisson::LaplaceOperator<dim, Number, 1>> laplace_operator;

  std::shared_ptr<Krylov::SolverBase<VectorType>> solver_pressure_block;

  // temporary vectors that are necessary when using preconditioners of block-triangular type
  VectorType mutable vec_tmp_pressure;
  VectorType mutable vec_tmp_velocity, vec_tmp_velocity_2;

  // temporary vectors that are necessary when applying the Schur-complement preconditioner (scp)
  VectorType mutable tmp_scp_pressure;
  VectorType mutable tmp_scp_velocity, tmp_scp_velocity_2;

  /*
   * Wall enrichment
   */

  // mass + viscous operator of the U_bar rows (no convective term), wrapped by the
  // EnrichedMomentumOperator of SpatialOperatorBase; the convective term (implicit) is added
  // separately by apply_linearized_convective_term_enriched()
  MomentumOperator<dim, Number> momentum_operator_mass_viscous;

  // linearization point (W_bar, W_tilde) of the convective term, and the evaluation time of
  // its boundary data
  mutable BlockVectorType linearization_enriched;
  mutable double          time_linearization_enriched = 0.0;

  NonlinearOperatorCoupledEnriched<dim, Number>   nonlinear_operator_enriched;
  LinearOperatorCoupledEnriched<dim, Number>      linear_operator_enriched;
  BlockPreconditionerCoupledEnriched<dim, Number> block_preconditioner_enriched;
  LinearSolverCoupledEnriched<dim, Number>        linear_solver_enriched;

  std::shared_ptr<Newton::Solver<BlockVectorType,
                                 NonlinearOperatorCoupledEnriched<dim, Number>,
                                 LinearOperatorCoupledEnriched<dim, Number>,
                                 LinearSolverCoupledEnriched<dim, Number>>>
    newton_solver_enriched;

  // temporary vectors: velocity block vectors (U_bar, U_tilde) and X = (U_bar, U_tilde, p^*)
  mutable BlockVectorType tmp_velocity_enriched, tmp_velocity_enriched_2,
    tmp_velocity_enriched_3;
  mutable BlockVectorType tmp_solution_enriched;
  mutable VectorType      tmp_pressure_enriched;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATOR_COUPLED_H_ \
        */
