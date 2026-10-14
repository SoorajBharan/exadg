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

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/numerics/vector_tools_mean_value.h>

#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_momentum.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/poisson/preconditioners/multigrid_preconditioner.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/utilities/check_multigrid.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
OperatorCoupled<dim, Number>::OperatorCoupled(
  std::shared_ptr<Grid<dim> const>                      grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>           mapping_in,
  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings_in,
  std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>            field_functions_in,
  Parameters const &                                    parameters_in,
  std::string const &                                   field_in,
  MPI_Comm const &                                      mpi_comm_in)
  : Base(grid_in,
         mapping_in,
         multigrid_mappings_in,
         boundary_descriptor_in,
         field_functions_in,
         parameters_in,
         field_in,
         mpi_comm_in),
    scaling_factor_continuity(1.0)
{
}

template<int dim, typename Number>
OperatorCoupled<dim, Number>::~OperatorCoupled()
{
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_derived()
{
  this->initialize_vector_velocity(temp_vector);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_preconditioners_and_solvers()
{
  Base::setup_preconditioners_and_solvers();

  if(this->param.apply_penalty_terms_in_postprocessing_step)
    Base::setup_projection_solver();

  setup_block_preconditioner();
  setup_solver_coupled();

  // wall enrichment: coupled system for (U_bar, U_tilde, p^*) of the unsteady solver
  // (TimeIntBDFCoupled); the standard solver above is used in the first time step
  // (U_tilde = 0). The steady solver (DriverSteadyProblems) keeps its own treatment.
  if(this->param.wall_enrichment_enabled and this->param.solver_type == SolverType::Unsteady)
    setup_wall_enrichment_coupled();
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_solver_coupled()
{
  linear_operator.initialize(*this);

  // setup linear solver
  if(this->param.solver_coupled == SolverCoupled::GMRES)
  {
    Krylov::SolverDataGMRES solver_data;
    solver_data.max_iter             = this->param.solver_data_coupled.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_coupled.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_coupled.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_coupled.max_krylov_size;
    solver_data.compute_eigenvalues  = false;

    if(this->param.preconditioner_coupled != PreconditionerCoupled::None)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver = std::make_shared<
      Krylov::SolverGMRES<LinearOperatorCoupled<dim, Number>, Preconditioner, BlockVectorType>>(
      linear_operator, block_preconditioner, solver_data, this->mpi_comm);
  }
  else if(this->param.solver_coupled == SolverCoupled::FGMRES)
  {
    Krylov::SolverDataFGMRES solver_data;
    solver_data.max_iter             = this->param.solver_data_coupled.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_coupled.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_coupled.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_coupled.max_krylov_size;

    if(this->param.preconditioner_coupled != PreconditionerCoupled::None)
    {
      solver_data.use_preconditioner = true;
    }

    linear_solver = std::make_shared<
      Krylov::SolverFGMRES<LinearOperatorCoupled<dim, Number>, Preconditioner, BlockVectorType>>(
      linear_operator, block_preconditioner, solver_data);
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("Specified solver for linearized problem is not implemented."));
  }

  // setup Newton solver
  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    nonlinear_operator.initialize(*this);

    newton_solver = std::make_shared<Newton::Solver<BlockVectorType,
                                                    NonlinearOperatorCoupled<dim, Number>,
                                                    LinearOperatorCoupled<dim, Number>,
                                                    Krylov::SolverBase<BlockVectorType>>>(
      this->param.newton_solver_data_coupled, nonlinear_operator, linear_operator, *linear_solver);
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::update_divergence_penalty_operator(VectorType const & velocity)
{
  this->div_penalty_operator.update(velocity);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::update_continuity_penalty_operator(VectorType const & velocity)
{
  this->conti_penalty_operator.update(velocity);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::set_scaling_factor_continuity(double const scaling_factor)
{
  scaling_factor_continuity = scaling_factor;
  this->gradient_operator.set_scaling_factor_pressure(scaling_factor);
}

template<int dim, typename Number>
unsigned int
OperatorCoupled<dim, Number>::solve_linear_stokes_problem(BlockVectorType &       dst,
                                                          BlockVectorType const & src,
                                                          bool const &   update_preconditioner,
                                                          double const & scaling_factor_mass)
{
  // Update momentum operator
  // We do not need to set the time here, because time affects the operator only in the form of
  // boundary conditions. The result of such boundary condition evaluations is handed over to this
  // function via the vector src.
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);

  linear_solver->update_preconditioner(update_preconditioner);

  return linear_solver->solve(dst, src);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::rhs_stokes_problem(BlockVectorType & dst, double const & time) const
{
  // velocity-block
  this->gradient_operator.rhs(dst.block(0), time);
  dst.block(0) *= scaling_factor_continuity;

  this->viscous_operator.set_time(time);
  this->viscous_operator.rhs_add(dst.block(0));

  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.rhs_add(dst.block(0), time);
  }

  if(this->param.right_hand_side == true)
    this->rhs_operator.evaluate_add(dst.block(0), time);

  // pressure-block
  this->divergence_operator.rhs(dst.block(1), time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_linearized_problem(BlockVectorType &       dst,
                                                       BlockVectorType const & src) const
{
  // (1,1) block of saddle point matrix
  this->momentum_operator.vmult(dst.block(0), src.block(0));

  // Divergence and continuity penalty operators
  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_divergence_penalty == true)
      this->div_penalty_operator.apply_add(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.apply_add(dst.block(0), src.block(0));
  }

  // (1,2) block of saddle point matrix
  // gradient operator: dst = velocity, src = pressure
  this->gradient_operator.apply(temp_vector, src.block(1));
  dst.block(0).add(scaling_factor_continuity, temp_vector);

  // (2,1) block of saddle point matrix
  // divergence operator: dst = pressure, src = velocity
  this->divergence_operator.apply(dst.block(1), src.block(0));
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
std::tuple<unsigned int, unsigned int>
OperatorCoupled<dim, Number>::solve_nonlinear_problem(BlockVectorType &  dst,
                                                      VectorType const & rhs_vector,
                                                      bool const &       update_preconditioner,
                                                      double const &     time,
                                                      double const &     scaling_factor_mass)
{
  // Update nonlinear operator
  nonlinear_operator.update(rhs_vector, time, scaling_factor_mass);

  // Update linearized momentum operator
  this->momentum_operator.set_time(time);
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);

  // Solve nonlinear problem
  Newton::UpdateData update;
  update.do_update                = update_preconditioner;
  update.update_every_newton_iter = this->param.update_preconditioner_coupled_every_newton_iter;

  auto const iter = newton_solver->solve(dst, update);

  return iter;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::evaluate_nonlinear_residual(BlockVectorType &       dst,
                                                          BlockVectorType const & src,
                                                          VectorType const *      rhs_vector,
                                                          double const &          time,
                                                          double const & scaling_factor_mass) const
{
  // update implicitly coupled variable viscosity
  if(this->param.nonlinear_viscous_problem())
  {
    this->update_viscosity(src.block(0));
  }

  // velocity-block

  if(this->unsteady_problem_has_to_be_solved())
    this->mass_operator.apply_scale(dst.block(0), scaling_factor_mass, src.block(0));
  else
    dst.block(0) = 0.0;

  if(this->param.implicit_convective_problem())
  {
    this->convective_operator.evaluate_nonlinear_operator_add(dst.block(0), src.block(0), time);
  }

  if(this->param.viscous_problem())
  {
    this->viscous_operator.set_time(time);
    this->viscous_operator.evaluate_add(dst.block(0), src.block(0));
  }

  // Divergence and continuity penalty operators
  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_divergence_penalty == true)
      this->div_penalty_operator.apply_add(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.evaluate_add(dst.block(0), src.block(0), time);
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate(temp_vector, src.block(1), time);
  dst.block(0).add(scaling_factor_continuity, temp_vector);

  // constant right-hand side vector (body force, sum_alphai_ui and explicit convective terms)
  dst.block(0).add(-1.0, *rhs_vector);

  // pressure-block

  this->divergence_operator.evaluate(dst.block(1), src.block(0), time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::evaluate_nonlinear_residual_steady(BlockVectorType &       dst,
                                                                 BlockVectorType const & src,
                                                                 double const &          time) const
{
  // update implicitly coupled variable viscosity
  if(this->param.nonlinear_viscous_problem())
  {
    this->update_viscosity(src.block(0));
  }

  // velocity-block

  // set dst.block(0) to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst.block(0) = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->rhs_operator.evaluate(dst.block(0), time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst.block(0) *= -1.0;
  }

  if(this->param.implicit_convective_problem())
  {
    this->convective_operator.evaluate_nonlinear_operator_add(dst.block(0), src.block(0), time);
  }

  if(this->param.viscous_problem())
  {
    this->viscous_operator.set_time(time);
    this->viscous_operator.evaluate_add(dst.block(0), src.block(0));
  }

  // Divergence and continuity penalty operators
  if(this->param.apply_penalty_terms_in_postprocessing_step == false)
  {
    if(this->param.use_divergence_penalty == true)
      this->div_penalty_operator.apply_add(dst.block(0), src.block(0));
    if(this->param.use_continuity_penalty == true)
      this->conti_penalty_operator.evaluate_add(dst.block(0), src.block(0), time);
  }

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate(temp_vector, src.block(1), time);
  dst.block(0).add(scaling_factor_continuity, temp_vector);


  // pressure-block

  this->divergence_operator.evaluate(dst.block(1), src.block(0), time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  // scale by scaling_factor_continuity
  dst.block(1) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_block_preconditioner()
{
  block_preconditioner.initialize(this);

  initialize_vectors();

  initialize_preconditioner_velocity_block();

  initialize_preconditioner_pressure_block();
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_vectors()
{
  auto type = this->param.preconditioner_coupled;

  if(type == PreconditionerCoupled::BlockTriangular)
  {
    this->initialize_vector_velocity(vec_tmp_velocity);
  }
  else if(type == PreconditionerCoupled::BlockTriangularFactorization)
  {
    this->initialize_vector_pressure(vec_tmp_pressure);
    this->initialize_vector_velocity(vec_tmp_velocity);
    this->initialize_vector_velocity(vec_tmp_velocity_2);
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_preconditioner_velocity_block()
{
  auto type = this->param.preconditioner_velocity_block;

  if(type == MomentumPreconditioner::PointJacobi)
  {
    preconditioner_momentum =
      std::make_shared<JacobiPreconditioner<MomentumOperator<dim, Number>>>(this->momentum_operator,
                                                                            false);
  }
  else if(type == MomentumPreconditioner::BlockJacobi)
  {
    preconditioner_momentum =
      std::make_shared<BlockJacobiPreconditioner<MomentumOperator<dim, Number>>>(
        this->momentum_operator, false);
  }
  else if(type == MomentumPreconditioner::InverseMassMatrix)
  {
    InverseMassOperatorData inverse_mass_operator_data;
    inverse_mass_operator_data.dof_index  = this->get_dof_index_velocity();
    inverse_mass_operator_data.quad_index = this->get_quad_index_velocity_standard();
    inverse_mass_operator_data.parameters = this->param.inverse_mass_preconditioner;

    preconditioner_momentum =
      std::make_shared<InverseMassPreconditioner<dim, dim, Number>>(this->get_matrix_free(),
                                                                    inverse_mass_operator_data);
  }
  else if(type == MomentumPreconditioner::Multigrid)
  {
    setup_multigrid_preconditioner_momentum();

    if(this->param.exact_inversion_of_velocity_block == true)
    {
      setup_iterative_solver_momentum();
    }
  }
  else
  {
    AssertThrow(type == MomentumPreconditioner::None, dealii::ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_multigrid_preconditioner_momentum()
{
  typedef MultigridPreconditioner<dim, Number> Multigrid;

  preconditioner_momentum = std::make_shared<Multigrid>(this->mpi_comm);

  std::shared_ptr<Multigrid> mg_preconditioner =
    std::dynamic_pointer_cast<Multigrid>(preconditioner_momentum);

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    dirichlet_boundary_conditions = this->momentum_operator.get_data().bc->dirichlet_bc;

  // We also need to add DirichletCached boundary conditions. From the
  // perspective of multigrid, there is no difference between standard
  // and cached Dirichlet BCs. Since multigrid does not need information
  // about inhomogeneous boundary data, we simply fill the map with
  // dealii::Functions::ZeroFunction for DirichletCached BCs.
  for(auto iter : this->momentum_operator.get_data().bc->dirichlet_cached_bc)
  {
    typedef std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> pair;

    dirichlet_boundary_conditions.insert(pair(iter, new dealii::Functions::ZeroFunction<dim>(dim)));
  }

  typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
  Map_DBC_ComponentMask                                               dirichlet_bc_component_mask;

  mg_preconditioner->initialize(this->param.multigrid_data_velocity_block,
                                this->grid,
                                this->multigrid_mappings,
                                this->get_dof_handler_u().get_fe(),
                                this->momentum_operator,
                                this->param.multigrid_operator_type_velocity_block,
                                this->param.ale_formulation,
                                dirichlet_boundary_conditions,
                                dirichlet_bc_component_mask);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_iterative_solver_momentum()
{
  AssertThrow(preconditioner_momentum.get() != 0,
              dealii::ExcMessage("preconditioner_momentum is uninitialized"));

  // use FGMRES for "exact" solution of velocity block system
  Krylov::SolverDataFGMRES gmres_data;
  gmres_data.use_preconditioner   = true;
  gmres_data.max_iter             = this->param.solver_data_velocity_block.max_iter;
  gmres_data.solver_tolerance_abs = this->param.solver_data_velocity_block.abs_tol;
  gmres_data.solver_tolerance_rel = this->param.solver_data_velocity_block.rel_tol;
  gmres_data.max_n_tmp_vectors    = this->param.solver_data_velocity_block.max_krylov_size;

  solver_velocity_block = std::make_shared<
    Krylov::SolverFGMRES<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>>(
    this->momentum_operator, *preconditioner_momentum, gmres_data);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_preconditioner_pressure_block()
{
  auto type = this->param.preconditioner_pressure_block;

  InverseMassOperatorData inverse_mass_operator_data;
  inverse_mass_operator_data.dof_index  = this->get_dof_index_pressure();
  inverse_mass_operator_data.quad_index = this->get_quad_index_pressure();
  inverse_mass_operator_data.parameters = this->param.inverse_mass_preconditioner;

  if(type == SchurComplementPreconditioner::InverseMassMatrix)
  {
    inverse_mass_preconditioner_schur_complement =
      std::make_shared<InverseMassPreconditioner<dim, 1, Number>>(this->get_matrix_free(),
                                                                  inverse_mass_operator_data);
  }
  else if(type == SchurComplementPreconditioner::LaplaceOperator)
  {
    setup_multigrid_preconditioner_schur_complement();

    if(this->param.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }
  }
  else if(type == SchurComplementPreconditioner::CahouetChabard)
  {
    AssertThrow(this->unsteady_problem_has_to_be_solved() == true,
                dealii::ExcMessage(
                  "Cahouet-Chabard preconditioner only makes sense for unsteady problems."));

    setup_multigrid_preconditioner_schur_complement();

    if(this->param.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }

    // inverse mass operator to also include the part of the preconditioner that is beneficial when
    // using large time steps and large viscosities.
    inverse_mass_preconditioner_schur_complement =
      std::make_shared<InverseMassPreconditioner<dim, 1, Number>>(this->get_matrix_free(),
                                                                  inverse_mass_operator_data);

    // initialize tmp vector
    this->initialize_vector_pressure(tmp_scp_pressure);
  }
  else if(type == SchurComplementPreconditioner::PressureConvectionDiffusion)
  {
    // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

    // I. multigrid for negative Laplace operator (classical or compatible discretization)
    setup_multigrid_preconditioner_schur_complement();

    if(this->param.exact_inversion_of_laplace_operator == true)
    {
      setup_iterative_solver_schur_complement();
    }

    // II. pressure convection-diffusion operator
    setup_pressure_convection_diffusion_operator();

    // III. inverse pressure mass operator
    inverse_mass_preconditioner_schur_complement =
      std::make_shared<InverseMassPreconditioner<dim, 1, Number>>(this->get_matrix_free(),
                                                                  inverse_mass_operator_data);

    // initialize tmp vector
    this->initialize_vector_pressure(tmp_scp_pressure);
  }
  else
  {
    AssertThrow(type == SchurComplementPreconditioner::None, dealii::ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_multigrid_preconditioner_schur_complement()
{
  // multigrid V-cycle for negative Laplace operator
  Poisson::LaplaceOperatorData<0, dim> laplace_operator_data;
  laplace_operator_data.dof_index             = this->get_dof_index_pressure();
  laplace_operator_data.quad_index            = this->get_quad_index_pressure();
  laplace_operator_data.operator_is_singular  = this->is_pressure_level_undefined();
  laplace_operator_data.kernel_data.IP_factor = 1.0;
  laplace_operator_data.bc                    = this->boundary_descriptor_laplace;

  MultigridData mg_data = this->param.multigrid_data_pressure_block;

  multigrid_preconditioner_schur_complement = std::make_shared<MultigridPoisson>(this->mpi_comm);

  std::shared_ptr<MultigridPoisson> mg_preconditioner =
    std::dynamic_pointer_cast<MultigridPoisson>(multigrid_preconditioner_schur_complement);

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    dirichlet_boundary_conditions = laplace_operator_data.bc->dirichlet_bc;

  typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
  Map_DBC_ComponentMask                                               dirichlet_bc_component_mask;

  auto & dof_handler = this->get_dof_handler_p();
  mg_preconditioner->initialize(mg_data,
                                this->grid,
                                this->multigrid_mappings,
                                dof_handler.get_fe(),
                                laplace_operator_data,
                                this->param.ale_formulation,
                                dirichlet_boundary_conditions,
                                dirichlet_bc_component_mask);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_iterative_solver_schur_complement()
{
  AssertThrow(
    multigrid_preconditioner_schur_complement.get() != 0,
    dealii::ExcMessage(
      "Setup of iterative solver for Schur complement preconditioner: Multigrid preconditioner is uninitialized"));

  Krylov::SolverDataCG solver_data;
  solver_data.max_iter             = this->param.solver_data_pressure_block.max_iter;
  solver_data.solver_tolerance_abs = this->param.solver_data_pressure_block.abs_tol;
  solver_data.solver_tolerance_rel = this->param.solver_data_pressure_block.rel_tol;
  solver_data.use_preconditioner   = true;

  Poisson::LaplaceOperatorData<0, dim> laplace_operator_data;
  laplace_operator_data.dof_index             = this->get_dof_index_pressure();
  laplace_operator_data.quad_index            = this->get_quad_index_pressure();
  laplace_operator_data.bc                    = this->boundary_descriptor_laplace;
  laplace_operator_data.kernel_data.IP_factor = 1.0;

  laplace_operator = std::make_shared<Poisson::LaplaceOperator<dim, Number, 1>>();
  laplace_operator->initialize(this->get_matrix_free(),
                               this->get_constraint_p(),
                               laplace_operator_data);

  solver_pressure_block =
    std::make_shared<Krylov::SolverCG<Poisson::LaplaceOperator<dim, Number, 1>,
                                      PreconditionerBase<Number>,
                                      VectorType>>(*laplace_operator,
                                                   *multigrid_preconditioner_schur_complement,
                                                   solver_data);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_pressure_convection_diffusion_operator()
{
  // pressure convection-diffusion operator

  // fill boundary descriptor
  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor;
  boundary_descriptor = std::make_shared<ConvDiff::BoundaryDescriptor<dim>>();

  // For the pressure convection-diffusion operator the homogeneous operators are applied, so there
  // is no need to specify functions for boundary conditions since they will never be used.
  std::shared_ptr<dealii::Function<dim>> dummy;

  // set boundary ID's for pressure convection-diffusion operator

  // Dirichlet BC for pressure
  for(typename std::map<dealii::types::boundary_id,
                        std::shared_ptr<dealii::Function<dim>>>::const_iterator it =
        this->boundary_descriptor->pressure->dirichlet_bc.begin();
      it != this->boundary_descriptor->pressure->dirichlet_bc.end();
      ++it)
  {
    boundary_descriptor->dirichlet_bc.insert(
      std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>(it->first,
                                                                                    dummy));
  }
  // Neumann BC for pressure
  for(typename std::set<dealii::types::boundary_id>::const_iterator it =
        this->boundary_descriptor->pressure->neumann_bc.begin();
      it != this->boundary_descriptor->pressure->neumann_bc.end();
      ++it)
  {
    boundary_descriptor->neumann_bc.insert(
      std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>(*it, dummy));
  }

  // convective operator:
  // use numerical velocity field with dof index of velocity field and local Lax-Friedrichs flux to
  // mimic the upwind-like discretization of the linearized convective term in the Navier-Stokes
  // equations.
  ConvDiff::Operators::ConvectiveKernelData<dim> convective_kernel_data;
  convective_kernel_data.velocity_type      = ConvDiff::TypeVelocityField::DoFVector;
  convective_kernel_data.dof_index_velocity = this->get_dof_index_velocity();
  convective_kernel_data.numerical_flux_formulation =
    ConvDiff::NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

  // diffusive operator:
  // take interior penalty factor of diffusivity of viscous operator, but use polynomial degree of
  // pressure shape functions.
  ConvDiff::Operators::DiffusiveKernelData diffusive_kernel_data;
  diffusive_kernel_data.IP_factor = this->param.IP_factor_viscous;
  // Note: the diffusive operator is initialized with constant viscosity. In case of spatially (and
  // temporally) varying viscosities the diffusive operator has to be extended so that it can deal
  // with variable coefficients (and should be updated in case of time dependent problems before
  // applying the preconditioner).
  diffusive_kernel_data.diffusivity = this->param.viscosity;

  // combined convection-diffusion operator
  ConvDiff::CombinedOperatorData<dim> operator_data;
  operator_data.dof_index  = this->get_dof_index_pressure();
  operator_data.quad_index = this->get_quad_index_pressure();

  operator_data.bc                   = boundary_descriptor;
  operator_data.use_cell_based_loops = this->param.use_cell_based_face_loops;

  operator_data.unsteady_problem   = this->unsteady_problem_has_to_be_solved();
  operator_data.convective_problem = this->param.implicit_convective_problem();
  operator_data.diffusive_problem  = this->param.viscous_problem();

  operator_data.convective_kernel_data = convective_kernel_data;
  operator_data.diffusive_kernel_data  = diffusive_kernel_data;

  pressure_conv_diff_operator = std::make_shared<ConvDiff::CombinedOperator<dim, 1, Number>>();
  pressure_conv_diff_operator->initialize(this->get_matrix_free(),
                                          this->get_constraint_p(),
                                          operator_data);
}

// clang-format off
/*
 * Consider the following saddle point matrix :
 *
 *       / A  B^{T} \
 *   M = |          |
 *       \ B    0   /
 *
 *  with block factorization
 *
 *       / I         0 \  / A   0 \ / I   A^{-1} B^{T} \
 *   M = |             |  |       | |                  |
 *       \ B A^{-1}  I /  \ 0   S / \ 0        I       /
 *
 *       / I         0 \  / A   B^{T} \
 *     = |             |  |           |
 *       \ B A^{-1}  I /  \ 0     S   /
 *
 *        / A  0 \  / I   A^{-1} B^{T} \
 *     =  |      |  |                  |
 *        \ B  S /  \ 0        I       /
 *
 *   with Schur complement S = -B A^{-1} B^{T}
 *
 *
 * - Block-diagonal preconditioner:
 *
 *                   / A   0 \                       / A^{-1}   0    \   / A^{-1}  0 \   / I      0    \
 *   -> P_diagonal = |       |  -> P_diagonal^{-1} = |               | = |           | * |             |
 *                   \ 0  -S /                       \   0   -S^{-1} /   \   0     I /   \ 0   -S^{-1} /
 *
 * - Block-triangular preconditioner:
 *
 *                     / A   B^{T} \                         / A^{-1}  0 \   / I  B^{T} \   / I      0    \
 *   -> P_triangular = |           |  -> P_triangular^{-1} = |           | * |          | * |             |
 *                     \ 0     S   /                         \   0     I /   \ 0   -I   /   \ 0   -S^{-1} /
 *
 * - Block-triangular factorization:
 *
 *                      / A  0 \  / I   A^{-1} B^{T} \
 *   -> P_tria-factor = |      |  |                  |
 *                      \ B  S /  \ 0        I       /
 *
 *                            / I  - A^{-1} B^{T} \   / I      0    \   / I   0 \   / A^{-1}  0 \
 *    -> P_tria-factor^{-1} = |                   | * |             | * |       | * |           |
 *                            \ 0          I      /   \ 0   -S^{-1} /   \ B  -I /   \   0     I /
 *
 *
 *  Main challenge: Development of efficient preconditioners for A and S that approximate
 *  the velocity block A and the Schur-complement block S in a spectrally equivalent way.
 *
 *
 *  Approximations of velocity block A = 1/dt M_u + C_lin(u) + nu (-L):
 *
 *   1. inverse mass preconditioner (dt small):
 *
 *     A = 1/dt M_u
 *
 *     -> A^{-1} = dt M_u^{-1}
 *
 *   2. Helmholtz operator H =  1/dt M_u + nu (-L) (neglecting the convective term):
 *
 *     -> A^{-1} = H^{-1} where H^{-1} is approximated by performing one multigrid V-cycle for the Helmholtz operator
 *
 *   3. Velocity convection-diffusion operator A = 1/dt M_u + C_lin(u) + nu (-L) including the convective term:
 *
 *      -> to approximately invert A consider one multigrid V-cycle
 *
 *  Approximations of pressure Schur-complement block S:
 *
 *   - S = - B A^{-1} B^T
 *       |
 *       |  apply method of pseudo-differential operators and neglect convective term
 *      \|/
 *       = - (- div ) * ( 1/dt * I - nu * laplace )^{-1} * grad
 *
 *   1. dt small, nu small:

 *      S = div * (1/dt I)^{-1} * grad = dt * laplace
 *
 *      -> - S^{-1} = 1/dt (-L)^{-1} (-L: negative Laplace operator)
 *
 *   2. dt large, nu large:
 *
 *      S = div * (- nu * laplace)^{-1} * grad = - 1/nu * I
 *
 *      -> - S^{-1} = nu M_p^{-1} (M_p: pressure mass operator)
 *
 *   3. Cahouet & Chabard (combines 1. and 2., robust preconditioner for whole range of time step sizes and visosities)
 *
 *      -> - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1}
 *
 *   4. Pressure convection-diffusion preconditioner
 *
 *      -> -S^{-1} = M_p^{-1} A_p (-L)^{-1} where A_p is a convection-diffusion operator for the pressure
 */
// clang-format on

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::update_block_preconditioner()
{
  // momentum block
  preconditioner_momentum->update();

  // pressure block
  if(this->param.ale_formulation) // only if mesh is moving
  {
    auto const type = this->param.preconditioner_pressure_block;

    // inverse mass operator
    if(type == SchurComplementPreconditioner::InverseMassMatrix or
       type == SchurComplementPreconditioner::CahouetChabard or
       type == SchurComplementPreconditioner::PressureConvectionDiffusion)
    {
      inverse_mass_preconditioner_schur_complement->update();
    }

    // Laplace operator
    if(type == SchurComplementPreconditioner::LaplaceOperator or
       type == SchurComplementPreconditioner::CahouetChabard or
       type == SchurComplementPreconditioner::PressureConvectionDiffusion)
    {
      if(this->param.exact_inversion_of_laplace_operator == true)
      {
        laplace_operator->update_penalty_parameter();
      }

      multigrid_preconditioner_schur_complement->update();
    }
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_block_preconditioner(BlockVectorType &       dst,
                                                         BlockVectorType const & src) const
{
  auto type = this->param.preconditioner_coupled;

  if(type == PreconditionerCoupled::BlockDiagonal)
  {
    /*                        / A^{-1}   0    \   / A^{-1}  0 \   / I      0    \
     *   -> P_diagonal^{-1} = |               | = |           | * |             |
     *                        \   0   -S^{-1} /   \   0     I /   \ 0   -S^{-1} /
     */

    /*
     *         / I      0    \
     *  temp = |             | * src
     *         \ 0   -S^{-1} /
     */

    // apply preconditioner for pressure/Schur-complement block
    apply_preconditioner_pressure_block(dst.block(1), src.block(1));

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * temp
     *        \   0     I /
     */

    // apply preconditioner for velocity/momentum block
    apply_preconditioner_velocity_block(dst.block(0), src.block(0));
  }
  else if(type == PreconditionerCoupled::BlockTriangular)
  {
    /*
     *                         / A^{-1}  0 \   / I  B^{T} \   / I      0    \
     *  -> P_triangular^{-1} = |           | * |          | * |             |
     *                         \   0     I /   \ 0   -I   /   \ 0   -S^{-1} /
     */

    /*
     *        / I      0    \
     *  dst = |             | * src
     *        \ 0   -S^{-1} /
     */

    // For the velocity block simply copy data from src to dst.
    dst.block(0) = src.block(0);
    // Apply preconditioner for pressure/Schur-complement block.
    apply_preconditioner_pressure_block(dst.block(1), src.block(1));

    /*
     *        / I  B^{T} \
     *  dst = |          | * dst
     *        \ 0   -I   /
     */

    // Apply gradient operator and add to dst vector.
    this->gradient_operator.apply(vec_tmp_velocity, dst.block(1));
    dst.block(0).add(this->scaling_factor_continuity, vec_tmp_velocity);
    dst.block(1) *= -1.0;

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * dst
     *        \   0     I /
     */

    // Copy data from dst.block(0) to vec_tmp_velocity before
    // applying the preconditioner for the velocity block.
    vec_tmp_velocity = dst.block(0);
    // Apply preconditioner for velocity/momentum block.
    apply_preconditioner_velocity_block(dst.block(0), vec_tmp_velocity);
  }
  else if(type == PreconditionerCoupled::BlockTriangularFactorization)
  {
    /*
     *                          / I  - A^{-1} B^{T} \   / I      0    \   / I   0 \   / A^{-1} 0 \
     *  -> P_tria-factor^{-1} = |                   | * |             | * |       | * |          |
     *                          \ 0          I      /   \ 0   -S^{-1} /   \ B  -I /   \   0    I /
     */

    /*
     *        / A^{-1}  0 \
     *  dst = |           | * src
     *        \   0     I /
     */

    // for the pressure block simply copy data from src to dst
    dst.block(1) = src.block(1);
    // apply preconditioner for velocity/momentum block
    apply_preconditioner_velocity_block(dst.block(0), src.block(0));

    /*
     *        / I   0 \
     *  dst = |       | * dst
     *        \ B  -I /
     */

    // dst.block(1) = B*dst.block(0) - dst.block(1)
    //              = -1.0 * (dst.block(1) + (-B) * dst.block(0));
    // I. dst.block(1) += (-B) * dst.block(0);
    // Note that B represents NEGATIVE divergence operator, i.e.,
    // applying -B is equal to applying the divergence operator
    this->divergence_operator.apply(vec_tmp_pressure, dst.block(0));
    dst.block(1).add(this->scaling_factor_continuity, vec_tmp_pressure);
    // II. dst.block(1) = -dst.block(1);
    dst.block(1) *= -1.0;

    /*
     *        / I      0    \
     *  dst = |             | * dst
     *        \ 0   -S^{-1} /
     */

    // Copy data from dst.block(1) to vec_tmp_pressure before
    // applying the preconditioner for the pressure block.
    vec_tmp_pressure = dst.block(1);
    // Apply preconditioner for pressure/Schur-complement block
    apply_preconditioner_pressure_block(dst.block(1), vec_tmp_pressure);

    /*
     *        / I  - A^{-1} B^{T} \
     *  dst = |                   | * dst
     *        \ 0          I      /
     */

    // vec_tmp_velocity = B^{T} * dst(1)
    this->gradient_operator.apply(vec_tmp_velocity, dst.block(1));

    // scaling factor continuity
    vec_tmp_velocity *= this->scaling_factor_continuity;

    // vec_tmp_velocity_2 = A^{-1} * vec_tmp_velocity
    apply_preconditioner_velocity_block(vec_tmp_velocity_2, vec_tmp_velocity);

    // dst(0) = dst(0) - vec_tmp_velocity_2
    dst.block(0).add(-1.0, vec_tmp_velocity_2);
  }
  else
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_preconditioner_velocity_block(VectorType &       dst,
                                                                  VectorType const & src) const
{
  auto type = this->param.preconditioner_velocity_block;

  if(type == MomentumPreconditioner::None)
  {
    dst = src;
  }
  else if(type == MomentumPreconditioner::PointJacobi or
          type == MomentumPreconditioner::BlockJacobi)
  {
    preconditioner_momentum->vmult(dst, src);
  }
  else if(type == MomentumPreconditioner::InverseMassMatrix)
  {
    // use the inverse mass operator as an approximation to the momentum block
    preconditioner_momentum->vmult(dst, src);
    dst *= 1. / this->momentum_operator.get_scaling_factor_mass_operator();
  }
  else if(type == MomentumPreconditioner::Multigrid)
  {
    if(this->param.exact_inversion_of_velocity_block == false)
    {
      // perform one multigrid V-cylce
      preconditioner_momentum->vmult(dst, src);
    }
    else // exact_inversion_of_velocity_block == true
    {
      // check correctness of multigrid V-cycle

      // clang-format off
      /*
      typedef MultigridPreconditioner<dim, degree_u, Number, MultigridNumber> MULTIGRID;

      std::shared_ptr<MULTIGRID> preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner_momentum);

      CheckMultigrid<dim, Number, MomentumOperator<dim, degree_u, Number>, MULTIGRID>
        check_multigrid(this->momentum_operator,preconditioner);

      check_multigrid.check();
      */
      // clang-format on

      // iteratively solve momentum equation up to given tolerance
      dst = 0.0;
      // Note that the preconditioner is not updated here since it has
      // already been updated in the function update_block_preconditioner().
      unsigned int const iterations = solver_velocity_block->solve(dst, src);

      // output
      bool const print_iterations = false;
      if(print_iterations)
      {
        dealii::ConditionalOStream pcout(std::cout,
                                         dealii::Utilities::MPI::this_mpi_process(this->mpi_comm) ==
                                           0);
        pcout << "Number of iterations for inner solver = " << iterations << std::endl;
      }
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_preconditioner_pressure_block(VectorType &       dst,
                                                                  VectorType const & src) const
{
  auto type = this->param.preconditioner_pressure_block;

  if(type == SchurComplementPreconditioner::None)
  {
    // No preconditioner for Schur-complement block
    dst = src;
  }
  else if(type == SchurComplementPreconditioner::InverseMassMatrix)
  {
    // - S^{-1} = nu M_p^{-1}
    // TODO consider variable viscosity here
    inverse_mass_preconditioner_schur_complement->vmult(dst, src);
    dst *= this->param.viscosity;
  }
  else if(type == SchurComplementPreconditioner::LaplaceOperator)
  {
    // -S^{-1} = 1/dt  (-L)^{-1}
    apply_inverse_negative_laplace_operator(dst, src);
    dst *= this->momentum_operator.get_scaling_factor_mass_operator();
  }
  else if(type == SchurComplementPreconditioner::CahouetChabard)
  {
    // - S^{-1} = nu M_p^{-1} + 1/dt (-L)^{-1}

    // I. 1/dt (-L)^{-1}
    apply_inverse_negative_laplace_operator(dst, src);
    dst *= this->momentum_operator.get_scaling_factor_mass_operator();

    // II. M_p^{-1}, apply inverse pressure mass operator to src-vector and store the result in a
    // temporary vector
    // TODO consider variable viscosity here
    inverse_mass_preconditioner_schur_complement->vmult(tmp_scp_pressure, src);

    // III. add temporary vector scaled by viscosity
    dst.add(this->param.viscosity, tmp_scp_pressure);
  }
  else if(type == SchurComplementPreconditioner::PressureConvectionDiffusion)
  {
    // -S^{-1} = M_p^{-1} A_p (-L)^{-1}

    // I. inverse, negative Laplace operator (-L)^{-1}
    apply_inverse_negative_laplace_operator(tmp_scp_pressure, src);

    // II. pressure convection-diffusion operator A_p
    if(this->unsteady_problem_has_to_be_solved())
    {
      pressure_conv_diff_operator->set_scaling_factor_mass_operator(
        this->momentum_operator.get_scaling_factor_mass_operator());
    }

    if(this->param.implicit_convective_problem())
    {
      pressure_conv_diff_operator->set_velocity_ptr(this->convective_kernel->get_velocity());
    }

    pressure_conv_diff_operator->apply(dst, tmp_scp_pressure);

    // III. inverse pressure mass operator M_p^{-1}
    // TODO consider variable viscosity here
    inverse_mass_preconditioner_schur_complement->vmult(dst, dst);
  }
  else
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }

  // scaling_factor_continuity: Since the Schur complement includes both the velocity divergence
  // and the pressure gradient operators as factors, we have to scale by
  // 1/(scaling_factor*scaling_factor) when applying (an approximation of) the inverse Schur
  // complement.
  double inverse_scaling_factor = 1.0 / scaling_factor_continuity;
  dst *= inverse_scaling_factor * inverse_scaling_factor;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_inverse_negative_laplace_operator(VectorType &       dst,
                                                                      VectorType const & src) const
{
  if(this->param.exact_inversion_of_laplace_operator == false)
  {
    // perform one multigrid V-cycle in order to approximately invert the negative Laplace
    // operator (classical or compatible)
    multigrid_preconditioner_schur_complement->vmult(dst, src);
  }
  else // exact_inversion_of_laplace_operator == true
  {
    // solve a linear system of equations for negative Laplace operator to given (relative)
    // tolerance using the PCG method
    VectorType const * pointer_to_src = &src;
    if(this->is_pressure_level_undefined())
    {
      VectorType vector_zero_mean;
      vector_zero_mean = src;

      if(laplace_operator->operator_is_singular())
      {
        dealii::VectorTools::subtract_mean_value(vector_zero_mean);
      }

      pointer_to_src = &vector_zero_mean;
    }

    dst = 0.0;
    // Note that the preconditioner is not updated here since it has
    // already been updated in the function update_block_preconditioner().
    solver_pressure_block->solve(dst, *pointer_to_src);
  }
}


/*
 * Wall enrichment (Krank et al. 2018): coupled system for X = (U_bar, U_tilde, p^*).
 */

template<int dim, typename Number>
unsigned int
LinearSolverCoupledEnriched<dim, Number>::solve(BlockVectorType &       dst,
                                                BlockVectorType const & rhs) const
{
  AssertThrow(linear_operator != nullptr and preconditioner != nullptr,
              dealii::ExcMessage("LinearSolverCoupledEnriched not initialized."));

  dealii::ReductionControl control(solver_data.max_iter, solver_data.abs_tol, solver_data.rel_tol);

  if(solver_type == SolverCoupled::GMRES)
  {
    typename dealii::SolverGMRES<BlockVectorType>::AdditionalData additional_data;
    additional_data.max_basis_size        = solver_data.max_krylov_size;
    additional_data.right_preconditioning = true;

    dealii::SolverGMRES<BlockVectorType> solver(control, additional_data);
    if(use_preconditioner)
      solver.solve(*linear_operator, dst, rhs, *preconditioner);
    else
      solver.solve(*linear_operator, dst, rhs, dealii::PreconditionIdentity());
  }
  else if(solver_type == SolverCoupled::FGMRES)
  {
    typename dealii::SolverFGMRES<BlockVectorType>::AdditionalData additional_data;
    additional_data.max_basis_size = solver_data.max_krylov_size;

    dealii::SolverFGMRES<BlockVectorType> solver(control, additional_data);
    if(use_preconditioner)
      solver.solve(*linear_operator, dst, rhs, *preconditioner);
    else
      solver.solve(*linear_operator, dst, rhs, dealii::PreconditionIdentity());
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("Specified solver for the enriched coupled problem is not "
                                   "implemented."));
  }

  return control.last_step();
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::setup_wall_enrichment_coupled()
{
  AssertThrow(this->unsteady_problem_has_to_be_solved(),
              dealii::ExcMessage("Coupled solver with wall enrichment: only unsteady problems "
                                 "are implemented."));
  AssertThrow(not this->param.ale_formulation,
              dealii::ExcMessage("Coupled solver with wall enrichment: ALE not implemented."));
  AssertThrow(this->param.viscous_problem() and not this->param.viscosity_is_variable(),
              dealii::ExcMessage("Coupled solver with wall enrichment: only a constant viscosity "
                                 "is implemented."));
  AssertThrow(this->param.apply_penalty_terms_in_postprocessing_step or
                (not this->param.use_divergence_penalty and not this->param.use_continuity_penalty),
              dealii::ExcMessage("Coupled solver with wall enrichment: penalty terms inside the "
                                 "coupled system are not implemented (use "
                                 "apply_penalty_terms_in_postprocessing_step)."));
  AssertThrow(not(this->param.use_continuity_penalty and
                  this->param.continuity_penalty_use_boundary_data),
              dealii::ExcMessage("Coupled solver with wall enrichment: continuity penalty with "
                                 "boundary data is not implemented."));
  AssertThrow(this->param.preconditioner_coupled != PreconditionerCoupled::None or
                this->param.solver_coupled == SolverCoupled::GMRES or
                this->param.solver_coupled == SolverCoupled::FGMRES,
              dealii::ExcMessage("Coupled solver with wall enrichment: unsupported solver."));

  // mass + viscous operator for the U_bar rows: same data and kernels as momentum_operator,
  // without the convective term (added by apply_linearized_convective_term_enriched())
  MomentumOperatorData<dim> data = this->momentum_operator.get_data();
  data.convective_problem        = false;

  dealii::AffineConstraints<Number> constraint_dummy;
  constraint_dummy.close();

  momentum_operator_mass_viscous.initialize(
    this->get_matrix_free(), constraint_dummy, data, this->viscous_kernel, this->convective_kernel);

  this->setup_wall_enrichment_solvers(momentum_operator_mass_viscous);

  // temporary vectors
  auto const init_velocity = [&](BlockVectorType & v) {
    v.reinit(2);
    this->initialize_vector_velocity(v.block(0));
    this->initialize_vector_enrichment(v.block(1));
    v.collect_sizes();
  };
  init_velocity(linearization_enriched);
  init_velocity(tmp_velocity_enriched);
  init_velocity(tmp_velocity_enriched_2);
  init_velocity(tmp_velocity_enriched_3);
  initialize_block_vector_velocity_pressure_enriched(tmp_solution_enriched);
  this->initialize_vector_pressure(tmp_pressure_enriched);
  if(this->param.preconditioner_coupled == PreconditionerCoupled::BlockTriangularFactorization)
    this->initialize_vector_pressure(vec_tmp_pressure);

  // linear solver, preconditioner and Newton solver
  linear_operator_enriched.initialize(*this);
  block_preconditioner_enriched.initialize(this);
  linear_solver_enriched.initialize(linear_operator_enriched,
                                    block_preconditioner_enriched,
                                    this->param.solver_data_coupled,
                                    this->param.solver_coupled,
                                    this->param.preconditioner_coupled !=
                                      PreconditionerCoupled::None);

  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    nonlinear_operator_enriched.initialize(*this);

    newton_solver_enriched =
      std::make_shared<Newton::Solver<BlockVectorType,
                                      NonlinearOperatorCoupledEnriched<dim, Number>,
                                      LinearOperatorCoupledEnriched<dim, Number>,
                                      LinearSolverCoupledEnriched<dim, Number>>>(
        this->param.newton_solver_data_coupled,
        nonlinear_operator_enriched,
        linear_operator_enriched,
        linear_solver_enriched);
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::initialize_block_vector_velocity_pressure_enriched(
  BlockVectorType & dst) const
{
  dst.reinit(3);
  this->initialize_vector_velocity(dst.block(0));
  this->initialize_vector_enrichment(dst.block(1));
  this->initialize_vector_pressure(dst.block(2));
  dst.collect_sizes();
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::constrain_velocity_enriched(BlockVectorType & dst) const
{
  // E_bar: strong no-slip, U_bar DoFs on enrichment walls fixed (homogeneous data)
  if(this->param.wall_bc_viscous == WallBoundaryConditionViscous::Strong)
    dst.block(0).scale(this->free_dofs_bar);

  // P: projection onto the well-posed enrichment modes
  this->get_wall_dg_coupler()->apply_enrichment_projector(dst.block(1), dst.block(1), false);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::copy_velocity_enriched(BlockVectorType &       velocity,
                                                     BlockVectorType const & src) const
{
  velocity.block(0) = src.block(0);
  velocity.block(1) = src.block(1);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::set_velocity_enriched(BlockVectorType &       dst,
                                                    BlockVectorType const & velocity) const
{
  dst.block(0) = velocity.block(0);
  dst.block(1) = velocity.block(1);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::rhs_stokes_problem_enriched(BlockVectorType & dst,
                                                          double const      time) const
{
  BlockVectorType & rhs_velocity = tmp_velocity_enriched;
  BlockVectorType & gradient     = tmp_velocity_enriched_2;

  /*
   * velocity rows (\bar{v}_h and \psi \tilde{v}_h)
   */

  // pressure gradient, boundary data g_p: s_c G(0; g_p / s_c) moved to the right-hand side
  VectorType zero_pressure;
  this->initialize_vector_pressure(zero_pressure);
  this->evaluate_pressure_gradient_term_enriched(gradient,
                                                 zero_pressure,
                                                 time,
                                                 1.0 / scaling_factor_continuity);
  rhs_velocity.equ(-scaling_factor_continuity, gradient);

  // viscous term, velocity Dirichlet data (U_bar rows: ExaDG's operator, U_tilde rows and
  // enrichment-aware penalty correction: EnrichedMomentumOperator)
  this->viscous_operator.set_time(time);
  this->viscous_operator.rhs_add(rhs_velocity.block(0));
  this->rhs_add_enriched_viscous_boundary_data(rhs_velocity.block(0),
                                               rhs_velocity.block(1),
                                               time);

  // body force
  if(this->param.right_hand_side)
    this->evaluate_add_body_force_term_enriched(rhs_velocity, time);

  set_velocity_enriched(dst, rhs_velocity);

  /*
   * continuity rows: velocity Dirichlet data of the divergence term (the \psi \tilde{u}_h part
   * has no inhomogeneous contribution); minus sign and scaling as in rhs_stokes_problem()
   */
  this->divergence_operator.rhs(dst.block(2), time);
  dst.block(2) *= -scaling_factor_continuity;
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_velocity_block_enriched(BlockVectorType &       dst,
                                                            BlockVectorType const & velocity) const
{
  // mass + viscous terms (U_bar-U_bar block: momentum_operator_mass_viscous)
  this->enriched_momentum_operator.vmult(dst, velocity);

  // linearized convective term c'(w_h) u_h (Newton)
  if(this->param.implicit_convective_problem())
  {
    BlockVectorType & convective = tmp_velocity_enriched_3;
    this->apply_linearized_convective_term_enriched(convective,
                                                    velocity,
                                                    linearization_enriched,
                                                    time_linearization_enriched);
    dst.add(1.0, convective);
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_linearized_problem_enriched(BlockVectorType &       dst,
                                                                BlockVectorType const & src) const
{
  // E X
  BlockVectorType & src_constrained = tmp_solution_enriched;
  src_constrained                   = src;
  constrain_velocity_enriched(src_constrained);

  BlockVectorType & velocity = tmp_velocity_enriched;
  BlockVectorType & result   = tmp_velocity_enriched_2;
  copy_velocity_enriched(velocity, src_constrained);

  // (1,1) block: A_h u_h
  apply_velocity_block_enriched(result, velocity);

  // (1,2) block: s_c G_h p^* (homogeneous), tested with \bar{v}_h and \psi \tilde{v}_h
  {
    BlockVectorType & gradient = tmp_velocity_enriched_3;
    this->apply_pressure_gradient_term_enriched(gradient, src_constrained.block(2));
    result.add(scaling_factor_continuity, gradient);
  }

  set_velocity_enriched(dst, result);

  // (2,1) block: -s_c D_h u_h (homogeneous)
  this->apply_velocity_divergence_term(dst.block(2), velocity);
  dst.block(2) *= -scaling_factor_continuity;

  // E K E X + (I - E) X on the velocity rows
  constrain_velocity_enriched(dst);
  for(unsigned int b = 0; b < 2; ++b)
  {
    dst.block(b).add(1.0, src.block(b));
    dst.block(b).add(-1.0, src_constrained.block(b));
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::evaluate_nonlinear_residual_enriched(
  BlockVectorType &       dst,
  BlockVectorType const & src,
  BlockVectorType const & rhs_velocity,
  double const            time,
  double const            scaling_factor_mass) const
{
  (void)scaling_factor_mass; // set in solve_nonlinear_problem_enriched()

  BlockVectorType & velocity = tmp_velocity_enriched;
  BlockVectorType & residual = tmp_velocity_enriched_2;
  BlockVectorType & term     = tmp_velocity_enriched_3;
  copy_velocity_enriched(velocity, src);

  /*
   * momentum rows: s M u_h + K u_h + c(u_h) + s_c G(p^*) - rhs, tested with \bar{v}_h and
   * \psi \tilde{v}_h
   */

  // mass + viscous, homogeneous
  this->enriched_momentum_operator.vmult(residual, velocity);

  // viscous term, velocity Dirichlet data
  term = 0.0;
  this->viscous_operator.set_time(time);
  this->viscous_operator.rhs_add(term.block(0));
  this->rhs_add_enriched_viscous_boundary_data(term.block(0), term.block(1), time);
  residual.add(-1.0, term);

  // implicit convective term c(u_h), incl. Dirichlet data
  if(this->param.implicit_convective_problem())
  {
    this->evaluate_convective_term_enriched(term, velocity, time);
    residual.add(1.0, term);
  }

  // pressure gradient, incl. Dirichlet data g_p / s_c
  this->evaluate_pressure_gradient_term_enriched(term,
                                                 src.block(2),
                                                 time,
                                                 1.0 / scaling_factor_continuity);
  residual.add(scaling_factor_continuity, term);

  // constant right-hand side (time derivative history, body force)
  residual.add(-1.0, rhs_velocity);

  set_velocity_enriched(dst, residual);

  /*
   * continuity rows: -s_c D(u_h), incl. velocity Dirichlet data
   */
  this->evaluate_velocity_divergence_term(dst.block(2), velocity, time);
  dst.block(2) *= -scaling_factor_continuity;

  // R_c(X) = E R(X) + (I - E) X on the velocity rows
  constrain_velocity_enriched(dst);
  BlockVectorType & src_constrained = tmp_solution_enriched;
  src_constrained                   = src;
  constrain_velocity_enriched(src_constrained);
  for(unsigned int b = 0; b < 2; ++b)
  {
    dst.block(b).add(1.0, src.block(b));
    dst.block(b).add(-1.0, src_constrained.block(b));
  }
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::set_linearization_enriched(
  BlockVectorType const & solution_linearization) const
{
  copy_velocity_enriched(linearization_enriched, solution_linearization);

  // U_bar rows of non-enriched cells: ExaDG's ConvectiveKernel, linearized about W_bar
  this->set_velocity_ptr(linearization_enriched.block(0));
}

template<int dim, typename Number>
unsigned int
OperatorCoupled<dim, Number>::solve_linear_stokes_problem_enriched(BlockVectorType &       dst,
                                                                   BlockVectorType const & rhs,
                                                                   double const scaling_factor_mass)
{
  AssertThrow(not this->param.implicit_convective_problem(),
              dealii::ExcMessage("solve_linear_stokes_problem_enriched(): use "
                                 "solve_nonlinear_problem_enriched() for an implicit convective "
                                 "term."));

  // mass factor of the velocity block, also used by the Schur-complement preconditioner
  momentum_operator_mass_viscous.set_scaling_factor_mass_operator(scaling_factor_mass);
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);

  // the block-Jacobi preconditioner depends on psi (u_tau) and on the mass factor
  update_block_preconditioner_enriched();

  // K_c X = E B, X_0 <- E X_0
  BlockVectorType rhs_constrained(rhs);
  constrain_velocity_enriched(rhs_constrained);
  constrain_velocity_enriched(dst);

  return linear_solver_enriched.solve(dst, rhs_constrained);
}

template<int dim, typename Number>
std::tuple<unsigned int, unsigned int>
OperatorCoupled<dim, Number>::solve_nonlinear_problem_enriched(
  BlockVectorType &       dst,
  BlockVectorType const & rhs_velocity,
  double const            time,
  double const            scaling_factor_mass)
{
  AssertThrow(newton_solver_enriched.get() != nullptr,
              dealii::ExcMessage("Newton solver for the enriched coupled problem not set up."));

  nonlinear_operator_enriched.update(rhs_velocity, time, scaling_factor_mass);

  // mass factor of the velocity block, also used by the Schur-complement preconditioner
  momentum_operator_mass_viscous.set_scaling_factor_mass_operator(scaling_factor_mass);
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);
  this->momentum_operator.set_time(time);

  // Dirichlet data of the linearization point in the linearized convective term
  time_linearization_enriched = time;

  // the block-Jacobi preconditioner depends on psi (u_tau) and on the mass factor, not on the
  // linearization point: update once per time step
  update_block_preconditioner_enriched();

  // initial guess in the constrained space
  constrain_velocity_enriched(dst);

  Newton::UpdateData update;
  update.do_update = false;

  return newton_solver_enriched->solve(dst, update);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::update_block_preconditioner_enriched()
{
  VectorType const * free_bar =
    (this->param.wall_bc_viscous == WallBoundaryConditionViscous::Strong) ? &this->free_dofs_bar :
                                                                           nullptr;

  // local operator on enriched cells: s M + nu (grad, grad) on (U_bar, U_tilde), restricted by E
  this->enriched_block_jacobi_preconditioner.update(
    momentum_operator_mass_viscous.get_scaling_factor_mass_operator(),
    this->param.viscosity,
    nullptr,
    free_bar);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_preconditioner_velocity_block_enriched(
  BlockVectorType &       dst,
  BlockVectorType const & src) const
{
  this->enriched_block_jacobi_preconditioner.vmult(dst, src);
}

template<int dim, typename Number>
void
OperatorCoupled<dim, Number>::apply_block_preconditioner_enriched(BlockVectorType &       dst,
                                                                  BlockVectorType const & src) const
{
  /*
   * Same block preconditioners as apply_block_preconditioner(): the velocity block A_h^{-1} is
   * approximated by the enriched block-Jacobi preconditioner, the pressure Schur complement by
   * the preconditioner of the standard solver (the pressure space is not enriched). The
   * pressure gradient and velocity divergence operators act on u_h = \bar{u}_h + \psi \tilde{u}_h.
   */
  auto const type = this->param.preconditioner_coupled;

  BlockVectorType & velocity_in  = tmp_velocity_enriched;
  BlockVectorType & velocity_out = tmp_velocity_enriched_2;

  if(type == PreconditionerCoupled::BlockDiagonal)
  {
    // dst = diag(A^{-1}, -S^{-1}) src
    apply_preconditioner_pressure_block(dst.block(2), src.block(2));

    copy_velocity_enriched(velocity_in, src);
    apply_preconditioner_velocity_block_enriched(velocity_out, velocity_in);
    set_velocity_enriched(dst, velocity_out);
  }
  else if(type == PreconditionerCoupled::BlockTriangular)
  {
    /*
     *                         / A^{-1}  0 \   / I  B^{T} \   / I      0    \
     *  -> P_triangular^{-1} = |           | * |          | * |             |
     *                         \   0     I /   \ 0   -I   /   \ 0   -S^{-1} /
     */
    apply_preconditioner_pressure_block(dst.block(2), src.block(2));

    // velocity rows: src_u + s_c E G_h dst_p
    copy_velocity_enriched(velocity_in, src);
    {
      BlockVectorType & gradient = tmp_velocity_enriched_3;
      this->apply_pressure_gradient_term_enriched(gradient, dst.block(2));
      gradient.block(0) *= scaling_factor_continuity;
      gradient.block(1) *= scaling_factor_continuity;
      if(this->param.wall_bc_viscous == WallBoundaryConditionViscous::Strong)
        gradient.block(0).scale(this->free_dofs_bar);
      this->get_wall_dg_coupler()->apply_enrichment_projector(gradient.block(1),
                                                              gradient.block(1),
                                                              false);
      velocity_in.add(1.0, gradient);
    }
    dst.block(2) *= -1.0;

    apply_preconditioner_velocity_block_enriched(velocity_out, velocity_in);
    set_velocity_enriched(dst, velocity_out);
  }
  else if(type == PreconditionerCoupled::BlockTriangularFactorization)
  {
    /*
     *                          / I  - A^{-1} B^{T} \   / I      0    \   / I   0 \   / A^{-1} 0 \
     *  -> P_tria-factor^{-1} = |                   | * |             | * |       | * |          |
     *                          \ 0          I      /   \ 0   -S^{-1} /   \ B  -I /   \   0    I /
     */

    // dst_u = A^{-1} src_u
    copy_velocity_enriched(velocity_in, src);
    apply_preconditioner_velocity_block_enriched(velocity_out, velocity_in);

    // dst_p = -(src_p - s_c D_h dst_u)  (B = -s_c D_h)
    this->apply_velocity_divergence_term(vec_tmp_pressure, velocity_out);
    tmp_pressure_enriched = src.block(2);
    tmp_pressure_enriched.add(scaling_factor_continuity, vec_tmp_pressure);
    tmp_pressure_enriched *= -1.0;

    // dst_p = -S^{-1} dst_p
    apply_preconditioner_pressure_block(dst.block(2), tmp_pressure_enriched);

    // dst_u = dst_u - A^{-1} s_c E G_h dst_p
    BlockVectorType & gradient = tmp_velocity_enriched_3;
    this->apply_pressure_gradient_term_enriched(gradient, dst.block(2));
    gradient.block(0) *= scaling_factor_continuity;
    gradient.block(1) *= scaling_factor_continuity;
    if(this->param.wall_bc_viscous == WallBoundaryConditionViscous::Strong)
      gradient.block(0).scale(this->free_dofs_bar);
    this->get_wall_dg_coupler()->apply_enrichment_projector(gradient.block(1),
                                                            gradient.block(1),
                                                            false);
    apply_preconditioner_velocity_block_enriched(velocity_in, gradient);
    velocity_out.add(-1.0, velocity_in);

    set_velocity_enriched(dst, velocity_out);
  }
  else
  {
    AssertThrow(false, dealii::ExcNotImplemented());
  }
}

template class LinearSolverCoupledEnriched<2, float>;
template class LinearSolverCoupledEnriched<2, double>;

template class LinearSolverCoupledEnriched<3, float>;
template class LinearSolverCoupledEnriched<3, double>;

template class OperatorCoupled<2, float>;
template class OperatorCoupled<2, double>;

template class OperatorCoupled<3, float>;
template class OperatorCoupled<3, double>;

} // namespace IncNS
} // namespace ExaDG
