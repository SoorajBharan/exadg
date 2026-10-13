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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_projection_methods.h>
#include <exadg/poisson/preconditioners/multigrid_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/utilities/check_multigrid.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/fe/fe_dgq.h>

#include <typeinfo>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
OperatorProjectionMethods<dim, Number>::OperatorProjectionMethods(
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
         mpi_comm_in)
{
}

template<int dim, typename Number>
OperatorProjectionMethods<dim, Number>::~OperatorProjectionMethods()
{
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::setup_derived()
{
  initialize_laplace_operator();
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::update_after_grid_motion(bool const update_matrix_free)
{
  Base::update_after_grid_motion(update_matrix_free);

  // update SIPG penalty parameter of Laplace operator which depends on the deformation
  // of elements
  laplace_operator.update_penalty_parameter();
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::setup_preconditioners_and_solvers()
{
  setup_preconditioner_pressure_poisson();
  setup_solver_pressure_poisson();

  Base::setup_projection_solver();

  setup_momentum_preconditioner();
  setup_momentum_solver();

  if(this->param.wall_enrichment_enabled)
  {
    enriched_momentum_operator.initialize(this->momentum_operator,
                                          this->get_function_enrichment());
    enriched_momentum_operator.set_enrichment_aware_penalty(this->param.use_enrichment_aware_penalty);
    enriched_momentum_operator.set_wall_penalty_factor(this->param.wall_bc_viscous == WallBoundaryConditionViscous::Weak ? this->param.wall_penalty_factor_viscous : 1.0);

    if(this->param.wall_bc_viscous == WallBoundaryConditionViscous::Strong)
      setup_strong_wall_dofs();

    enriched_momentum_preconditioner.initialize(*this);
    enriched_block_jacobi_preconditioner.initialize(*this);
    if(this->param.use_divergence_penalty or this->param.use_continuity_penalty)
      enriched_projection_operator.initialize(*this->projection_operator,
                                              this->get_function_enrichment(),
                                              this->param.degree_u,
                                              this->param.use_divergence_penalty,
                                              this->param.use_continuity_penalty,
                                              this->param.divergence_penalty_factor,
                                              this->param.continuity_penalty_factor);
  }
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::initialize_laplace_operator()
{
  // setup Laplace operator
  Poisson::LaplaceOperatorData<0, dim> laplace_operator_data;
  laplace_operator_data.dof_index  = this->get_dof_index_pressure();
  laplace_operator_data.quad_index = this->get_quad_index_pressure();

  /*
   * In case no Dirichlet boundary conditions as prescribed for the pressure, the pressure Poisson
   * equation is singular (i.e. vectors describing a constant pressure state form the nullspace
   * of the discrete pressure Poisson operator). To solve the pressure Poisson equation in that
   * case, a Krylov subspace projection is applied during the solution of the linear system of
   * equations.
   *
   * Strictly speaking, this projection is only needed if the linear system of equations
   *
   *  A * p = rhs (with nullspace spanned by vector p0)
   *
   * is not consistent, meaning that
   *
   *  p0^T * rhs != 0 ,
   *
   *  so that
   *
   *  p0^T * A * p = (A^T * p0)^T * p = 0 != p0 * rhs (since A is symmetric).
   */
  if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    // the Krylov subspace projection is needed for the dual splitting scheme since the linear
    // system of equations is not consistent for this splitting method (due to the boundary
    // conditions).
    laplace_operator_data.operator_is_singular = this->is_pressure_level_undefined();
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    // One can show that the linear system of equations of the pressure Poisson equation is
    // consistent in case of the pressure-correction scheme if the velocity Dirichlet BC is
    // consistent. So there should be no need to solve a transformed linear system of equations.
    //    laplace_operator_data.operator_is_singular = false;

    // In principle, it works (since the linear system of equations is consistent)
    // but we detected no convergence for some test cases and specific parameters.
    // Hence, for reasons of robustness we also solve a transformed linear system of equations
    // in case of the pressure-correction scheme.
    laplace_operator_data.operator_is_singular = this->is_pressure_level_undefined();
  }

  laplace_operator_data.bc                   = this->boundary_descriptor_laplace;
  laplace_operator_data.use_cell_based_loops = this->param.use_cell_based_face_loops;
  laplace_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    this->param.implement_block_diagonal_preconditioner_matrix_free;

  laplace_operator_data.kernel_data.IP_factor = this->param.IP_factor_pressure;

  laplace_operator.initialize(this->get_matrix_free(),
                              this->get_constraint_p(),
                              laplace_operator_data);
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::setup_preconditioner_pressure_poisson()
{
  // setup preconditioner
  if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::None)
  {
    // do nothing, preconditioner will not be used
  }
  else if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::PointJacobi)
  {
    preconditioner_pressure_poisson =
      std::make_shared<JacobiPreconditioner<Poisson::LaplaceOperator<dim, Number, 1>>>(
        laplace_operator, true);
  }
  else if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::BlockJacobi)
  {
    preconditioner_pressure_poisson =
      std::make_shared<BlockJacobiPreconditioner<Poisson::LaplaceOperator<dim, Number, 1>>>(
        laplace_operator, true);
  }
  else if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Multigrid)
  {
    MultigridData mg_data;
    mg_data = this->param.multigrid_data_pressure_poisson;

    typedef Poisson::MultigridPreconditioner<dim, Number, 1> Multigrid;

    preconditioner_pressure_poisson = std::make_shared<Multigrid>(this->mpi_comm);

    std::shared_ptr<Multigrid> mg_preconditioner =
      std::dynamic_pointer_cast<Multigrid>(preconditioner_pressure_poisson);

    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      dirichlet_boundary_conditions = laplace_operator.get_data().bc->dirichlet_bc;

    typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
    Map_DBC_ComponentMask                                               dirichlet_bc_component_mask;

    auto & dof_handler = this->get_dof_handler_p();
    mg_preconditioner->initialize(mg_data,
                                  this->grid,
                                  this->multigrid_mappings,
                                  dof_handler.get_fe(),
                                  laplace_operator.get_data(),
                                  this->param.ale_formulation,
                                  dirichlet_boundary_conditions,
                                  dirichlet_bc_component_mask);
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "Specified preconditioner for pressure Poisson equation not implemented"));
  }
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::setup_solver_pressure_poisson()
{
  if(this->param.solver_pressure_poisson == SolverPressurePoisson::CG)
  {
    // setup solver data
    Krylov::SolverDataCG solver_data;
    solver_data.max_iter             = this->param.solver_data_pressure_poisson.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_pressure_poisson.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_pressure_poisson.rel_tol;
    // use default value of update_preconditioner (=false)

    if(this->param.preconditioner_pressure_poisson != PreconditionerPressurePoisson::None)
    {
      solver_data.use_preconditioner = true;
    }

    // setup solver
    pressure_poisson_solver =
      std::make_shared<Krylov::SolverCG<Poisson::LaplaceOperator<dim, Number, 1>,
                                        PreconditionerBase<Number>,
                                        VectorType>>(laplace_operator,
                                                     *preconditioner_pressure_poisson,
                                                     solver_data);
  }
  else if(this->param.solver_pressure_poisson == SolverPressurePoisson::FGMRES)
  {
    Krylov::SolverDataFGMRES solver_data;
    solver_data.max_iter             = this->param.solver_data_pressure_poisson.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_pressure_poisson.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_pressure_poisson.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_pressure_poisson.max_krylov_size;
    // use default value of update_preconditioner (=false)

    if(this->param.preconditioner_pressure_poisson != PreconditionerPressurePoisson::None)
    {
      solver_data.use_preconditioner = true;
    }

    pressure_poisson_solver =
      std::make_shared<Krylov::SolverFGMRES<Poisson::LaplaceOperator<dim, Number, 1>,
                                            PreconditionerBase<Number>,
                                            VectorType>>(laplace_operator,
                                                         *preconditioner_pressure_poisson,
                                                         solver_data);
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "Specified solver for pressure Poisson equation is not implemented."));
  }
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::setup_momentum_preconditioner()
{
  if(this->param.preconditioner_momentum == MomentumPreconditioner::InverseMassMatrix)
  {
    InverseMassOperatorData inverse_mass_operator_data;
    inverse_mass_operator_data.dof_index  = this->get_dof_index_velocity();
    inverse_mass_operator_data.quad_index = this->get_quad_index_velocity_standard();
    inverse_mass_operator_data.parameters = this->param.inverse_mass_preconditioner;

    momentum_preconditioner =
      std::make_shared<InverseMassPreconditioner<dim, dim, Number>>(this->get_matrix_free(),
                                                                    inverse_mass_operator_data);
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::PointJacobi)
  {
    momentum_preconditioner =
      std::make_shared<JacobiPreconditioner<MomentumOperator<dim, Number>>>(this->momentum_operator,
                                                                            false);
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::BlockJacobi)
  {
    momentum_preconditioner =
      std::make_shared<BlockJacobiPreconditioner<MomentumOperator<dim, Number>>>(
        this->momentum_operator, false);
  }
  else if(this->param.preconditioner_momentum == MomentumPreconditioner::Multigrid)
  {
    typedef MultigridPreconditioner<dim, Number> Multigrid;

    momentum_preconditioner = std::make_shared<Multigrid>(this->mpi_comm);

    std::shared_ptr<Multigrid> mg_preconditioner =
      std::dynamic_pointer_cast<Multigrid>(momentum_preconditioner);

    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      dirichlet_boundary_conditions = this->momentum_operator.get_data().bc->dirichlet_bc;

    // We also need to add DirichletCached boundary conditions. From the
    // perspective of multigrid, there is no difference between standard
    // and cached Dirichlet BCs. Since multigrid does not need information
    // about inhomogeneous boundary data, we simply fill the map with
    // dealii::Functions::ZeroFunction for DirichletCached BCs.
    for(auto iter : this->momentum_operator.get_data().bc->dirichlet_cached_bc)
    {
      typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        pair;

      dirichlet_boundary_conditions.insert(
        pair(iter, new dealii::Functions::ZeroFunction<dim>(dim)));
    }

    typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
    Map_DBC_ComponentMask                                               dirichlet_bc_component_mask;

    mg_preconditioner->initialize(this->param.multigrid_data_momentum,
                                  this->grid,
                                  this->multigrid_mappings,
                                  this->get_dof_handler_u().get_fe(),
                                  this->momentum_operator,
                                  this->param.multigrid_operator_type_momentum,
                                  this->param.ale_formulation,
                                  dirichlet_boundary_conditions,
                                  dirichlet_bc_component_mask);
  }
  else
  {
    AssertThrow(this->param.preconditioner_momentum == MomentumPreconditioner::None,
                dealii::ExcNotImplemented());
  }
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::setup_momentum_solver()
{
  if(this->param.solver_momentum == SolverMomentum::CG)
  {
    // setup solver data
    Krylov::SolverDataCG solver_data;
    solver_data.max_iter             = this->param.solver_data_momentum.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_momentum.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_momentum.rel_tol;
    if(this->param.preconditioner_momentum != MomentumPreconditioner::None)
      solver_data.use_preconditioner = true;

    // setup solver
    momentum_linear_solver = std::make_shared<
      Krylov::SolverCG<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>>(
      this->momentum_operator, *momentum_preconditioner, solver_data);
  }
  else if(this->param.solver_momentum == SolverMomentum::GMRES)
  {
    // setup solver data
    Krylov::SolverDataGMRES solver_data;
    solver_data.max_iter             = this->param.solver_data_momentum.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_momentum.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_momentum.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_momentum.max_krylov_size;
    solver_data.compute_eigenvalues  = false;
    if(this->param.preconditioner_momentum != MomentumPreconditioner::None)
      solver_data.use_preconditioner = true;

    // setup solver
    momentum_linear_solver = std::make_shared<
      Krylov::SolverGMRES<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>>(
      this->momentum_operator, *momentum_preconditioner, solver_data, this->mpi_comm);
  }
  else if(this->param.solver_momentum == SolverMomentum::FGMRES)
  {
    Krylov::SolverDataFGMRES solver_data;
    solver_data.max_iter             = this->param.solver_data_momentum.max_iter;
    solver_data.solver_tolerance_abs = this->param.solver_data_momentum.abs_tol;
    solver_data.solver_tolerance_rel = this->param.solver_data_momentum.rel_tol;
    solver_data.max_n_tmp_vectors    = this->param.solver_data_momentum.max_krylov_size;
    if(this->param.preconditioner_momentum != MomentumPreconditioner::None)
      solver_data.use_preconditioner = true;

    momentum_linear_solver = std::make_shared<
      Krylov::SolverFGMRES<MomentumOperator<dim, Number>, PreconditionerBase<Number>, VectorType>>(
      this->momentum_operator, *momentum_preconditioner, solver_data);
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("Specified solver for momentum equation is not implemented."));
  }


  // Navier-Stokes equations with an implicit treatment of the convective term
  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    // nonlinear_operator
    nonlinear_operator.initialize(*this);

    // setup Newton solver
    momentum_newton_solver = std::make_shared<Newton::Solver<VectorType,
                                                             NonlinearMomentumOperator<dim, Number>,
                                                             MomentumOperator<dim, Number>,
                                                             Krylov::SolverBase<VectorType>>>(
      this->param.newton_solver_data_momentum,
      nonlinear_operator,
      this->momentum_operator,
      *momentum_linear_solver);
  }
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::rhs_add_viscous_term(VectorType & dst,
                                                             double const time) const
{
  this->viscous_operator.set_time(time);
  this->viscous_operator.rhs_add(dst);
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::do_rhs_ppe_laplace_add(VectorType &   dst,
                                                               double const & time) const
{
  this->laplace_operator.set_time(time);
  this->laplace_operator.rhs_add(dst);
}

template<int dim, typename Number>
unsigned int
OperatorProjectionMethods<dim, Number>::do_solve_pressure(VectorType &       dst,
                                                          VectorType const & src,
                                                          bool const update_preconditioner) const
{
  // Check multigrid algorithm
  //  std::shared_ptr<MultigridPoisson> mg_preconditioner
  //    = std::dynamic_pointer_cast<MultigridPoisson>(preconditioner_pressure_poisson);
  //
  //  CheckMultigrid<dim, Number, Poisson::LaplaceOperator<dim, Number>, MultigridPoisson,
  //  MultigridNumber>
  //    check_multigrid(this->laplace_operator, mg_preconditioner, this->mpi_comm);
  //  check_multigrid.check();

  // Use multigrid as a solver (use double precision here)
  //  std::shared_ptr<MultigridPoisson> mg_preconditioner
  //    = std::dynamic_pointer_cast<MultigridPoisson>(preconditioner_pressure_poisson);
  //  unsigned int n_iter = mg_preconditioner->solve(dst,src);

  // update preconditioner
  this->pressure_poisson_solver->update_preconditioner(update_preconditioner);

  // call pressure Poisson solver
  unsigned int n_iter = this->pressure_poisson_solver->solve(dst, src);

  return n_iter;
}


template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::apply_laplace_operator(VectorType &       dst,
                                                               VectorType const & src) const
{
  this->laplace_operator.vmult(dst, src);
}

template<int dim, typename Number>
unsigned int
OperatorProjectionMethods<dim, Number>::solve_linear_momentum_equation(
  VectorType &       solution,
  VectorType const & rhs,
  bool const &       update_preconditioner,
  double const &     scaling_factor_mass)
{
  // We do not need to set the time here, because time affects the operator only in the form of
  // boundary conditions. The result of such boundary condition evaluations is handed over to this
  // function via the vector rhs.
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);

  // Note that there is no need to set the evaluation time for the momentum_operator
  // because this function is only called if the convective term is not considered
  // in the momentum_operator (Stokes eq. or explicit treatment of convective term).

  this->momentum_linear_solver->update_preconditioner(update_preconditioner);

  auto linear_iterations = this->momentum_linear_solver->solve(solution, rhs);

  return linear_iterations;
}

template<int dim, typename Number>
unsigned int
OperatorProjectionMethods<dim, Number>::solve_linear_momentum_equation(
  BlockVectorType &       solution,
  BlockVectorType const & rhs,
  double const &          scaling_factor_mass)
{
  AssertThrow(this->param.wall_enrichment_enabled,
              dealii::ExcMessage("Block momentum solve requires wall_enrichment_enabled."));

  // EnrichedMomentumOperator reads the mass scaling from momentum_operator
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);

  // Step 7c: strong no-slip -> U_bar DoFs on enrichment walls fixed to zero (E_bar)
  VectorType const * free_bar =
    (this->param.wall_bc_viscous == WallBoundaryConditionViscous::Strong) ? &free_dofs_bar : nullptr;

  enriched_momentum_preconditioner.set_scaling_factor_mass(scaling_factor_mass);
  // local operator on enriched cells: s M + nu (grad, grad)
  enriched_block_jacobi_preconditioner.update(scaling_factor_mass,
                                              this->param.viscosity,
                                              nullptr,
                                              free_bar);

  dealii::ReductionControl control(this->param.solver_data_momentum.max_iter,
                                   this->param.solver_data_momentum.abs_tol,
                                   this->param.solver_data_momentum.rel_tol);

  // Galerkin problem on the well-posed enrichment space (Step 5b):
  //   K_c X = E B,  X_0 <- E X_0,  K_c = E K E + diag(0, I - P),  E = diag(I, P)
  WallDGCoupler<dim, Number> const & coupler = *this->get_wall_dg_coupler();

  EnrichmentConstrainedOperator<dim, Number, EnrichedMomentumOperator<dim, Number>> const
  constrained_operator(enriched_momentum_operator, coupler, free_bar);

  BlockVectorType rhs_constrained(rhs);
  constrained_operator.constrain(rhs_constrained); // E B (homogeneous no-slip: fixed entries 0)
  constrained_operator.constrain(solution);        // E X_0

  if(this->param.solver_momentum == SolverMomentum::CG)
  {
    dealii::SolverCG<BlockVectorType> solver(control);
    /*************************************DEBUG*********************************/
    solver.connect_eigenvalues_slot(
      [&](std::vector<double> const & eigenvalues) {
        if(not eigenvalues.empty())
          this->pcout << "  [viscous debug] eigenvalues of P^{-1}A in ["
                      << *std::min_element(eigenvalues.begin(), eigenvalues.end()) << ", "
                      << *std::max_element(eigenvalues.begin(), eigenvalues.end()) << "]" << std::endl;
      },
      false);
    /************************************************************************************/
    // solver.solve(constrained_operator, s
    // olution, rhs_constrained, enriched_block_jacobi_preconditioner);
    /*************************************DEBUG*********************************/
    try
    {
      solver.solve(constrained_operator, solution, rhs_constrained, enriched_block_jacobi_preconditioner);
    }
    catch(dealii::SolverControl::NoConvergence const & e)
    {
      this->pcout << "  [viscous debug] NO CONVERGENCE: initial residual = " << control.initial_value()
                  << "   last residual = " << e.last_residual << "   iterations = " << e.last_step
                  << std::endl;
      throw;
    }
    this->pcout << "  [viscous debug] initial residual = " << control.initial_value()
                << "   final residual = " << control.last_value()
                << "   iterations = " << control.last_step() << std::endl;
    /************************************************************************************/
  }
  else
  {
    typename dealii::SolverGMRES<BlockVectorType>::AdditionalData data;
    data.max_basis_size = this->param.solver_data_momentum.max_krylov_size;

    dealii::SolverGMRES<BlockVectorType> solver(control, data);
    // solver.solve(constrained_operator, solution, rhs_constrained, enriched_block_jacobi_preconditioner);
    /*************************************DEBUG*********************************/
    try
    {
      solver.solve(constrained_operator, solution, rhs_constrained, enriched_block_jacobi_preconditioner);
    }
    catch(dealii::SolverControl::NoConvergence const & e)
    {
      this->pcout << "  [viscous debug] NO CONVERGENCE: initial residual = " << control.initial_value()
                  << "   last residual = " << e.last_residual << "   iterations = " << e.last_step
                  << std::endl;
      throw;
    }
    this->pcout << "  [viscous debug] initial residual = " << control.initial_value()
                << "   final residual = " << control.last_value()
                << "   iterations = " << control.last_step() << std::endl;
    /************************************************************************************/
  }

  return control.last_step();
}

/*
 * Strong no-slip on enrichment walls in the viscous step (Krank et al. 2018, App. C; thesis
 * Sec. 5.3.1).
 *
 * u_h = \bar u_h + \psi \tilde u_h with \psi = \kappa u^+(0) = 0 at the wall, so
 * u_h|_wall = \bar u_h|_wall. FE_DGQ uses Lagrange polynomials \ell_i on Gauss-Lobatto nodes,
 * \ell_i(x_d = side) = 0 for every node i not on the face x_d = side, so the trace of \bar u_h
 * depends only on the nodal values on that face. Fixing
 *   \bar U_i = 0   for all nodes i on enrichment-wall faces (all components)
 * gives u_h = 0 on the wall exactly. The remaining test functions vanish on the wall as well,
 * so all NIPG wall face terms drop out: u_h^+ = u_h^- = 0, \nabla u_h^+ = \nabla u_h^- on the
 * face terms, mass term unchanged.
 *
 * FEEvaluation stores the DoFs of FE_DGQ lexicographically per component:
 *   local index = c (k+1)^dim + \sum_d i_d (k+1)^d,
 * and face f of a hypercube has normal direction d = f / 2 and nodes i_d = (f % 2) k.
 */
template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::setup_strong_wall_dofs()
{
  AssertThrow(typeid(this->get_fe_u().base_element(0)) == typeid(dealii::FE_DGQ<dim>),
              dealii::ExcMessage("Strong wall boundary condition requires FE_DGQ (Gauss-Lobatto nodes)."));
  AssertThrow(this->param.degree_u >= 1,
              dealii::ExcMessage("Strong wall boundary condition requires degree_u >= 1."));

  this->initialize_vector_velocity(free_dofs_bar);
  free_dofs_bar = 1.0;

  auto const & mf = this->get_matrix_free();
  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi(mf,
                                                    this->get_dof_index_velocity(),
                                                    this->get_quad_index_velocity_standard());

  unsigned int const n_1d     = this->param.degree_u + 1;
  unsigned int const n_scalar = dealii::Utilities::pow(n_1d, dim);
  auto const &       wall_ids = this->boundary_descriptor->velocity->wall_enrichment_bc;

  for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
  {
    phi.reinit(cell);
    phi.read_dof_values_plain(free_dofs_bar);
    for(unsigned int v = 0; v < mf.n_active_entries_per_cell_batch(cell); ++v)
    {
      auto const cell_it = mf.get_cell_iterator(cell, v);
      for(auto const f : cell_it->face_indices())
      {
        if(not cell_it->at_boundary(f) or cell_it->has_periodic_neighbor(f))
          continue;
        if(wall_ids.find(cell_it->face(f)->boundary_id()) == wall_ids.end())
          continue;

        unsigned int const d       = f / 2;
        unsigned int const index_d = (f % 2) * (n_1d - 1);
        unsigned int const stride  = dealii::Utilities::pow(n_1d, d);

        for(unsigned int i = 0; i < n_scalar; ++i)
          if((i / stride) % n_1d == index_d)
            for(unsigned int c = 0; c < dim; ++c)
              phi.begin_dof_values()[c * n_scalar + i][v] = 0.0;
      }
    }
    phi.set_dof_values_plain(free_dofs_bar);
  }

  double const n_fixed = double(free_dofs_bar.size()) - free_dofs_bar.l1_norm();
  this->pcout << "  [strong wall BC] fixed U_bar DoFs: " << n_fixed << " of "
              << free_dofs_bar.size() << std::endl;
}

template<int dim, typename Number>
std::tuple<unsigned int, unsigned int>
OperatorProjectionMethods<dim, Number>::solve_nonlinear_momentum_equation(
  VectorType &       dst,
  VectorType const & rhs_vector,
  double const &     time,
  bool const &       update_preconditioner,
  double const &     scaling_factor_mass)
{
  // update nonlinear operator
  this->nonlinear_operator.update(rhs_vector, time, scaling_factor_mass);

  // Set time and scaling_factor_mass for linear operator
  this->momentum_operator.set_time(time);
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);

  // Solve nonlinear problem
  Newton::UpdateData update;
  update.do_update                = update_preconditioner;
  update.update_every_newton_iter = this->param.update_preconditioner_momentum_every_newton_iter;

  std::tuple<unsigned int, unsigned int> iter = this->momentum_newton_solver->solve(dst, update);

  return iter;
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::evaluate_nonlinear_residual(
  VectorType &       dst,
  VectorType const & src,
  VectorType const * rhs_vector,
  double const &     time,
  double const &     scaling_factor_mass) const
{
  // update implicitly coupled viscosity
  if(this->param.nonlinear_viscous_problem())
  {
    this->update_viscosity(src);
  }

  this->mass_operator.apply_scale(dst, scaling_factor_mass, src);

  // implicitly treated convective term
  if(this->param.implicit_convective_problem())
  {
    this->convective_operator.evaluate_nonlinear_operator_add(dst, src, time);
  }

  // viscous term
  this->viscous_operator.set_time(time);
  this->viscous_operator.evaluate_add(dst, src);

  // rhs vector
  dst.add(-1.0, *rhs_vector);
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::apply_projection_operator(VectorType &       dst,
                                                                  VectorType const & src) const
{
  AssertThrow(this->projection_operator.get() != 0,
              dealii::ExcMessage("Projection operator is not initialized correctly."));

  this->projection_operator->vmult(dst, src);
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::apply_enriched_momentum_operator_test(
  BlockVectorType &       dst,
  BlockVectorType const & src,
  double const            scaling_factor_mass)
{
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);
  enriched_momentum_operator.vmult(dst, src);
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::apply_momentum_operator_test(
  VectorType &       dst,
  VectorType const & src,
  double const       scaling_factor_mass)
{
  this->momentum_operator.set_scaling_factor_mass_operator(scaling_factor_mass);
  this->momentum_operator.apply(dst, src);
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::rhs_add_enriched_viscous_boundary_data(VectorType & dst_bar,
                                                                              VectorType & dst_tilde,
                                                                              double const time) const
{
  enriched_momentum_operator.update_enriched_penalty_parameter();
  enriched_momentum_operator.rhs_add_boundary_data(dst_bar, dst_tilde, time);
}

template<int dim, typename Number>
void
OperatorProjectionMethods<dim, Number>::update_projection_operator_enriched(
  VectorType const & velocity_total,
  double const       time_step_size)
{
  // block(0): ExaDG's projection operator, tau from the full velocity
  this->update_projection_operator(velocity_total, time_step_size);
  // coupling terms: same tau formulas on matrix_free_en
  enriched_projection_operator.update(velocity_total, time_step_size);
}

template<int dim, typename Number>
unsigned int
OperatorProjectionMethods<dim, Number>::solve_projection_enriched(BlockVectorType &       dst,
                                                                  BlockVectorType const & rhs)
{
  // operator = M + dt (D + C) ~ M  ->  block inverse mass is a good preconditioner
  enriched_momentum_preconditioner.set_scaling_factor_mass(1.0);

  // local operator on enriched cells: M + dt tau_D (div, div)
  dealii::AlignedVector<dealii::VectorizedArray<Number>> const tau_divergence_times_dt =
    enriched_projection_operator.get_tau_divergence_times_dt();
  enriched_block_jacobi_preconditioner.update(
    1.0,
    0.0,
    enriched_projection_operator.divergence_penalty_used() ? &tau_divergence_times_dt : nullptr);

  // The enriched block system is far worse conditioned than ExaDG's projection operator
  // (small Schur complement), so a relative tolerance of 1e-12 is not reachable in double
  // precision. Use at most the tolerance of the momentum step.
  double const rel_tol = std::max(this->param.solver_data_projection.rel_tol,
                                  this->param.solver_data_momentum.rel_tol);

  dealii::ReductionControl control(this->param.solver_data_projection.max_iter,
                                   this->param.solver_data_projection.abs_tol,
                                   rel_tol);

  // Galerkin problem on the well-posed enrichment space (Step 5b):
  //   K_c X = E B,  X_0 <- E X_0,  K_c = E K E + diag(0, I - P),  E = diag(I, P)
  WallDGCoupler<dim, Number> const & coupler = *this->get_wall_dg_coupler();

  BlockVectorType rhs_constrained(rhs);
  coupler.apply_enrichment_projector(rhs_constrained.block(1), rhs.block(1), false); // E B
  coupler.apply_enrichment_projector(dst.block(1), dst.block(1), false);             // E X_0

  EnrichmentConstrainedOperator<dim, Number, EnrichedProjectionOperator<dim, Number>> const
    constrained_operator(enriched_projection_operator, coupler);                     // K_c

  /************************************DEBUG************************************/
  // ---- DEBUG 1: symmetry  x.(A y) == y.(A x) -------------------------------------
  {
    BlockVectorType x(rhs), y(rhs), Ax(rhs), Ay(rhs);
    for(unsigned int b = 0; b < 2; ++b)
      for(unsigned int i = 0; i < x.block(b).locally_owned_size(); ++i)
      {
        x.block(b).local_element(i) = std::sin(1.0 + 3.0 * i + b);
        y.block(b).local_element(i) = std::cos(2.0 + 7.0 * i + b);
      }
    constrained_operator.vmult(Ax, x);
    constrained_operator.vmult(Ay, y);
    double const xAy = x * Ay, yAx = y * Ax;
    this->pcout << "  [penalty debug] x.Ay = " << xAy << "  y.Ax = " << yAx
                << "  rel. asymmetry = " << std::abs(xAy - yAx) / std::abs(xAy) << std::endl;
  }

  // ---- DEBUG 2: convergence, without aborting the run ------------------------------
  dealii::SolverCG<BlockVectorType> solver(control);
  try
  {
    solver.solve(constrained_operator, dst, rhs_constrained, enriched_block_jacobi_preconditioner);
  }
  catch(dealii::SolverControl::NoConvergence const &)
  {
    this->pcout << "  [penalty debug] NOT converged" << std::endl;
  }
  this->pcout << "  [penalty debug] initial residual = " << control.initial_value()
              << "   final residual = " << control.last_value()
              << "   reduction = " << control.last_value() / control.initial_value()
              << "   iterations = " << control.last_step() << std::endl;
  /*****************************************************************************/

  return control.last_step();
}

template class OperatorProjectionMethods<2, float>;
template class OperatorProjectionMethods<2, double>;

template class OperatorProjectionMethods<3, float>;
template class OperatorProjectionMethods<3, double>;

} // namespace IncNS
} // namespace ExaDG
