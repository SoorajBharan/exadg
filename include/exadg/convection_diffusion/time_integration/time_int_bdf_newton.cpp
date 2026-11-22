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

#include <array>
#include <string>

#include <exadg/convection_diffusion/spatial_discretization/operator.h>
#include <exadg/convection_diffusion/time_integration/time_int_bdf_newton.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/utilities/print_solver_results.h>
#include <exadg/utilities/section_profile.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
TimeIntBDFNewton<dim, Number>::TimeIntBDFNewton(
  std::shared_ptr<Operator<dim, Number>>          operator_in,
  std::shared_ptr<HelpersALE<dim, Number> const>  helpers_ale_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
  Parameters const &                              param_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in)
  : TimeIntBDF<dim, Number>(operator_in,
                            helpers_ale_in,
                            postprocessor_in,
                            param_in,
                            mpi_comm_in,
                            is_test_in),
    time_steps_since_preconditioner_update(0),
    preconditioner_updated(false),
    newton_iterations({0, 0})
{
}

template<int dim, typename Number>
void
TimeIntBDFNewton<dim, Number>::evaluate_residual(BlockVectorType &       residual,
                                                 BlockVectorType const & solution,
                                                 BlockVectorType const & constant_part,
                                                 VectorType const &      velocity_np) const
{
#ifdef EXADG_PROFILE
  auto t_prof = SectionProfile::Clock::now();
#endif

#ifdef EXADG_DEBUG
  // parts of the residual per block (projected as the residual), first residual of a time step:
  // b, f M Q (f = gamma_0/dt, block mass), A Q (conv-diff operator without the mass), N(Q)
  if(print_residual_decomposition)
  {
    print_residual_decomposition = false;

    BlockVectorType mass, linear, turbulence, rhs;
    this->pde_operator->initialize_dof_vector(mass);
    this->pde_operator->initialize_dof_vector(linear);
    this->pde_operator->initialize_dof_vector(turbulence);
    rhs = constant_part;

    double const factor = this->bdf.get_gamma0() / this->get_time_step_size();
    this->pde_operator->apply_mass_operator(mass, solution);
    mass *= factor;
    this->pde_operator->apply_conv_diff_operator(linear, solution);
    linear.add(-1.0, mass);
    if(this->param.turbulence_model_data.is_active)
      this->pde_operator->evaluate_add_turbulence_newton_explicit(turbulence,
                                                                  solution,
                                                                  this->get_next_time(),
                                                                  &velocity_np);

    std::array<BlockVectorType *, 4> const parts = {{&rhs, &mass, &linear, &turbulence}};
    std::array<std::string, 4> const       names = {{"b", "f M Q", "A Q", "N(Q)"}};
    this->pcout << "  [residual parts] time step " << this->get_time_step_number()
                << ", residual = b - f M Q - A Q + N(Q), per block:" << std::endl;
    for(unsigned int p = 0; p < parts.size(); ++p)
    {
      this->pde_operator->apply_enrichment_projector(*parts[p]);
      this->pcout << "    " << names[p] << ":";
      for(unsigned int b = 0; b < parts[p]->n_blocks(); ++b)
        this->pcout << "  " << parts[p]->block(b).l2_norm();
      this->pcout << std::endl;
    }

    // b - f M Q: the time-derivative part alone
    rhs.add(-1.0, mass);
    this->pcout << "    b - f M Q:";
    for(unsigned int b = 0; b < rhs.n_blocks(); ++b)
      this->pcout << "  " << rhs.block(b).l2_norm();
    this->pcout << std::endl;
  }
#endif

  // (gamma_0/dt M + A) solution, homogeneous operator (mass scaling set by
  // update_conv_diff_operator())
  this->pde_operator->apply_conv_diff_operator(residual, solution);

  // constant_part - (gamma_0/dt M + A) solution
  residual.sadd(-1.0, 1.0, constant_part);

#ifdef EXADG_PROFILE
  SectionProfile::lap("scal 2a residual: conv-diff operator", t_prof);
#endif

  // + Newton_Explicit part of the turbulence terms treated Implicit, at solution
  if(this->param.turbulence_model_data.is_active)
    this->pde_operator->evaluate_add_turbulence_newton_explicit(residual,
                                                                solution,
                                                                this->get_next_time(),
                                                                &velocity_np);

  // enrichment of RANS variables: W_tilde rows in range(P) (well-posed enrichment modes)
  this->pde_operator->apply_enrichment_projector(residual);

#ifdef EXADG_PROFILE
  SectionProfile::lap("scal 2b residual: turbulence Newton_Explicit", t_prof);
#endif
}

#ifdef EXADG_DEBUG
template<int dim, typename Number>
void
TimeIntBDFNewton<dim, Number>::check_jacobian(BlockVectorType const & residual,
                                              BlockVectorType const & constant_part,
                                              VectorType const &      velocity_np,
                                              BlockVectorType const * direction_in,
                                              double const            eps_scale) const
{
  BlockVectorType direction, jacobian_direction, difference, solution_eps;
  this->pde_operator->initialize_dof_vector(direction);
  this->pde_operator->initialize_dof_vector(jacobian_direction);
  this->pde_operator->initialize_dof_vector(difference);
  this->pde_operator->initialize_dof_vector(solution_eps);

  // fixed direction d with O(1) entries in all blocks (k, omega_bar, omega_tilde, ...)
  if(direction_in != nullptr)
  {
    // given direction (e.g. the Newton increment), scaled to |d|_inf = 1
    direction = *direction_in;
    double const norm = direction.linfty_norm();
    if(norm > 0.0)
      direction *= 1.0 / norm;
  }
  else
  {
    for(unsigned int b = 0; b < direction.n_blocks(); ++b)
      for(unsigned int i = 0; i < direction.block(b).locally_owned_size(); ++i)
        direction.block(b).local_element(i) =
          std::sin(1.0 + 0.37 * double(i + 1) + 1.3 * double(b));
  }

  // enrichment of RANS variables: W_tilde in range(P), the space of the linear systems
  this->pde_operator->apply_enrichment_projector(direction);

  // J d at the linearization point Q (set by the last evaluate_residual())
  this->pde_operator->vmult(jacobian_direction, direction);

  // finite difference: J d ~ -(r(Q + eps d) - r(Q)) / eps
  double const eps = eps_scale * std::max(1.0, double(this->solution_np.linfty_norm()));
  solution_eps     = this->solution_np;
  solution_eps.add(eps, direction);
  evaluate_residual(difference, solution_eps, constant_part, velocity_np);
  difference.add(-1.0, residual);
  difference *= -1.0 / eps;

  this->pcout << std::endl << "[Jacobian check] initial Newton residual per block:";
  for(unsigned int b = 0; b < residual.n_blocks(); ++b)
    this->pcout << "  " << residual.block(b).l2_norm();
  this->pcout << "  (total " << residual.l2_norm() << ")" << std::endl;

  this->pcout << "[Jacobian check] eps = " << eps
              << ", per block: |J d|, |FD|, |J d - FD| / max(|J d|, |FD|)" << std::endl;
  for(unsigned int b = 0; b < direction.n_blocks(); ++b)
  {
    double const norm_jacobian   = jacobian_direction.block(b).l2_norm();
    double const norm_difference = difference.block(b).l2_norm();
    VectorType   error(jacobian_direction.block(b));
    error.add(-1.0, difference.block(b));
    double const scale = std::max({norm_jacobian, norm_difference, 1.e-300});
    this->pcout << "  block " << b << ": " << norm_jacobian << "  " << norm_difference << "  "
                << error.l2_norm() / scale << std::endl;
  }

  // linearization point back at Q
  evaluate_residual(difference, this->solution_np, constant_part, velocity_np);
}

template<int dim, typename Number>
void
TimeIntBDFNewton<dim, Number>::print_residual_location(BlockVectorType const & residual) const
{
  auto const & dof_handler = this->pde_operator->get_dof_handler();
  auto const & fer         = this->pde_operator->get_function_enrichment_rans();
  MPI_Comm const comm      = dof_handler.get_triangulation().get_communicator();

  // categories: 0 enriched + active, 1 enriched + switched off, 2 halo, 3 other
  std::array<std::string, 4> const names = {
    {"enriched (active)", "enriched (switched off)", "halo", "other"}};

  auto const category = [&](auto const & cell) -> unsigned int {
    if(not fer)
      return 3;
    dealii::types::global_cell_index const index = cell->global_active_cell_index();
    if(fer->is_enriched_cell(index))
      return fer->is_enrichment_active_cell(index) ? 0 : 1;
    for(unsigned int f = 0; f < cell->n_faces(); ++f)
      if(not cell->at_boundary(f) and fer->is_enriched_cell(cell->neighbor(f)->global_active_cell_index()))
        return 2;
    return 3;
  };

  unsigned int const n_blocks = this->pde_operator->get_n_blocks();
  std::vector<dealii::types::global_dof_index> dof_indices(dof_handler.get_fe().n_dofs_per_cell());

  this->pcout << "[residual location] time step " << this->get_time_step_number()
              << ": |r| per category, largest entry (value, cell centre, category)" << std::endl;

  for(unsigned int b = 0; b < n_blocks; ++b)
  {
    std::array<double, 4> sum_squares = {{0.0, 0.0, 0.0, 0.0}};
    std::array<double, 4> local_max   = {{0.0, 0.0, 0.0, 3.0}}; // |r|, x, y, category

    for(auto const & cell : dof_handler.active_cell_iterators())
    {
      if(not cell->is_locally_owned())
        continue;
      unsigned int const c = category(cell);
      cell->get_dof_indices(dof_indices);
      for(auto const i : dof_indices)
      {
        double const value = std::abs(double(residual.block(b)(i)));
        sum_squares[c] += value * value;
        if(value > local_max[0])
          local_max = {{value, cell->center()[0], cell->center()[1], double(c)}};
      }
    }

    this->pcout << "  block " << b << ":";
    for(unsigned int c = 0; c < 4; ++c)
      this->pcout << "  " << names[c] << " "
                  << std::sqrt(dealii::Utilities::MPI::sum(sum_squares[c], comm));

    // largest entry over all processes
    std::vector<std::array<double, 4>> const all = dealii::Utilities::MPI::all_gather(comm, local_max);
    std::array<double, 4> global_max = all[0];
    for(auto const & m : all)
      if(m[0] > global_max[0])
        global_max = m;
    this->pcout << ";  max " << global_max[0] << " at (" << global_max[1] << ", " << global_max[2]
                << "), " << names[static_cast<unsigned int>(global_max[3])] << std::endl;
  }
}

template<int dim, typename Number>
void
TimeIntBDFNewton<dim, Number>::print_newton_debug(unsigned int const      iteration,
                                                  BlockVectorType const & residual,
                                                  double const            step_length,
                                                  unsigned int const      linear_iterations) const
{
  this->pcout << "  [Newton debug] time step " << this->get_time_step_number() << ", iteration "
              << iteration << ": |r| per block";
  for(unsigned int b = 0; b < residual.n_blocks(); ++b)
    this->pcout << "  " << residual.block(b).l2_norm();
  this->pcout << ";  |Q| per block";
  for(unsigned int b = 0; b < this->solution_np.n_blocks(); ++b)
    this->pcout << "  " << this->solution_np.block(b).l2_norm();
  if(iteration > 0)
    this->pcout << ";  step length " << step_length << ", GMRES iterations " << linear_iterations;
  this->pcout << std::endl;
}
#endif

template<int dim, typename Number>
void
TimeIntBDFNewton<dim, Number>::do_timestep_solve()
{
  dealii::Timer timer;
  timer.restart();

  Parameters const &         param       = this->param;
  Newton::SolverData const & newton_data = param.newton_solver_data;

  double const scaling_factor_mass = this->bdf.get_gamma0() / this->get_time_step_size();

  // transport velocity at t_{n+1}
  VectorType velocity_np;
  this->prepare_timestep(velocity_np);

  if(param.turbulence_model_data.is_active)
    AssertThrow(velocity_np.size() > 0,
                dealii::ExcMessage("Turbulence model: the velocity field is not initialized."));

#ifdef EXADG_PROFILE
  auto t_prof_rhs = SectionProfile::Clock::now();
#endif

  // constant part of the right-hand side: b (no turbulence) + Explicit turbulence terms at Q^n
  this->assemble_right_hand_side(velocity_np);

  if(param.turbulence_model_data.is_active)
    this->pde_operator->evaluate_add_turbulence_explicit(this->rhs_vector,
                                                         this->solution[0],
                                                         this->get_time(),
                                                         &velocity_np);

#ifdef EXADG_PROFILE
  SectionProfile::lap("scal 1 rhs + Explicit turbulence terms", t_prof_rhs);
#endif

  // time t_{n+1}, mass scaling gamma_0/dt and velocity of the convection-diffusion operator
  this->pde_operator->update_conv_diff_operator(this->get_next_time(),
                                                scaling_factor_mass,
                                                &velocity_np);

  BlockVectorType residual, increment, solution_trial;
  this->pde_operator->initialize_dof_vector(residual);
  this->pde_operator->initialize_dof_vector(increment);
  if(param.newton_damping)
    this->pde_operator->initialize_dof_vector(solution_trial);

  // Q^0 = Q^n
  this->solution_np = this->solution[0];

#ifdef EXADG_DEBUG
  print_residual_decomposition = this->param.newton_diagnostics;
#endif

  evaluate_residual(residual, this->solution_np, this->rhs_vector, velocity_np);

  double       residual_norm         = residual.l2_norm();
  double const residual_norm_initial = residual_norm;

#ifdef EXADG_DEBUG
  if(this->param.newton_diagnostics)
    print_newton_debug(0, residual);

  // Jacobian against the finite difference of the residual (first three time steps)
  if(this->param.newton_diagnostics and n_jacobian_checks < 3)
  {
    ++n_jacobian_checks;
    check_jacobian(residual, this->rhs_vector, velocity_np);
  }
#endif

  auto const converged = [&](double const norm) {
    return norm <= newton_data.abs_tol or
           (residual_norm_initial > 0.0 and norm / residual_norm_initial <= newton_data.rel_tol);
  };

  // Eq. 43: update the preconditioner in the first time step or after
  // preconditioner_update_max_time_steps time steps
  ++time_steps_since_preconditioner_update;
  bool update_preconditioner =
    param.update_preconditioner and
    (not preconditioner_updated or
     time_steps_since_preconditioner_update > param.preconditioner_update_max_time_steps);

  unsigned int n_iter_newton = 0, n_iter_linear = 0;
  double       increment_norm_previous = 0.0;

  while(not converged(residual_norm) and n_iter_newton < newton_data.max_iter)
  {
    // J(Q^j) dQ = r(Q^j), linearization point set by evaluate_residual()
    increment = 0.0;

#ifdef EXADG_PROFILE
    auto t_prof_solve = SectionProfile::Clock::now();
#endif

    unsigned int const n_iter_linear_this = this->pde_operator->solve(increment,
                                                                     residual,
                                                                     update_preconditioner,
                                                                     scaling_factor_mass,
                                                                     this->get_next_time(),
                                                                     &velocity_np);
    n_iter_linear += n_iter_linear_this;

#ifdef EXADG_PROFILE
    SectionProfile::lap("scal 3 linear solve (total)", t_prof_solve);
#endif

    if(update_preconditioner)
    {
      preconditioner_updated                 = true;
      time_steps_since_preconditioner_update = 0;
    }

    double step_length = 1.0;

    if(param.newton_damping)
    {
      // line search: halve the step until the residual norm decreases sufficiently
      double const       tau               = 0.5;
      unsigned int const max_iter_damping  = 10;
      unsigned int       n_iter_damping    = 0;
      double             residual_norm_new = 0.0;

      while(true)
      {
        solution_trial = this->solution_np;
        solution_trial.add(step_length, increment);
        this->pde_operator->enforce_positivity(solution_trial);

        evaluate_residual(residual, solution_trial, this->rhs_vector, velocity_np);
        residual_norm_new = residual.l2_norm();

        ++n_iter_damping;
        if(residual_norm_new < (1.0 - tau * step_length) * residual_norm or
           n_iter_damping >= max_iter_damping)
          break;

        step_length *= 0.5;
      }

      AssertThrow(residual_norm_new < (1.0 - tau * step_length) * residual_norm,
                  dealii::ExcMessage("Damped Newton iteration did not converge. "
                                     "Maximum number of damping iterations exceeded."));

      this->solution_np = solution_trial;
      residual_norm     = residual_norm_new;
    }
    else
    {
      // Q^{j+1} = Q^j + dQ
      this->solution_np.add(1.0, increment);
      this->pde_operator->enforce_positivity(this->solution_np);

      evaluate_residual(residual, this->solution_np, this->rhs_vector, velocity_np);
      residual_norm = residual.l2_norm();
    }

    ++n_iter_newton;

#ifdef EXADG_DEBUG
    if(this->param.newton_diagnostics)
      print_newton_debug(n_iter_newton, residual, step_length, n_iter_linear_this);

    // Newton slow (6th iteration of a time step): Jacobian check at the current iterate, fixed
    // direction and Newton increment, each with two eps. A finite difference that changes much
    // with eps points to a non-smooth term (limiter, clipping); a difference independent of eps
    // to a missing or wrong Jacobian term.
    if(this->param.newton_diagnostics and n_iter_newton == 6)
    {
      this->pcout << std::endl;
      print_residual_location(residual);
      this->pcout << "[Jacobian check at slow Newton] time step " << this->get_time_step_number()
                  << ", iteration " << n_iter_newton << std::endl;
      for(double const eps_scale : {1.e-6, 1.e-8})
      {
        this->pcout << "  fixed direction, eps_scale " << eps_scale << ":";
        check_jacobian(residual, this->rhs_vector, velocity_np, nullptr, eps_scale);
        this->pcout << "  Newton increment, eps_scale " << eps_scale << ":";
        check_jacobian(residual, this->rhs_vector, velocity_np, &increment, eps_scale);
      }
    }
#endif

    // Eq. 43: update the preconditioner for the next Newton iteration if the convergence rate is
    // poor, ||dQ^{j+1}|| / ||dQ^j|| >= preconditioner_update_tolerance
    double const increment_norm = step_length * increment.l2_norm();

    update_preconditioner =
      param.update_preconditioner and n_iter_newton > 1 and
      increment_norm >= param.preconditioner_update_tolerance * increment_norm_previous;

    increment_norm_previous = increment_norm;
  }

  AssertThrow(converged(residual_norm),
              dealii::ExcMessage("Newton solver did not reach the tolerance within the maximum "
                                 "number of iterations."));

#ifdef EXADG_DEBUG
  // growth diagnostics: d = Q^{n+1} - Q^n (projected), per Jacobian part and block
  //   R = <d, J_part d> / <d, f M d>   (damping parts: R >= 0)
  if(this->param.newton_diagnostics and this->param.turbulence_model_data.is_active)
  {
    BlockVectorType d, mass_d, part_d;
    this->pde_operator->initialize_dof_vector(d);
    this->pde_operator->initialize_dof_vector(mass_d);
    this->pde_operator->initialize_dof_vector(part_d);
    d = this->solution_np;
    d.add(-1.0, this->solution[0]);
    this->pde_operator->apply_enrichment_projector(d);

    this->pde_operator->apply_mass_operator(mass_d, d);
    mass_d *= scaling_factor_mass;

    // per block: numerator <d_b, (J_part d)_b>; total: sum over blocks / <d, f M d>
    auto const print_ratio = [&](std::string const & name, BlockVectorType & jd) {
      this->pde_operator->apply_enrichment_projector(jd);
      double numerator_total = 0.0, denominator_total = 0.0;
      this->pcout << "    " << name << ": numerator per block";
      for(unsigned int b = 0; b < d.n_blocks(); ++b)
      {
        double const numerator = d.block(b) * jd.block(b);
        numerator_total += numerator;
        denominator_total += d.block(b) * mass_d.block(b);
        this->pcout << "  " << numerator;
      }
      this->pcout << ";  total R = "
                  << (std::abs(denominator_total) > 0.0 ? numerator_total / denominator_total : 0.0)
                  << std::endl;
    };

    this->pcout << "  [growth] <d_b, (f M d)_b> per block:";
    for(unsigned int b = 0; b < d.n_blocks(); ++b)
      this->pcout << "  " << d.block(b) * mass_d.block(b);
    this->pcout << std::endl;

    this->pcout << "  [growth] time step " << this->get_time_step_number()
                << ", R = <d, J_part d> / <d, f M d> per block, d = Q^{n+1} - Q^n, |d| per block:";
    for(unsigned int b = 0; b < d.n_blocks(); ++b)
      this->pcout << "  " << d.block(b).l2_norm();
    this->pcout << std::endl;

    // A: conv-diff operator without the mass (masked molecular diffusion)
    this->pde_operator->apply_conv_diff_operator(part_d, d);
    part_d.add(-1.0, mass_d);
    print_ratio("A (conv-diff)", part_d);

    std::array<std::string, 5> const names = {
      {"production", "dissipation", "square gradient", "cross diffusion", "diffusion"}};
    for(unsigned int p = 0; p < names.size(); ++p)
    {
      part_d = 0.0;
      this->pde_operator->apply_add_turbulence_jacobian_part(part_d, d, p);
      print_ratio(names[p], part_d);
    }

    // dissipation split by columns: increment of k only (block 0) / of omega only (blocks >= 1)
    BlockVectorType d_column(d);
    d_column = 0.0;
    d_column.block(0) = d.block(0);
    part_d = 0.0;
    this->pde_operator->apply_add_turbulence_jacobian_part(part_d, d_column, 1);
    print_ratio("dissipation, k column", part_d);
    d_column = d;
    d_column.block(0) = 0.0;
    part_d = 0.0;
    this->pde_operator->apply_add_turbulence_jacobian_part(part_d, d_column, 1);
    print_ratio("dissipation, omega columns", part_d);

    // all turbulence parts together (check: sum of the parts)
    part_d = 0.0;
    this->pde_operator->vmult(part_d, d);
    part_d.add(-1.0, mass_d);
    print_ratio("J - f M (total)", part_d);
  }

  // term balance of the enriched wall cells at the x stations, every 10th time step (printed by the
  // next evaluation of the turbulence terms, i.e. at the converged state of this step)
  if(this->param.newton_diagnostics and not this->param.newton_diagnostics_x_stations.empty() and
     this->get_time_step_number() % 10 == 0)
    this->pde_operator->request_omega_balance(this->param.newton_diagnostics_x_stations,
                                              this->get_time_step_number());

  // wall layer of the enriched variables: first 10 time steps, then every 20th
  if(this->param.newton_diagnostics and
     (this->get_time_step_number() <= 10 or this->get_time_step_number() % 20 == 0))
    this->pde_operator->print_enrichment_wall_diagnostics(this->solution_np,
                                                          this->get_time_step_number());
#endif

  this->iterations.first += 1;
  this->iterations.second += n_iter_linear;

  newton_iterations.first += 1;
  newton_iterations.second += n_iter_newton;

#ifdef EXADG_PROFILE
  auto t_prof_explicit = SectionProfile::Clock::now();
#endif

  this->evaluate_explicit_terms_at_new_time(velocity_np);

#ifdef EXADG_PROFILE
  SectionProfile::lap("scal 4 explicit terms at t_{n+1}", t_prof_explicit);
#endif

  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Solve convection-diffusion equation (Newton):";
    print_solver_info_nonlinear(this->pcout, n_iter_newton, n_iter_linear, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Solve"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFNewton<dim, Number>::print_iterations() const
{
  std::vector<std::string> names = {"Newton iterations",
                                    "Linear iterations (accumulated)",
                                    "Linear iterations (per Newton iteration)"};

  double const n_time_steps = std::max(1., (double)newton_iterations.first);
  double const n_newton     = std::max(1., (double)newton_iterations.second);

  std::vector<double> iterations_avg = {(double)newton_iterations.second / n_time_steps,
                                        (double)this->iterations.second / n_time_steps,
                                        (double)this->iterations.second / n_newton};

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

template class TimeIntBDFNewton<2, float>;
template class TimeIntBDFNewton<2, double>;

template class TimeIntBDFNewton<3, float>;
template class TimeIntBDFNewton<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
