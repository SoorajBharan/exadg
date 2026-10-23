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

#include <exadg/convection_diffusion/spatial_discretization/operator.h>
#include <exadg/convection_diffusion/time_integration/time_int_bdf_newton.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/utilities/print_solver_results.h>

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
  // (gamma_0/dt M + A) solution, homogeneous operator (mass scaling set by
  // update_conv_diff_operator())
  this->pde_operator->apply_conv_diff_operator(residual, solution);

  // constant_part - (gamma_0/dt M + A) solution
  residual.sadd(-1.0, 1.0, constant_part);

  // + Newton_Explicit part of the turbulence terms treated Implicit, at solution
  if(this->param.turbulence_model_data.is_active)
    this->pde_operator->evaluate_add_turbulence_newton_explicit(residual,
                                                                solution,
                                                                this->get_next_time(),
                                                                &velocity_np);
}

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

  // constant part of the right-hand side: b (no turbulence) + Explicit turbulence terms at Q^n
  this->assemble_right_hand_side(velocity_np);

  if(param.turbulence_model_data.is_active)
    this->pde_operator->evaluate_add_turbulence_explicit(this->rhs_vector,
                                                         this->solution[0],
                                                         this->get_time(),
                                                         &velocity_np);

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

  evaluate_residual(residual, this->solution_np, this->rhs_vector, velocity_np);

  double       residual_norm         = residual.l2_norm();
  double const residual_norm_initial = residual_norm;

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

    n_iter_linear += this->pde_operator->solve(increment,
                                               residual,
                                               update_preconditioner,
                                               scaling_factor_mass,
                                               this->get_next_time(),
                                               &velocity_np);

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

  this->iterations.first += 1;
  this->iterations.second += n_iter_linear;

  newton_iterations.first += 1;
  newton_iterations.second += n_iter_newton;

  // explicit convective and source terms at t_{n+1} for the next time steps
  this->evaluate_explicit_terms_at_new_time(velocity_np);

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
