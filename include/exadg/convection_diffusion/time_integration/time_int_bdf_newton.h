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

#ifndef INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_NEWTON_H_
#define INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_NEWTON_H_

// ExaDG
#include <exadg/convection_diffusion/time_integration/time_int_bdf.h>

namespace ExaDG
{
namespace ConvDiff
{
/*
 * BDF time integration with a Newton solver for the nonlinear system of each time step
 * (Bassi et al. 2014, Eqs. 42-43). The unknown of each Newton iteration j is the increment dQ:
 *
 *   J(Q^j) dQ = r(Q^j),   Q^{j+1} = Q^j + dQ,   Q^0 = Q^n
 *
 *   r(Q) = b - (gamma_0/dt M + A) Q + Explicit(Q^n) + Newton_Explicit(Q)
 *   J(Q) = gamma_0/dt M + A + Newton_Implicit(Q)
 *
 * with A the convection-diffusion operator (homogeneous part), b the constant right-hand side of
 * the time step (f, inhomogeneous boundary terms, explicit convective and source terms, BDF
 * history terms), Explicit the turbulence terms treated Explicit and Newton_Explicit /
 * Newton_Implicit the two parts of the turbulence terms treated Implicit. r = -G is the negative
 * residual of the nonlinear system G(Q) = 0.
 *
 * Convergence: ||r|| <= abs_tol or ||r|| / ||r(Q^0)|| <= rel_tol.
 *
 * Preconditioner (Eq. 43, only if update_preconditioner is true): updated in the first time step,
 * if ||dQ^{j+1}|| / ||dQ^j|| >= preconditioner_update_tolerance, or if
 * preconditioner_update_max_time_steps time steps have passed since the last update.
 */
template<int dim, typename Number>
class TimeIntBDFNewton : public TimeIntBDF<dim, Number>
{
public:
  using VectorType      = typename TimeIntBDF<dim, Number>::VectorType;
  using BlockVectorType = typename TimeIntBDF<dim, Number>::BlockVectorType;

  TimeIntBDFNewton(std::shared_ptr<Operator<dim, Number>>          operator_in,
                   std::shared_ptr<HelpersALE<dim, Number> const>  helpers_ale_in,
                   std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
                   Parameters const &                              param_in,
                   MPI_Comm const &                                mpi_comm_in,
                   bool const                                      is_test_in);

  void
  print_iterations() const final;

private:
  void
  do_timestep_solve() final;

  /*
   * residual = r(solution) = constant_part - (gamma_0/dt M + A) solution + Newton_Explicit(solution)
   *
   * constant_part = b + Explicit(Q^n). The turbulence call sets solution as the linearization point
   * of J (vmult of the operator).
   */
  void
  evaluate_residual(BlockVectorType &       residual,
                    BlockVectorType const & solution,
                    BlockVectorType const & constant_part,
                    VectorType const &      velocity_np) const;

  // number of time steps since the last preconditioner update (Eq. 43)
  unsigned int time_steps_since_preconditioner_update;

  // false until the preconditioner has been updated once
  bool preconditioner_updated;

  // Newton iteration counts: (number of time steps, accumulated Newton iterations)
  std::pair<unsigned int, unsigned long long> newton_iterations;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_NEWTON_H_ */
