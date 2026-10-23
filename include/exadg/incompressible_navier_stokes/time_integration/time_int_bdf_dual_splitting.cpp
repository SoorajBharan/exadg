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

#include <deal.II/numerics/vector_tools_mean_value.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/time_integration/push_back_vectors.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>
#include <exadg/utilities/section_profile.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
TimeIntBDFDualSplitting<dim, Number>::TimeIntBDFDualSplitting(
  std::shared_ptr<Operator>                       operator_in,
  std::shared_ptr<HelpersALE<dim, Number> const>  helpers_ale_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
  Parameters const &                              param_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in)
  : Base(operator_in, helpers_ale_in, postprocessor_in, param_in, mpi_comm_in, is_test_in),
    pde_operator(operator_in),
    velocity(this->order),
    pressure(this->order),
    velocity_dbc(this->order),
    iterations_pressure({0, 0}),
    iterations_projection({0, 0}),
    iterations_viscous({0, {0, 0}}),
    iterations_penalty({0, 0}),
    iterations_mass({0, 0}),
    extra_pressure_nbc(this->param.order_extrapolation_pressure_nbc,
                       this->param.start_with_low_order)
{
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  Base::update_time_integrator_constants();

  // update time integrator constants for extrapolation scheme of pressure Neumann bc
  extra_pressure_nbc.update(this->get_time_step_number(),
                            this->adaptive_time_stepping,
                            this->get_time_step_vector());

  // use this function to check the correctness of the time integrator constants
  //    std::cout << "Coefficients extrapolation scheme pressure NBC:" << std::endl;
  //    extra_pressure_nbc.print(this->pcout);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::setup_derived()
{
  Base::setup_derived();

  // velocity_dbc vectors do not have to be initialized in case of a restart, where
  // the vectors are read from restart files.
  if(this->param.restarted_simulation == false)
  {
    initialize_velocity_dbc();
  }

  // wall enrichment: initial convective history (U_tilde = 0 at the start -> standard form)
  if(this->param.wall_enrichment_enabled and this->param.convective_problem() and this->param.restarted_simulation == false)
  {
    compute_convective_history_entry(convective_history[0], velocity[0], this->get_time(), false);

    if(this->param.start_with_low_order == false)
      for(unsigned int i = 1; i < convective_history.size(); ++i)
        compute_convective_history_entry(convective_history[i],
                                         velocity[i],
                                         this->get_previous_time(i),
                                         false);
  }

#ifdef EXADG_PROFILE
  // load balance of the enriched work: locally owned cells per active_fe_index   on each rank
  if(this->param.wall_enrichment_enabled)
  {
    auto const & mf_en = *pde_operator->get_function_enrichment()->get_matrix_free();

    std::array<double, 3> n_cells = {{0.0, 0.0, 0.0}};
    for(unsigned int cell = 0; cell < mf_en.n_cell_batches(); ++cell)
    {
      unsigned int const fe_index = mf_en.get_cell_iterator(cell, 0)->active_fe_index();
      if(fe_index < 3)
        n_cells[fe_index] += mf_en.n_active_entries_per_cell_batch(cell);
    }

    std::array<std::string, 3> const names = {{"standard (fe 0)", "enriched (fe 1)", "halo (fe 2)"}};
    this->pcout << std::endl
      << "Locally owned cells per rank (min / avg / max, rank of max):" << std::endl;
    for(unsigned int i = 0; i < 3; ++i)
    {
      auto const r = dealii::Utilities::MPI::min_max_avg(n_cells[i], this->mpi_comm);
      this->pcout << "  " << names[i] << ": " << r.min << " / " << r.avg << " / " << r.max << ", rank " << r.max_index << std::endl;
    }
  }
#endif
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  Base::read_restart_vectors(ia);

  for(unsigned int i = 0; i < velocity_dbc.size(); i++)
  {
    ia >> velocity_dbc[i];
  }

  if(this->param.wall_enrichment_enabled)
  {
    // must be in sync with write_restart_vectors()
    for(unsigned int i = 0; i < velocity.size(); i++)
      ia >> velocity[i].block(1);

    if(this->param.convective_problem())
      for(unsigned int i = 0; i < convective_history.size(); i++)
      {
        ia >> convective_history[i].block(0);
        ia >> convective_history[i].block(1);
      }

    auto const & fe = pde_operator->get_function_enrichment();
    ia >> fe->friction_velocity;
    ia >> fe->wall_shear_stress;
    fe->friction_velocity.update_ghost_values();
    fe->wall_shear_stress.update_ghost_values();

    fe->enrichment_velocity = velocity[0].block(1);
    fe->enrichment_velocity.update_ghost_values();

    // psi of the space the restored solution is expressed in; called twice so that
    // psi_old == psi and the first project_enrichment_history() is the identity
    fe->update_enrichment_function();
    fe->update_enrichment_function();
  } 
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::write_restart_vectors(
  boost::archive::binary_oarchive & oa) const
{
  Base::write_restart_vectors(oa);

  for(unsigned int i = 0; i < velocity_dbc.size(); i++)
  {
    oa << velocity_dbc[i];
  }

  if(this->param.wall_enrichment_enabled)
  {
    // enriched state: U_tilde of all levels, convective history (both blocks) wall shear stress
    for(unsigned int i = 0; i < velocity.size(); i++)
      oa << velocity[i].block(1);

    if(this->param.convective_problem())
      for(unsigned int i = 0; i < convective_history.size(); i++)
      {
        oa << convective_history[i].block(0);
        oa << convective_history[i].block(1);
      } 

    auto const & fe = pde_operator->get_function_enrichment();
    oa << fe->friction_velocity;
    oa << fe->wall_shear_stress;
  } 
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::allocate_vectors()
{
  Base::allocate_vectors();

  // velocity
  // for(unsigned int i = 0; i < velocity.size(); ++i)
  //   pde_operator->initialize_vector_velocity(velocity[i]);
  // pde_operator->initialize_vector_velocity(velocity_np);
  auto const init_velocity = [&](BlockVectorType & v){
    v.reinit(2);
    pde_operator->initialize_vector_velocity(v.block(0));

    if(this->param.wall_enrichment_enabled)
    {
      pde_operator->initialize_vector_enrichment(v.block(1));
    }
    else {
      v.block(1).reinit(dealii::IndexSet(0), v.block(0).get_mpi_communicator());
    }

    v.collect_sizes();
  };

  for(unsigned int i = 0; i < velocity.size(); ++i)
    init_velocity(velocity[i]);
  init_velocity(velocity_np);

  // pressure
  for(unsigned int i = 0; i < pressure.size(); ++i)
    pde_operator->initialize_vector_pressure(pressure[i]);
  pde_operator->initialize_vector_pressure(pressure_np);

  // velocity_dbc
  for(unsigned int i = 0; i < velocity_dbc.size(); ++i)
    pde_operator->initialize_vector_velocity(velocity_dbc[i]);
  pde_operator->initialize_vector_velocity(velocity_dbc_np);

  // wall enrichment: convective term history in velocity space
  if(this->param.wall_enrichment_enabled and this->param.convective_problem())
  {
    convective_history.resize(this->vec_convective_term.size());
    for(unsigned int i = 0; i < convective_history.size(); ++i)
      init_velocity(convective_history[i]);
    init_velocity(convective_history_np);
  }
}


template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::initialize_current_solution()
{
  if(this->param.ale_formulation)
    this->helpers_ale->move_grid(this->get_time());

  // pde_operator->prescribe_initial_conditions(velocity[0], pressure[0], this->get_time());
  pde_operator->prescribe_initial_conditions(velocity[0].block(0), pressure[0], this->get_time());

  if(this->param.wall_enrichment_enabled)
    velocity[0].block(1) = 0.0;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::initialize_former_multistep_dof_vectors()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    if(this->param.ale_formulation)
      this->helpers_ale->move_grid(this->get_previous_time(i));

    // pde_operator->prescribe_initial_conditions(velocity[i],
    //                                            pressure[i],
    //                                            this->get_previous_time(i));
    pde_operator->prescribe_initial_conditions(velocity[i].block(0),
                                               pressure[i],
                                               this->get_previous_time(i));
    if(this->param.wall_enrichment_enabled)
      velocity[i].block(1) = 0.0;
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::initialize_velocity_dbc()
{
  // fill vector velocity_dbc: The first entry [0] is already needed if start_with_low_order == true
  if(this->param.ale_formulation)
  {
    this->helpers_ale->move_grid(this->get_time());
    this->helpers_ale->update_pde_operator_after_grid_motion();
  }
  pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc[0], this->get_time());
  // ... and previous times if start_with_low_order == false
  if(this->start_with_low_order == false)
  {
    for(unsigned int i = 1; i < velocity_dbc.size(); ++i)
    {
      double const time = this->get_time() - double(i) * this->get_time_step_size();
      if(this->param.ale_formulation)
      {
        this->helpers_ale->move_grid(time);
        this->helpers_ale->update_pde_operator_after_grid_motion();
      }
      pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc[i], time);
    }
  }
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
TimeIntBDFDualSplitting<dim, Number>::get_velocity() const
{
  return velocity[0].block(0);
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
TimeIntBDFDualSplitting<dim, Number>::get_velocity_np() const
{
  return velocity_np.block(0);
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
TimeIntBDFDualSplitting<dim, Number>::get_pressure() const
{
  return pressure[0];
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
TimeIntBDFDualSplitting<dim, Number>::get_pressure_np() const
{
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
TimeIntBDFDualSplitting<dim, Number>::get_velocity(unsigned int i) const
{
  return velocity[i].block(0);
}

template<int dim, typename Number>
typename TimeIntBDFDualSplitting<dim, Number>::VectorType const &
TimeIntBDFDualSplitting<dim, Number>::get_pressure(unsigned int i) const
{
  return pressure[i];
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::set_velocity(VectorType const & velocity_in,
                                                   unsigned int const i)
{
  velocity[i].block(0) = velocity_in;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::set_pressure(VectorType const & pressure_in,
                                                   unsigned int const i)
{
  pressure[i] = pressure_in;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::postprocessing_stability_analysis()
{
  AssertThrow(this->order == 1,
              dealii::ExcMessage("Order of BDF scheme has to be 1 for this stability analysis."));

  AssertThrow(this->param.convective_problem() == false,
              dealii::ExcMessage(
                "Stability analysis can not be performed for nonlinear convective problems."));

  AssertThrow(velocity[0].l2_norm() < 1.e-15 and pressure[0].l2_norm() < 1.e-15,
              dealii::ExcMessage("Solution vector has to be zero for this stability analysis."));

  AssertThrow(dealii::Utilities::MPI::n_mpi_processes(this->mpi_comm) == 1,
              dealii::ExcMessage("Number of MPI processes has to be 1."));

  std::cout << std::endl << "Analysis of eigenvalue spectrum:" << std::endl;

  unsigned int const size = velocity[0].block(0).locally_owned_size();

  dealii::LAPACKFullMatrix<Number> propagation_matrix(size, size);

  // loop over all columns of propagation matrix
  for(unsigned int j = 0; j < size; ++j)
  {
    // set j-th element to 1
    velocity[0].block(0).local_element(j) = 1.0;

    // solve time step
    this->do_timestep_solve();

    // dst-vector velocity_np is j-th column of propagation matrix
    for(unsigned int i = 0; i < size; ++i)
    {
      propagation_matrix(i, j) = velocity_np.block(0).local_element(i);
    }

    // reset j-th element to 0
    velocity[0].block(0).local_element(j) = 0.0;
  }

  // compute eigenvalues
  propagation_matrix.compute_eigenvalues();

  double norm_max = 0.0;

  std::cout << "List of all eigenvalues:" << std::endl;

  for(unsigned int i = 0; i < size; ++i)
  {
    double norm = std::abs(propagation_matrix.eigenvalue(i));
    if(norm > norm_max)
      norm_max = norm;

    // print eigenvalues
    std::cout << std::scientific << std::setprecision(5) << propagation_matrix.eigenvalue(i)
              << std::endl;
  }

  std::cout << std::endl << std::endl << "Maximum eigenvalue = " << norm_max << std::endl;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::do_timestep_solve()
{
#ifdef EXADG_PROFILE
  // one profile cycle: start of this do_timestep_solve() to the start of the next one, i.e.
  // including prepare_vectors_for_next_timestep(), CFL, restart, postprocessing   and output
  StepProfile & profile = step_profile();
  if(profile.cycle_running)
  {
    profile.add("full step", profile.cycle_timer);
    ++profile.n_steps;
    if(this->print_solver_info() and not(this->is_test))
      profile.print_and_reset(this->pcout, this->mpi_comm);
  }
  profile.start(profile.cycle_timer, this->mpi_comm);
  profile.cycle_running = true;

  dealii::Timer timer_profile;
#endif
  // pre-computations
  pde_operator->interpolate_velocity_dirichlet_bc(velocity_dbc_np, this->get_next_time());

  enrichment_coupling = this->param.wall_enrichment_enabled and
    (this->get_time_step_number() > 1 or
    this->param.restarted_simulation);
  // if(this->param.wall_enrichment_enabled and this->get_time() >= t_start_enrichment)
  // {
  //   double const y_plus_max = pde_operator->get_function_enrichment()->get_max_y_plus();
  //   enrichment_coupling       = (y_plus_max >= y_plus_min_enrichment);
  // }
  if(this->param.wall_enrichment_enabled and enrichment_coupling)
  {
#ifdef EXADG_PROFILE
    profile.start(timer_profile, this->mpi_comm);
#endif
    pde_operator->precompute_schur_matrices();

#ifdef EXADG_PROFILE
    profile.add("Schur precompute", timer_profile);
    profile.start(timer_profile, this->mpi_comm);
#endif

    for(unsigned int i = 0; i < velocity.size(); ++i)
      pde_operator->project_enrichment_history(velocity[i]);

    // the convective history lives in velocity space as well (w_i = M^{-1} c_i)
    for(unsigned int i = 0; i < convective_history.size(); ++i)
      pde_operator->project_enrichment_history(convective_history[i]);
#ifdef EXADG_PROFILE
    profile.add("history projection", timer_profile);
#endif
  }

  // perform the sub-steps of the dual-splitting method
#ifdef EXADG_PROFILE
  profile.start(timer_profile, this->mpi_comm);
#endif
  convective_step();
#ifdef EXADG_PROFILE
  profile.add("convective step", timer_profile);
#endif
  debug_norms("convective");

#ifdef EXADG_PROFILE
  profile.start(timer_profile, this->mpi_comm);
#endif
  pressure_step();
#ifdef EXADG_PROFILE
  profile.add("pressure step", timer_profile);
  profile.start(timer_profile, this->mpi_comm);
#endif

  if(this->param.wall_enrichment_enabled and enrichment_coupling)
    projection_step_enriched();
  else
    projection_step();
#ifdef EXADG_PROFILE
  profile.add("projection step", timer_profile);
#endif
  debug_norms("projection");

#ifdef EXADG_PROFILE
  profile.start(timer_profile, this->mpi_comm);
#endif
  if(this->param.wall_enrichment_enabled and enrichment_coupling)
    viscous_step_enriched();
  else
    viscous_step();
#ifdef EXADG_PROFILE
  profile.add("viscous step (incl. nu)", timer_profile);
#endif
  debug_norms("viscous");

  if(this->param.apply_penalty_terms_in_postprocessing_step)
  {
#ifdef EXADG_PROFILE
    profile.start(timer_profile, this->mpi_comm);
#endif
    if(this->param.wall_enrichment_enabled and enrichment_coupling)
      penalty_step_enriched();
    else
      penalty_step();
#ifdef EXADG_PROFILE
    profile.add("penalty step", timer_profile);
#endif
  }
  debug_norms("penalty");

#ifdef EXADG_PROFILE
  profile.start(timer_profile, this->mpi_comm);
#endif
  if(this->param.wall_enrichment_enabled)
    pde_operator->compute_total_velocity(this->velocity_total, velocity_np);
#ifdef EXADG_PROFILE
  profile.add("total velocity", timer_profile);
  profile.start(timer_profile, this->mpi_comm);
#endif

  // evaluate convective term once the final solution at time
  // t_{n+1} is known
  evaluate_convective_term();
#ifdef EXADG_PROFILE
  profile.add("convective term (end)", timer_profile);
#endif
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::convective_step()
{
  dealii::Timer timer;
  timer.restart();

  velocity_np = 0.0;

  unsigned int n_iter_mass = 0;

  if(this->param.wall_enrichment_enabled)
  {
    AssertThrow(not this->param.ale_formulation,
                dealii::ExcMessage("Enriched convective step: ALE not implemented."));

    // history is stored as M^{-1} c (velocity space): no inverse mass here
    if(this->param.convective_problem())
      for(unsigned int i = 0; i < convective_history.size(); ++i)
        velocity_np.add(-this->extra.get_beta(i), convective_history[i]);

    if(this->param.right_hand_side)
    {
      BlockVectorType body_force(velocity_np), inverse_mass_body_force(velocity_np);
      body_force = 0.0;

      if(enrichment_coupling)
      {
        // Krank et al. (2018), Eq. (37): (v_h, f) with v_h in the enriched space
        pde_operator->evaluate_add_body_force_term_enriched(body_force, this->get_next_time());
        pde_operator->apply_inverse_mass_operator(inverse_mass_body_force, body_force);
      }
      else
    {
        // enrichment not coupled in this step: polynomial space, U_tilde = 0
        pde_operator->evaluate_add_body_force_term(body_force.block(0), this->get_next_time());
        n_iter_mass = pde_operator->apply_inverse_mass_operator(inverse_mass_body_force.block(0),
                                                                body_force.block(0));
        inverse_mass_body_force.block(1) = 0.0;                 
      }

      velocity_np.add(1.0, inverse_mass_body_force);
    }
  }
  else
{
    // compute convective term and extrapolate convective term (if not Stokes equations)
    if(this->param.convective_problem())
    {
      if(this->param.ale_formulation)
      {
        for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
        {
          // in a general setting, we only know the boundary conditions at time t_{n+1}
          pde_operator->evaluate_convective_term(this->vec_convective_term[i],
                                                 velocity[i].block(0),
                                                 this->get_next_time());
        }
      }

      for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
        velocity_np.block(0).add(-this->extra.get_beta(i), this->vec_convective_term[i]);
    }

    // compute body force vector
    if(this->param.right_hand_side == true)
    {
      pde_operator->evaluate_add_body_force_term(velocity_np.block(0), this->get_next_time());
    }

    // apply inverse mass operator
    n_iter_mass =
      pde_operator->apply_inverse_mass_operator(velocity_np.block(0), velocity_np.block(0));
  }
  iterations_mass.first += 1;
  iterations_mass.second += n_iter_mass;

  // calculate sum (alpha_i/dt * u_i) and add to velocity_np
  for(unsigned int i = 0; i < velocity.size(); ++i)
  {
    velocity_np.add(this->bdf.get_alpha(i) / this->get_time_step_size(), velocity[i]);
  }

  // solve discrete temporal derivative term for intermediate velocity u_hat
  velocity_np *= this->get_time_step_size() / this->bdf.get_gamma0();

  if(this->print_solver_info() and not(this->is_test))
  {
    if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
    {
      this->pcout << std::endl << "Convective step:";
      print_solver_info_linear(this->pcout, n_iter_mass, timer.wall_time());
    }
    else if(this->param.spatial_discretization == SpatialDiscretization::L2)
    {
      this->pcout << std::endl << "Explicit convective step:";
      print_wall_time(this->pcout, timer.wall_time());
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  this->timer_tree->insert({"Timeloop", "Convective step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::evaluate_convective_term()
{
  dealii::Timer timer;
  timer.restart();

  if(this->param.convective_problem())
  {
    if(this->param.ale_formulation == false) // Eulerian case
    {
      if(this->param.wall_enrichment_enabled)
      {
        // w_np = M^{-1} c(u_h); enriched form once the coupling is active in this step
        compute_convective_history_entry(convective_history_np,
                                         velocity_np,
                                         this->get_next_time(),
                                         enrichment_coupling);
#ifdef EXADG_DEBUG
        /*************************************DEBUG*********************************/
        this->pcout << "  [convective debug] |M^{-1} c(u_h)|: U_bar = "
          << convective_history_np.block(0).l2_norm()
          << "   U_tilde = " <<
          convective_history_np.block(1).l2_norm() << std::endl;

        /****************************************************************************/
#endif
      }
      else
      {
        pde_operator->evaluate_convective_term(this->convective_term_np,
                                               velocity_np.block(0),
                                               this->get_next_time());
      }
    }
  }

  this->timer_tree->insert({"Timeloop", "Convective step"}, timer.wall_time());
}


template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim,
Number>::compute_convective_history_entry(BlockVectorType &       w,

                                          BlockVectorType const & u,
                                          double
                                          const evaluation_time,
                                          bool const
                                          enriched) const
{ 
  BlockVectorType c(w);
  if(enriched)
  {
    pde_operator->evaluate_convective_term_enriched(c, u, evaluation_time);
    pde_operator->apply_inverse_mass_operator(w, c);
  }
  else
{ 
    pde_operator->evaluate_convective_term(c.block(0), u.block(0),
                                           evaluation_time);
    pde_operator->apply_inverse_mass_operator(w.block(0), c.block(0));
    w.block(1) = 0.0;
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::pressure_step()
{
  dealii::Timer timer;
  timer.restart();

  // compute right-hand-side vector
  VectorType rhs(pressure_np);
  rhs_pressure(rhs);

  // extrapolate old solution to get a good initial estimate for the solver
  if(this->use_extrapolation)
  {
    pressure_np = 0;
    for(unsigned int i = 0; i < pressure.size(); ++i)
    {
      pressure_np.add(this->extra.get_beta(i), pressure[i]);
    }
  }
  else
  {
    pressure_np = pressure_last_iter;
  }

  // solve linear system of equations
  bool const update_preconditioner =
    this->param.update_preconditioner_pressure_poisson and
    ((this->time_step_number - 1) %
       this->param.update_preconditioner_pressure_poisson_every_time_steps ==
     0);

  unsigned int const n_iter = pde_operator->solve_pressure(pressure_np, rhs, update_preconditioner);
  iterations_pressure.first += 1;
  iterations_pressure.second += n_iter;

  // special case: pressure level is undefined
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  pde_operator->adjust_pressure_level_if_undefined(pressure_np, this->get_next_time());

  if(this->store_solution)
    pressure_last_iter = pressure_np;

  // write output
  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Solve pressure step:";
    print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Pressure step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::rhs_pressure(VectorType & rhs) const
{
  /*
   *  I. calculate divergence term
   */
  // homogeneous part of velocity divergence operator
  if(this->param.wall_enrichment_enabled)
    pde_operator->apply_velocity_divergence_term(rhs, velocity_np); // u_h = U_bar + psi*U_tilde
  else
    pde_operator->apply_velocity_divergence_term(rhs, velocity_np.block(0));

  rhs *= -this->bdf.get_gamma0() / this->get_time_step_size();

  // inhomogeneous parts of boundary face integrals of velocity divergence operator
  if(this->param.divu_integrated_by_parts == true and this->param.divu_use_boundary_data == true)
  {
    VectorType temp(rhs);

    // sum alpha_i * u_i term
    for(unsigned int i = 0; i < velocity.size(); ++i)
    {
      pde_operator->rhs_velocity_divergence_term_dirichlet_bc_from_dof_vector(temp,
                                                                              velocity_dbc[i]);

      // note that the minus sign related to this term is already taken into account
      // in the function rhs() of the divergence operator
      rhs.add(this->bdf.get_alpha(i) / this->get_time_step_size(), temp);
    }

    // convective term
    if(this->param.convective_problem())
    {
      for(unsigned int i = 0; i < velocity.size(); ++i)
      {
        temp = 0.0;


        if(this->param.wall_enrichment_enabled and enrichment_coupling)
        {
          // Krank et al. (2018), Eq. (41): u_h = U_bar + psi U_tilde on the boundary faces
          pde_operator->rhs_ppe_div_term_convective_term_add(temp, velocity[i]);
        } 
        else
      {
          pde_operator->rhs_ppe_div_term_convective_term_add(temp,
                                                             velocity[i].block(0));
        }

        rhs.add(this->extra.get_beta(i), temp);
      }
    }

    // body force term
    if(this->param.right_hand_side)
      pde_operator->rhs_ppe_div_term_body_forces_add(rhs, this->get_next_time());
  }

  /*
   *  II. calculate terms originating from inhomogeneous parts of boundary face integrals of Laplace
   * operator
   */

  // II.1. pressure Dirichlet boundary conditions
  pde_operator->rhs_ppe_laplace_add(rhs, this->get_next_time());

  // II.2. pressure Neumann boundary condition: body force vector
  if(this->param.right_hand_side)
  {
    pde_operator->rhs_ppe_nbc_body_force_term_add(rhs, this->get_next_time());
  }

  // II.3. pressure Neumann boundary condition: temporal derivative of velocity
  VectorType acceleration(velocity_dbc_np);
  compute_bdf_time_derivative(acceleration, velocity_dbc_np, velocity_dbc, this->bdf, this->get_time_step_size());
  pde_operator->rhs_ppe_nbc_numerical_time_derivative_add(rhs, acceleration);

  // II.4. viscous term of pressure Neumann boundary condition on Gamma_D:
  //       extrapolate velocity, evaluate vorticity, and subsequently evaluate boundary
  //       face integral (this is possible since pressure Neumann BC is linear in vorticity)

  if(this->param.viscous_problem())
  {
    if(this->param.order_extrapolation_pressure_nbc > 0)
    {
      if(this->param.wall_enrichment_enabled and enrichment_coupling)
      {
        // Krank et al. (2018), Eq. (41); thesis Eq. (3.16): omega_h in the         enriched space.
        // All velocity[i] are already projected onto the current psi           (do_timestep_solve).
        BlockVectorType velocity_extra(velocity[0]);
        velocity_extra = 0.0;
        for(unsigned int i = 0; i < extra_pressure_nbc.get_order(); ++i)
          velocity_extra.add(this->extra_pressure_nbc.get_beta(i), velocity[i]);

        BlockVectorType vorticity(velocity_extra);
        pde_operator->compute_vorticity(vorticity, velocity_extra);

        pde_operator->rhs_ppe_nbc_viscous_add(rhs, vorticity);
      }
      else
    {
        VectorType velocity_extra(velocity[0].block(0));
        velocity_extra = 0.0;
        for(unsigned int i = 0; i < extra_pressure_nbc.get_order(); ++i)
        {
          velocity_extra.add(this->extra_pressure_nbc.get_beta(i),
                             velocity[i].block(0));
        }

        VectorType vorticity(velocity_extra);
        pde_operator->compute_vorticity(vorticity, velocity_extra);

        pde_operator->rhs_ppe_nbc_viscous_add(rhs, vorticity);
      }
    }
  }

  // II.5. convective term of pressure Neumann boundary condition on Gamma_D:
  //       evaluate convective term and subsequently extrapolate rhs vectors
  //       (the convective term is nonlinear!)
  if(this->param.convective_problem())
  {
    if(this->param.order_extrapolation_pressure_nbc > 0)
    {
      VectorType temp(rhs);
      for(unsigned int i = 0; i < extra_pressure_nbc.get_order(); ++i)
      {
        temp = 0.0;


        if(this->param.wall_enrichment_enabled and enrichment_coupling)
        {
          // Krank et al. (2018), Eq. (41): u_h = U_bar + psi U_tilde on the boundary faces
          pde_operator->rhs_ppe_nbc_convective_add(temp, velocity[i]);
        } 
        else
      {
          pde_operator->rhs_ppe_nbc_convective_add(temp, velocity[i].block(0));
        }
        rhs.add(this->extra_pressure_nbc.get_beta(i), temp);
      }
    }
  }

  // special case: pressure level is undefined
  // Set mean value of rhs to zero in order to obtain a consistent linear system of equations.
  // This is really necessary for the dual-splitting scheme in contrast to the pressure-correction
  // scheme and coupled solution approach due to the Dirichlet BC prescribed for the intermediate
  // velocity field and the pressure Neumann BC in case of the dual-splitting scheme.
  if(pde_operator->is_pressure_level_undefined())
    dealii::VectorTools::subtract_mean_value(rhs);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::projection_step()
{
  dealii::Timer timer;
  timer.restart();

  // compute right-hand-side vector
  VectorType rhs(velocity_np.block(0));
  rhs_projection(rhs);

  // apply inverse mass operator: this is the solution if no penalty terms are applied
  // and serves as a good initial guess for the case with penalty terms
  unsigned int const n_iter_mass = pde_operator->apply_inverse_mass_operator(velocity_np.block(0), rhs);
  iterations_mass.first += 1;
  iterations_mass.second += n_iter_mass;

  // penalty terms
  if(this->param.apply_penalty_terms_in_postprocessing_step == false and
     (this->param.use_divergence_penalty == true or this->param.use_continuity_penalty == true))
  {
    // extrapolate velocity to time t_n+1 and use this velocity field to
    // calculate the penalty parameter for the divergence and continuity penalty term
    VectorType velocity_extrapolated;
    if(this->use_extrapolation)
    {
      velocity_extrapolated.reinit(velocity[0].block(0));
      for(unsigned int i = 0; i < velocity.size(); ++i)
        velocity_extrapolated.add(this->extra.get_beta(i), velocity[i].block(0));
    }
    else
    {
      velocity_extrapolated = velocity_projection_last_iter;
    }

    pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_projection and
      ((this->time_step_number - 1) %
         this->param.update_preconditioner_projection_every_time_steps ==
       0);

    if(this->use_extrapolation == false)
      velocity_np.block(0) = velocity_projection_last_iter;

    unsigned int n_iter = pde_operator->solve_projection(velocity_np.block(0), rhs, update_preconditioner);
    iterations_projection.first += 1;
    iterations_projection.second += n_iter;

    if(this->store_solution)
      velocity_projection_last_iter = velocity_np.block(0);

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve projection step:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }
  }
  else // no penalty terms
  {
    if(this->print_solver_info() and not(this->is_test))
    {
      if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
      {
        this->pcout << std::endl << "Projection step:";
        print_solver_info_linear(this->pcout, n_iter_mass, timer.wall_time());
      }
      else if(this->param.spatial_discretization == SpatialDiscretization::L2)
      {
        this->pcout << std::endl << "Explicit projection step:";
        print_wall_time(this->pcout, timer.wall_time());
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }
    }
  }

  this->timer_tree->insert({"Timeloop", "Pojection step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::rhs_projection(VectorType & rhs) const
{
  /*
   *  I. calculate pressure gradient term
   */
  pde_operator->evaluate_pressure_gradient_term(rhs, pressure_np, this->get_next_time());

  rhs *= -this->get_time_step_size() / this->bdf.get_gamma0();

  /*
   *  II. add mass operator term
   */
  pde_operator->apply_mass_operator_add(rhs, velocity_np.block(0));
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::projection_step_enriched()
{
  dealii::Timer timer;
  timer.restart();

  AssertThrow(this->param.apply_penalty_terms_in_postprocessing_step or
                (not this->param.use_divergence_penalty and not this->param.use_continuity_penalty),
              dealii::ExcMessage("projection_step_enriched: penalty terms inside the projection "
                                 "step are not supported (use apply_penalty_terms_in_postprocessing_step)."));

  // rhs = M u_hat - dt/gamma0 * G p     (both blocks)
  BlockVectorType rhs(velocity_np);
  pde_operator->evaluate_pressure_gradient_term_enriched(rhs, pressure_np, this->get_next_time());
  rhs *= -this->get_time_step_size() / this->bdf.get_gamma0();

  BlockVectorType mass_u(velocity_np);
  pde_operator->apply_mass_operator(mass_u, velocity_np);
  rhs.add(1.0, mass_u);

  // u = M^{-1} rhs   (block inverse mass: Schur complement on enriched cells)
  pde_operator->apply_inverse_mass_operator(velocity_np, rhs);

  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Explicit projection step (enriched):";
    print_wall_time(this->pcout, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Pojection step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::viscous_step()
{
  dealii::Timer timer;
  timer.restart();

  // in case we need to iteratively solve a linear or nonlinear system of equations
  if(this->param.viscous_problem() or this->param.implicit_convective_problem())
  {
    // store the velocity vector that is needed to compute the rhs vector
    VectorType velocity_rhs = velocity_np.block(0);

    // Extrapolate old solution to get a good initial estimate for the solver.
    if(this->use_extrapolation)
    {
      velocity_np = 0;
      for(unsigned int i = 0; i < velocity.size(); ++i)
        velocity_np.add(this->extra.get_beta(i), velocity[i]);
    }
    else
    {
      velocity_np.block(0) = velocity_viscous_last_iter;
    }

    /*
     *  update variable viscosity
     */
    if(this->param.viscous_problem() and this->param.viscosity_is_variable() and
       this->param.treatment_of_variable_viscosity == TreatmentOfVariableViscosity::Explicit)
    {
      dealii::Timer timer_viscosity_update;
      timer_viscosity_update.restart();

      pde_operator->update_viscosity(velocity_np.block(0));

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Update of variable viscosity:";
        print_wall_time(this->pcout, timer_viscosity_update.wall_time());
      }
    }

    bool const update_preconditioner =
      this->param.update_preconditioner_momentum and
      ((this->time_step_number - 1) % this->param.update_preconditioner_momentum_every_time_steps ==
       0);

    if(this->param.nonlinear_problem_has_to_be_solved())
    {
      /*
       *  Calculate the vector that is constant when solving the nonlinear momentum equation
       *  (where constant means that the vector does not change from one Newton iteration
       *  to the next, i.e., it does not depend on the current solution of the nonlinear solver)
       */
      VectorType rhs(velocity_rhs);
      rhs_viscous(rhs, velocity_rhs);

      // solve non-linear system of equations
      auto const iter = pde_operator->solve_nonlinear_momentum_equation(
        velocity_np.block(0),
        rhs,
        this->get_next_time(),
        update_preconditioner,
        this->get_scaling_factor_time_derivative_term());

      iterations_viscous.first += 1;
      std::get<0>(iterations_viscous.second) += std::get<0>(iter);
      std::get<1>(iterations_viscous.second) += std::get<1>(iter);

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Solve momentum step:";
        print_solver_info_nonlinear(this->pcout,
                                    std::get<0>(iter),
                                    std::get<1>(iter),
                                    timer.wall_time());
      }
    }
    else // linear problem
    {
      AssertThrow(this->param.viscous_problem(), dealii::ExcMessage("Should not arrive here."));

      /*
       *  Calculate the right-hand side of the linear system of equations.
       */
      VectorType rhs(velocity_rhs);
      rhs_viscous(rhs, velocity_rhs);

      // solve linear system of equations
      unsigned int const n_iter = pde_operator->solve_linear_momentum_equation(
        velocity_np.block(0), rhs, update_preconditioner, this->get_scaling_factor_time_derivative_term());

      iterations_viscous.first += 1;
      std::get<1>(iterations_viscous.second) += n_iter;

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Solve viscous step:";
        print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
      }
    }

    if(this->store_solution)
      velocity_viscous_last_iter = velocity_np.block(0);
  }
  else // no viscous term and no implicit convective term, i.e. there is nothing to do in this step
       // of the dual splitting scheme
  {
    // nothing to do
    AssertThrow(this->param.equation_type == EquationType::Euler and
                  this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit,
                dealii::ExcMessage("Logical error."));
  }

  this->timer_tree->insert({"Timeloop", "Viscous step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::rhs_viscous(VectorType &       rhs,
                                                  VectorType const & velocity_rhs) const
{
  /*
   *  I. apply mass operator
   */
  pde_operator->apply_mass_operator(rhs, velocity_rhs);
  rhs *= this->bdf.get_gamma0() / this->get_time_step_size();

  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    // for a nonlinear problem, inhomogeneous contributions are taken into account when evaluating
    // the nonlinear residual

    // compensate for explicit convective term
    if(this->param.convective_problem())
    {
      for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
        rhs.add(this->extra.get_beta(i), this->vec_convective_term[i]);
    }
  }
  else
  {
    /*
     *  II. inhomogeneous parts of boundary face integrals of viscous operator
     */
    pde_operator->rhs_add_viscous_term(rhs, this->get_next_time());
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::viscous_step_enriched()
{
  dealii::Timer timer;
  timer.restart();

  AssertThrow(this->param.viscous_problem() and
                not this->param.nonlinear_problem_has_to_be_solved(),
              dealii::ExcMessage("viscous_step_enriched: only the linear viscous step is supported."));

  // velocity after the projection step: (U_bar, U_tilde)
  BlockVectorType velocity_rhs(velocity_np);
  velocity_projected  = velocity_rhs;
  mass_factor_viscous = this->get_scaling_factor_time_derivative_term();
  residual_available  = true;

  // initial guess
  if(this->use_extrapolation)
  {
    velocity_np = 0.0;
    for(unsigned int i = 0; i < velocity.size(); ++i)
      velocity_np.add(this->extra.get_beta(i), velocity[i]);
  }

  if(this->param.viscosity_is_variable())
  {
    dealii::Timer timer_viscosity_update;
    timer_viscosity_update.restart();

    pde_operator->update_viscosity_enriched(velocity_np);

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Update of variable viscosity (enriched):";
      print_wall_time(this->pcout, timer_viscosity_update.wall_time());
    }
  }

  // rhs = gamma0/dt * M * velocity_rhs  (both blocks)
  BlockVectorType rhs(velocity_np);
  pde_operator->apply_mass_operator(rhs, velocity_rhs);
  rhs *= this->bdf.get_gamma0() / this->get_time_step_size();

  // inhomogeneous boundary terms of the viscous operator (U_bar row only;
  // at enrichment walls the prescribed velocity is zero)
  pde_operator->rhs_add_viscous_term(rhs.block(0), this->get_next_time());
  pde_operator->rhs_add_enriched_viscous_boundary_data(rhs.block(0), rhs.block(1), this->get_next_time());

  unsigned int const n_iter = pde_operator->solve_linear_momentum_equation(velocity_np, rhs, this->get_scaling_factor_time_derivative_term());

  iterations_viscous.first += 1;
  std::get<1>(iterations_viscous.second) += n_iter;

  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Solve viscous step (enriched):";
    print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Viscous step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::penalty_step()
{
  if(this->param.use_divergence_penalty == true or this->param.use_continuity_penalty == true)
  {
    dealii::Timer timer;
    timer.restart();

    // compute right-hand-side vector
    VectorType rhs(velocity_np.block(0));
    pde_operator->apply_mass_operator(rhs, velocity_np.block(0));

    // extrapolate velocity to time t_n+1 and use this velocity field to
    // calculate the penalty parameter for the divergence and continuity penalty term
    VectorType velocity_extrapolated(velocity_np.block(0));
    velocity_extrapolated = 0.0;
    for(unsigned int i = 0; i < velocity.size(); ++i)
      velocity_extrapolated.add(this->extra.get_beta(i), velocity[i].block(0));

    pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

    // right-hand side term: add inhomogeneous contributions of continuity penalty operator to
    // rhs-vector if desired
    if(this->param.use_continuity_penalty and this->param.continuity_penalty_use_boundary_data)
      pde_operator->rhs_add_projection_operator(rhs, this->get_next_time());

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_projection and
      ((this->time_step_number - 1) %
         this->param.update_preconditioner_projection_every_time_steps ==
       0);

    if(this->use_extrapolation == false)
      velocity_np.block(0) = velocity_projection_last_iter;

    unsigned int const n_iter =
      pde_operator->solve_projection(velocity_np.block(0), rhs, update_preconditioner);

    iterations_penalty.first += 1;
    iterations_penalty.second += n_iter;

    if(this->store_solution)
      velocity_projection_last_iter = velocity_np.block(0);

    // write output
    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve penalty step:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }

    this->timer_tree->insert({"Timeloop", "Penalty step"}, timer.wall_time());
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::prepare_vectors_for_next_timestep()
{
  Base::prepare_vectors_for_next_timestep();

  push_back(velocity);
  velocity[0].swap(velocity_np);

  if(this->param.wall_enrichment_enabled and this->param.convective_problem())
  {
    push_back(convective_history);
    convective_history[0].swap(convective_history_np);
  }

  push_back(pressure);
  pressure[0].swap(pressure_np);

  // We also have to care about the history of velocity Dirichlet boundary conditions.
  // Note that velocity_dbc_np has already been updated.
  push_back(velocity_dbc);
  velocity_dbc[0].swap(velocity_dbc_np);

  if(this->param.wall_enrichment_enabled )
  {
#ifdef EXADG_PROFILE
    StepProfile & profile = step_profile();
    dealii::Timer timer_profile;
    profile.start(timer_profile, this->mpi_comm);
#endif
    auto const & fe = this->pde_operator->get_function_enrichment();   // NEW
    fe->enrichment_velocity = velocity[0].block(1);                    // NEW
    fe->enrichment_velocity.update_ghost_values();

    bool const residual_based_wss = this->param.wall_shear_stress_computation == WallShearStressComputation::MomentumResidual;
    if(residual_based_wss and residual_available)
    {
      // u_tau from the residual of the enriched viscous step (robust against large U_bar/U_tilde)
      this->pde_operator->update_friction_velocity_from_residual(velocity[0],
                                                                 velocity_projected,
                                                                 mass_factor_viscous);
    }
    else
    {
      // no enriched step yet (step 1): gradient-based u_tau
      this->pde_operator->update_wall_enrichment_vectors(get_velocity(0));
    }
    residual_available = false;
#ifdef EXADG_PROFILE
    profile.add("u_tau + psi update", timer_profile);
#endif
#ifdef EXADG_DEBUG
    /**************************************DEBUG*********************************************/
    // DEBUG: largest friction velocity after the update
    this->pcout << "  [u_tau debug] max u_tau = "
      << this->pde_operator->get_function_enrichment()->friction_velocity.linfty_norm()
      << std::endl;
    /*********************************************************************************************/
#endif
  }
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::solve_steady_problem()
{
  this->pcout << std::endl << "Starting time loop ..." << std::endl;

  // pseudo-time integration in order to solve steady-state problem
  bool converged = false;

  if(this->param.convergence_criterion_steady_problem ==
     ConvergenceCriterionSteadyProblem::SolutionIncrement)
  {
    VectorType velocity_tmp;
    VectorType pressure_tmp;

    while(not(converged) and this->time < (this->end_time - this->eps) and
          this->get_time_step_number() <= this->param.max_number_of_time_steps)
    {
      // save solution from previous time step
      velocity_tmp = this->velocity[0].block(0);
      pressure_tmp = this->pressure[0];

      // calculate norm of solution
      double const norm_u = velocity_tmp.l2_norm();
      double const norm_p = pressure_tmp.l2_norm();
      double const norm   = std::sqrt(norm_u * norm_u + norm_p * norm_p);

      // solve time step
      this->do_timestep();

      // calculate increment:
      // increment = solution_{n+1} - solution_{n}
      //           = solution[0] - solution_tmp
      velocity_tmp *= -1.0;
      pressure_tmp *= -1.0;
      velocity_tmp.add(1.0, this->velocity[0].block(0));
      pressure_tmp.add(1.0, this->pressure[0]);

      double const incr_u   = velocity_tmp.l2_norm();
      double const incr_p   = pressure_tmp.l2_norm();
      double const incr     = std::sqrt(incr_u * incr_u + incr_p * incr_p);
      double       incr_rel = 1.0;
      if(norm > 1.0e-10)
        incr_rel = incr / norm;

      // write output
      if(this->print_solver_info())
      {
        this->pcout << std::endl
                    << "Norm of solution increment:" << std::endl
                    << "  ||incr_abs|| = " << std::scientific << std::setprecision(10) << incr
                    << std::endl
                    << "  ||incr_rel|| = " << std::scientific << std::setprecision(10) << incr_rel
                    << std::endl;
      }

      // check convergence
      if(incr < this->param.abs_tol_steady or incr_rel < this->param.rel_tol_steady)
      {
        converged = true;
      }
    }
  }
  else if(this->param.convergence_criterion_steady_problem ==
          ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes)
  {
    AssertThrow(this->param.convergence_criterion_steady_problem !=
                  ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes,
                dealii::ExcMessage(
                  "This option is not available for the dual splitting scheme. "
                  "Due to splitting errors the solution does not fulfill the "
                  "residual of the steady, incompressible Navier-Stokes equations."));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("not implemented."));
  }

  AssertThrow(
    converged == true,
    dealii::ExcMessage(
      "Maximum number of time steps or end time exceeded! This might be due to the fact that "
      "(i) the maximum number of time steps is simply too small to reach a steady solution, "
      "(ii) the problem is unsteady so that the applied solution approach is inappropriate, "
      "(iii) some of the solver tolerances are in conflict."));

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::print_iterations() const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(this->param.nonlinear_problem_has_to_be_solved())
  {
    names = {"Convective step",
             "Pressure step",
             "Projection step",
             "Viscous step (nonlinear)",
             "Viscous step (accumulated)",
             "Viscous step (linear per nonlinear)"};

    iterations_avg.resize(6);
    iterations_avg[0] = 0.0;
    iterations_avg[1] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[2] =
      (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
    iterations_avg[3] = (double)std::get<0>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
    iterations_avg[4] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);

    if(iterations_avg[3] > std::numeric_limits<double>::min())
      iterations_avg[5] = iterations_avg[4] / iterations_avg[3];
    else
      iterations_avg[5] = iterations_avg[4];
  }
  else
  {
    names = {"Convective step", "Pressure step", "Projection step", "Viscous step"};

    iterations_avg.resize(4);
    iterations_avg[0] = 0.0;
    iterations_avg[1] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[2] =
      (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
    iterations_avg[3] = (double)std::get<1>(iterations_viscous.second) /
                        std::max(1., (double)iterations_viscous.first);
  }

  if(this->param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    names.push_back("Mass solver");
    iterations_avg.push_back(iterations_mass.second / std::max(1., (double)iterations_mass.first));
  }

  if(this->param.apply_penalty_terms_in_postprocessing_step)
  {
    names.push_back("Penalty step");
    iterations_avg.push_back((double)iterations_penalty.second /
                             std::max(1., (double)iterations_penalty.first));
  }

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::debug_norms(std::string const & label) const
{
#ifdef EXADG_DEBUG
  if(not this->param.wall_enrichment_enabled)
    return;

  VectorType u_total;
  pde_operator->initialize_vector_velocity(u_total);
  pde_operator->compute_total_velocity(u_total, velocity_np);

  this->pcout << "  [norms after " << label << "]  |U_bar| = " << velocity_np.block(0).linfty_norm()
    << "   |U_tilde| = " << velocity_np.block(1).linfty_norm()
    << "   |u_h| = " << u_total.linfty_norm() << std::endl;
#else
  (void)label;
#endif
}

template<int dim, typename Number>
void
TimeIntBDFDualSplitting<dim, Number>::penalty_step_enriched()
{
  if(not(this->param.use_divergence_penalty or this->param.use_continuity_penalty))
    return;

  dealii::Timer timer;
  timer.restart();

  AssertThrow(not(this->param.use_continuity_penalty and
                  this->param.continuity_penalty_use_boundary_data),
              dealii::ExcMessage("penalty_step_enriched: continuity penalty with boundary data "
                                 "is not implemented."));

  // rhs = M u*   (both blocks)
  BlockVectorType rhs(velocity_np);
  pde_operator->apply_mass_operator(rhs, velocity_np);

  // penalty parameters from the extrapolated FULL velocity u_h
  VectorType velocity_extrapolated, velocity_total_i;
  pde_operator->initialize_vector_velocity(velocity_extrapolated);
  pde_operator->initialize_vector_velocity(velocity_total_i);
  velocity_extrapolated = 0.0;
  for(unsigned int i = 0; i < velocity.size(); ++i)
  {
    pde_operator->compute_total_velocity(velocity_total_i, velocity[i]);
    velocity_extrapolated.add(this->extra.get_beta(i), velocity_total_i);
  }

  pde_operator->update_projection_operator_enriched(velocity_extrapolated,
                                                    this->get_time_step_size());

  unsigned int const n_iter = pde_operator->solve_projection_enriched(velocity_np, rhs);

  iterations_penalty.first += 1;
  iterations_penalty.second += n_iter;

  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Solve penalty step (enriched):";
    print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Penalty step"}, timer.wall_time());
}

// instantiations

template class TimeIntBDFDualSplitting<2, float>;
template class TimeIntBDFDualSplitting<2, double>;

template class TimeIntBDFDualSplitting<3, float>;
template class TimeIntBDFDualSplitting<3, double>;

} // namespace IncNS
} // namespace ExaDG
