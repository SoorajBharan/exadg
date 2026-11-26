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
#include <deal.II/fe/mapping_q_cache.h>

#include <exadg/convection_diffusion/time_integration/create_time_integrator.h>
#include <exadg/grid/grid_utilities.h>
#include <exadg/incompressible_flow_with_transport/driver.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/utilities/print_solver_results.h>
#include <exadg/operators/adaptive_mesh_refinement.h>

namespace ExaDG
{
namespace FTI
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const &                              comm,
                            std::shared_ptr<ApplicationBase<dim, Number>> app,
                            bool const                                    is_test)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    is_test(is_test),
    application(app),
    use_adaptive_time_stepping(false),
    N_time_steps(0),
    rans_enabled(false),
    rans_scalar_index(dealii::numbers::invalid_unsigned_int)
{
  print_general_info<Number>(pcout, mpi_comm, is_test);
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up incompressible flow with scalar transport solver:" << std::endl;

  application->setup(grid, mapping, multigrid_mappings);

  // additional parameter check: This driver does not implement steady
  // flow-transport problems. Note, however, that ProblemType and
  // SolverType might be Steady for the fluid problem in order to be able
  // to neglect the inertial term in the fluid equations. The overall coupled
  // flow-transport problem is then still a transient/unsteady problem.
  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    AssertThrow(application->scalars[i]->get_parameters().problem_type ==
                  ConvDiff::ProblemType::Unsteady,
                dealii::ExcMessage("ProblemType must be unsteady!"));
  }

  bool const ale = application->fluid->get_parameters().ale_formulation;

  if(ale) // moving mesh
  {
    AssertThrow(application->fluid->get_parameters().mesh_movement_type ==
                  IncNS::MeshMovementType::Function,
                dealii::ExcMessage("not implemented."));

    std::shared_ptr<dealii::Function<dim>> mesh_motion;
    mesh_motion = application->fluid->create_mesh_movement_function();
    ale_mapping = std::make_shared<DeformedMappingFunction<dim, Number>>(
      mapping,
      application->fluid->get_parameters().degree_u,
      *grid->triangulation,
      mesh_motion,
      application->fluid->get_parameters().start_time);

    ale_multigrid_mappings = std::make_shared<MultigridMappings<dim, Number>>(
      ale_mapping, application->fluid->get_parameters().mapping_degree_coarse_grids);

    helpers_ale = std::make_shared<HelpersALE<dim, Number>>();

    helpers_ale->move_grid = [&](double const & time) {
      ale_mapping->update(time,
                          false /* print_solver_info */,
                          this->fluid_time_integrator->get_number_of_time_steps());
    };

    helpers_ale->update_pde_operator_after_grid_motion = [&]() {
      // Since we use the same MatrixFree object for the fluid field and the scalar transport field,
      // we need to update MatrixFree here in the Driver.
      matrix_free->update_mapping(*ale_mapping->get_mapping());

      fluid_operator->update_after_grid_motion(false /* update_matrix_free */);
      for(unsigned int i = 0; i < application->scalars.size(); ++i)
        scalar_operator[i]->update_after_grid_motion(false /* update_matrix_free */);
    };

    helpers_ale->fill_grid_coordinates_vector = [&](VectorType & grid_coordinates,
                                                    dealii::DoFHandler<dim> const & dof_handler) {
      ale_mapping->fill_grid_coordinates_vector(grid_coordinates, dof_handler);
    };
  }

  std::shared_ptr<dealii::Mapping<dim> const> dynamic_mapping =
    ale ? ale_mapping->get_mapping() : mapping;

  // initialize fluid_operator
  if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    fluid_operator =
      IncNS::create_operator<dim, Number>(grid,
                                          dynamic_mapping,
                                          ale ? ale_multigrid_mappings : multigrid_mappings,
                                          application->fluid->get_boundary_descriptor(),
                                          application->fluid->get_field_functions(),
                                          application->fluid->get_parameters(),
                                          "fluid",
                                          mpi_comm);
  }
  else if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    fluid_operator = std::make_shared<IncNS::OperatorCoupled<dim, Number>>(
      grid,
      dynamic_mapping,
      ale ? ale_multigrid_mappings : multigrid_mappings,
      application->fluid->get_boundary_descriptor(),
      application->fluid->get_field_functions(),
      application->fluid->get_parameters(),
      "fluid",
      mpi_comm);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  unsigned int const n_scalars = application->scalars.size();

  scalar_operator.resize(n_scalars);
  scalar_postprocessor.resize(n_scalars);
  scalar_time_integrator.resize(n_scalars);

  // initialize convection-diffusion operator
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_operator[i] = std::make_shared<ConvDiff::Operator<dim, Number>>(
      grid,
      dynamic_mapping,
      ale ? ale_multigrid_mappings : multigrid_mappings,
      application->scalars[i]->get_boundary_descriptor(),
      application->scalars[i]->get_field_functions(),
      application->scalars[i]->get_parameters(),
      "scalar" + std::to_string(i),
      mpi_comm);
  }

  // initialize matrix_free (one object for the fluid and all scalars)
  setup_matrix_free();

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    AssertThrow(
      application->scalars[i]->get_parameters().use_cell_based_face_loops ==
        application->fluid->get_parameters().use_cell_based_face_loops,
      dealii::ExcMessage(
        "Parameter use_cell_based_face_loops should be the same for fluid and scalar transport."));
  }
  
  // Find the RANS scalar (the scalar with an active turbulence model, all its variables, e.g. k
  // and omega, are the blocks of one BlockVector). It has to be known before the fluid operator
  // is set up. Only one RANS scalar is allowed.
  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    if(application->scalars[i]->get_parameters().turbulence_model_data.is_active)
    {
      AssertThrow(not rans_enabled,
                  dealii::ExcMessage("Only one scalar can solve the turbulence model."));

      rans_enabled      = true;
      rans_scalar_index = i;
    }
  }

  if(application->fluid->get_parameters().boussinesq_term)
  {
    // scalar 0 is the temperature: a single transport variable, not the RANS scalar
    AssertThrow(n_scalars > 0,
                dealii::ExcMessage("Boussinesq term: the temperature (scalar 0) is missing."));
    AssertThrow(not(rans_enabled and rans_scalar_index == 0),
                dealii::ExcMessage("Boussinesq term: scalar 0 is the temperature, the turbulence model has to be solved by another scalar."));
    AssertThrow(application->scalars[0]->get_parameters().number_of_transport_variables == 1,
                dealii::ExcMessage("Boussinesq term: the temperature (scalar 0) has to be a single transport variable."));
  }

  if(rans_enabled)
  {
    AssertThrow(application->scalars[rans_scalar_index]->get_parameters().temporal_discretization ==
                ConvDiff::TemporalDiscretization::BDF,
                dealii::ExcMessage("Turbulence model: use TemporalDiscretization::BDF for the RANS scalar."));
  }

  if(application->fluid->get_parameters().enable_adaptivity)
  {
    for(unsigned int i = 0; i < n_scalars; ++i)
    {
      AssertThrow(not application->scalars[i]->get_parameters().enable_adaptivity,
                  dealii::ExcMessage("Adaptive mesh refinement of the coupled solver is set in the fluid parameters; set enable_adaptivity = false for the scalars."));
      AssertThrow(application->scalars[i]->get_parameters().temporal_discretization == ConvDiff::TemporalDiscretization::BDF,
                  dealii::ExcMessage("Adaptive mesh refinement: the scalars need BDF time integration."));
    }


    if(rans_enabled)
      AssertThrow(not application->scalars[rans_scalar_index]->get_parameters().turbulence_model_data.omega_enrichment,
                  dealii::ExcMessage("Combination of adaptive mesh refinement and the enrichment of omega not implemented yet."));
  }

  // setup of the fluid and scalar operators with the shared MatrixFree object
  setup_operators();

  // DoF vectors of the coupling between fluid and scalars (temperature, RANS -> fluid, wall model)
  initialize_coupling_vectors();

  // Enrichment of omega in the wall cells of the velocity enrichment:
  //   omega_h = omega_bar + psi_omega omega_tilde,  W_h = W_bar + ln(psi_omega) W_tilde (ln omega)
  if(rans_enabled and
     application->scalars[rans_scalar_index]->get_parameters().turbulence_model_data.omega_enrichment)
  {
    auto const & turbulence_data =
      application->scalars[rans_scalar_index]->get_parameters().turbulence_model_data;

    AssertThrow(fluid_operator->get_wall_enrichment_enabled(),
                dealii::ExcMessage("omega_enrichment requires the wall enrichment of the fluid."));

    // model coefficients: [1] beta (beta_0 for 2006), [2] beta*
    std::vector<double> const coefficients =
      turbulence_data.turbulence_data_base->get_all_coefficients();

    using EnrichmentRANS = FunctionEnrichmentRANS<dim, Number>;

    typename EnrichmentRANS::Variable omega;
    omega.rans_index = 1;
    omega.shape      = EnrichmentRANS::Shape::Omega;
    omega.log_form   = turbulence_data.positivity_preserving_limiter.at(1) ==
                     ConvDiff::PositivityPreservingLimiter::LogarithmicTransportVariable;
    omega.omega_wall_law.beta          = coefficients.at(1);
    omega.omega_wall_law.beta_star     = coefficients.at(2);
    omega.omega_wall_law.cap           = turbulence_data.omega_wall_cap;
    omega.omega_wall_law.y_plus_shift  = turbulence_data.omega_wall_cap_y_plus_shift;
    omega.omega_wall_law.menter_factor = turbulence_data.omega_wall_cap_menter_factor;
    omega.omega_wall_law.y_plus_cap    = turbulence_data.omega_wall_cap_y_plus;
    omega.wall_condition               = turbulence_data.omega_enrichment_wall_condition;
    omega.face_formulation             = turbulence_data.omega_enrichment_face_formulation;

    auto function_enrichment_rans =
      std::make_shared<EnrichmentRANS>(*fluid_operator->get_function_enrichment(),
                                       application->fluid->get_parameters().viscosity);
    function_enrichment_rans->add_variable(omega);
    function_enrichment_rans->set_rank_tolerance(turbulence_data.omega_enrichment_rank_tolerance);
    function_enrichment_rans->set_k_criterion(turbulence_data.omega_enrichment_k_plus_on,
                                              turbulence_data.omega_enrichment_k_plus_off,
                                              turbulence_data.omega_enrichment_off_wall_condition);
    function_enrichment_rans->initialize(*grid->triangulation,
                                         scalar_operator[rans_scalar_index]->get_mapping(),
                                         scalar_operator[rans_scalar_index]->get_dof_handler(),
                                         scalar_operator[rans_scalar_index]->get_dof_handler_eddy_viscosity());

    scalar_operator[rans_scalar_index]->set_function_enrichment_rans(function_enrichment_rans);
  }

  // after set_function_enrichment_rans(): same blocks as the solution of the time integrator
  if(rans_enabled)
    scalar_operator[rans_scalar_index]->initialize_dof_vector(rans_solution_extrapolated);

  // setup postprocessor
  fluid_postprocessor = application->fluid->create_postprocessor();
  fluid_postprocessor->setup(*fluid_operator);

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    scalar_postprocessor[i] = application->scalars[i]->create_postprocessor();
    scalar_postprocessor[i]->setup(*scalar_operator[i]);
  }

  if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    fluid_time_integrator =
      IncNS::create_time_integrator<dim, Number>(fluid_operator,
                                                 helpers_ale,
                                                 fluid_postprocessor,
                                                 application->fluid->get_parameters(),
                                                 mpi_comm,
                                                 is_test);

    if(application->fluid->get_parameters().restarted_simulation == false)
    {
      // The parameter start_with_low_order for BDF time integration has to be true.
      // This is due to the fact that the setup function of the time integrator initializes
      // the solution at previous time instants t_0 - dt, t_0 - 2*dt, ... in case of
      // start_with_low_order == false. However, the combined time step size
      // is not known at this point since we have to first communicate the time step size
      // in order to find the minimum time step size. Overwriting the time step size would
      // imply that the time step sizes are non constant for a time integrator, but the
      // time integration constants are initialized based on the assumption of constant
      // time step size. Hence, the easiest way to avoid this kind of
      // inconsistency is to preclude the case start_with_low_order == false.
      // In case of a restart, start_with_low_order = false is possible since it has been
      // enforced that all previous time steps are identical for fluid and scalar transport.
      AssertThrow(application->fluid->get_parameters().start_with_low_order == true,
                  dealii::ExcMessage("start_with_low_order has to be true for this solver."));
    }

    fluid_time_integrator->setup(application->fluid->get_parameters().restarted_simulation);
  }
  else if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    std::shared_ptr<IncNS::OperatorCoupled<dim, Number>> fluid_operator_coupled =
      std::dynamic_pointer_cast<IncNS::OperatorCoupled<dim, Number>>(fluid_operator);

    // initialize driver for steady state problem that depends on fluid_operator
    fluid_driver_steady = std::make_shared<DriverSteady>(fluid_operator_coupled,
                                                         fluid_postprocessor,
                                                         application->fluid->get_parameters(),
                                                         mpi_comm,
                                                         is_test);

    fluid_driver_steady->setup();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  for(unsigned int i = 0; i < n_scalars; ++i)
  {
    // initialize time integrator
    scalar_time_integrator[i] =
      ConvDiff::create_time_integrator<dim, Number>(scalar_operator[i],
                                                    helpers_ale,
                                                    scalar_postprocessor[i],
                                                    application->scalars[i]->get_parameters(),
                                                    mpi_comm,
                                                    is_test);

    if(application->scalars[i]->get_parameters().restarted_simulation == false and
       application->scalars[i]->get_parameters().temporal_discretization ==
         ConvDiff::TemporalDiscretization::BDF)
    {
      // See comment above for fluid field.
      AssertThrow(application->scalars[i]->get_parameters().start_with_low_order == true,
                  dealii::ExcMessage("start_with_low_order has to be true for this solver."));
    }

    scalar_time_integrator[i]->setup(
      application->scalars[i]->get_parameters().restarted_simulation);

    AssertThrow(application->scalars[i]->get_parameters().analytical_velocity_field == false,
                dealii::ExcMessage(
                  "An analytical velocity field can not be used for this coupled solver."));
  }

  // enrichment of RANS variables: initial psi and block inverse mass (with the initial u_tau);
  // the first update_enrichment_space() then projects with psi_old = psi (identity)
  if(rans_enabled and scalar_operator[rans_scalar_index]->get_function_enrichment_rans())
  {
    // velocity of the fluid for the exact grad u_h of the wall-cell source terms
    scalar_operator[rans_scalar_index]->get_function_enrichment_rans()->set_velocity(
      fluid_time_integrator->get_velocity_np_enriched());

    // k of the initial RANS solution: k criterion of the omega enrichment for the first psi
    std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_rans_setup =
      std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(
        scalar_time_integrator[rans_scalar_index]);
    AssertThrow(time_int_rans_setup != nullptr,
                dealii::ExcMessage("Enrichment of RANS variables requires the BDF time integrator."));
    time_int_rans_setup->set_enrichment_turbulent_kinetic_energy();

    scalar_operator[rans_scalar_index]->get_function_enrichment_rans()->update();
    scalar_operator[rans_scalar_index]->get_function_enrichment_rans()->precompute_inverse_mass();
  }

  // Initialize member variable use_adaptive_time_stepping
  if(application->fluid->get_parameters().adaptive_time_stepping == true)
  {
    use_adaptive_time_stepping = true;
  }

  timer_tree.insert({"Flow + transport", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup_matrix_free() const
{
  bool const ale = application->fluid->get_parameters().ale_formulation;

  std::shared_ptr<dealii::Mapping<dim> const> dynamic_mapping =
    ale ? ale_mapping->get_mapping() : mapping;

  // one MatrixFree object for the fluid and all scalars: DoFHandlers, constraints and quadratures
  // of all operators
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(fluid_operator);
  for(unsigned int i = 0; i < application->scalars.size(); ++i)
    matrix_free_data->append(scalar_operator[i]);

  matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
  if(application->fluid->get_parameters().use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*grid->triangulation, matrix_free_data->data);

  matrix_free->reinit(*dynamic_mapping,
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup_operators() const
{
  // setup Navier-Stokes operator
  if(application->fluid->get_parameters().boussinesq_term && rans_enabled)
  {
    // assume that the first scalar field with index 0 is the active scalar that
    // couples to the incompressible Navier-Stokes equations
    fluid_operator->setup(matrix_free,
                          matrix_free_data,
                          scalar_operator[0]->get_dof_name(),
                          scalar_operator[rans_scalar_index]->get_dof_name_eddy_viscosity());
  }
  else if(application->fluid->get_parameters().boussinesq_term)
  {
    // Pass only temperature (eddy viscosity defaults to "")
    fluid_operator->setup(matrix_free, matrix_free_data, scalar_operator[0]->get_dof_name());
  }
  else if(rans_enabled)
  {
    // Pass empty temperature to reach the 4th argument for eddy viscosity
    fluid_operator->setup(matrix_free,
                          matrix_free_data,
                          "",
                          scalar_operator[rans_scalar_index]->get_dof_name_eddy_viscosity());
  }
  else
  {
    fluid_operator->setup(matrix_free, matrix_free_data);
  }

  // setup convection-diffusion operator
  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    scalar_operator[i]->setup(matrix_free, matrix_free_data, fluid_operator->get_dof_name_velocity());
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::initialize_coupling_vectors() const
{
  // Initialize DoF-vector temperature in case of transport->fluid coupling with Boussinesq term:
  if(application->fluid->get_parameters().boussinesq_term)
  {
    // assume that the first scalar quantity with index 0 is the active scalar coupled to the
    // incompressible Navier-Stokes equations via the Boussinesq term
    scalar_operator[0]->initialize_dof_vector(temperature);
  }

  // Initialize DoF-vectors of the RANS -> fluid coupling (eddy viscosity and turbulent kinetic
  // energy, both live on the eddy viscosity DoFHandler of the RANS scalar)
  if(rans_enabled)
  {
    scalar_operator[rans_scalar_index]->initialize_dof_vector_eddy_viscosity(eddy_viscosity);
    scalar_operator[rans_scalar_index]->initialize_dof_vector_eddy_viscosity(
      turbulent_kinetic_energy);
    scalar_operator[rans_scalar_index]->initialize_dof_vector_eddy_viscosity(
      eddy_viscosity_wall_law);
    scalar_operator[rans_scalar_index]->initialize_dof_vector_eddy_viscosity(
      tke_production_from_wall_law);
    scalar_operator[rans_scalar_index]->initialize_dof_vector_eddy_viscosity(
      wall_cell_indicator);

    velocity_gradient_enriched.resize(dim * dim);
    for(auto & vec : velocity_gradient_enriched)
      scalar_operator[rans_scalar_index]->initialize_dof_vector_eddy_viscosity(vec);
    scalar_operator[rans_scalar_index]->initialize_dof_vector_eddy_viscosity(
      velocity_gradient_indicator);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::mark_cells_coarsening_and_refinement() const
{
  dealii::Timer timer;
  timer.restart();

  AdaptiveMeshRefinementData const & amr_data = application->fluid->get_parameters().amr_data;

  // error indicators of all fields and their weights: the velocity (fluid), then the transport
  // variables of each scalar (e.g. temperature, k, omega)
  std::vector<dealii::Vector<float>> indicators;
  std::vector<double>                weights;

  auto const append_solver = [&](std::vector<dealii::Vector<float>> const & solver_indicators,
                                 std::vector<double> const &                solver_weights) {
      for(unsigned int field = 0; field < solver_indicators.size(); ++field)
      {
        indicators.push_back(solver_indicators[field]);
        weights.push_back(solver_weights.empty() ? 1.0 : solver_weights[field]);
      }
    };
  append_solver(fluid_operator->compute_error_indicators(fluid_time_integrator->get_velocity_np()),
                amr_data.indicator_weights);

  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> const time_integrator_scalar =
      std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
    AssertThrow(time_integrator_scalar != nullptr,
                dealii::ExcMessage("Adaptive mesh refinement: the scalars need BDF time "
                                   "integration."));

    append_solver(
      scalar_operator[i]->compute_error_indicators(time_integrator_scalar->get_solution_np()),
      application->scalars[i]->get_parameters().amr_data.indicator_weights);
  }

  // one indicator per cell (normalized, weighted, combined), marking with the fluid parameters
  dealii::Vector<float> const indicator =
    combine_indicators(indicators, weights, amr_data.indicator_combination, mpi_comm);

  mark_cells(*grid->triangulation, indicator, amr_data, get_frozen_cells());

  timer_tree.insert({"Flow + transport", "Adaptive mesh refinement", "Mark cells (Kelly)"},
                    timer.wall_time());
}

template<int dim, typename Number>
std::vector<bool>
Driver<dim, Number>::get_frozen_cells() const
{
  std::vector<bool> frozen_cells;

  IncNS::Parameters const & fluid_param = application->fluid->get_parameters();

  bool const preserve_boundary_cells = fluid_param.amr_data.preserve_boundary_cells;
  bool const wall_enrichment         = fluid_param.wall_enrichment_enabled;

  if(not preserve_boundary_cells and not wall_enrichment)
    return frozen_cells;

  frozen_cells.resize(grid->triangulation->n_active_cells(), false);

  // boundary cells (with or without wall enrichment): otherwise the 2:1 rule of
  // prepare_coarsening_and_refinement() could refine them again (e.g. the RANS omega wall value
  // is computed from the size of the first cell at the wall)
  if(preserve_boundary_cells)
  {
    for(auto const & cell : grid->triangulation->active_cell_iterators())
    {
      if(not cell->is_artificial() and cell->at_boundary())
      {
        frozen_cells[cell->active_cell_index()] = true;
      }
    }
  }
  // wall enrichment layer: active_fe_index of the enrichment DoFHandler, 0 not enriched,
  // 1 enriched, 2 halo (also known on ghost cells)
  if(wall_enrichment)
  {
    dealii::DoFHandler<dim> const & dof_handler_enrichment = fluid_operator->get_function_enrichment()->get_dof_handler_en_vector();

    for(auto const & cell : dof_handler_enrichment.active_cell_iterators())
    {
      if(not cell->is_artificial() and cell->active_fe_index() != 0)
      {
        frozen_cells[cell->active_cell_index()] = true;
      }
    }
  }

  return frozen_cells;
}

template<int dim, typename Number>
void
Driver<dim, Number>::do_adaptive_refinement() const
{
  dealii::Timer timer;
  timer.restart();

  IncNS::Parameters const & fluid_param = application->fluid->get_parameters();

  // cells of the wall enrichment layer on the current mesh (empty without wall enrichment)
  std::vector<bool> const frozen_cells = get_frozen_cells();

  // remove flags at the minimum/maximum refinement level, on the boundary if requested, and on the
  // cells of the wall enrichment layer
  limit_coarsening_and_refinement(*grid->triangulation, fluid_param.amr_data, frozen_cells);

  if(any_cells_flagged_for_coarsening_or_refinement(*grid->triangulation))
  {
    AssertThrow(std::dynamic_pointer_cast<dealii::MappingQCache<dim>>(mapping) == nullptr,
                dealii::ExcMessage("Combination of adaptive mesh refinement and "
                                   "dealii::MappingQCache not implemented."));

    grid->triangulation->prepare_coarsening_and_refinement();
    check_frozen_cells_not_flagged(*grid->triangulation, frozen_cells);

    fluid_time_integrator->prepare_coarsening_and_refinement();
    for(unsigned int i = 0; i < application->scalars.size(); ++i)
      scalar_time_integrator[i]->prepare_coarsening_and_refinement();

    grid->triangulation->execute_coarsening_and_refinement();

    bool involves_h_multigrid = fluid_param.involves_h_multigrid();
    for(unsigned int i = 0; i < application->scalars.size(); ++i)
      involves_h_multigrid = involves_h_multigrid or application->scalars[i]->get_parameters().involves_h_multigrid();

    if(involves_h_multigrid)
    {
      GridUtilities::create_coarse_triangulations_after_coarsening_and_refinement(
        *grid->triangulation,
        grid->periodic_face_pairs,
        grid->coarse_triangulations,
        grid->coarse_periodic_face_pairs,
        fluid_param.grid,
        fluid_param.amr_data.preserve_boundary_cells);
    }

    std::shared_ptr<dealii::MatrixFree<dim, Number> const> const matrix_free_old = matrix_free;
    std::shared_ptr<MatrixFreeData<dim, Number> const> const     matrix_free_data_old = matrix_free_data;

    fluid_operator->distribute_dofs_after_coarsening_and_refinement();
    for(unsigned int i = 0; i < application->scalars.size(); ++i)
      scalar_operator[i]->distribute_dofs_after_coarsening_and_refinement();

    setup_matrix_free();
    setup_operators();
    initialize_coupling_vectors();
    if(rans_enabled)
      scalar_operator[rans_scalar_index]->initialize_dof_vector(rans_solution_extrapolated);

    fluid_postprocessor->setup_after_coarsening_and_refinement();
    for(unsigned int i = 0; i < application->scalars.size(); ++i)
      scalar_postprocessor[i]->setup_after_coarsening_and_refinement();

    fluid_time_integrator->interpolate_after_coarsening_and_refinement();
    for(unsigned int i = 0; i < application->scalars.size(); ++i)
      scalar_time_integrator[i]->interpolate_after_coarsening_and_refinement();
  }
  timer_tree.insert({"Flow + transport", "Adaptive mesh refinement", "Refine and transfer"},
                    timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::set_start_time() const
{
  double time = std::numeric_limits<double>::max();

  // Setup time integrator and get time step size
  if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
    time = std::min(time, fluid_time_integrator->get_time());

  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    time = std::min(time, scalar_time_integrator[i]->get_time());
  }

  // Set the same start time for both solvers

  // fluid
  if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
    fluid_time_integrator->reset_time(time);

  // scalar transport
  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    scalar_time_integrator[i]->reset_time(time);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::synchronize_time_step_size() const
{
  double const EPSILON = 1.e-10;

  double time_step_size = std::numeric_limits<double>::max();

  if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    // Setup time integrator and get time step size
    double time_step_size_fluid = std::numeric_limits<double>::max();

    IncNS::Parameters const & fluid_param = application->fluid->get_parameters();

    // fluid
    if(fluid_time_integrator->get_time() > fluid_param.start_time - EPSILON)
      time_step_size_fluid = fluid_time_integrator->get_time_step_size();

    if(use_adaptive_time_stepping == false)
    {
      // decrease time_step in order to exactly hit end_time
      time_step_size_fluid =
        (fluid_param.end_time - fluid_param.start_time) /
        (1 + int((fluid_param.end_time - fluid_param.start_time) / time_step_size_fluid));
    }

    time_step_size = std::min(time_step_size, time_step_size_fluid);
  }

  // scalar transport
  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    ConvDiff::Parameters const & scalar_param_i = application->scalars[i]->get_parameters();

    double time_step_size_scalar = std::numeric_limits<double>::max();
    if(scalar_time_integrator[i]->get_time() > scalar_param_i.start_time - EPSILON)
      time_step_size_scalar = scalar_time_integrator[i]->get_time_step_size();

    if(use_adaptive_time_stepping == false)
    {
      // decrease time_step in order to exactly hit end_time
      time_step_size_scalar =
        (scalar_param_i.end_time - scalar_param_i.start_time) /
        (1 + int((scalar_param_i.end_time - scalar_param_i.start_time) / time_step_size_scalar));
    }

    time_step_size = std::min(time_step_size, time_step_size_scalar);
  }

  if(use_adaptive_time_stepping == false)
  {
    pcout << std::endl << "Combined time step size dt = " << time_step_size << std::endl;
  }

  // Set the same time step size for both solvers

  // fluid
  if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    fluid_time_integrator->set_current_time_step_size(time_step_size);
  }

  // scalar transport
  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    scalar_time_integrator[i]->set_current_time_step_size(time_step_size);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::communicate_scalar_to_fluid() const
{
  // We need to communicate between fluid solver and scalar transport solver, i.e., ask the
  // scalar transport solver (scalar 0 by definition) for the temperature and hand it over to the
  // fluid solver.
  if(application->fluid->get_parameters().boussinesq_term)
  {
    // assume that the first scalar quantity with index 0 is the active scalar coupled to
    // the incompressible Navier-Stokes equations via the Boussinesq term
    if(application->scalars[0]->get_parameters().temporal_discretization ==
       ConvDiff::TemporalDiscretization::ExplRK)
    {
      std::shared_ptr<ConvDiff::TimeIntExplRK<Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntExplRK<Number>>(scalar_time_integrator[0]);
      time_int_scalar->extrapolate_solution(temperature);
    }
    else if(application->scalars[0]->get_parameters().temporal_discretization ==
            ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[0]);
      time_int_scalar->extrapolate_solution(temperature);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    fluid_operator->set_temperature(temperature.block(0));
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::communicate_fluid_to_all_scalars() const
{
  // We need to communicate between fluid solver and scalar transport solver, i.e., ask the
  // fluid solver for the velocity field and hand it over to all scalar transport solvers.
  std::vector<dealii::LinearAlgebra::distributed::Vector<Number> const *> velocities;
  std::vector<double>                                                     times;

  if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    fluid_time_integrator->get_velocities_and_times_np(velocities, times);
  }
  else if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Steady)
  {

    AssertThrow(not application->fluid->get_parameters().wall_enrichment_enabled, dealii::ExcMessage("Wall enrichment: steady fluid solver coupled to scalar transport is not implemented."));

    velocities.resize(1);
    times.resize(1);

    velocities.at(0) = &fluid_driver_steady->get_velocity();
    AssertThrow(scalar_time_integrator[0].get() != nullptr, dealii::ExcMessage("Not implemented."));
    times.at(0) = scalar_time_integrator[0]->get_next_time();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    if(application->scalars[i]->get_parameters().temporal_discretization ==
       ConvDiff::TemporalDiscretization::ExplRK)
    {
      AssertThrow(not application->fluid->get_parameters().wall_enrichment_enabled, dealii::ExcMessage("Wall enrichment: use TemporalDiscretization::BDF for the scalar transport (ExplRK needs several velocity levels)."));

      std::shared_ptr<ConvDiff::TimeIntExplRK<Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntExplRK<Number>>(scalar_time_integrator[i]);
      time_int_scalar->set_velocities_and_times(velocities, times);
    }
    else if(application->scalars[i]->get_parameters().temporal_discretization ==
            ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_scalar =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      time_int_scalar->set_velocities_and_times(velocities, times);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  // wall model: upper bound of the RANS eddy viscosity on enriched cells (Spalding's law with the
  // current tau_w), used by the turbulence source terms of the RANS scalar
  if(rans_enabled and application->fluid->get_parameters().wall_enrichment_enabled and
     application->fluid->get_parameters().wall_layer_nu_t_wall_law)
  {
    fluid_operator->compute_eddy_viscosity_wall_law(eddy_viscosity_wall_law);
    scalar_operator[rans_scalar_index]->set_eddy_viscosity_wall_law(eddy_viscosity_wall_law);
  }

  // enrichment of RANS variables: new psi from the new u_tau of the fluid step; stored RANS
  // solutions and explicit-term histories are re-expressed in the new space before the RANS solve
  if(rans_enabled and scalar_operator[rans_scalar_index]->get_function_enrichment_rans())
  {
    auto const & function_enrichment_rans =
      scalar_operator[rans_scalar_index]->get_function_enrichment_rans();

    std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_rans =
      std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(
        scalar_time_integrator[rans_scalar_index]);
    AssertThrow(time_int_rans != nullptr,
                dealii::ExcMessage("Enrichment of RANS variables requires the BDF time integrator."));

    // velocity of the fluid at the new time for the exact grad u_h of the wall-cell source terms
    function_enrichment_rans->set_velocity(fluid_time_integrator->get_velocity_np_enriched());

    time_int_rans->update_enrichment_space([&]() {
      function_enrichment_rans->update();
      function_enrichment_rans->precompute_inverse_mass();
    });
  }

  if(rans_enabled and application->fluid->get_parameters().wall_enrichment_enabled and application->fluid->get_parameters().wall_layer_tke_production_from_wall_law)
  {
    fluid_operator->compute_tke_production_from_wall_law(tke_production_from_wall_law,
                                                         wall_cell_indicator);
    scalar_operator[rans_scalar_index]->set_tke_production_from_wall_law(tke_production_from_wall_law, wall_cell_indicator);
  }

  // wall model: exact velocity gradient grad(U_bar + psi U_tilde) of the enriched cells for the
  // cell terms of the RANS scalar (instead of the gradient of the projected velocity_total)
  if(rans_enabled and application->fluid->get_parameters().wall_enrichment_enabled and
     application->fluid->get_parameters().wall_layer_exact_velocity_gradient)
  {
    fluid_operator->compute_velocity_gradient_rans(velocity_gradient_enriched,
                                                   velocity_gradient_indicator,
                                                   fluid_time_integrator->get_velocity_np_enriched());
    scalar_operator[rans_scalar_index]->set_velocity_gradient_enriched(velocity_gradient_enriched,
                                                                       velocity_gradient_indicator);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::communicate_eddy_viscosity_to_fluid() const
{
  // We need to communicate between the RANS scalar and the fluid solver: the RANS solution
  // (all blocks, e.g. k and omega) is extrapolated to t_{n+1}, the time of the fluid solve. From
  // it, the eddy viscosity and the turbulent kinetic energy are computed and handed over to the
  // fluid solver. Nothing else of the turbulence model is needed by the fluid solver.
  if(rans_enabled)
  {
    // BDF only (checked in setup())
    std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_scalar =
      std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(
        scalar_time_integrator[rans_scalar_index]);
    time_int_scalar->extrapolate_solution(rans_solution_extrapolated);

    // The stored nodal eddy viscosity of the RANS scalar is only used for this coupling and
    // postprocessing. The RANS time integrator recomputes it from Q^n in its own time step.
    scalar_operator[rans_scalar_index]->update_eddy_viscosity(rans_solution_extrapolated);
    scalar_operator[rans_scalar_index]->get_eddy_viscosity(eddy_viscosity);

    scalar_operator[rans_scalar_index]->get_turbulent_kinetic_energy(turbulent_kinetic_energy,
                                                                     rans_solution_extrapolated);

    fluid_operator->set_eddy_viscosity(eddy_viscosity);
    fluid_operator->set_turbulent_kinetic_energy(turbulent_kinetic_energy);
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::ale_update() const
{
  dealii::Timer timer;
  timer.restart();

  dealii::Timer sub_timer;

  sub_timer.restart();
  helpers_ale->move_grid(fluid_time_integrator->get_next_time());
  timer_tree.insert({"Flow + transport", "ALE", "Reinit mapping"}, sub_timer.wall_time());

  sub_timer.restart();
  helpers_ale->update_pde_operator_after_grid_motion();
  timer_tree.insert({"Flow + transport", "ALE", "Update matrix-free / PDE operators"},
                    sub_timer.wall_time());

  sub_timer.restart();
  fluid_time_integrator->ale_update();
  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_int_bdf =
      std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
    time_int_bdf->ale_update();
  }
  timer_tree.insert({"Flow + transport", "ALE", "Update all time integrators"},
                    sub_timer.wall_time());

  timer_tree.insert({"Flow + transport", "ALE"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  AdaptivityStartCriterion adaptivity_start(application->fluid->get_parameters().amr_data.start_time_solution_ratio);
  bool adaptivity_start_reported = false;
  // time loop
  bool finished = false;
  while(not finished)
  {
    /*
     * pre solve
     */
    if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
      fluid_time_integrator->advance_one_timestep_pre_solve(true);

    for(unsigned int i = 0; i < application->scalars.size(); ++i)
      scalar_time_integrator[i]->advance_one_timestep_pre_solve(false);

    /*
     * ALE: move the mesh and update dependent data structures
     */
    if(application->fluid->get_parameters().ale_formulation)
      ale_update();

    /*
     *  solve
     */

    // Communicate scalar -> fluid
    communicate_scalar_to_fluid();

    // Communicate scalar -> fluid (turbulence eddy viscosity)
    communicate_eddy_viscosity_to_fluid();

    // fluid: advance one time step
    if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
    {
      fluid_time_integrator->advance_one_timestep_solve();
    }
    else if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Steady)
    {
      fluid_driver_steady->solve(scalar_time_integrator[0]->get_next_time(), true /*unsteady*/);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    // Communicate fluid -> all scalars
    communicate_fluid_to_all_scalars();

    // scalar transport: advance one time step
    for(unsigned int i = 0; i < application->scalars.size(); ++i)
      scalar_time_integrator[i]->advance_one_timestep_solve();

    // start-up phase: relative change of the fluid velocity in this time step
    if(application->fluid->get_parameters().enable_adaptivity)
    {
      adaptivity_start.update(
        compute_relative_change_rate(fluid_time_integrator->get_velocity_np(),
                                     fluid_time_integrator->get_velocity(),
                                     fluid_time_integrator->get_time_step_size()));
      if(adaptivity_start.adaptivity_started() and not adaptivity_start_reported)
      {
        pcout << std::endl
              << "Adaptive mesh refinement starts at t = "
              << fluid_time_integrator->get_next_time() << " (time step "
              << fluid_time_integrator->get_number_of_time_steps() << ")" << std::endl;
        adaptivity_start_reported = true;
      }
    }

    /*
       * adaptive mesh refinement: before post solve, so that the history vectors and the next time
       * step size of the fluid and the scalars use the new mesh
       */
    if(application->fluid->get_parameters().enable_adaptivity and
      adaptivity_start.adaptivity_started() and
      fluid_time_integrator->get_next_time() >= application->fluid->get_parameters().amr_data.start_time and
      trigger_coarsening_and_refinement_now(
        application->fluid->get_parameters().amr_data.trigger_every_n_time_steps,
        fluid_time_integrator->get_number_of_time_steps()))
    {
      dealii::Timer timer_amr;
      timer_amr.restart();

      mark_cells_coarsening_and_refinement();

      do_adaptive_refinement();

      timer_tree.insert({"Flow + transport", "Adaptive mesh refinement"}, timer_amr.wall_time());
    }

    /*
     * post solve
     */
    if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
      fluid_time_integrator->advance_one_timestep_post_solve();

    for(unsigned int i = 0; i < application->scalars.size(); ++i)
      scalar_time_integrator[i]->advance_one_timestep_post_solve();

    // Both solvers have already calculated the new, adaptive time step size individually in
    // function advance_one_timestep(). Here, we have to synchronize the time step size.
    if(use_adaptive_time_stepping == true)
      synchronize_time_step_size();

    // check if all finished
    if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
      finished = fluid_time_integrator->finished();
    else
      finished = true;

    for(unsigned int i = 0; i < application->scalars.size(); ++i)
      finished = finished and scalar_time_integrator[i]->finished();

    ++N_time_steps;
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  pcout << "Performance results for coupled flow-transport solver:" << std::endl;

  // Iterations
  this->pcout << std::endl << "Average number of iterations:" << std::endl;

  // Fluid
  this->pcout << std::endl << "Incompressible Navier-Stokes solver:" << std::endl;

  if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    this->fluid_time_integrator->print_iterations();
  }
  else if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    fluid_driver_steady->print_iterations();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  // Scalar
  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    this->pcout << std::endl << "Convection-diffusion solver for scalar " << i << ":" << std::endl;

    // only relevant for BDF time integrator
    if(application->scalars[i]->get_parameters().temporal_discretization ==
       ConvDiff::TemporalDiscretization::BDF)
    {
      std::shared_ptr<ConvDiff::TimeIntBDF<dim, Number>> time_integrator_bdf =
        std::dynamic_pointer_cast<ConvDiff::TimeIntBDF<dim, Number>>(scalar_time_integrator[i]);
      time_integrator_bdf->print_iterations();
    }
    else if(application->scalars[i]->get_parameters().temporal_discretization ==
            ConvDiff::TemporalDiscretization::ExplRK)
    {
      this->pcout << "  Explicit solver (no systems of equations have to be solved)" << std::endl;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }

  // Wall times

  timer_tree.insert({"Flow + transport"}, total_time);

  if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Unsteady)
  {
    timer_tree.insert({"Flow + transport"}, fluid_time_integrator->get_timings(), "Timeloop fluid");
  }
  else if(application->fluid->get_parameters().solver_type == IncNS::SolverType::Steady)
  {
    timer_tree.insert({"Flow + transport"}, fluid_driver_steady->get_timings(), "Timeloop fluid");
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    timer_tree.insert({"Flow + transport"},
                      scalar_time_integrator[i]->get_timings(),
                      "Timeloop scalar " + std::to_string(i));
  }

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  dealii::types::global_dof_index DoFs = this->fluid_operator->get_number_of_dofs();

  for(unsigned int i = 0; i < application->scalars.size(); ++i)
  {
    DoFs += this->scalar_operator[i]->get_number_of_dofs() *
            this->scalar_operator[i]->get_n_blocks();
  }

  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg overall_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const overall_time_avg = overall_time_data.avg;

  print_throughput_unsteady(pcout, DoFs, overall_time_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, overall_time_avg, N_mpi_processes);

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace FTI
} // namespace ExaDG
