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

// deal.II
#include "spatial_operator_base.h"
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/vector_tools.h>

#include <typeinfo>

// ExaDG
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_projection.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/grid_related_time_step_restrictions.h>
#include <exadg/operators/quadrature.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/utilities/exceptions.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
SpatialOperatorBase<dim, Number>::SpatialOperatorBase(
  std::shared_ptr<Grid<dim> const>                      grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>           mapping_in,
  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings_in,
  std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>            field_functions_in,
  Parameters const &                                    parameters_in,
  std::string const &                                   field_in,
  MPI_Comm const &                                      mpi_comm_in)
  : dealii::Subscriptor(),
    grid(grid_in),
    mapping(mapping_in),
    multigrid_mappings(multigrid_mappings_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    param(parameters_in),
    field(field_in),
    dof_index_first_point(0),
    evaluation_time(0.0),
    dof_handler_u(*grid_in->triangulation),
    dof_handler_p(*grid_in->triangulation),
    dof_handler_u_scalar(*grid_in->triangulation),
    pressure_level_is_undefined(false),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    velocity_ptr(nullptr),
    pressure_ptr(nullptr)
{
  pcout << std::endl
        << "Construct incompressible Navier-Stokes operator ..." << std::endl
        << std::flush;

  initialize_dof_handler_and_constraints();

  initialize_boundary_descriptor_laplace();

  initialization_pure_dirichlet_bc();

  pcout << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_dof_handler_and_constraints()
{
  fe_p = create_finite_element<dim>(param.grid.element_type,
                                    true,
                                    1,
                                    param.get_degree_p(param.degree_u));

  fe_u_scalar = create_finite_element<dim>(param.grid.element_type, true, 1, param.degree_u);

  if(param.spatial_discretization == SpatialDiscretization::L2)
  {
    fe_u = create_finite_element<dim>(param.grid.element_type, true, dim, param.degree_u);
  }
  else if(param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    AssertThrow(
      param.grid.element_type == ElementType::Hypercube,
      dealii::ExcMessage(
        "SpatialDiscretization::HDIV is currently only implemented for hypercube elements. "
        "You might want to change the element type of the grid, or the function space, "
        "or implement HDIV for element types other than hypercube."));

    // The constructor of FE_RaviartThomas takes the degree in tangential direction as an
    // argument.
    fe_u = std::make_shared<dealii::FE_RaviartThomasNodal<dim>>(param.degree_u - 1);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("FE not implemented."));
  }

  // enumerate degrees of freedom
  dof_handler_u.distribute_dofs(*fe_u);
  dof_handler_p.distribute_dofs(*fe_p);
  dof_handler_u_scalar.distribute_dofs(*fe_u_scalar);

  // Strong imposition of boundary conditions in the case of HDIV
  if(param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    // Periodic boundaries
    // We need to make sure the normal dofs are shared between cells on the periodic boundaries,
    // since these are continuous for HDIV.
    dealii::IndexSet relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_u, relevant_dofs);
    constraint_u.reinit(relevant_dofs);

    for(auto const & face : grid->periodic_face_pairs)
      dealii::DoFTools::make_periodicity_constraints(
        dof_handler_u,
        face.cell[0]->face(face.face_idx[0])->boundary_id(),
        face.cell[1]->face(face.face_idx[1])->boundary_id(),
        face.face_idx[0] / 2,
        constraint_u);

    // Symmetry boundaries
    // Constraints the normal components of the velocity, where "0" as second argument indicates the
    // first component in the dof_handler.
    if(not(boundary_descriptor->velocity->symmetry_bc.empty()))
    {
      for(auto bc : boundary_descriptor->velocity->symmetry_bc)
        dealii::VectorTools::project_boundary_values_div_conforming(
          dof_handler_u, 0, *(bc.second), bc.first, constraint_u, *get_mapping());
    }

    // Dirichlet boundaries
    if(not(boundary_descriptor->velocity->dirichlet_bc.empty()))
    {
      AssertThrow(
        false,
        dealii::ExcMessage(
          "Dirichlet BCs are not properly implemented for HDIV. The normal component of the velocity field needs to be strongly applied."));
      // We would like to do something similar to above. Probably would work with the same function
      // i.e dealii::VectorTools::project_boundary_values_div_conforming(). Otherwise one might want
      // to look into: dealii::VectorTools::interpolate_boundary_values
      // dealii::VectorTools::project_boundary_values
      // dealii::DoFTools::make_zero_boundary_constraints
    }
  }

  constraint_u.close();
  constraint_p.close();
  constraint_u_scalar.close();
  constraint_enrichment.close();

  // Output to pcout
  pcout << "Velocity:" << std::endl;
  if(param.spatial_discretization == SpatialDiscretization::L2)
  {
    print_parameter(pcout, "degree of 1D polynomials", param.degree_u);
  }
  else if(param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    print_parameter(pcout, "degree of 1D polynomials (normal)", param.degree_u);
    print_parameter(pcout, "degree of 1D polynomials (tangential)", (param.degree_u - 1));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("FE not implemented."));
  }
  print_parameter(pcout, "number of dofs per cell", fe_u->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", dof_handler_u.n_dofs());
  if(param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    pcout << "NOTE. Continuity constraints in case of periodic boundary conditions"
          << "are not taken into account regarding the number of total DoFs." << std::endl;
  }

  pcout << "Pressure:" << std::endl;
  print_parameter(pcout, "degree of 1D polynomials", param.get_degree_p(param.degree_u));
  print_parameter(pcout, "number of dofs per cell", fe_p->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", dof_handler_p.n_dofs());

  pcout << "Velocity and pressure:" << std::endl;
  print_parameter(pcout,
                  "number of dofs per cell",
                  fe_u->n_dofs_per_cell() + fe_p->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", get_number_of_dofs());

  pcout << std::flush;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_boundary_descriptor_laplace()
{
  boundary_descriptor_laplace = std::make_shared<Poisson::BoundaryDescriptor<0, dim>>();

  // Dirichlet BCs for pressure
  boundary_descriptor_laplace->dirichlet_bc = boundary_descriptor->pressure->dirichlet_bc;

  // Neumann BCs for pressure: These boundary conditions are empty.
  // However, when using projection methods with the solution of a pressure Poisson
  // equation, the interface of the Laplace operator requires to set functions on
  // Neumann boundaries, which we simply fill by ZeroFunction. In case that a
  // projection method prescribes inhomogeneous Neumann boundary conditions for the
  // pressure (e.g. dual splitting projection scheme), this is done by separate
  // routines.
  for(typename std::set<dealii::types::boundary_id>::const_iterator it =
        boundary_descriptor->pressure->neumann_bc.begin();
      it != boundary_descriptor->pressure->neumann_bc.end();
      ++it)
  {
    std::shared_ptr<dealii::Function<dim>> zero_function;
    zero_function = std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);
    boundary_descriptor_laplace->neumann_bc.insert(
      std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>(*it,
                                                                                    zero_function));
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialization_pure_dirichlet_bc()
{ // Erroneously, the boundary descriptor might contain too many boundary IDs which
  // do not even exist in the triangulation. Here, we make sure that each entry of
  // the boundary descriptor has indeed a counterpart in the triangulation.
  std::vector<dealii::types::boundary_id> boundary_ids = grid->triangulation->get_boundary_ids();
  for(auto it = boundary_descriptor->pressure->dirichlet_bc.begin();
      it != boundary_descriptor->pressure->dirichlet_bc.end();
      ++it)
  {
    bool const triangulation_has_boundary_id =
      std::find(boundary_ids.begin(), boundary_ids.end(), it->first) != boundary_ids.end();

    AssertThrow(triangulation_has_boundary_id,
                dealii::ExcMessage("The boundary descriptor for the pressure contains boundary IDs "
                                   "that are not part of the triangulation."));
  }

  pressure_level_is_undefined = boundary_descriptor->pressure->dirichlet_bc.empty();

  if(is_pressure_level_undefined())
  {
    if(param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
    {
      dof_index_first_point = 0;
      for(unsigned int d = 0; d < dim; ++d)
        first_point[d] = 0.0;

      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        typename dealii::DoFHandler<dim>::active_cell_iterator first_cell;

        bool processor_has_active_cells = false;
        for(auto const & cell : dof_handler_p.active_cell_iterators())
        {
          if(cell->is_locally_owned())
          {
            first_cell = cell;

            processor_has_active_cells = true;
            break;
          }
        }

        AssertThrow(processor_has_active_cells == true,
                    dealii::ExcMessage("No active cells on Processor with ID=0"));

        dealii::FEValues<dim> fe_values(dof_handler_p.get_fe(),
                                        dealii::Quadrature<dim>(
                                          dof_handler_p.get_fe().get_unit_support_points()),
                                        dealii::update_quadrature_points);

        fe_values.reinit(first_cell);

        first_point = fe_values.quadrature_point(0);
        std::vector<dealii::types::global_dof_index> dof_indices(
          dof_handler_p.get_fe().dofs_per_cell);
        first_cell->get_dof_indices(dof_indices);
        dof_index_first_point = dof_indices[0];
      }
      dof_index_first_point = dealii::Utilities::MPI::sum(dof_index_first_point, mpi_comm);
      for(unsigned int d = 0; d < dim; ++d)
      {
        first_point[d] = dealii::Utilities::MPI::sum(first_point[d], mpi_comm);
      }
    }
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, Number> & matrix_free_data) const
{
  // append mapping flags
  matrix_free_data.append_mapping_flags(MassKernel<dim, Number>::get_mapping_flags());
  matrix_free_data.append_mapping_flags(
    Operators::DivergenceKernel<dim, Number>::get_mapping_flags());
  matrix_free_data.append_mapping_flags(
    Operators::GradientKernel<dim, Number>::get_mapping_flags());

  if(param.convective_problem())
    matrix_free_data.append_mapping_flags(
      Operators::ConvectiveKernel<dim, Number>::get_mapping_flags());

  if(param.viscous_problem())
    matrix_free_data.append_mapping_flags(
      Operators::ViscousKernel<dim, Number>::get_mapping_flags(true, true));

  if(param.right_hand_side)
    matrix_free_data.append_mapping_flags(Operators::RHSKernel<dim, Number>::get_mapping_flags());

  if(param.use_divergence_penalty)
    matrix_free_data.append_mapping_flags(
      Operators::DivergencePenaltyKernel<dim, Number>::get_mapping_flags());

  if(param.use_continuity_penalty)
    matrix_free_data.append_mapping_flags(
      Operators::ContinuityPenaltyKernel<dim, Number>::get_mapping_flags());

  // mapping flags required for CFL condition
  MappingFlags flags_cfl;
  flags_cfl.cells = dealii::update_quadrature_points;
  matrix_free_data.append_mapping_flags(flags_cfl);

  // dof handler
  matrix_free_data.insert_dof_handler(&dof_handler_u, field + dof_index_u);
  matrix_free_data.insert_dof_handler(&dof_handler_p, field + dof_index_p);
  matrix_free_data.insert_dof_handler(&dof_handler_u_scalar, field + dof_index_u_scalar);

  // constraint
  matrix_free_data.insert_constraint(&constraint_u, field + dof_index_u);
  matrix_free_data.insert_constraint(&constraint_p, field + dof_index_p);
  matrix_free_data.insert_constraint(&constraint_u_scalar, field + dof_index_u_scalar);

  // quadrature
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_u =
    create_quadrature<dim>(param.grid.element_type, param.degree_u + 1);
  matrix_free_data.insert_quadrature(*quadrature_u, field + quad_index_u);
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_p =
    create_quadrature<dim>(param.grid.element_type, param.get_degree_p(param.degree_u) + 1);
  matrix_free_data.insert_quadrature(*quadrature_p, field + quad_index_p);
  std::shared_ptr<dealii::Quadrature<dim>> quadrature_u_overintegration =
    create_quadrature<dim>(param.grid.element_type, param.degree_u + (param.degree_u + 2) / 2);
  matrix_free_data.insert_quadrature(*quadrature_u_overintegration,
                                     field + quad_index_u_overintegration);

  // TODO create these quadrature rules only when needed
  matrix_free_data.insert_quadrature(dealii::QGaussLobatto<1>(param.degree_u + 1),
                                     field + quad_index_u_gauss_lobatto);
  matrix_free_data.insert_quadrature(dealii::QGaussLobatto<1>(param.get_degree_p(param.degree_u) +
                                                              1),
                                     field + quad_index_p_gauss_lobatto);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_dirichlet_cached_bc()
{
  // initialize data container for DirichletCached boundary conditions
  if(not(boundary_descriptor->velocity->dirichlet_cached_bc.empty()))
  {
    std::vector<unsigned int> quad_indices;
    quad_indices.emplace_back(get_quad_index_velocity_standard());
    quad_indices.emplace_back(get_quad_index_velocity_overintegration());
    quad_indices.emplace_back(get_quad_index_velocity_gauss_lobatto());

    interface_data_dirichlet_cached = std::make_shared<ContainerInterfaceData<1, dim, double>>();
    interface_data_dirichlet_cached->setup(*matrix_free,
                                           get_dof_index_velocity(),
                                           quad_indices,
                                           boundary_descriptor->velocity->dirichlet_cached_bc);

    boundary_descriptor->velocity->set_dirichlet_cached_data(interface_data_dirichlet_cached);
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_operators(std::string const & dof_index_temperature,
                                                       std::string const & dof_index_eddy_viscosity)
{
  // wall enrichment
  if(param.wall_enrichment_enabled)
  {
    function_enrichment = std::make_shared<FunctionEnrichment<dim, Number>>(param.viscosity);
    function_enrichment->set_minimum_wall_shear_stress_fraction(param.wall_shear_stress_min_fraction);
    wall_dg_coupler     = std::make_shared<WallDGCoupler<dim, Number>>();
    wall_dg_coupler->set_rank_tolerance(param.wall_enrichment_rank_tolerance);
    function_enrichment->set_global_pressure_dof_handler(get_dof_handler_p());

    function_enrichment->initialize(*grid->triangulation,
                                    get_mapping(),
                                    get_dof_handler_u(),
                                    param.degree_u,
                                    param.fe_degree_enrichment,
                                    boundary_descriptor->velocity,
                                    grid->periodic_face_pairs,
                                    get_quad_index_velocity_standard(),
                                    param.enrichment_quadrature_points,
                                    param.wall_enrichment_layers);

    wall_dg_coupler->initialize(*matrix_free, 
                                get_dof_index_velocity(), 
                                function_enrichment);

    function_enrichment->setup_wall_distance(*get_mapping(), param.wall_enrichment_layers);
    wall_dg_coupler->precompute_schur_matrices();
  }
  // mass operator
  MassOperatorData<dim> mass_operator_data;
  mass_operator_data.dof_index  = get_dof_index_velocity();
  mass_operator_data.quad_index = get_quad_index_velocity_standard();
  mass_operator.initialize(*matrix_free, constraint_u, mass_operator_data);

  InverseMassOperatorDataHdiv inverse_mass_data_hdiv;
  inverse_mass_data_hdiv.dof_index  = get_dof_index_velocity();
  inverse_mass_data_hdiv.quad_index = get_quad_index_velocity_standard();
  inverse_mass_data_hdiv.parameters = this->param.inverse_mass_operator_hdiv;

  inverse_mass_hdiv.initialize(*matrix_free, constraint_u, inverse_mass_data_hdiv);

  // inverse mass operator
  if(param.spatial_discretization == SpatialDiscretization::L2)
  {
    InverseMassOperatorData inverse_mass_operator_data_velocity;
    inverse_mass_operator_data_velocity.dof_index  = get_dof_index_velocity();
    inverse_mass_operator_data_velocity.quad_index = get_quad_index_velocity_standard();
    inverse_mass_operator_data_velocity.parameters = param.inverse_mass_operator;
    inverse_mass_velocity.initialize(*matrix_free, inverse_mass_operator_data_velocity);
  }
  // inverse mass operator velocity scalar
  InverseMassOperatorData inverse_mass_operator_data_velocity_scalar;
  inverse_mass_operator_data_velocity_scalar.dof_index  = get_dof_index_velocity_scalar();
  inverse_mass_operator_data_velocity_scalar.quad_index = get_quad_index_velocity_standard();
  inverse_mass_operator_data_velocity_scalar.parameters = param.inverse_mass_operator;
  inverse_mass_velocity_scalar.initialize(*matrix_free, inverse_mass_operator_data_velocity_scalar);

  // body force operator
  RHSOperatorData<dim> rhs_data;
  rhs_data.dof_index = get_dof_index_velocity();
  if(param.boussinesq_term)
    rhs_data.dof_index_scalar = matrix_free_data->get_dof_index(dof_index_temperature);
  rhs_data.quad_index                                = get_quad_index_velocity_standard();
  rhs_data.kernel_data.f                             = field_functions->right_hand_side;
  rhs_data.kernel_data.boussinesq_term               = param.boussinesq_term;
  rhs_data.kernel_data.boussinesq_dynamic_part_only  = param.boussinesq_dynamic_part_only;
  rhs_data.kernel_data.thermal_expansion_coefficient = param.thermal_expansion_coefficient;
  rhs_data.kernel_data.reference_temperature         = param.reference_temperature;
  rhs_data.kernel_data.gravitational_force           = field_functions->gravitational_force;

  rhs_operator.initialize(*matrix_free, rhs_data);

  // gradient operator
  GradientOperatorData<dim> gradient_operator_data;
  gradient_operator_data.dof_index_velocity   = get_dof_index_velocity();
  gradient_operator_data.dof_index_pressure   = get_dof_index_pressure();
  gradient_operator_data.quad_index           = get_quad_index_velocity_standard();
  gradient_operator_data.integration_by_parts = param.gradp_integrated_by_parts;
  gradient_operator_data.formulation          = param.gradp_formulation;
  gradient_operator_data.use_boundary_data    = param.gradp_use_boundary_data;
  gradient_operator_data.bc                   = boundary_descriptor->pressure;
  gradient_operator.initialize(*matrix_free, gradient_operator_data);

  // divergence operator
  DivergenceOperatorData<dim> divergence_operator_data;
  divergence_operator_data.dof_index_velocity   = get_dof_index_velocity();
  divergence_operator_data.dof_index_pressure   = get_dof_index_pressure();
  divergence_operator_data.quad_index           = get_quad_index_velocity_standard();
  divergence_operator_data.integration_by_parts = param.divu_integrated_by_parts;
  divergence_operator_data.formulation          = param.divu_formulation;
  divergence_operator_data.use_boundary_data    = param.divu_use_boundary_data;
  divergence_operator_data.bc                   = boundary_descriptor->velocity;
  divergence_operator.initialize(*matrix_free, divergence_operator_data);

  // convective operator
  convective_kernel_data.formulation             = param.formulation_convective_term;
  convective_kernel_data.upwind_factor           = param.upwind_factor;
  convective_kernel_data.use_outflow_bc          = param.use_outflow_bc_convective_term;
  convective_kernel_data.type_dirichlet_bc       = param.type_dirichlet_bc_convective;
  convective_kernel_data.ale                     = param.ale_formulation;
  convective_kernel_data.wall_enrichment_enabled = param.wall_enrichment_enabled;
  convective_kernel = std::make_shared<Operators::ConvectiveKernel<dim, Number>>();
  convective_kernel->reinit(*matrix_free,
                            convective_kernel_data,
                            get_dof_index_velocity(),
                            get_quad_index_velocity_linearized(),
                            false /* is_mg */);

  dealii::AffineConstraints<Number> constraint_dummy;
  constraint_dummy.close();

  ConvectiveOperatorData<dim> convective_operator_data;
  convective_operator_data.kernel_data          = convective_kernel_data;
  convective_operator_data.dof_index            = get_dof_index_velocity();
  convective_operator_data.quad_index           = this->get_quad_index_velocity_linearized();
  convective_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  convective_operator_data.quad_index_nonlinear = get_quad_index_velocity_overintegration();
  convective_operator_data.bc                   = boundary_descriptor->velocity;
  if(param.wall_enrichment_enabled)
  {
    convective_operator.initialize(*matrix_free,
                                   constraint_dummy,
                                   convective_operator_data,
                                   convective_kernel,
                                   this->function_enrichment);
  }
  else{
    convective_operator.initialize(*matrix_free,
                                   constraint_dummy,
                                   convective_operator_data,
                                   convective_kernel);
  }

  // viscous operator
  viscous_kernel_data.IP_factor                    = param.IP_factor_viscous;
  viscous_kernel_data.viscosity                    = param.viscosity;
  viscous_kernel_data.formulation_viscous_term     = param.formulation_viscous_term;
  viscous_kernel_data.penalty_term_div_formulation = param.penalty_term_div_formulation;
  viscous_kernel_data.IP_formulation               = param.IP_formulation_viscous;
  viscous_kernel_data.viscosity_is_variable        = param.viscosity_is_variable();
  viscous_kernel_data.variable_normal_vector       = param.neumann_with_variable_normal_vector;
  viscous_kernel = std::make_shared<Operators::ViscousKernel<dim, Number>>();
  viscous_kernel->reinit(*matrix_free,
                         viscous_kernel_data,
                         get_dof_index_velocity(),
                         get_quad_index_velocity_standard());

  // initialize and check turbulence model data
  if(param.turbulence_model_data.is_active)
  {
    if(param.turbulence_model_data.rans_model)
    {
      // Fetch the exact integer index using the string provided by the Driver!
      unsigned int const eddy_viscosity_index = matrix_free_data->get_dof_index(dof_index_eddy_viscosity);
      turbulence_model.initialize(*matrix_free,
                                  *get_mapping(),
                                  viscous_kernel,
                                  param.turbulence_model_data,
                                  get_dof_index_velocity(),
                                  eddy_viscosity_index);
    }
    else
  {
      // Standard LES models
      turbulence_model.initialize(*matrix_free,
                                  *get_mapping(),
                                  viscous_kernel,
                                  param.turbulence_model_data,
                                  get_dof_index_velocity());
    }
  }

  // initialize and check generalized Newtonian model data
  if(param.generalized_newtonian_model_data.is_active)
  {
    generalized_newtonian_model.initialize(*matrix_free,
                                           viscous_kernel,
                                           param.generalized_newtonian_model_data,
                                           get_dof_index_velocity());
  }

  ViscousOperatorData<dim> viscous_operator_data;
  viscous_operator_data.kernel_data          = viscous_kernel_data;
  viscous_operator_data.bc                   = boundary_descriptor->velocity;
  viscous_operator_data.dof_index            = get_dof_index_velocity();
  viscous_operator_data.quad_index           = get_quad_index_velocity_standard();
  viscous_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  viscous_operator.initialize(*matrix_free,
                              constraint_dummy,
                              viscous_operator_data,
                              viscous_kernel);

  // Momentum operator
  MomentumOperatorData<dim> data;

  data.unsteady_problem = unsteady_problem_has_to_be_solved();
  if(param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    data.convective_problem = false;
  }
  else
  {
    data.convective_problem = param.implicit_convective_problem();
  }
  data.viscous_problem = param.viscous_problem();

  data.convective_kernel_data = convective_kernel_data;
  data.viscous_kernel_data    = viscous_kernel_data;

  data.bc = boundary_descriptor->velocity;

  data.dof_index  = get_dof_index_velocity();
  data.quad_index = get_quad_index_velocity_linearized();

  data.use_cell_based_loops = param.use_cell_based_face_loops;
  data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;
  if(data.convective_problem)
    data.solver_block_diagonal = Elementwise::Solver::GMRES;
  else
    data.solver_block_diagonal = Elementwise::Solver::CG;
  data.preconditioner_block_diagonal = Elementwise::Preconditioner::InverseMassMatrix;
  data.solver_data_block_diagonal    = param.solver_data_block_diagonal;

  momentum_operator.initialize(
    *matrix_free, constraint_dummy, data, viscous_kernel, convective_kernel);

  if(param.use_divergence_penalty)
  {
    // Kernel
    Operators::DivergencePenaltyKernelData div_penalty_data;
    div_penalty_data.type_penalty_parameter = param.type_penalty_parameter;
    div_penalty_data.viscosity              = param.viscosity;
    div_penalty_data.degree                 = param.degree_u;
    div_penalty_data.penalty_factor         = param.divergence_penalty_factor;

    div_penalty_kernel = std::make_shared<Operators::DivergencePenaltyKernel<dim, Number>>();
    div_penalty_kernel->reinit(*matrix_free,
                               get_dof_index_velocity(),
                               get_quad_index_velocity_standard(),
                               div_penalty_data);

    // Operator
    DivergencePenaltyData operator_data;
    operator_data.dof_index  = get_dof_index_velocity();
    operator_data.quad_index = get_quad_index_velocity_standard();

    div_penalty_operator.initialize(*matrix_free, operator_data, div_penalty_kernel);
  }

  if(param.use_continuity_penalty)
  {
    // Kernel
    Operators::ContinuityPenaltyKernelData kernel_data;

    kernel_data.type_penalty_parameter = param.type_penalty_parameter;
    kernel_data.which_components       = param.continuity_penalty_components;
    kernel_data.viscosity              = param.viscosity;
    kernel_data.degree                 = param.degree_u;
    kernel_data.penalty_factor         = param.continuity_penalty_factor;

    conti_penalty_kernel = std::make_shared<Operators::ContinuityPenaltyKernel<dim, Number>>();
    conti_penalty_kernel->reinit(*matrix_free,
                                 get_dof_index_velocity(),
                                 get_quad_index_velocity_standard(),
                                 kernel_data);

    // Operator
    ContinuityPenaltyData<dim> operator_data;
    operator_data.dof_index         = get_dof_index_velocity();
    operator_data.quad_index        = get_quad_index_velocity_standard();
    operator_data.use_boundary_data = param.continuity_penalty_use_boundary_data;
    operator_data.bc                = this->boundary_descriptor->velocity;

    conti_penalty_operator.initialize(*matrix_free, operator_data, conti_penalty_kernel);
  }

  if(param.use_divergence_penalty or param.use_continuity_penalty)
  {
    if(param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme or
       param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection or
       (param.temporal_discretization == TemporalDiscretization::BDFCoupledSolution and
        param.apply_penalty_terms_in_postprocessing_step == true))
    {
      // setup projection operator
      ProjectionOperatorData<dim> data;
      data.use_divergence_penalty = param.use_divergence_penalty;
      data.use_continuity_penalty = param.use_continuity_penalty;
      data.use_boundary_data      = param.continuity_penalty_use_boundary_data;
      data.bc                     = this->boundary_descriptor->velocity;
      data.dof_index              = get_dof_index_velocity();
      data.quad_index             = get_quad_index_velocity_standard();
      data.use_cell_based_loops   = param.use_cell_based_face_loops;
      data.implement_block_diagonal_preconditioner_matrix_free =
        param.implement_block_diagonal_preconditioner_matrix_free;
      data.solver_block_diagonal         = Elementwise::Solver::CG;
      data.preconditioner_block_diagonal = param.preconditioner_block_diagonal_projection;
      data.solver_data_block_diagonal    = param.solver_data_block_diagonal_projection;

      projection_operator = std::make_shared<ProjOperator>();

      projection_operator->initialize(
        *matrix_free, constraint_dummy, data, div_penalty_kernel, conti_penalty_kernel);
    }
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_calculators_for_derived_quantities()
{
  vorticity_calculator.initialize(*matrix_free,
                                  get_dof_index_velocity(),
                                  get_quad_index_velocity_standard());
  divergence_calculator.initialize(*matrix_free,
                                   get_dof_index_velocity(),
                                   get_dof_index_velocity_scalar(),
                                   get_quad_index_velocity_standard());
  shear_rate_calculator.initialize(*matrix_free,
                                   get_dof_index_velocity(),
                                   get_dof_index_velocity_scalar(),
                                   get_quad_index_velocity_standard());
  magnitude_calculator.initialize(*matrix_free,
                                  get_dof_index_velocity(),
                                  get_dof_index_velocity_scalar(),
                                  get_quad_index_velocity_standard());
  q_criterion_calculator.initialize(*matrix_free,
                                    get_dof_index_velocity(),
                                    get_dof_index_velocity_scalar(),
                                    get_quad_index_velocity_standard(),
                                    false /*compressible_flow*/);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::setup()
{
  // initialize MatrixFree and MatrixFreeData
  std::shared_ptr<dealii::MatrixFree<dim, Number>> mf =
    std::make_shared<dealii::MatrixFree<dim, Number>>();
  std::shared_ptr<MatrixFreeData<dim, Number>> mf_data =
    std::make_shared<MatrixFreeData<dim, Number>>();

  fill_matrix_free_data(*mf_data);

  if(param.use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*grid->triangulation, mf_data->data);
  mf->reinit(*get_mapping(),
             mf_data->get_dof_handler_vector(),
             mf_data->get_constraint_vector(),
             mf_data->get_quadrature_vector(),
             mf_data->data);

  if(param.ale_formulation)
    matrix_free_own_storage = mf;

  // Subsequently, call the other setup function with MatrixFree/MatrixFreeData objects as
  // arguments.
  this->setup(mf, mf_data);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::setup(
  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data_in,
  std::string const &                                    dof_index_temperature,
  std::string const &                                    dof_index_eddy_viscosity)
{
  pcout << std::endl
        << "Setup incompressible Navier-Stokes operator ..." << std::endl
        << std::flush;

  // MatrixFree
  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;

  // Next, initialize data structures depending on MatrixFree:

  initialize_dirichlet_cached_bc();

  initialize_operators(dof_index_temperature, dof_index_eddy_viscosity);

  initialize_calculators_for_derived_quantities();

  // Finally, do set up of derived classes
  setup_derived();

  setup_preconditioners_and_solvers();

  pcout << std::endl << "... done!" << std::endl << std::flush;
}

template<int dim, typename Number>
dealii::types::global_dof_index
SpatialOperatorBase<dim, Number>::get_number_of_dofs() const
{
  return dof_handler_u.n_dofs() + dof_handler_p.n_dofs();
}

template<int dim, typename Number>
dealii::MatrixFree<dim, Number> const &
SpatialOperatorBase<dim, Number>::get_matrix_free() const
{
  return *matrix_free;
}

template<int dim, typename Number>
std::string
SpatialOperatorBase<dim, Number>::get_dof_name_velocity() const
{
  return field + dof_index_u;
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_dof_index_velocity() const
{
  return matrix_free_data->get_dof_index(get_dof_name_velocity());
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_dof_index_pressure() const
{
  return matrix_free_data->get_dof_index(field + dof_index_p);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_dof_index_velocity_scalar() const
{
  return matrix_free_data->get_dof_index(field + dof_index_u_scalar);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_velocity_standard() const
{
  return matrix_free_data->get_quad_index(field + quad_index_u);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_pressure() const
{
  return matrix_free_data->get_quad_index(field + quad_index_p);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_velocity_overintegration() const
{
  return matrix_free_data->get_quad_index(field + quad_index_u_overintegration);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_velocity_gauss_lobatto() const
{
  return matrix_free_data->get_quad_index(field + quad_index_u_gauss_lobatto);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_pressure_gauss_lobatto() const
{
  return matrix_free_data->get_quad_index(field + quad_index_p_gauss_lobatto);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::get_quad_index_velocity_linearized() const
{
  if(param.quad_rule_linearization == QuadratureRuleLinearization::Standard)
  {
    return get_quad_index_velocity_standard();
  }
  else if(param.quad_rule_linearization == QuadratureRuleLinearization::Overintegration32k)
  {
    if(param.implicit_convective_problem())
      return get_quad_index_velocity_overintegration();
    else
      return get_quad_index_velocity_standard();
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented"));
    return get_quad_index_velocity_overintegration();
  }
}

template<int dim, typename Number>
std::shared_ptr<dealii::Mapping<dim> const>
SpatialOperatorBase<dim, Number>::get_mapping() const
{
  return mapping;
}

template<int dim, typename Number>
dealii::FiniteElement<dim> const &
SpatialOperatorBase<dim, Number>::get_fe_u() const
{
  return *fe_u;
}

template<int dim, typename Number>
dealii::FiniteElement<dim> const &
SpatialOperatorBase<dim, Number>::get_fe_p() const
{
  return *fe_p;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperatorBase<dim, Number>::get_dof_handler_u() const
{
  return dof_handler_u;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperatorBase<dim, Number>::get_dof_handler_u_scalar() const
{
  return dof_handler_u_scalar;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperatorBase<dim, Number>::get_dof_handler_p() const
{
  return dof_handler_p;
}

template<int dim, typename Number>
dealii::AffineConstraints<Number> const &
SpatialOperatorBase<dim, Number>::get_constraint_p() const
{
  return constraint_p;
}

template<int dim, typename Number>
dealii::AffineConstraints<Number> const &
SpatialOperatorBase<dim, Number>::get_constraint_u() const
{
  return constraint_u;
}

template<int dim, typename Number>
dealii::VectorizedArray<Number>
SpatialOperatorBase<dim, Number>::get_viscosity_boundary_face(unsigned int const face,
                                                              unsigned int const q) const
{
  if(param.viscosity_is_variable())
  {
    return viscous_kernel->get_coefficient_face(face, q);
  }
  else
  {
    return dealii::make_vectorized_array<Number>(param.viscosity);
  }
}

template<int dim, typename Number>
std::shared_ptr<ContainerInterfaceData<1, dim, double>>
SpatialOperatorBase<dim, Number>::get_container_interface_data()
{
  return interface_data_dirichlet_cached;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::set_velocity_ptr(VectorType const & velocity) const
{
  convective_kernel->set_velocity_ptr(velocity);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_vector_velocity(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index_velocity());
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_vector_velocity_scalar(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index_velocity_scalar());
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_vector_pressure(VectorType & src) const
{
  matrix_free->initialize_dof_vector(src, get_dof_index_pressure());
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_vector_enrichment(VectorType & src) const
{
  AssertThrow(param.wall_enrichment_enabled,
              dealii::ExcMessage("initialize_vector_enrichment requires wall_enrichment_enabled."));
  AssertThrow(function_enrichment != nullptr,
              dealii::ExcMessage("function_enrichment not set up yet. The operator's setup() "
                                 "must run before the time integrator allocates vectors."));

  function_enrichment->get_matrix_free()->initialize_dof_vector(src, function_enrichment->get_dof_index_en());
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::initialize_block_vector_velocity_pressure(
  BlockVectorType & src) const
{
  // velocity (1st block) + pressure (2nd block)
  src.reinit(2);

  matrix_free->initialize_dof_vector(src.block(0), get_dof_index_velocity());
  matrix_free->initialize_dof_vector(src.block(1), get_dof_index_pressure());

  src.collect_sizes();
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::prescribe_initial_conditions(VectorType & velocity,
                                                               VectorType & pressure,
                                                               double const time) const
{
  field_functions->initial_solution_velocity->set_time(time);
  field_functions->initial_solution_pressure->set_time(time);

  // This is necessary if Number == float
  typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble velocity_double;
  VectorTypeDouble pressure_double;
  velocity_double = velocity;
  pressure_double = pressure;

  dealii::VectorTools::interpolate(*get_mapping(),
                                   dof_handler_u,
                                   *(field_functions->initial_solution_velocity),
                                   velocity_double);

  dealii::VectorTools::interpolate(*get_mapping(),
                                   dof_handler_p,
                                   *(field_functions->initial_solution_pressure),
                                   pressure_double);

  velocity = velocity_double;
  pressure = pressure_double;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::interpolate_analytical_solution(VectorType & velocity,
                                                                  VectorType & pressure,
                                                                  double const time) const
{
  AssertThrow(field_functions->analytical_solution_velocity,
              dealii::ExcMessage("FieldFunctions::analytical_solution_velocity not set"));
  AssertThrow(field_functions->analytical_solution_pressure,
              dealii::ExcMessage("FieldFunctions::analytical_solution_pressure not set"));

  field_functions->analytical_solution_velocity->set_time(time);
  field_functions->analytical_solution_pressure->set_time(time);

  // This is necessary if Number == float
  using VectorTypeDouble = dealii::LinearAlgebra::distributed::Vector<double>;

  VectorTypeDouble velocity_double;
  VectorTypeDouble pressure_double;
  velocity_double = velocity;
  pressure_double = pressure;

  dealii::VectorTools::interpolate(*get_mapping(),
                                   get_dof_handler_u(),
                                   *(field_functions->analytical_solution_velocity),
                                   velocity_double);

  dealii::VectorTools::interpolate(*get_mapping(),
                                   get_dof_handler_p(),
                                   *(field_functions->analytical_solution_pressure),
                                   pressure_double);

  velocity = velocity_double;
  pressure = pressure_double;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::interpolate_stress_bc(VectorType &       stress,
                                                        VectorType const & velocity,
                                                        VectorType const & pressure) const
{
  velocity_ptr = &velocity;
  pressure_ptr = &pressure;

  stress = 0.0;

  VectorType src_dummy;
  matrix_free->loop(&This::cell_loop_empty,
                    &This::face_loop_empty,
                    &This::local_interpolate_stress_bc_boundary_face,
                    this,
                    stress,
                    src_dummy);

  velocity_ptr = nullptr;
  pressure_ptr = nullptr;
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_minimum_element_length() const
{
  return calculate_minimum_vertex_distance(dof_handler_u.get_triangulation(),
                                           *get_mapping(),
                                           mpi_comm);
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_time_step_max_efficiency(
  unsigned int const order_time_integrator) const
{
  double const h_min = calculate_minimum_element_length();

  return ExaDG::calculate_time_step_max_efficiency(h_min, param.degree_u, order_time_integrator);
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_time_step_cfl_global() const
{
  std::shared_ptr<dealii::Function<dim>> const velocity_field =
    std::make_shared<dealii::Functions::ConstantFunction<dim>>(param.max_velocity, dim);

  return calculate_time_step_cfl_local<dim, Number>(
    *matrix_free,
    get_dof_index_velocity(),
    get_quad_index_velocity_standard(),
    velocity_field,
    param.start_time /* will not be used (ConstantFunction) */,
    param.degree_u,
    param.cfl_exponent_fe_degree_velocity,
    CFLConditionType::VelocityComponents,
    mpi_comm);
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_time_step_cfl(VectorType const & velocity) const
{
  // Need to update ghost values in the case of continuity constraints.
  if(param.spatial_discretization == SpatialDiscretization::HDIV)
    velocity.update_ghost_values();

  return calculate_time_step_cfl_local<dim, Number>(*matrix_free,
                                                    get_dof_index_velocity(),
                                                    get_quad_index_velocity_standard(),
                                                    velocity,
                                                    param.degree_u,
                                                    param.cfl_exponent_fe_degree_velocity,
                                                    param.adaptive_time_stepping_cfl_type,
                                                    mpi_comm);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::calculate_cfl_from_time_step(VectorType &       cfl,
                                                               VectorType const & velocity,
                                                               double const time_step_size) const
{
  calculate_cfl<dim, Number>(cfl,
                             *grid->triangulation,
                             *matrix_free,
                             get_dof_index_velocity(),
                             get_quad_index_velocity_standard(),
                             velocity,
                             time_step_size,
                             param.degree_u,
                             param.cfl_exponent_fe_degree_velocity);
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::get_characteristic_element_length() const
{
  double const h_min = calculate_minimum_element_length();

  return calculate_high_order_element_length(h_min, param.degree_u, true /* is_dg */);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::apply_mass_operator(VectorType &       dst,
                                                      VectorType const & src) const
{
  mass_operator.apply(dst, src);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::apply_mass_operator_add(VectorType &       dst,
                                                          VectorType const & src) const
{
  mass_operator.apply_add(dst, src);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::apply_mass_operator(BlockVectorType &       dst,
                                                      BlockVectorType const & src) const
{
  dst.block(0).zero_out_ghost_values();
  dst.block(1).zero_out_ghost_values();

  mass_operator.apply(dst.block(0), src.block(0));

  auto const & mf_en = *function_enrichment->get_matrix_free();

  function_enrichment->copy_global_dg_to_wall_layout(src.block(0),
                                                     function_enrichment->shadow_velocity);
  function_enrichment->shadow_velocity.update_ghost_values();
  src.block(1).update_ghost_values();

  VectorType dst_bar_shadow;
  mf_en.initialize_dof_vector(dst_bar_shadow, function_enrichment->get_dof_index_shadow_vector());
  dst_bar_shadow = 0.0;

  dst.block(1) = 0.0;

  using CellIntegratorU = dealii::FEEvaluation<dim, -1, 0, dim, Number>;
  FEEnrichedEvaluation<dim, Number, CellIntegratorU> phi(mf_en, *function_enrichment);

  for(unsigned int cell = 0; cell < mf_en.n_cell_batches(); ++cell)
  {
    if(mf_en.get_cell_iterator(cell, 0)->active_fe_index() != 1)
      continue; // non-enriched batch: nothing to do here

    phi.reinit(cell);
    phi.gather_evaluate(function_enrichment->shadow_velocity,
                        src.block(1),
                        dealii::EvaluationFlags::values);

    for(unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      // u_h = U_bar + psi * U_tilde, tested against both v_bar and psi * v_tilde
      phi.submit_value(phi.get_value(q), q);
    }

    phi.integrate(dealii::EvaluationFlags::values);

    phi.phi_dg.distribute_local_to_global(dst_bar_shadow); // M_bb*Ub + M_bt*Ut
    phi.phi_en.distribute_local_to_global(dst.block(1));   // M_tb*Ub + M_tt*Ut
  }

  dst_bar_shadow.compress(dealii::VectorOperation::add);
  dst.block(1).compress(dealii::VectorOperation::add);

  // (4) Overwrite enriched cells in block(0) with the complete row.
  //     Valid because the DG mass matrix is cell-local (no face terms).
  function_enrichment->copy_wall_layout_to_global_dg(dst_bar_shadow, dst.block(0));

  function_enrichment->shadow_velocity.zero_out_ghost_values();
  src.block(1).zero_out_ghost_values();
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::apply_mass_operator_add(BlockVectorType &       dst,
                                                          BlockVectorType const & src) const
{
  BlockVectorType tmp(dst);
  apply_mass_operator(tmp, src);
  dst.add(1.0, tmp);
}

template<int dim, typename Number>
bool
SpatialOperatorBase<dim, Number>::is_pressure_level_undefined() const
{
  return pressure_level_is_undefined;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::adjust_pressure_level_if_undefined(VectorType &   pressure,
                                                                     double const & time) const
{
  if(is_pressure_level_undefined())
  {
    // If an analytical solution is available: shift pressure so that the numerical pressure
    // solution coincides with the analytical pressure solution in an arbitrary point. Note that the
    // parameter 'time' is only needed for unsteady problems.
    if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
    {
      field_functions->analytical_solution_pressure->set_time(time);
      double const exact = field_functions->analytical_solution_pressure->value(first_point);

      double current = 0.;
      if(pressure.locally_owned_elements().is_element(dof_index_first_point))
        current = pressure(dof_index_first_point);
      current = dealii::Utilities::MPI::sum(current, mpi_comm);

      VectorType vec_temp(pressure);
      for(unsigned int i = 0; i < vec_temp.locally_owned_size(); ++i)
        vec_temp.local_element(i) = 1.;

      pressure.add(exact - current, vec_temp);
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyZeroMeanValue)
    {
      dealii::VectorTools::subtract_mean_value(pressure);
    }
    // If an analytical solution is available: shift pressure so that the numerical pressure
    // solution has a mean value identical to the "exact pressure solution" obtained by
    // interpolation of analytical solution. Note that the parameter 'time' is only needed for
    // unsteady problems.
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalMeanValue)
    {
      // one cannot use Number as template here since Number might be float
      // while analytical_solution_pressure is of type dealii::Function<dim,double>
      typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

      VectorTypeDouble vec_double;
      vec_double = pressure; // initialize

      field_functions->analytical_solution_pressure->set_time(time);
      dealii::VectorTools::interpolate(*get_mapping(),
                                       dof_handler_p,
                                       *(field_functions->analytical_solution_pressure),
                                       vec_double);

      double const exact   = vec_double.mean_value();
      double const current = pressure.mean_value();

      VectorType vec_temp(pressure);
      for(unsigned int i = 0; i < vec_temp.locally_owned_size(); ++i)
        vec_temp.local_element(i) = 1.;

      pressure.add(exact - current, vec_temp);
    }
    else
    {
      AssertThrow(
        false, dealii::ExcMessage("Specified method to adjust pressure level is not implemented."));
    }
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::set_temperature(VectorType const & temperature)
{
  AssertThrow(param.boussinesq_term, dealii::ExcMessage("Invalid parameters detected."));

  rhs_operator.set_temperature(temperature);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_vorticity(VectorType & dst, VectorType const & src) const
{
  vorticity_calculator.compute_vorticity(dst, src);
  this->apply_inverse_mass_operator(dst, dst);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_vorticity(BlockVectorType &       dst,
                                                    BlockVectorType const & src)
const
{ 
  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number>      CellIntegratorU;
  typedef FEEnrichedEvaluation<dim, Number, CellIntegratorU>
  EnrichedCellIntegrator;

  auto const & mf_en = *function_enrichment->get_matrix_free();

  BlockVectorType rhs(dst);
  rhs.block(0).zero_out_ghost_values();
  rhs.block(1).zero_out_ghost_values();

  // (1)
  vorticity_calculator.compute_vorticity(rhs.block(0), src.block(0));

  // (2)
  VectorType u_bar_shadow, rhs_bar_shadow;
  mf_en.initialize_dof_vector(u_bar_shadow,
                              function_enrichment->get_dof_index_shadow_vector());
  mf_en.initialize_dof_vector(rhs_bar_shadow,
                              function_enrichment->get_dof_index_shadow_vector());
  function_enrichment->copy_global_dg_to_wall_layout(src.block(0), u_bar_shadow);
  u_bar_shadow.update_ghost_values();
  src.block(1).update_ghost_values();

  rhs.block(1) = 0.0;

  EnrichedCellIntegrator phi(mf_en, *function_enrichment);
  for(unsigned int cell = 0; cell < mf_en.n_cell_batches(); ++cell)
  {
    if(mf_en.get_cell_iterator(cell, 0)->active_fe_index() != 1)
      continue;

    phi.reinit(cell);
    // values are needed as well: grad u_h contains U_tilde (x) grad psi
    phi.gather_evaluate(u_bar_shadow,
                        src.block(1),
                        dealii::EvaluationFlags::values |
                        dealii::EvaluationFlags::gradients);

    for(unsigned int q = 0; q < phi.n_q_points; ++q)
      phi.submit_value(vorticity_from_gradient<dim, Number>(phi.get_gradient(q)), q);

    phi.integrate(dealii::EvaluationFlags::values);
    phi.phi_dg.distribute_local_to_global(rhs_bar_shadow);
    phi.phi_en.distribute_local_to_global(rhs.block(1));
  }

  rhs_bar_shadow.compress(dealii::VectorOperation::add);
  rhs.block(1).compress(dealii::VectorOperation::add);
  src.block(1).zero_out_ghost_values();

  // enriched cells: overwrite with the complete rows
  function_enrichment->copy_wall_layout_to_global_dg(rhs_bar_shadow,
                                                     rhs.block(0));


  // (3)
  this->apply_inverse_mass_operator(dst, rhs);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_divergence(VectorType & dst, VectorType const & src) const
{
  divergence_calculator.compute_divergence(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_shear_rate(VectorType & dst, VectorType const & src) const
{
  shear_rate_calculator.compute_shear_rate(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_velocity_magnitude(VectorType &       dst,
                                                             VectorType const & src) const
{
  magnitude_calculator.compute(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_vorticity_magnitude(VectorType &       dst,
                                                              VectorType const & src) const
{
  magnitude_calculator.compute(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

/*
 *  Streamfunction psi (2D only): defined as u1 = d(psi)/dx2, u2 = - d(psi)/dx1
 *
 *  Vorticity: omega = du2/dx1 - du1/dx2
 *
 *  --> laplace(psi) = (d²/dx1²+d²/dx2²)(psi)
 *                   = d(d(psi)/dx1)/dx1 + d(d(psi)/dx2)/dx2
 *                   = d(-u2)/dx1 + d(u1)/dx2 = - omega
 *
 *  or
 *      - laplace(psi) = omega
 *
 *  with homogeneous Dirichlet BC's (assumption: whole boundary == streamline)
 */
template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_streamfunction(VectorType &       dst,
                                                         VectorType const & src) const
{
  AssertThrow(dim == 2,
              dealii::ExcMessage("Calculation of streamfunction can only be used for dim==2."));

  // compute rhs vector
  StreamfunctionCalculatorRHSOperator<dim, Number> rhs_operator;
  rhs_operator.initialize(*matrix_free,
                          get_dof_index_velocity(),
                          get_dof_index_velocity_scalar(),
                          get_quad_index_velocity_standard());
  VectorType rhs;
  initialize_vector_velocity_scalar(rhs);
  rhs_operator.apply(rhs, src);

  // setup Laplace operator for scalar velocity vector
  Poisson::LaplaceOperatorData<0, dim> laplace_operator_data;
  laplace_operator_data.dof_index  = get_dof_index_velocity_scalar();
  laplace_operator_data.quad_index = get_quad_index_velocity_standard();

  std::shared_ptr<Poisson::BoundaryDescriptor<0, dim>> boundary_descriptor_streamfunction;
  boundary_descriptor_streamfunction = std::make_shared<Poisson::BoundaryDescriptor<0, dim>>();

  // fill boundary descriptor: Assumption: only Dirichlet BC's
  boundary_descriptor_streamfunction->dirichlet_bc = boundary_descriptor->velocity->dirichlet_bc;

  AssertThrow(boundary_descriptor->velocity->neumann_bc.empty() == true,
              dealii::ExcMessage("Assumption is not fulfilled. Streamfunction calculator is "
                                 "not implemented for this type of boundary conditions."));
  AssertThrow(boundary_descriptor->velocity->symmetry_bc.empty() == true,
              dealii::ExcMessage("Assumption is not fulfilled. Streamfunction calculator is "
                                 "not implemented for this type of boundary conditions."));

  laplace_operator_data.bc = boundary_descriptor_streamfunction;

  laplace_operator_data.kernel_data.IP_factor = 1.0;

  typedef Poisson::LaplaceOperator<dim, Number, 1> Laplace;
  Laplace                                          laplace_operator;
  dealii::AffineConstraints<Number>                constraint_dummy;
  laplace_operator.initialize(*matrix_free, constraint_dummy, laplace_operator_data);

  // setup preconditioner
  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  // use multigrid preconditioner with Chebyshev smoother
  MultigridData mg_data;

  preconditioner = std::make_shared<MultigridPoisson>(mpi_comm);

  std::shared_ptr<MultigridPoisson> mg_preconditioner =
    std::dynamic_pointer_cast<MultigridPoisson>(preconditioner);

  typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
  Map_DBC_ComponentMask                                               dirichlet_bc_component_mask;

  mg_preconditioner->initialize(mg_data,
                                grid,
                                multigrid_mappings,
                                dof_handler_u_scalar.get_fe(),
                                laplace_operator.get_data(),
                                param.ale_formulation,
                                laplace_operator.get_data().bc->dirichlet_bc,
                                dirichlet_bc_component_mask);

  // setup solver
  Krylov::SolverDataCG solver_data;
  solver_data.solver_tolerance_rel = 1.e-10;
  solver_data.use_preconditioner   = true;

  Krylov::SolverCG<Laplace, PreconditionerBase<Number>, VectorType> poisson_solver(laplace_operator,
                                                                                   *preconditioner,
                                                                                   solver_data);

  // solve Poisson problem
  poisson_solver.solve(dst, rhs);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_q_criterion(VectorType &       dst,
                                                      VectorType const & src) const
{
  q_criterion_calculator.compute(dst, src);

  inverse_mass_velocity_scalar.apply(dst, dst);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::apply_inverse_mass_operator(VectorType &       dst,
                                                              VectorType const & src) const
{
  if(param.spatial_discretization == SpatialDiscretization::L2)
  {
    inverse_mass_velocity.apply(dst, src);
  }
  else if(param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    inverse_mass_hdiv.apply(dst, src);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  return 0;
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::apply_inverse_mass_operator(BlockVectorType &       dst,
                                                              BlockVectorType const & src) const
{
  dst.block(0).zero_out_ghost_values();
  dst.block(1).zero_out_ghost_values();
  AssertThrow(param.wall_enrichment_enabled,
              dealii::ExcMessage("Block inverse mass operator requires wall_enrichment_enabled."));
  AssertThrow(param.spatial_discretization == SpatialDiscretization::L2,
              dealii::ExcMessage("Wall enrichment is only implemented for L2 (DG) velocity."));

  // (1) Copy the enriched parts of src FIRST. dst and src may be the same vector
  //     (e.g. the convective step calls M^{-1} in place), so src must not be read
  //     after dst has been written.
  function_enrichment->copy_global_dg_to_wall_layout(src.block(0),
                                                     function_enrichment->shadow_velocity_residual);
  VectorType src_tilde(src.block(1));

  // (2) Whole domain, U_bar only: plain inverse mass.
  //     Correct on non-enriched cells; overwritten on enriched cells in step (4).
  inverse_mass_velocity.apply(dst.block(0), src.block(0));

  // (3) Enriched cells: exact 2x2 inverse via Schur complement (Eq. 6.16-6.17).
  function_enrichment->shadow_velocity = 0.0;
  dst.block(1)                         = 0.0;

  wall_dg_coupler->apply_schur_inverse_mass(function_enrichment->shadow_velocity,          // U_bar  (out)
                                            function_enrichment->shadow_velocity_residual, // R_bar  (in)
                                            dst.block(1),                                  // U_tilde(out)
                                            src_tilde);                                    // R_tilde(in)

  // (4) Overwrite enriched cells of block(0) with the coupled result.
  function_enrichment->copy_wall_layout_to_global_dg(function_enrichment->shadow_velocity,
                                                     dst.block(0));

  return 0;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_add_body_force_term(VectorType & dst,
                                                               double const time) const
{
  this->rhs_operator.evaluate_add(dst, time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_convective_term(VectorType &       dst,
                                                           VectorType const & src,
                                                           Number const       time) const
{
  convective_operator.evaluate_nonlinear_operator(dst, src, time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_convective_term_enriched(BlockVectorType &       dst,
                                                                    BlockVectorType const & src,
                                                                    double const evaluation_time) const
{
  auto const & mf_en = *function_enrichment->get_matrix_free();

  dst.block(0).zero_out_ghost_values();
  dst.block(1).zero_out_ghost_values();

  // (1) standard operator on U_bar, whole domain
  evaluate_convective_term(dst.block(0), src.block(0), evaluation_time);

  // (2) enriched rows (complete), halo corrections and U_tilde rows on matrix_free_en
  VectorType u_bar_shadow, bar_enriched_shadow, bar_halo_shadow;
  mf_en.initialize_dof_vector(u_bar_shadow, function_enrichment->get_dof_index_shadow_vector());
  mf_en.initialize_dof_vector(bar_enriched_shadow, function_enrichment->get_dof_index_shadow_vector());
  mf_en.initialize_dof_vector(bar_halo_shadow, function_enrichment->get_dof_index_shadow_vector());

  function_enrichment->copy_global_dg_to_wall_layout(src.block(0), u_bar_shadow);
  u_bar_shadow.update_ghost_values();
  src.block(1).update_ghost_values();

  convective_operator.evaluate_nonlinear_operator_enriched(bar_enriched_shadow,
                                                           bar_halo_shadow,
                                                           dst.block(1),
                                                           u_bar_shadow,
                                                           src.block(1),
                                                           *function_enrichment,
                                                           evaluation_time);

  src.block(1).zero_out_ghost_values();

  // enriched cells: OVERWRITE with the complete rows (same quadrature as the U_tilde rows)
  function_enrichment->copy_wall_layout_to_global_dg(bar_enriched_shadow, dst.block(0));
  // halo cells: ADD the correction of their faces to enriched cells
  function_enrichment->add_wall_layout_to_global_dg(bar_halo_shadow, dst.block(0));
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_pressure_gradient_term(VectorType &       dst,
                                                                  VectorType const & src,
                                                                  double const       time) const
{
  gradient_operator.evaluate(dst, src, time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_velocity_divergence_term(VectorType &       dst,
                                                                    VectorType const & src,
                                                                    double const       time) const
{
  divergence_operator.evaluate(dst, src, time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::update_viscosity(VectorType const & velocity) const
{
  AssertThrow(param.viscous_problem() and param.viscosity_is_variable(),
              dealii::ExcMessage(
                "Updating viscosity reasonable for variable viscosity models only."));

  // reset the viscosity stored
  // viscosity = viscosity_newtonian_limit
  viscous_kernel->set_constant_coefficient(viscous_kernel_data.viscosity);

  // add contribution from generalized Newtonian model
  // viscosity += generalized_newtonian_viscosity(viscosity_newtonian_limit)
  if(param.generalized_newtonian_model_data.is_active)
    generalized_newtonian_model.add_viscosity(velocity);

  // add contribution from turbulence model
  // viscosity += turbulent_viscosity(viscosity)
  // note that the apparent viscosity is used to compute the turbulent viscosity, such that the
  // *sequence of calls matters*, i.e., we can only compute the turbulent viscosity once the laminar
  // viscosity has been computed
  if(param.turbulence_model_data.is_active)
    turbulence_model.add_viscosity(velocity);
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_dissipation_convective_term(VectorType const & velocity,
                                                                        double const time) const
{
  if(param.convective_problem())
  {
    VectorType dst;
    dst.reinit(velocity, false);
    convective_operator.evaluate_nonlinear_operator(dst, velocity, time);
    return velocity * dst;
  }
  else
  {
    return 0.0;
  }
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_dissipation_viscous_term(
  VectorType const & velocity) const
{
  if(param.viscous_problem())
  {
    VectorType dst;
    dst.reinit(velocity, false);
    viscous_operator.apply(dst, velocity);
    return velocity * dst;
  }
  else
  {
    return 0.0;
  }
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_dissipation_divergence_term(
  VectorType const & velocity) const
{
  if(param.use_divergence_penalty == true)
  {
    VectorType dst;
    dst.reinit(velocity, false);
    div_penalty_operator.apply(dst, velocity);
    return velocity * dst;
  }
  else
  {
    return 0.0;
  }
}

template<int dim, typename Number>
double
SpatialOperatorBase<dim, Number>::calculate_dissipation_continuity_term(
  VectorType const & velocity) const
{
  if(param.use_continuity_penalty == true)
  {
    VectorType dst;
    dst.reinit(velocity, false);
    conti_penalty_operator.apply(dst, velocity);
    return velocity * dst;
  }
  else
  {
    return 0.0;
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::update_after_grid_motion(bool const update_matrix_free)
{
  if(update_matrix_free)
  {
    // Since matrix_free points to matrix_free_own_storage, we also update the actual/main
    // MatrixFree object called matrix_free.
    matrix_free_own_storage->update_mapping(*get_mapping());
  }

  if(param.turbulence_model_data.is_active)
  {
    // the mesh (and hence the filter width) changes in case of an ALE formulation
    turbulence_model.calculate_filter_width(*get_mapping());
  }

  if(this->param.viscous_problem())
  {
    // update SIPG penalty parameter of viscous operator which depends on the deformation
    // of elements
    viscous_kernel->calculate_penalty_parameter(*matrix_free, get_dof_index_velocity());
  }

  // The inverse mass operator might contain matrix-based components, in which cases it needs to be
  // updated after the grid has been deformed.
  inverse_mass_velocity.update();
  inverse_mass_velocity_scalar.update();

  // note that the update of div-div and continuity penalty terms is done separately
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::set_grid_velocity(VectorType const & u_grid_in)
{
  convective_kernel->set_grid_velocity_ptr(u_grid_in);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::setup_projection_solver()
{
  // setup projection solver

  // divergence penalty only -> local, elementwise problem
  if(param.use_divergence_penalty == true and param.use_continuity_penalty == false)
  {
    // elementwise operator
    elementwise_projection_operator =
      std::make_shared<ELEMENTWISE_PROJ_OPERATOR>(*projection_operator);

    // elementwise preconditioner
    if(param.preconditioner_projection == PreconditionerProjection::None)
    {
      typedef Elementwise::PreconditionerIdentity<dealii::VectorizedArray<Number>> IDENTITY;

      elementwise_preconditioner_projection =
        std::make_shared<IDENTITY>(elementwise_projection_operator->get_problem_size());
    }
    else if(param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
    {
      typedef Elementwise::InverseMassPreconditioner<dim, dim, Number> INVERSE_MASS;

      elementwise_preconditioner_projection =
        std::make_shared<INVERSE_MASS>(projection_operator->get_matrix_free(),
                                       projection_operator->get_dof_index(),
                                       projection_operator->get_quad_index());
    }
    else if(param.preconditioner_projection == PreconditionerProjection::PointJacobi)
    {
      typedef Elementwise::JacobiPreconditioner<dim, dim, Number, ProjOperator> JACOBI;

      elementwise_preconditioner_projection =
        std::make_shared<JACOBI>(projection_operator->get_matrix_free(),
                                 projection_operator->get_dof_index(),
                                 projection_operator->get_quad_index(),
                                 *projection_operator,
                                 false);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("The specified preconditioner is not implemented."));
    }

    // elementwise solver
    if(param.solver_projection == SolverProjection::CG)
    {
      Elementwise::IterativeSolverData projection_solver_data;
      projection_solver_data.solver_type         = Elementwise::Solver::CG;
      projection_solver_data.solver_data.abs_tol = param.solver_data_projection.abs_tol;
      projection_solver_data.solver_data.rel_tol = param.solver_data_projection.rel_tol;

      typedef Elementwise::PreconditionerBase<dealii::VectorizedArray<Number>> PROJ_PRECONDITIONER;

      typedef Elementwise::
        IterativeSolver<dim, dim, Number, ELEMENTWISE_PROJ_OPERATOR, PROJ_PRECONDITIONER>
          PROJ_SOLVER;

      projection_solver = std::make_shared<PROJ_SOLVER>(
        *std::dynamic_pointer_cast<ELEMENTWISE_PROJ_OPERATOR>(elementwise_projection_operator),
        *std::dynamic_pointer_cast<PROJ_PRECONDITIONER>(elementwise_preconditioner_projection),
        projection_solver_data);
    }
    else
    {
      AssertThrow(param.solver_projection == SolverProjection::CG,
                  dealii::ExcMessage("Specified projection solver not implemented."));
    }
  }
  // continuity penalty term with/without divergence penalty term -> globally coupled problem
  else if(param.use_continuity_penalty == true)
  {
    // preconditioner
    if(param.preconditioner_projection == PreconditionerProjection::None)
    {
      // do nothing, preconditioner will not be used
    }
    else if(param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
    {
      InverseMassOperatorData inverse_mass_operator_data;
      inverse_mass_operator_data.dof_index  = get_dof_index_velocity();
      inverse_mass_operator_data.quad_index = get_quad_index_velocity_standard();
      inverse_mass_operator_data.parameters = param.inverse_mass_preconditioner;

      preconditioner_projection =
        std::make_shared<InverseMassPreconditioner<dim, dim, Number>>(*matrix_free,
                                                                      inverse_mass_operator_data);
    }
    else if(param.preconditioner_projection == PreconditionerProjection::PointJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner and calculating the
      // diagonal) the penalty parameter of the projection operator has not been calculated and the
      // time step size has not been set. Hence, 'update_preconditioner = true' should be used for
      // the Jacobi preconditioner in order to use to correct diagonal for preconditioning.
      preconditioner_projection =
        std::make_shared<JacobiPreconditioner<ProjOperator>>(*projection_operator, false);
    }
    else if(param.preconditioner_projection == PreconditionerProjection::BlockJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner)
      // the penalty parameter of the projection operator has not been calculated and the time step
      // size has not been set. Hence, 'update_preconditioner = true' should be used for the Jacobi
      // preconditioner in order to use to correct diagonal blocks for preconditioning.
      preconditioner_projection =
        std::make_shared<BlockJacobiPreconditioner<ProjOperator>>(*projection_operator, false);
    }
    else if(param.preconditioner_projection == PreconditionerProjection::Multigrid)
    {
      typedef MultigridPreconditionerProjection<dim, Number> Multigrid;

      preconditioner_projection = std::make_shared<Multigrid>(this->mpi_comm);

      std::shared_ptr<Multigrid> mg_preconditioner =
        std::dynamic_pointer_cast<Multigrid>(preconditioner_projection);

      typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        pair;

      std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        dirichlet_boundary_conditions = this->projection_operator->get_data().bc->dirichlet_bc;

      // We also need to add DirichletCached boundary conditions. From the
      // perspective of multigrid, there is no difference between standard
      // and cached Dirichlet BCs. Since multigrid does not need information
      // about inhomogeneous boundary data, we simply fill the map with
      // dealii::Functions::ZeroFunction for DirichletCached BCs.
      for(auto iter : this->projection_operator->get_data().bc->dirichlet_cached_bc)
        dirichlet_boundary_conditions.insert(
          pair(iter, new dealii::Functions::ZeroFunction<dim>(dim)));

      typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;
      Map_DBC_ComponentMask dirichlet_bc_component_mask;

      auto const & dof_handler = this->get_dof_handler_u();
      mg_preconditioner->initialize(this->param.multigrid_data_projection,
                                    grid,
                                    multigrid_mappings,
                                    dof_handler.get_fe(),
                                    *this->projection_operator,
                                    this->param.ale_formulation,
                                    dirichlet_boundary_conditions,
                                    dirichlet_bc_component_mask);
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "Preconditioner specified for projection step is not implemented."));
    }

    // solver
    if(param.solver_projection == SolverProjection::CG)
    {
      // setup solver data
      Krylov::SolverDataCG solver_data;
      solver_data.max_iter             = param.solver_data_projection.max_iter;
      solver_data.solver_tolerance_abs = param.solver_data_projection.abs_tol;
      solver_data.solver_tolerance_rel = param.solver_data_projection.rel_tol;
      // default value of use_preconditioner = false
      if(param.preconditioner_projection != PreconditionerProjection::None)
      {
        solver_data.use_preconditioner = true;
      }

      // setup solver
      projection_solver =
        std::make_shared<Krylov::SolverCG<ProjOperator, PreconditionerBase<Number>, VectorType>>(
          *projection_operator, *preconditioner_projection, solver_data);
    }
    else if(param.solver_projection == SolverProjection::FGMRES)
    {
      // setup solver data
      Krylov::SolverDataFGMRES solver_data;
      solver_data.max_iter             = param.solver_data_projection.max_iter;
      solver_data.solver_tolerance_abs = param.solver_data_projection.abs_tol;
      solver_data.solver_tolerance_rel = param.solver_data_projection.rel_tol;
      solver_data.max_n_tmp_vectors    = param.solver_data_projection.max_krylov_size;

      // default value of use_preconditioner = false
      if(param.preconditioner_projection != PreconditionerProjection::None)
      {
        solver_data.use_preconditioner = true;
      }

      // setup solver
      projection_solver = std::make_shared<
        Krylov::SolverFGMRES<ProjOperator, PreconditionerBase<Number>, VectorType>>(
        *projection_operator, *preconditioner_projection, solver_data);
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Specified projection solver not implemented."));
    }
  }
  else
  {
    AssertThrow(
      param.use_divergence_penalty == false and param.use_continuity_penalty == false,
      dealii::ExcMessage(
        "Specified combination of divergence and continuity penalty operators not implemented."));
  }
}

template<int dim, typename Number>
bool
SpatialOperatorBase<dim, Number>::unsteady_problem_has_to_be_solved() const
{
  return (this->param.solver_type == SolverType::Unsteady);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::update_projection_operator(VectorType const & velocity,
                                                             double const time_step_size) const
{
  AssertThrow(projection_operator.get() != 0,
              dealii::ExcMessage("Projection operator is not initialized."));

  // Update projection operator, i.e., the penalty parameters that depend on the velocity field
  // and the time step size
  projection_operator->update(velocity, time_step_size);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::rhs_add_projection_operator(VectorType & dst,
                                                              double const time) const
{
  projection_operator->set_time(time);
  projection_operator->rhs_add(dst);
}

template<int dim, typename Number>
unsigned int
SpatialOperatorBase<dim, Number>::solve_projection(VectorType &       dst,
                                                   VectorType const & src,
                                                   bool const &       update_preconditioner) const
{
  Assert(projection_solver.get() != 0,
         dealii::ExcMessage("Projection solver has not been initialized."));

  projection_solver->update_preconditioner(update_preconditioner);

  unsigned int n_iter = projection_solver->solve(dst, src);

  return n_iter;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::local_interpolate_stress_bc_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int const dof_index_u = this->get_dof_index_velocity();
  unsigned int const dof_index_p = this->get_dof_index_pressure();
  unsigned int const quad_index  = this->get_quad_index_velocity_gauss_lobatto();

  FaceIntegratorU integrator_u(matrix_free, true, dof_index_u, quad_index);
  FaceIntegratorP integrator_p(matrix_free, true, dof_index_p, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    dealii::types::boundary_id const boundary_id = matrix_free.get_boundary_id(face);

    BoundaryTypeU const boundary_type =
      this->boundary_descriptor->velocity->get_boundary_type(boundary_id);

    // a Dirichlet boundary for the fluid is a stress boundary for the structure
    if(boundary_type == BoundaryTypeU::DirichletCached)
    {
      integrator_u.reinit(face);
      integrator_u.gather_evaluate(*velocity_ptr, dealii::EvaluationFlags::gradients);

      integrator_p.reinit(face);
      integrator_p.gather_evaluate(*pressure_ptr, dealii::EvaluationFlags::values);

      for(unsigned int q = 0; q < integrator_u.n_q_points; ++q)
      {
        unsigned int const local_face_number = matrix_free.get_face_info(face).interior_face_no;

        unsigned int const index = matrix_free.get_shape_info(dof_index_u, quad_index)
                                     .face_to_cell_index_nodal[local_face_number][q];

        // compute traction acting on structure with normal vector in opposite direction
        // as compared to the fluid domain
        vector normal = integrator_u.get_normal_vector(q);
        tensor grad_u = integrator_u.get_gradient(q);
        scalar p      = integrator_p.get_value(q);

        scalar viscosity = get_viscosity_boundary_face(face, q);

        // incompressible flow solver is formulated in terms of kinematic viscosity and kinematic
        // pressure
        // -> multiply by density to get true traction in N/m^2.
        vector traction =
          param.density * (viscosity * (grad_u + transpose(grad_u)) * normal - p * normal);

        integrator_u.submit_dof_value(traction, index);
      }

      integrator_u.set_dof_values(dst);
    }
    else
    {
      AssertThrow(boundary_type == BoundaryTypeU::Dirichlet or
                    boundary_type == BoundaryTypeU::Neumann or
                    boundary_type == BoundaryTypeU::Symmetry,
                  dealii::ExcMessage("BoundaryTypeU not implemented."));
    }
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::distribute_constraint_u(VectorType & velocity) const
{
  if(param.spatial_discretization == SpatialDiscretization::HDIV)
  {
    constraint_u.distribute(velocity);
  }
  else if(param.spatial_discretization == SpatialDiscretization::L2)
  {
    // Do nothing
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::set_eddy_viscosity(VectorType const & eddy_viscosity_in)
{
  turbulence_model.set_eddy_viscosity(eddy_viscosity_in);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::set_turbulent_kinetic_energy(VectorType const & tke_in)
{
  this->tke = &tke_in;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::update_wall_enrichment_vectors(VectorType const & velocity) const
{
  if(param.wall_enrichment_enabled)
  {
    wall_dg_coupler->update_wall_enrichment_vectors(velocity);
  }
  else{
    AssertThrow(false, dealii::ExcMessage("update_wall_enrichment_vectors is only necessary when wall_enrichment_enabled is true"));
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::precompute_schur_matrices() const
{
  if(param.wall_enrichment_enabled)
  {
    wall_dg_coupler->precompute_schur_matrices();
  }
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::set_constant_friction_velocity(Number const u_tau) const
{
  AssertThrow(param.wall_enrichment_enabled,
              dealii::ExcMessage("set_constant_friction_velocity requires wall_enrichment_enabled."));

  function_enrichment->friction_velocity = u_tau;
  function_enrichment->friction_velocity.update_ghost_values();

  function_enrichment->wall_shear_stress = u_tau * u_tau;
  function_enrichment->wall_shear_stress.update_ghost_values();

  function_enrichment->update_enrichment_function();

  wall_dg_coupler->precompute_schur_matrices();
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperatorBase<dim, Number>::get_dof_handler_en_cg_scalar() const
{
  return function_enrichment->get_dof_handler_cg_scalar();
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperatorBase<dim, Number>::get_dof_handler_en_scalar() const
{
  return function_enrichment->get_dof_handler_en_scalar();
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperatorBase<dim, Number>::get_dof_handler_en_vector() const
{
  return function_enrichment->get_dof_handler_en_vector();
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
SpatialOperatorBase<dim, Number>::get_dof_handler_en_shadow_vector() const
{
  return function_enrichment->get_dof_handler_shadow_vector();
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
SpatialOperatorBase<dim, Number>::get_friction_velocity() const
{
  return function_enrichment->friction_velocity;
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
SpatialOperatorBase<dim, Number>::get_wall_distance() const
{
  return function_enrichment->wall_distance;
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
SpatialOperatorBase<dim, Number>::get_enrichment_velocity() const
{
  return wall_dg_coupler->get_enrichment_velocity();
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::get_enrichment_component(
  VectorType & dst,
  dealii::DoFHandler<dim> const & dof_handler_out) const
{
  AssertThrow(param.wall_enrichment_enabled,
              dealii::ExcMessage("compute_enrichment_component requires wall_enrichment_enabled."));

  function_enrichment->compute_enrichment_component(dst,
                                                    get_enrichment_velocity(),
                                                    dof_handler_out,
                                                    *get_mapping());
}

template<int dim, typename Number>
std::shared_ptr<FunctionEnrichment<dim, Number>> const &
SpatialOperatorBase<dim, Number>::get_function_enrichment() const
{
  return function_enrichment;
}

template<int dim, typename Number>
std::shared_ptr<WallDGCoupler<dim, Number>> const &
SpatialOperatorBase<dim, Number>::get_wall_dg_coupler() const
{
  return wall_dg_coupler;
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
SpatialOperatorBase<dim, Number>::get_shadow_velocity() const
{
  return wall_dg_coupler->get_shadow_velocity();
}

template<int dim, typename Number>
bool
SpatialOperatorBase<dim, Number>::get_wall_enrichment_enabled() const
{
  return param.wall_enrichment_enabled;
}

template<int dim, typename Number>
Number
SpatialOperatorBase<dim, Number>::get_kinematic_viscosity() const
{
  return param.viscosity;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::reinit_enrichment_residual()
{
  function_enrichment->enrichment_residual = 0.0;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::update_schur_matrices() const
{
  wall_dg_coupler->precompute_schur_matrices();
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::compute_total_velocity(VectorType &            dst,
                                                         BlockVectorType const & src) const
{
  // block(0) of the block mass = M_bb U_bar + M_bt U_tilde  (= int v_bar . u_h)
  BlockVectorType mass_u(src);
  apply_mass_operator(mass_u, src);

  // plain DG inverse mass -> best polynomial approximation of u_h
  apply_inverse_mass_operator(dst, mass_u.block(0));
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::update_friction_velocity_from_residual(BlockVectorType const & u,
                                                                         BlockVectorType const & u_hat,
                                                                         double const mass_factor) const
{
  function_enrichment->evaluate_friction_velocity_from_residual(
    u.block(0), u.block(1), u_hat.block(0), u_hat.block(1), mass_factor);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::project_enrichment_history(BlockVectorType & u) const
{
  auto const & mf_en = *function_enrichment->get_matrix_free();

  // right-hand side, block(0) = int v_bar . u_h^old, block(1) = int psi_new v_tilde . u_h^old
  BlockVectorType rhs(u);
  rhs.block(0).zero_out_ghost_values();
  rhs.block(1).zero_out_ghost_values();

  // (1) non-enriched cells: standard mass, u_h^old = U_bar there
  mass_operator.apply(rhs.block(0), u.block(0));

  // (2) enriched cells: trial function with psi_old, test function with psi_new
  VectorType u_bar_shadow, rhs_bar_shadow;
  mf_en.initialize_dof_vector(u_bar_shadow, function_enrichment->get_dof_index_shadow_vector());
  mf_en.initialize_dof_vector(rhs_bar_shadow, function_enrichment->get_dof_index_shadow_vector());

  function_enrichment->copy_global_dg_to_wall_layout(u.block(0), u_bar_shadow);
  u_bar_shadow.update_ghost_values();
  u.block(1).update_ghost_values();

  function_enrichment->apply_mass_old_to_new(rhs_bar_shadow, rhs.block(1), u_bar_shadow, u.block(1));

  u.block(1).zero_out_ghost_values();

  // overwrite the enriched cells of block(0) (mass is cell-local)
  function_enrichment->copy_wall_layout_to_global_dg(rhs_bar_shadow, rhs.block(0));

  // (3) inverse block mass with psi_new
  apply_inverse_mass_operator(u, rhs);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_add_body_force_term_enriched(BlockVectorType & dst,
                                                                        double const      time) const
{
  auto const & mf_en = *function_enrichment->get_matrix_free();

  VectorType f_bar;
  initialize_vector_velocity(f_bar);
  rhs_operator.evaluate(f_bar, time);

  VectorType f_bar_enriched_shadow, f_tilde;
  mf_en.initialize_dof_vector(f_bar_enriched_shadow,
                              function_enrichment->get_dof_index_shadow_vector());
  mf_en.initialize_dof_vector(f_tilde, function_enrichment->get_dof_index_en());

  rhs_operator.evaluate_add_enriched(f_bar_enriched_shadow, f_tilde, *function_enrichment, time);

  function_enrichment->copy_wall_layout_to_global_dg(f_bar_enriched_shadow, f_bar);

  dst.block(0) += f_bar;
  dst.block(1) += f_tilde;
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::apply_linearized_convective_term_enriched(
  BlockVectorType &       dst,
  BlockVectorType const & delta_u,
  BlockVectorType const & linearization,
  double const            time) const
{
  auto const & mf_en = *function_enrichment->get_matrix_free();

  dst.block(0).zero_out_ghost_values();
  dst.block(1).zero_out_ghost_values();

  // (1) ExaDG's linearized operator c'(W_bar) delta_U_bar, whole domain (the linearization point
  //     W_bar of the ConvectiveKernel is set by the caller)
  convective_operator.set_time(time);
  convective_operator.apply(dst.block(0), delta_u.block(0));

  // (2) enriched rows (complete), halo corrections and U_tilde rows on matrix_free_en
  VectorType w_bar_shadow, delta_u_bar_shadow, bar_enriched_shadow, bar_halo_shadow;
  unsigned int const dof_shadow = function_enrichment->get_dof_index_shadow_vector();
  mf_en.initialize_dof_vector(w_bar_shadow, dof_shadow);
  mf_en.initialize_dof_vector(delta_u_bar_shadow, dof_shadow);
  mf_en.initialize_dof_vector(bar_enriched_shadow, dof_shadow);
  mf_en.initialize_dof_vector(bar_halo_shadow, dof_shadow);

  function_enrichment->copy_global_dg_to_wall_layout(linearization.block(0), w_bar_shadow);
  function_enrichment->copy_global_dg_to_wall_layout(delta_u.block(0), delta_u_bar_shadow);
  w_bar_shadow.update_ghost_values();
  delta_u_bar_shadow.update_ghost_values();
  linearization.block(1).update_ghost_values();
  delta_u.block(1).update_ghost_values();

  convective_operator.apply_linearized_operator_enriched(bar_enriched_shadow,
                                                         bar_halo_shadow,
                                                         dst.block(1),
                                                         w_bar_shadow,
                                                         linearization.block(1),
                                                         delta_u_bar_shadow,
                                                         delta_u.block(1),
                                                         *function_enrichment,
                                                         time);

  linearization.block(1).zero_out_ghost_values();
  delta_u.block(1).zero_out_ghost_values();

  // enriched cells: OVERWRITE with the complete rows (same quadrature as the U_tilde rows)
  function_enrichment->copy_wall_layout_to_global_dg(bar_enriched_shadow, dst.block(0));
  // halo cells: ADD the correction of their faces to enriched cells
  function_enrichment->add_wall_layout_to_global_dg(bar_halo_shadow, dst.block(0));
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::setup_wall_enrichment_solvers(
  MomentumOperator<dim, Number> const & momentum_operator_mass_viscous)
{
  AssertThrow(param.wall_enrichment_enabled,
              dealii::ExcMessage("setup_wall_enrichment_solvers() requires wall enrichment."));

  // mass + viscous operator for (U_bar, U_tilde); the viscous wall condition is weak (Nitsche,
  // penalty scaled by wall_penalty_factor_viscous) or strong (fixed U_bar DoFs, E_bar)
  enriched_momentum_operator.initialize(momentum_operator_mass_viscous, function_enrichment);
  enriched_momentum_operator.set_enrichment_aware_penalty(param.use_enrichment_aware_penalty);
  enriched_momentum_operator.set_wall_penalty_factor(
    param.wall_bc_viscous == WallBoundaryConditionViscous::Weak ? param.wall_penalty_factor_viscous :
                                                                  1.0);

  if(param.wall_bc_viscous == WallBoundaryConditionViscous::Strong)
    setup_strong_wall_dofs();

  enriched_momentum_preconditioner.initialize(*this);
  enriched_block_jacobi_preconditioner.initialize(*this);

  // penalty step operator (requires ExaDG's projection operator for the U_bar-U_bar block)
  if(param.use_divergence_penalty or param.use_continuity_penalty)
  {
    AssertThrow(projection_operator.get() != nullptr,
                dealii::ExcMessage("Wall enrichment: the projection operator has to be set up "
                                   "before the enriched penalty operator."));

    enriched_projection_operator.initialize(*projection_operator,
                                            function_enrichment,
                                            param.degree_u,
                                            param.use_divergence_penalty,
                                            param.use_continuity_penalty,
                                            param.divergence_penalty_factor,
                                            param.continuity_penalty_factor);
  }
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
SpatialOperatorBase<dim, Number>::setup_strong_wall_dofs()
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
void
SpatialOperatorBase<dim, Number>::rhs_add_enriched_viscous_boundary_data(VectorType & dst_bar,
                                                                              VectorType & dst_tilde,
                                                                              double const time) const
{
  enriched_momentum_operator.update_enriched_penalty_parameter();
  enriched_momentum_operator.rhs_add_boundary_data(dst_bar, dst_tilde, time);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::update_projection_operator_enriched(
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
SpatialOperatorBase<dim, Number>::solve_projection_enriched(BlockVectorType &       dst,
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

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::apply_velocity_divergence_term(VectorType &       dst,
                                                                   VectorType const & src) const
{
  this->divergence_operator.apply(dst, src);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::apply_velocity_divergence_term(
  VectorType &            dst,
  BlockVectorType const & src) const
{
  // (1) D(q, U_bar) on the whole domain: existing ExaDG operator
  this->divergence_operator.apply(dst, src.block(0));

  AssertThrow(this->param.divu_integrated_by_parts and
                this->param.divu_formulation == FormulationVelocityDivergenceTerm::Weak,
              dealii::ExcMessage("Enriched divergence: only weak formulation with "
                                 "integration by parts is implemented."));

  // (2) D(q, psi*U_tilde) on enriched cells and faces touching them
  auto const & fe_en = this->get_function_enrichment();
  auto const & mf_en = *fe_en->get_matrix_free();

  unsigned int const dof_p = fe_en->get_dof_index_shadow_pressure();
  unsigned int const quad  = fe_en->get_quad_index();

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number>     CellIntU;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, dim, Number> FaceIntU;
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number>       CellIntP;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number>   FaceIntP;

  // U_bar = 0, so the enriched evaluators return exactly psi * U_tilde
  VectorType zero_bar;
  mf_en.initialize_dof_vector(zero_bar, fe_en->get_dof_index_shadow_vector());
  zero_bar.update_ghost_values();
  src.block(1).update_ghost_values();

  VectorType correction_p;
  mf_en.initialize_dof_vector(correction_p, dof_p);

  // --- cell term: -int grad(q) . (psi U_tilde) ------------------------------
  {
    FEEnrichedEvaluation<dim, Number, CellIntU> phi_u(mf_en, *fe_en);
    CellIntP                                    phi_p(mf_en, dof_p, quad, 0, 1);

    for(unsigned int cell = 0; cell < mf_en.n_cell_batches(); ++cell)
    {
      if(mf_en.get_cell_iterator(cell, 0)->active_fe_index() != 1)
        continue;

      phi_u.reinit(cell);
      phi_u.gather_evaluate(zero_bar, src.block(1), dealii::EvaluationFlags::values);

      phi_p.reinit(cell);
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q)
        phi_p.submit_gradient(-phi_u.get_value(q), q);
      phi_p.integrate(dealii::EvaluationFlags::gradients);
      phi_p.distribute_local_to_global(correction_p);
    }
  }

  // psi*U_tilde on one side of a face; zero if that side is not enriched (halo)
  auto const eval_side = [&](bool const interior, unsigned int const fe_index, unsigned int const face,
                             unsigned int const n_q, std::vector<dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>> & vals) {
    vals.assign(n_q, dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>());
    if(fe_index != 1)
      return;
    FEFaceEnrichedEvaluation<dim, Number, FaceIntU> phi(mf_en, *fe_en, interior, 1);
    phi.reinit(face);
    phi.gather_evaluate(zero_bar, src.block(1), dealii::EvaluationFlags::values);
    for(unsigned int q = 0; q < n_q; ++q)
      vals[q] = phi.get_value(q);
  };

  std::vector<dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>> u_m, u_p;

  // --- interior faces: int [q] * avg(psi U_tilde) . n ------------------------
  for(unsigned int face = 0; face < mf_en.n_inner_face_batches(); ++face)
  {
    auto const         range = std::make_pair(face, face + 1);
    unsigned int const fe_m  = mf_en.get_face_active_fe_index(range, true);
    unsigned int const fe_p  = mf_en.get_face_active_fe_index(range, false);

    if(fe_m != 1 and fe_p != 1)
      continue;

    AssertThrow(fe_m != 0 and fe_p != 0,
                dealii::ExcMessage("Enriched cell has a neighbour outside the halo layer."));

    FaceIntP p_m(mf_en, true, dof_p, quad, 0, fe_m);
    FaceIntP p_p(mf_en, false, dof_p, quad, 0, fe_p);
    p_m.reinit(face);
    p_p.reinit(face);

    eval_side(true, fe_m, face, p_m.n_q_points, u_m);
    eval_side(false, fe_p, face, p_m.n_q_points, u_p);

    for(unsigned int q = 0; q < p_m.n_q_points; ++q)
    {
      auto const flux = (0.5 * (u_m[q] + u_p[q])) * p_m.get_normal_vector(q);
      p_m.submit_value(flux, q);
      p_p.submit_value(-flux, q);
    }

    p_m.integrate(dealii::EvaluationFlags::values);
    p_m.distribute_local_to_global(correction_p);
    p_p.integrate(dealii::EvaluationFlags::values);
    p_p.distribute_local_to_global(correction_p);
  }

  // --- boundary faces (homogeneous part) -------------------------------------
  //   wall: psi = 0 -> nothing;  Dirichlet: avg = 0 -> nothing;  Neumann: u_m . n
  unsigned int const b_begin = mf_en.n_inner_face_batches();
  unsigned int const b_end   = b_begin + mf_en.n_boundary_face_batches();

  for(unsigned int face = b_begin; face < b_end; ++face)
  {
    if(mf_en.get_face_active_fe_index(std::make_pair(face, face + 1), true) != 1)
      continue;

    BoundaryTypeU const type =
      this->boundary_descriptor->velocity->get_boundary_type(mf_en.get_boundary_id(face));

    if(type != BoundaryTypeU::Neumann)
      continue; // WallEnrichment, Dirichlet, ...: homogeneous contribution of psi*U_tilde is zero

    FaceIntP p_m(mf_en, true, dof_p, quad, 0, 1);
    p_m.reinit(face);
    eval_side(true, 1, face, p_m.n_q_points, u_m);

    for(unsigned int q = 0; q < p_m.n_q_points; ++q)
      p_m.submit_value(u_m[q] * p_m.get_normal_vector(q), q);

    p_m.integrate(dealii::EvaluationFlags::values);
    p_m.distribute_local_to_global(correction_p);
  }

  correction_p.compress(dealii::VectorOperation::add);

  // (3) add onto the global pressure-space result
  fe_en->add_wall_layout_to_global_pressure(correction_p, dst);

  src.block(1).zero_out_ghost_values();
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_pressure_gradient_term_enriched(
  BlockVectorType &  dst,
  VectorType const & src,
  double const       time,
  double const       inverse_scaling_factor_pressure) const
{
  dst.block(0).zero_out_ghost_values();
  dst.block(1).zero_out_ghost_values();

  // (1) U_bar rows: existing ExaDG operator, full (incl. boundary data)
  this->evaluate_pressure_gradient_term(dst.block(0), src, time);

  // (2) U_tilde rows: test functions psi * v_tilde, incl. boundary data
  pressure_gradient_term_enriched_tilde_rows(
    dst.block(1), src, time, true /* include_boundary_data */, inverse_scaling_factor_pressure);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::apply_pressure_gradient_term_enriched(BlockVectorType &  dst,
                                                                       VectorType const & src) const
{
  dst.block(0).zero_out_ghost_values();
  dst.block(1).zero_out_ghost_values();

  // (1) U_bar rows: existing ExaDG operator, homogeneous
  gradient_operator.apply(dst.block(0), src);

  // (2) U_tilde rows: test functions psi * v_tilde, homogeneous
  pressure_gradient_term_enriched_tilde_rows(
    dst.block(1), src, 0.0 /* time */, false /* include_boundary_data */, 1.0);
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::pressure_gradient_term_enriched_tilde_rows(
  VectorType &       dst_tilde,
  VectorType const & src,
  double const       time,
  bool const         include_boundary_data,
  double const       inverse_scaling_factor_pressure) const
{
  AssertThrow(this->param.gradp_integrated_by_parts and
                this->param.gradp_formulation == FormulationPressureGradientTerm::Weak and
                this->param.gradp_use_boundary_data,
              dealii::ExcMessage("Enriched pressure gradient: only weak formulation with "
                                 "integration by parts and boundary data is implemented."));

  dst_tilde.zero_out_ghost_values();

  auto const & fe_en = this->get_function_enrichment();
  auto const & mf_en = *fe_en->get_matrix_free();

  unsigned int const dof_p = fe_en->get_dof_index_shadow_pressure();
  unsigned int const quad  = fe_en->get_quad_index();

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number>     CellIntU;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, dim, Number> FaceIntU;
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number>       CellIntP;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number>   FaceIntP;
  typedef dealii::VectorizedArray<Number>                   scalar;

  // pressure in shadow layout (enriched + halo cells)
  VectorType & p_shadow = fe_en->shadow_pressure;
  fe_en->copy_global_pressure_to_wall_layout(src, p_shadow);
  p_shadow.update_ghost_values();

  // zero velocity vectors: the enriched evaluators are only used as test functions,
  // gather_evaluate is needed so that y and u_tau (-> psi) are evaluated
  VectorType zero_bar, zero_tilde;
  mf_en.initialize_dof_vector(zero_bar, fe_en->get_dof_index_shadow_vector());
  mf_en.initialize_dof_vector(zero_tilde, fe_en->get_dof_index_en());
  zero_bar.update_ghost_values();
  zero_tilde.update_ghost_values();

  dst_tilde = 0.0;

  // --- cell term: -int p div(psi v_tilde) ------------------------------------
  {
    FEEnrichedEvaluation<dim, Number, CellIntU> phi_u(mf_en, *fe_en);
    CellIntP                                    phi_p(mf_en, dof_p, quad, 0, 1);

    for(unsigned int cell = 0; cell < mf_en.n_cell_batches(); ++cell)
    {
      if(mf_en.get_cell_iterator(cell, 0)->active_fe_index() != 1)
        continue;

      phi_u.reinit(cell);
      phi_u.gather_evaluate(zero_bar, zero_tilde, dealii::EvaluationFlags::values);

      phi_p.reinit(cell);
      phi_p.gather_evaluate(p_shadow, dealii::EvaluationFlags::gradients);

      // strong form: int psi v_tilde . grad(p)   (no grad(psi), no large p values)
      for(unsigned int q = 0; q < phi_u.n_q_points; ++q)
        phi_u.submit_value(phi_p.get_gradient(q), q); // submit_value tests with v_bar and psi v_tilde

      phi_u.integrate(dealii::EvaluationFlags::values);

      phi_u.phi_en.distribute_local_to_global(dst_tilde); // U_tilde rows only
    }
  }

  // test one side of a face with psi*v_tilde: submits value_n[q] (a vector) and scatters
  auto const test_side = [&](bool const interior, unsigned int const face,
                             std::vector<dealii::Tensor<1, dim, scalar>> const & value_n) {
    FEFaceEnrichedEvaluation<dim, Number, FaceIntU> phi(mf_en, *fe_en, interior, 1);
    phi.reinit(face);
    phi.gather_evaluate(zero_bar, zero_tilde, dealii::EvaluationFlags::values);
    for(unsigned int q = 0; q < phi.n_q_points; ++q)
      phi.submit_value(value_n[q], q);
    phi.integrate(dealii::EvaluationFlags::values);
    phi.phi_en.distribute_local_to_global(dst_tilde);
  };

  std::vector<dealii::Tensor<1, dim, scalar>> flux_n_m, flux_n_p;

  // --- interior faces: int {p} [psi v_tilde] . n -----------------------------
  for(unsigned int face = 0; face < mf_en.n_inner_face_batches(); ++face)
  {
    auto const         range = std::make_pair(face, face + 1);
    unsigned int const fe_m  = mf_en.get_face_active_fe_index(range, true);
    unsigned int const fe_p  = mf_en.get_face_active_fe_index(range, false);

    if(fe_m != 1 and fe_p != 1)
      continue;

    AssertThrow(fe_m != 0 and fe_p != 0,
                dealii::ExcMessage("Enriched cell has a neighbour outside the halo layer."));

    FaceIntP pm(mf_en, true, dof_p, quad, 0, fe_m);
    FaceIntP pp(mf_en, false, dof_p, quad, 0, fe_p);
    pm.reinit(face);
    pp.reinit(face);
    pm.gather_evaluate(p_shadow, dealii::EvaluationFlags::values);
    pp.gather_evaluate(p_shadow, dealii::EvaluationFlags::values);

    flux_n_m.resize(pm.n_q_points);
    flux_n_p.resize(pm.n_q_points);
    // strong form: - int psi v_tilde . n (p_side - {p}); for both sides this is
    //   - 0.5 (p_m - p_p) psi v_tilde . n_m   (only the JUMP of p enters)
    for(unsigned int q = 0; q < pm.n_q_points; ++q)
    {
      scalar const jump = pm.get_value(q) - pp.get_value(q);
      flux_n_m[q]       = -0.5 * jump * pm.get_normal_vector(q);
      flux_n_p[q]       = flux_n_m[q];
    }

    if(fe_m == 1)
      test_side(true, face, flux_n_m);
    if(fe_p == 1)
      test_side(false, face, flux_n_p);
  }

  // --- boundary faces -----------------------------------------------------------
  //   WallEnrichment: psi = 0 -> nothing
  //   velocity Dirichlet (inflow)  = pressure Neumann   -> p* = p_m
  //   velocity Neumann  (outflow)  = pressure Dirichlet -> p* = g_p(x,t)
  unsigned int const b_begin = mf_en.n_inner_face_batches();
  unsigned int const b_end   = b_begin + mf_en.n_boundary_face_batches();

  for(unsigned int face = b_begin; face < b_end; ++face)
  {
    if(mf_en.get_face_active_fe_index(std::make_pair(face, face + 1), true) != 1)
      continue;

    auto const          boundary_id = mf_en.get_boundary_id(face);
    BoundaryTypeU const type        = this->boundary_descriptor->velocity->get_boundary_type(boundary_id);

    if(type == BoundaryTypeU::WallEnrichment)
      continue;

    FaceIntP pm(mf_en, true, dof_p, quad, 0, 1);
    pm.reinit(face);
    pm.gather_evaluate(p_shadow, dealii::EvaluationFlags::values);

    flux_n_m.resize(pm.n_q_points);

    if(type == BoundaryTypeU::Dirichlet)
    {
      // inflow = pressure Neumann: p* = p_m -> no contribution in the strong form
      continue;
    }
    else if(type == BoundaryTypeU::Neumann)
    {
      // outflow = pressure Dirichlet: p* = g_p -> - int psi v_tilde . n (p_m - g_p),
      // g_p scaled like in GradientOperator (homogeneous operator: g_p = 0)
      auto const & g_p = this->boundary_descriptor->pressure->dirichlet_bc.find(boundary_id)->second;
      g_p->set_time(time);

      for(unsigned int q = 0; q < pm.n_q_points; ++q)
      {
        scalar g = dealii::make_vectorized_array<Number>(0.0);
        if(include_boundary_data)
        {
          auto const xq = pm.quadrature_point(q);
          for(unsigned int l = 0; l < scalar::size(); ++l)
          {
            dealii::Point<dim> pt;
            for(unsigned int d = 0; d < dim; ++d)
              pt[d] = xq[d][l];
            g[l] = inverse_scaling_factor_pressure * g_p->value(pt);
          }
        }
        flux_n_m[q] = -(pm.get_value(q) - g) * pm.get_normal_vector(q);
      }
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage("Enriched pressure gradient: boundary type not supported "
                                     "on enriched cells."));
    }

    test_side(true, face, flux_n_m);
  }

  dst_tilde.compress(dealii::VectorOperation::add);
  p_shadow.zero_out_ghost_values();
}

template<int dim, typename Number>
void
SpatialOperatorBase<dim, Number>::evaluate_velocity_divergence_term(
  VectorType &            dst,
  BlockVectorType const & src,
  double const            time) const
{
  // homogeneous part: D(q, \bar{u}_h + \psi \tilde{u}_h); the \psi \tilde{u}_h part has no
  // boundary contribution on walls (\psi = 0) and velocity Dirichlet boundaries ({u} = g_u)
  apply_velocity_divergence_term(dst, src);

  // inhomogeneous part (velocity Dirichlet data g_u(t)):
  // DivergenceOperator::evaluate = apply - rhs, rhs() contains -(boundary data term)
  VectorType rhs_boundary_data;
  rhs_boundary_data.reinit(dst, false);
  this->divergence_operator.rhs(rhs_boundary_data, time);
  dst.add(-1.0, rhs_boundary_data);
}

template class SpatialOperatorBase<2, float>;
template class SpatialOperatorBase<3, float>;

template class SpatialOperatorBase<2, double>;
template class SpatialOperatorBase<3, double>;

} // namespace IncNS
} // namespace ExaDG
