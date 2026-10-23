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
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/convection_diffusion/preconditioners/multigrid_preconditioner.h>
#include <exadg/convection_diffusion/spatial_discretization/operator.h>
#include <exadg/convection_diffusion/spatial_discretization/project_velocity.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/grid_related_time_step_restrictions.h>
#include <exadg/operators/quadrature.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
Operator<dim, Number>::Operator(
  std::shared_ptr<Grid<dim> const>                      grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>           mapping_in,
  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings_in,
  std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>            field_functions_in,
  Parameters const &                                    param_in,
  std::string const &                                   field_in,
  MPI_Comm const &                                      mpi_comm_in)
  : dealii::Subscriptor(),
    grid(grid_in),
    mapping(mapping_in),
    multigrid_mappings(multigrid_mappings_in),
    boundary_descriptor(boundary_descriptor_in),
    field_functions(field_functions_in),
    param(param_in),
    field(field_in),
    n_blocks(param_in.number_of_transport_variables),
    dof_handler(*grid_in->triangulation),
    mpi_comm(mpi_comm_in),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_in) == 0)
{
  pcout << std::endl << "Construct convection-diffusion operator ..." << std::endl;

  fe = create_finite_element<dim>(ElementType::Hypercube, true, 1, param.degree);
  AssertThrow(n_blocks >= 1,
              dealii::ExcMessage("number_of_transport_variables must be at least 1."));

  boundary_descriptor_block.resize(n_blocks);
  for(unsigned int b = 0; b < n_blocks; ++b)
    boundary_descriptor_block[b] = get_boundary_descriptor_of_component(*boundary_descriptor, b);

  if(needs_own_dof_handler_velocity())
  {
    fe_velocity = create_finite_element<dim>(ElementType::Hypercube, true, dim, param.degree);
    dof_handler_velocity = std::make_shared<dealii::DoFHandler<dim>>(*grid->triangulation);
  }

  if(param.turbulence_model_data.is_active)
  {
    // Create a 1-component element with the same polynomial degree as the solution
    fe_eddy_viscosity = create_finite_element<dim>(ElementType::Hypercube, true, 1, param.degree);
    dof_handler_eddy_viscosity = std::make_shared<dealii::DoFHandler<dim>>(*grid->triangulation);

    dealii::QGaussLobatto<dim> const check_quad(param.degree + 1);
    auto const &                     support_points = fe_eddy_viscosity->get_unit_support_points();
    AssertDimension(support_points.size(), check_quad.size());
    for(unsigned int i = 0; i < support_points.size(); ++i)
      AssertThrow(support_points[i].distance(check_quad.point(i)) < 1e-12,
             dealii::ExcMessage(
             "fe_eddy_viscosity's unit support points do not coincide with "
             "quad_index_eddy_velocity (QGaussLobatto). The nodal collocation "
             "in TurbulenceModel::cell_loop() is invalid for this deal.II "
             "version/FE choice -- build quad_index_eddy_velocity from "
             "fe_eddy_viscosity->get_unit_support_points() directly instead."));
  }

  initialize_dof_handler_and_constraints();

  pcout << std::endl
        << "Discontinuous Galerkin finite element discretization:" << std::endl
        << std::endl;

  print_parameter(pcout, "degree of 1D polynomials", param.degree);
  print_parameter(pcout, "number of dofs per cell", fe->n_dofs_per_cell());
  print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_handler_and_constraints()
{
  dof_handler.distribute_dofs(*fe);

  if(needs_own_dof_handler_velocity())
  {
    dof_handler_velocity->distribute_dofs(*fe_velocity);
  }

  if(param.turbulence_model_data.is_active)
  {
    dof_handler_eddy_viscosity->distribute_dofs(*fe_eddy_viscosity);
  }

  affine_constraints.close();
}

template<int dim, typename Number>
void
Operator<dim, Number>::fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const
{
  // append mapping flags
  if(param.problem_type == ProblemType::Unsteady)
  {
    matrix_free_data.append_mapping_flags(MassKernel<dim, Number>::get_mapping_flags());
  }

  if(param.right_hand_side)
  {
    matrix_free_data.append_mapping_flags(
      Operators::RHSKernel<dim, Number>::get_mapping_flags());
  }

  if(param.convective_problem())
  {
    matrix_free_data.append_mapping_flags(
      Operators::ConvectiveKernel<dim, Number>::get_mapping_flags());
  }

  if(param.diffusive_problem())
  {
    matrix_free_data.append_mapping_flags(
      Operators::DiffusiveKernel<dim, Number>::get_mapping_flags(true, true));
  }

  if(param.turbulence_model_data.is_active)
  { 
    matrix_free_data.append_mapping_flags(
      Operators::TurbulenceSourceKernel<dim, Number>::get_mapping_flags());
  }

  // mapping flags required for CFL condition
  MappingFlags flags_cfl;
  flags_cfl.cells = dealii::update_quadrature_points;
  matrix_free_data.append_mapping_flags(flags_cfl);

  // dealii::DoFHandler, dealii::AffineConstraints
  matrix_free_data.insert_dof_handler(&dof_handler, get_dof_name());
  matrix_free_data.insert_constraint(&affine_constraints, get_dof_name());

  if(needs_own_dof_handler_velocity())
  {
    matrix_free_data.insert_dof_handler(&(*dof_handler_velocity), get_dof_name_velocity());
    matrix_free_data.insert_constraint(&affine_constraints, get_dof_name_velocity());
  }

  if(param.turbulence_model_data.is_active)
  {
    matrix_free_data.insert_dof_handler(&(*dof_handler_eddy_viscosity), get_dof_name_eddy_viscosity());
    matrix_free_data.insert_constraint(&affine_constraints, get_dof_name_eddy_viscosity());
  }

  // dealii::Quadrature
  std::shared_ptr<dealii::Quadrature<dim>> quadrature =
    create_quadrature<dim>(param.grid.element_type, param.degree + 1);
  matrix_free_data.insert_quadrature(*quadrature, get_quad_name());

  if(param.use_overintegration)
  {
    std::shared_ptr<dealii::Quadrature<dim>> quadrature_over =
      create_quadrature<dim>(param.grid.element_type, param.degree + (param.degree + 2) / 2);
    matrix_free_data.insert_quadrature(*quadrature_over, get_quad_name_overintegration());
  }

  if(param.turbulence_model_data.is_active)
  {
    std::shared_ptr<dealii::Quadrature<dim>> quadrature_eddy_velocity = std::make_shared<dealii::QGaussLobatto<dim>>(param.degree + 1);
    matrix_free_data.insert_quadrature(*quadrature_eddy_velocity, get_quad_name_eddy_velocity());
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_operators()
{
  // mass operator
  MassOperatorData<dim> mass_operator_data;
  mass_operator_data.dof_index            = get_dof_index();
  mass_operator_data.quad_index           = get_quad_index();
  mass_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
  mass_operator_data.implement_block_diagonal_preconditioner_matrix_free =
    param.implement_block_diagonal_preconditioner_matrix_free;

  mass_operator.initialize(*matrix_free, affine_constraints, mass_operator_data);

  // inverse mass operator
  InverseMassOperatorData inverse_mass_operator_data;
  inverse_mass_operator_data.dof_index  = get_dof_index();
  inverse_mass_operator_data.quad_index = get_quad_index();
  inverse_mass_operator_data.parameters = param.inverse_mass_operator;

  inverse_mass_operator.initialize(*matrix_free, inverse_mass_operator_data);

  // convective operator
  unsigned int const quad_index_convective =
    param.use_overintegration ? get_quad_index_overintegration() : get_quad_index();

  Operators::ConvectiveKernelData<dim> convective_kernel_data;

  if(param.convective_problem())
  {
    convective_kernel_data.formulation                = param.formulation_convective_term;
    convective_kernel_data.velocity_type              = param.get_type_velocity_field();
    convective_kernel_data.dof_index_velocity         = get_dof_index_velocity();
    convective_kernel_data.numerical_flux_formulation = param.numerical_flux_convective_operator;
    convective_kernel_data.velocity                   = field_functions->velocity;

    convective_kernel = std::make_shared<Operators::ConvectiveKernel<dim, Number>>();
    convective_kernel->reinit(*matrix_free,
                              convective_kernel_data,
                              quad_index_convective,
                              false /* is_mg */);

    convective_operator.resize(n_blocks);

    for(unsigned int b = 0; b < n_blocks; ++b)
    {
      ConvectiveOperatorData<dim> convective_operator_data;
      convective_operator_data.dof_index            = get_dof_index();
      convective_operator_data.quad_index           = quad_index_convective;
      convective_operator_data.bc                   = boundary_descriptor_block[b];
      convective_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
      convective_operator_data.implement_block_diagonal_preconditioner_matrix_free =
        param.implement_block_diagonal_preconditioner_matrix_free;
      convective_operator_data.kernel_data = convective_kernel_data;

      convective_operator[b] = std::make_shared<ConvectiveOperator<dim, Number>>();
      convective_operator[b]->initialize(*matrix_free,
                                     affine_constraints,
                                     convective_operator_data,
                                     convective_kernel);
    }
  }

  if(param.turbulence_model_data.is_active)
  {

    AssertThrow(not param.use_overintegration,
                dealii::ExcMessage("Overintegration is not implemented in combination with the turbulence source terms."));

    turbulence_model_ptr->diffusivity              = param.diffusivity;

    turbulence_model_ptr->initialize(*matrix_free,
                                     param.turbulence_model_data,
                                     get_dof_index(),
                                     get_dof_index_eddy_viscosity(),
                                     get_dof_index_velocity(),
                                     get_quad_index(),
                                     get_quad_index_eddy_velocity());
    turbulence_model_ptr->model_coefficients       = param.turbulence_model_data.turbulence_data_base->get_all_coefficients();
  }

  std::vector<double> eddy_viscosity_factor(n_blocks, 0.0);

  if(param.turbulence_model_data.is_active)
  {
    std::vector<double> const & coefficients =
      turbulence_model_ptr->model_coefficients;

    if(param.turbulence_model_data.turbulence_model ==
      TurbulenceEddyViscosityModel::StandardKEpsilon)
    {  
      eddy_viscosity_factor = {1.0 / coefficients[0],  // 1/sigma_k
        1.0 / coefficients[4]}; // 1/sigma_epsilon
    }
    else if(param.turbulence_model_data.turbulence_model ==
      TurbulenceEddyViscosityModel::StandardKOmega1988)
    {
      eddy_viscosity_factor = {coefficients[4],  // sigma*
        coefficients[3]}; // sigma
    }
  }

  // diffusive operator
  std::vector<Operators::DiffusiveKernelData> diffusive_kernel_data(n_blocks);

  if(param.diffusive_problem())
  {
    diffusive_kernel.resize(n_blocks);
    diffusive_operator.resize(n_blocks);

    for(unsigned int b = 0; b < n_blocks; ++b)
    {
      // molecular viscosity only; the turbulent diffusion is part of TurbulenceSourceOperator
      diffusive_kernel_data[b].IP_factor   = param.IP_factor;
      diffusive_kernel_data[b].diffusivity = param.diffusivity;

      diffusive_kernel[b] = std::make_shared<Operators::DiffusiveKernel<dim, Number>>();
      diffusive_kernel[b]->reinit(*matrix_free, diffusive_kernel_data[b], get_dof_index());


      DiffusiveOperatorData<dim> diffusive_operator_data;
      diffusive_operator_data.dof_index            = get_dof_index();
      diffusive_operator_data.quad_index           = get_quad_index();
      diffusive_operator_data.bc                   = boundary_descriptor_block[b];
      diffusive_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
      diffusive_operator_data.implement_block_diagonal_preconditioner_matrix_free = param.implement_block_diagonal_preconditioner_matrix_free;
      diffusive_operator_data.kernel_data = diffusive_kernel_data[b];

      diffusive_operator[b] = std::make_shared<DiffusiveOperator<dim, Number>>();
      diffusive_operator[b]->initialize(*matrix_free,
                                        affine_constraints,
                                        diffusive_operator_data,
                                        diffusive_kernel[b]);
    }
  }

  rhs_kernel.resize(n_blocks);
  rhs_operator.resize(n_blocks);

  for(unsigned int b = 0; b < n_blocks; ++b)
  {
    Operators::RHSKernelData<dim> rhs_kernel_data;
    rhs_kernel_data.f = get_component_function(field_functions->right_hand_side,  b);

    rhs_kernel[b] = std::make_shared<Operators::RHSKernel<dim, Number>>();
    rhs_kernel[b]->reinit(rhs_kernel_data);

    // rhs operator
    RHSOperatorData<dim> rhs_operator_data;
    rhs_operator_data.dof_index     = get_dof_index();
    rhs_operator_data.quad_index    = get_quad_index();

    rhs_operator[b] = std::make_shared<RHSOperator<dim, Number>>();
    rhs_operator[b]->initialize(*matrix_free, rhs_operator_data, rhs_kernel[b]);
  }

  if(param.turbulence_model_data.is_active)
  {
    AssertThrow(param.problem_type == ProblemType::Unsteady and
                  param.temporal_discretization == TemporalDiscretization::BDF,
                dealii::ExcMessage("Turbulence model: only unsteady problems with BDF time "
                                   "integration are implemented (no explicit Runge-Kutta, no "
                                   "steady solver)."));

    TurbulenceSourceOperatorData<dim> turbulence_source_operator_data;
    turbulence_source_operator_data.dof_index  = get_dof_index();
    turbulence_source_operator_data.quad_index = get_quad_index();

    Operators::TurbulenceSourceKernelData<dim> turbulence_source_kernel_data;
    turbulence_source_kernel_data.dof_index                = get_dof_index();
    turbulence_source_kernel_data.dof_index_velocity       = get_dof_index_velocity();
    turbulence_source_kernel_data.turbulence_model_data    = param.turbulence_model_data;
    turbulence_source_kernel_data.diffusivity              = param.diffusivity;
    turbulence_source_kernel_data.k_log                    = param.turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::LogarithmicTransportVariable;
    turbulence_source_kernel_data.omega_log                = param.turbulence_model_data.positivity_preserving_limiter[1] == PositivityPreservingLimiter::LogarithmicTransportVariable;

    turbulence_source_kernel_data.production_implicit =
      param.treatment_of_production == TreatmentOfProduction::Implicit;
    turbulence_source_kernel_data.dissipation_implicit =
      param.treatment_of_dissipation == TreatmentOfDissipation::Implicit;
    turbulence_source_kernel_data.square_gradient_implicit =
      param.treatment_of_square_gradient == TreatmentOfSquareGradient::Implicit;
    turbulence_source_kernel_data.viscosity_implicit =
      param.treatment_of_variable_viscosity == TreatmentOfVariableViscosity::Implicit;

    turbulence_source_kernel_data.eddy_viscosity_factor = eddy_viscosity_factor;
    turbulence_source_kernel_data.IP_factor             = param.IP_factor;

    turbulence_source_kernel_data.bc = boundary_descriptor_block;

    turbulence_source_kernel = std::make_shared<Operators::TurbulenceSourceKernel<dim, Number>>();
    turbulence_source_kernel->turbulence_model_ptr = turbulence_model_ptr;
    turbulence_source_kernel->reinit(*matrix_free, turbulence_source_kernel_data, get_quad_index());

    turbulence_source_operator.initialize(*matrix_free,
                                          turbulence_source_operator_data,
                                          turbulence_source_kernel);
  }

  // merged operator
  if(param.temporal_discretization == TemporalDiscretization::BDF or
     (param.temporal_discretization == TemporalDiscretization::ExplRK and
      param.use_combined_operator == true))
  {
    combined_operator.resize(n_blocks);

    for(unsigned int b = 0; b < n_blocks; ++b)
    {
      CombinedOperatorData<dim> combined_operator_data;
      combined_operator_data.bc                   = boundary_descriptor_block[b];
      combined_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
      combined_operator_data.implement_block_diagonal_preconditioner_matrix_free =  param.implement_block_diagonal_preconditioner_matrix_free;
      combined_operator_data.solver_block_diagonal         = param.solver_block_diagonal;
      combined_operator_data.preconditioner_block_diagonal = param.preconditioner_block_diagonal;
      combined_operator_data.solver_data_block_diagonal    = param.solver_data_block_diagonal;

      // linear system of equations has to be solved: the problem is either steady or
    // an unsteady problem is solved with BDF time integration (semi-implicit or fully implicit
    // formulation of convective and diffusive terms)
    if(param.problem_type == ProblemType::Steady or
       param.temporal_discretization == TemporalDiscretization::BDF)
    {
      if(param.problem_type == ProblemType::Unsteady)
        combined_operator_data.unsteady_problem = true;

      if(param.convective_problem() and
         param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
        combined_operator_data.convective_problem = true;

      if(param.diffusive_problem())
        combined_operator_data.diffusive_problem = true;
    }
    else if(param.temporal_discretization == TemporalDiscretization::ExplRK)
    {
      // always false
      combined_operator_data.unsteady_problem = false;

      if(this->param.equation_type == EquationType::Convection or
         this->param.equation_type == EquationType::ConvectionDiffusion)
        combined_operator_data.convective_problem = true;

      if(this->param.equation_type == EquationType::Diffusion or
         this->param.equation_type == EquationType::ConvectionDiffusion)
        combined_operator_data.diffusive_problem = true;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    combined_operator_data.convective_kernel_data = convective_kernel_data;
    combined_operator_data.diffusive_kernel_data  = diffusive_kernel_data[b];

    combined_operator_data.dof_index = get_dof_index();
    combined_operator_data.quad_index =
      (param.use_overintegration and combined_operator_data.convective_problem) ?
        get_quad_index_overintegration() :
        get_quad_index();

      combined_operator[b] = std::make_shared<CombinedOperator<dim, Number>>();
      combined_operator[b]->initialize(*matrix_free,
                                      affine_constraints,
                                      combined_operator_data,
                                       convective_kernel,
                                       param.diffusive_problem() ? diffusive_kernel[b] : nullptr);
    }
  }

  // modal filter (scalar, acts on all blocks)
  modal_filter.initialize(*matrix_free,
                          get_dof_index(),
                          get_quad_index(),
                          param.cut_off_degree,
                          param.gradient_threshold);
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup()
{
  pcout << std::endl << "Setup convection-diffusion operator ..." << std::endl;

  do_setup();

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
void
Operator<dim, Number>::do_setup()
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
Operator<dim, Number>::setup(std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free_in,
                             std::shared_ptr<MatrixFreeData<dim, Number> const> matrix_free_data_in,
                             std::string const & dof_index_velocity_external_in)
{
  matrix_free      = matrix_free_in;
  matrix_free_data = matrix_free_data_in;

  dof_index_velocity_external = dof_index_velocity_external_in;

  setup_operators();

  if(param.linear_system_has_to_be_solved())
  {
    setup_preconditioner();

    setup_solver();
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_after_coarsening_and_refinement()
{
  initialize_dof_handler_and_constraints();

  print_parameter(pcout,
                  "number of dofs (total) after adaptive mesh refinement",
                  dof_handler.n_dofs());

  do_setup();
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_preconditioner()
{
  preconditioner.assign(n_blocks, nullptr);
  if(param.preconditioner == Preconditioner::InverseMassMatrix)
  {
    InverseMassOperatorData inverse_mass_operator_data;
    inverse_mass_operator_data.dof_index  = get_dof_index();
    inverse_mass_operator_data.quad_index = get_quad_index();
    inverse_mass_operator_data.parameters = param.inverse_mass_preconditioner;


    std::shared_ptr<PreconditionerBase<Number>> const inverse_mass_preconditioner = std::make_shared<InverseMassPreconditioner<dim, 1, Number>>(*matrix_free, inverse_mass_operator_data);


    for(unsigned int b = 0; b < n_blocks; ++b)
    {
      preconditioner[b] = inverse_mass_preconditioner;
    }
  }
  else if(param.preconditioner == Preconditioner::PointJacobi)
  {
    for(unsigned int b = 0; b < n_blocks; ++b)
      preconditioner[b] = std::make_shared<JacobiPreconditioner<CombinedOperator<dim, Number>>>(*combined_operator[b], false);
  }
  else if(param.preconditioner == Preconditioner::BlockJacobi)
  {
    for(unsigned int b = 0; b < n_blocks; ++b)
      preconditioner[b] = std::make_shared<BlockJacobiPreconditioner<CombinedOperator<dim, Number>>>(*combined_operator[b], false);
  }
  else if(param.preconditioner == Preconditioner::Multigrid)
  {
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    {
      AssertThrow(param.mg_operator_type != MultigridOperatorType::ReactionConvection and
                    param.mg_operator_type != MultigridOperatorType::ReactionConvectionDiffusion,
                  dealii::ExcMessage(
                  "Invalid solver parameters. The convective term is treated explicitly."));
    }

    if(param.mg_operator_type == MultigridOperatorType::ReactionConvection or param.mg_operator_type ==  MultigridOperatorType::ReactionConvectionDiffusion)
    {
      if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        unsigned int const degree_scalar = dof_handler.get_fe().degree;
        unsigned int const degree_velocity = matrix_free_data->get_dof_handler(get_dof_name_velocity()).get_fe().degree;
        AssertThrow(degree_scalar == degree_velocity,
                    dealii::ExcMessage("When using a multigrid preconditioner for the scalar  convection-diffusion equation, the scalar field and the velocity field must have the same polynomial degree."));
      }
    }

    MultigridData mg_data;
    mg_data = param.multigrid_data;

    typedef MultigridPreconditioner<dim, Number> Multigrid;

    typedef std::map<dealii::types::boundary_id, dealii::ComponentMask> Map_DBC_ComponentMask;

    for(unsigned int b = 0; b < n_blocks; ++b)
    {
      std::shared_ptr<Multigrid> mg_preconditioner = std::make_shared<Multigrid>(this->mpi_comm);

      std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> dirichlet_boundary_conditions = boundary_descriptor_block[b]->dirichlet_bc;

      Map_DBC_ComponentMask dirichlet_bc_component_mask;

      mg_preconditioner->initialize(mg_data,
                                    grid,
                                    multigrid_mappings,
                                    dof_handler.get_fe(),
                                    *combined_operator[b],
                                    param.mg_operator_type,
                                    param.ale_formulation,
                                    dirichlet_boundary_conditions,
                                    dirichlet_bc_component_mask);

      preconditioner[b] = mg_preconditioner;
    }
  }
  else
  {
    AssertThrow(param.preconditioner == Preconditioner::None or
                  param.preconditioner == Preconditioner::InverseMassMatrix or
                  param.preconditioner == Preconditioner::PointJacobi or
                  param.preconditioner == Preconditioner::BlockJacobi or
                  param.preconditioner == Preconditioner::Multigrid,
                dealii::ExcMessage("Specified preconditioner is not implemented!"));
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::setup_solver()
{
  block_preconditioner.initialize(preconditioner);
  if(param.solver == Solver::CG)
  {
    // initialize solver_data
    Krylov::SolverDataCG solver_data;
    solver_data.solver_tolerance_abs = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel = param.solver_data.rel_tol;
    solver_data.max_iter             = param.solver_data.max_iter;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver =
      std::make_shared<Krylov::SolverCG<This, BlockPreconditioner<Number>,
      BlockVectorType>>(
        *this, block_preconditioner, solver_data);
  }
  else if(param.solver == Solver::GMRES)
  {
    // initialize solver_data
    Krylov::SolverDataGMRES solver_data;
    solver_data.solver_tolerance_abs = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel = param.solver_data.rel_tol;
    solver_data.max_iter             = param.solver_data.max_iter;
    solver_data.max_n_tmp_vectors    = param.solver_data.max_krylov_size;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
    iterative_solver =
      std::make_shared<Krylov::SolverGMRES<This, BlockPreconditioner<Number>,
      BlockVectorType>>(
        *this, block_preconditioner, solver_data, mpi_comm);
  }
  else if(param.solver == Solver::FGMRES)
  {
    // initialize solver_data
    Krylov::SolverDataFGMRES solver_data;
    solver_data.solver_tolerance_abs = param.solver_data.abs_tol;
    solver_data.solver_tolerance_rel = param.solver_data.rel_tol;
    solver_data.max_iter             = param.solver_data.max_iter;
    solver_data.max_n_tmp_vectors    = param.solver_data.max_krylov_size;

    if(param.preconditioner != Preconditioner::None)
      solver_data.use_preconditioner = true;

    // initialize solver
      iterative_solver =
        std::make_shared<Krylov::SolverFGMRES<This, BlockPreconditioner<Number>,
      BlockVectorType>>(
          *this, block_preconditioner, solver_data);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Specified solver is not implemented!"));
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_vector(BlockVectorType & src) const
{
  src.reinit(n_blocks);
  for(unsigned int b = 0; b < n_blocks; ++b)
    matrix_free->initialize_dof_vector(src.block(b), get_dof_index());
  src.collect_sizes();
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_vector_eddy_viscosity(VectorType & eddy_viscosity) const
{
  matrix_free->initialize_dof_vector(eddy_viscosity, get_dof_index_eddy_viscosity());
}

template<int dim, typename Number>
void
Operator<dim, Number>::initialize_dof_vector_velocity(VectorType & velocity) const
{
  matrix_free->initialize_dof_vector(velocity, get_dof_index_velocity());
}

template<int dim, typename Number>
void
Operator<dim, Number>::interpolate_velocity(VectorType & velocity, double const time) const
{
  field_functions->velocity->set_time(time);

  // This is necessary if Number == float
  typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble vector_double;
  vector_double = velocity;

  dealii::VectorTools::interpolate(get_dof_handler_velocity(),
                                   *(field_functions->velocity),
                                   vector_double);

  velocity = vector_double;
}

template<int dim, typename Number>
void
Operator<dim, Number>::project_velocity(VectorType & velocity, double const time) const
{
  VelocityProjection<dim, Number> l2_projection;

  InverseMassOperatorData inverse_mass_operator_data_l2_projection;
  inverse_mass_operator_data_l2_projection.dof_index  = get_dof_index_velocity();
  inverse_mass_operator_data_l2_projection.quad_index = get_quad_index();

  l2_projection.apply(*matrix_free,
                      inverse_mass_operator_data_l2_projection,
                      field_functions->velocity,
                      time,
                      velocity);
}

template<int dim, typename Number>
void
Operator<dim, Number>::prescribe_initial_conditions(BlockVectorType & src, double const time) const
{
  field_functions->initial_solution->set_time(time);

  // This is necessary if Number == float
  typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  for(unsigned int b = 0; b < n_blocks; ++b)
  {
    std::shared_ptr<dealii::Function<dim>> const initial_solution_b =
      get_component_function(field_functions->initial_solution, b);

    VectorTypeDouble src_double;
    src_double = src.block(b);

    dealii::VectorTools::interpolate(dof_handler, *initial_solution_b,
                                     src_double);
    src.block(b) = src_double;
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_explicit_time_int(BlockVectorType &       dst,
                                                  BlockVectorType const & src,
                                                  double const       time,
                                                  VectorType const * velocity) const
{
  for(unsigned int b = 0; b < n_blocks; ++b)
  {
    // evaluate each operator separately
    if(param.use_combined_operator == false)
    {
      // set dst to zero
      dst.block(b) = 0.0;

      // diffusive operator
      if(param.diffusive_problem())
      {
        diffusive_operator[b]->set_time(time);
        diffusive_operator[b]->evaluate_add(dst.block(b), src.block(b));
      }

      // convective operator
      if(param.convective_problem())
      {
        if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
        {
          AssertThrow(velocity != nullptr,
                      dealii::ExcMessage("velocity pointer is not initialized."));
          convective_operator[b]->set_velocity_ptr(*velocity);
        }

        convective_operator[b]->set_time(time);
        convective_operator[b]->evaluate_add(dst.block(b), src.block(b));
      }
    } else {
        if(param.convective_problem())
        {
          if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
          {
            AssertThrow(velocity != nullptr,
                        dealii::ExcMessage("velocity pointer is not initialized."));
  
            combined_operator[b]->set_velocity_ptr(*velocity);
          }
        }
  
        combined_operator[b]->set_time(time);
        combined_operator[b]->evaluate(dst.block(b), src.block(b));
    }

      // shift diffusive and convective term to the rhs of the equation
      dst.block(b) *= -1.0;

      if(param.right_hand_side == true)
      {
        rhs_operator[b]->evaluate_add(dst.block(b), time);
      }
  }

  // apply inverse mass operator
  for(unsigned int b = 0; b < n_blocks; ++b)
    inverse_mass_operator.apply(dst.block(b), dst.block(b));
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_convective_term(BlockVectorType &       dst,
                                                BlockVectorType const & src,
                                                double const       time,
                                                VectorType const * velocity) const
{
  for(unsigned int b = 0; b < n_blocks; ++b)    
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      AssertThrow(velocity != nullptr, dealii::ExcMessage("velocity pointer is not initialized."));

      convective_operator[b]->set_velocity_ptr(*velocity);
    }

    convective_operator[b]->set_time(time);
    convective_operator[b]->evaluate(dst.block(b), src.block(b));
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::rhs(BlockVectorType & dst,
                           BlockVectorType const & src,
                           double const time,
                           VectorType const * velocity) const
{
  // no need to set scaling_factor_mass because the mass operator does not contribute to rhs

  for(unsigned int b = 0; b < n_blocks; ++b)
  {
    if(param.linear_system_including_convective_term_has_to_be_solved())
    {
      if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
      {
        AssertThrow(velocity != nullptr, dealii::ExcMessage("velocity pointer is not initialized."));

        combined_operator[b]->set_velocity_ptr(*velocity);
      }
    }

    combined_operator[b]->set_time(time);
    combined_operator[b]->rhs(dst.block(b));

    // rhs operator f(t)
    if(param.right_hand_side == true)
    {
      rhs_operator[b]->evaluate_add(dst.block(b), time);
    }
  }

}

template<int dim, typename Number>
void
Operator<dim, Number>::rhs_add_turbulence(BlockVectorType &       dst,
                                          BlockVectorType const & solution_n,
                                          double const            time_n,
                                          double const            time,
                                          VectorType const *      velocity) const
{
  AssertThrow(param.turbulence_model_data.is_active,
              dealii::ExcMessage("Turbulence model is not active."));
  AssertThrow(velocity != nullptr and velocity->size() > 0,
              dealii::ExcMessage("Turbulence model: the velocity field is not initialized."));

  turbulence_source_operator.set_velocity_ptr(*velocity);

  // terms treated Explicit, at the previous time step Q^n
  turbulence_source_operator.evaluate_add_explicit(dst, solution_n, time_n);

  // terms treated Implicit, linearized at Q^n: Newton_Explicit + J(Q^n) Q^n, since the unknown of
  // the linear system is Q^{n+1} = Q^n + dQ (J(Q^n) Q^{n+1} is applied in vmult())
  turbulence_source_operator.evaluate_add_newton_explicit(dst, solution_n, time);
  turbulence_source_operator.apply_add_newton_implicit(dst, solution_n);
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_add_turbulence_explicit(BlockVectorType &       dst,
                                                        BlockVectorType const & solution_n,
                                                        double const            time_n,
                                                        VectorType const *      velocity) const
{
  AssertThrow(velocity != nullptr, dealii::ExcMessage("velocity pointer is not initialized."));

  turbulence_source_operator.set_velocity_ptr(*velocity);
  turbulence_source_operator.evaluate_add_explicit(dst, solution_n, time_n);
}

template<int dim, typename Number>
void
Operator<dim, Number>::evaluate_add_turbulence_newton_explicit(BlockVectorType &       dst,
                                                               BlockVectorType const & solution,
                                                               double const            time,
                                                               VectorType const * velocity) const
{
  AssertThrow(velocity != nullptr, dealii::ExcMessage("velocity pointer is not initialized."));

  turbulence_source_operator.set_velocity_ptr(*velocity);
  turbulence_source_operator.evaluate_add_newton_explicit(dst, solution, time);
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_mass_operator(BlockVectorType & dst,
                                           BlockVectorType const & src) const
{
  for(unsigned int b = 0; b < n_blocks; ++b)
    mass_operator.apply(dst.block(b), src.block(b));
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_mass_operator_add(BlockVectorType & dst,
                                               BlockVectorType const & src) const
{
  for(unsigned int b = 0; b < n_blocks; ++b) 
    mass_operator.apply_add(dst.block(b), src.block(b));
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_convective_term(BlockVectorType & dst,
                                             BlockVectorType const & src) const
{
  for(unsigned int b = 0; b < n_blocks; ++b) 
    convective_operator[b]->apply(dst.block(b), src.block(b));
}

template<int dim, typename Number>
void
Operator<dim, Number>::update_convective_term(double const time, VectorType const * velocity) const
{

  for(unsigned int b = 0; b < n_blocks; ++b)
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      AssertThrow(velocity != nullptr, dealii::ExcMessage("velocity pointer is not initialized."));

      convective_operator[b]->set_velocity_ptr(*velocity);
    }

    convective_operator[b]->set_time(time);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_diffusive_term(BlockVectorType & dst,
                                            BlockVectorType const & src) const
{
  for(unsigned int b = 0; b < n_blocks; ++b)  
    diffusive_operator[b]->apply(dst.block(b), src.block(b));
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_conv_diff_operator(BlockVectorType & dst,
                                                BlockVectorType const & src) const
{
  for(unsigned int b = 0; b < n_blocks; ++b)  
    combined_operator[b]->apply(dst.block(b), src.block(b));
}

template<int dim, typename Number>
void
Operator<dim, Number>::update_conv_diff_operator(double const       time,
                                                 double const       scaling_factor,
                                                 VectorType const * velocity)
{
  for(unsigned int b = 0; b < n_blocks; ++b)
  {
    combined_operator[b]->set_scaling_factor_mass_operator(scaling_factor);
    combined_operator[b]->set_time(time);
  }

  if(param.linear_system_including_convective_term_has_to_be_solved())
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      AssertThrow(velocity != nullptr, dealii::ExcMessage("velocity pointer is not initialized."));

      for(unsigned int b = 0; b < n_blocks; ++b)
      {
        combined_operator[b]->set_velocity_ptr(*velocity);
      }

      if(param.turbulence_model_data.is_active)
      {
        turbulence_model_ptr->set_velocity_ptr(*velocity);
      }
    }
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::update_after_grid_motion(bool const update_matrix_free)
{
  if(update_matrix_free)
  {
    // Since matrix_free points to matrix_free_own_storage, we also update the actual/main
    // MatrixFree object called matrix_free.
    matrix_free_own_storage->update_mapping(*get_mapping());
  }


  // update SIPG penalty parameter of diffusive operator which depends on the deformation
  // of elements
  if(param.diffusive_problem())
  {
    for(auto const & diffusive_kernel_block : diffusive_kernel)
    {
      diffusive_kernel_block->calculate_penalty_parameter(*matrix_free, get_dof_index());
    }
  }

  // The inverse mass operator might contain matrix-based components, in which cases it needs to be
  // updated after the grid has been deformed.
  inverse_mass_operator.update();
}

template<int dim, typename Number>
void
Operator<dim, Number>::prepare_coarsening_and_refinement(std::vector<VectorType *> & vectors)
{
  solution_transfer = std::make_shared<ExaDG::SolutionTransfer<dim, VectorType>>(dof_handler);

  solution_transfer->prepare_coarsening_and_refinement(vectors);
}

template<int dim, typename Number>
void
Operator<dim, Number>::interpolate_after_coarsening_and_refinement(
  std::vector<VectorType *> & vectors)
{
  solution_transfer->interpolate_after_coarsening_and_refinement(vectors);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::solve(BlockVectorType &       sol,
                             BlockVectorType const & rhs,
                             bool const         update_preconditioner,
                             double const       scaling_factor,
                             double const       time,
                             VectorType const * velocity)
{
  update_conv_diff_operator(time, scaling_factor, velocity);

  iterative_solver->update_preconditioner(update_preconditioner);

  unsigned int const iterations = iterative_solver->solve(sol, rhs);

  return iterations;
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_n_blocks() const
{
  return n_blocks;
}

template<int dim, typename Number>
TurbulenceModelData const &
Operator<dim, Number>::get_turbulence_model_data() const
{
  return param.turbulence_model_data;
}

template<int dim, typename Number>
void
Operator<dim, Number>::vmult(BlockVectorType & dst, BlockVectorType const & src)
const
{
  // block-diagonal part: operator of each transport variable acting on its own  block
  for(unsigned int b = 0; b < n_blocks; ++b)
    combined_operator[b]->vmult(dst.block(b), src.block(b));


  // Newton_Implicit part of the turbulence terms treated Implicit
  if(param.turbulence_model_data.is_active)
    turbulence_source_operator.apply_add_newton_implicit(dst, src);
}

template<int dim, typename Number>
double
Operator<dim, Number>::calculate_time_step_cfl_global(double const time) const
{
  double max_velocity = 0.0;
  if(param.analytical_velocity_field)
  {
    max_velocity = calculate_maximum_velocity(time);
  }

  // max_velocity computed above might be zero depending on the initial velocity field -> dt would
  // tend to infinity
  max_velocity = std::max(max_velocity, param.max_velocity);

  std::shared_ptr<dealii::Function<dim>> const velocity_field =
    std::make_shared<dealii::Functions::ConstantFunction<dim>>(max_velocity, dim);

  return calculate_time_step_cfl_local<dim, Number>(*matrix_free,
                                                    get_dof_index_velocity(),
                                                    get_quad_index(),
                                                    velocity_field,
                                                    time,
                                                    param.degree,
                                                    param.exponent_fe_degree_convection,
                                                    CFLConditionType::VelocityComponents,
                                                    mpi_comm);
}

template<int dim, typename Number>
double
Operator<dim, Number>::calculate_time_step_max_efficiency(
  unsigned int const order_time_integrator) const
{
  double const h_min = calculate_minimum_element_length();

  return ExaDG::calculate_time_step_max_efficiency(h_min, param.degree, order_time_integrator);
}

template<int dim, typename Number>
double
Operator<dim, Number>::calculate_time_step_diffusion() const
{
  double const h_min = calculate_minimum_element_length();

  return ExaDG::calculate_const_time_step_diff(param.diffusivity,
                                               h_min,
                                               param.degree,
                                               param.exponent_fe_degree_diffusion);
}

template<int dim, typename Number>
double
Operator<dim, Number>::calculate_time_step_cfl_numerical_velocity(VectorType const & velocity) const
{
  return calculate_time_step_cfl_local<dim, Number>(*matrix_free,
                                                    get_dof_index_velocity(),
                                                    get_quad_index(),
                                                    velocity,
                                                    param.degree,
                                                    param.exponent_fe_degree_convection,
                                                    param.adaptive_time_stepping_cfl_type,
                                                    mpi_comm);
}

template<int dim, typename Number>
double
Operator<dim, Number>::calculate_time_step_cfl_analytical_velocity(double const time) const
{
  return calculate_time_step_cfl_local<dim, Number>(*matrix_free,
                                                    get_dof_index_velocity(),
                                                    get_quad_index(),
                                                    field_functions->velocity,
                                                    time,
                                                    param.degree,
                                                    param.exponent_fe_degree_convection,
                                                    param.adaptive_time_stepping_cfl_type,
                                                    mpi_comm);
}

template<int dim, typename Number>
double
Operator<dim, Number>::calculate_maximum_velocity(double const time) const
{
  return calculate_max_velocity(dof_handler.get_triangulation(),
                                field_functions->velocity,
                                time,
                                mpi_comm);
}

template<int dim, typename Number>
double
Operator<dim, Number>::calculate_minimum_element_length() const
{
  return calculate_minimum_vertex_distance(dof_handler.get_triangulation(),
                                           *get_mapping(),
                                           mpi_comm);
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
Operator<dim, Number>::get_dof_handler() const
{
  return dof_handler;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
Operator<dim, Number>::get_dof_handler_eddy_viscosity() const
{
  return matrix_free_data->get_dof_handler(get_dof_name_eddy_viscosity());
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
Operator<dim, Number>::get_dof_handler_velocity() const
{
  return matrix_free_data->get_dof_handler(get_dof_name_velocity());
}

template<int dim, typename Number>
dealii::types::global_dof_index
Operator<dim, Number>::get_number_of_dofs() const
{
  return dof_handler.n_dofs();
}

template<int dim, typename Number>
dealii::MatrixFree<dim, Number> const &
Operator<dim, Number>::get_matrix_free() const
{
  return *matrix_free;
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_dof_name() const
{
  return field + dof_index_std;
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_quad_name() const
{
  return field + quad_index_std;
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_quad_name_overintegration() const
{
  return field + quad_index_overintegration;
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_quad_name_eddy_velocity() const
{
  return field + quad_index_eddy_velocity;
}

template<int dim, typename Number>
bool
Operator<dim, Number>::needs_own_dof_handler_velocity() const
{
  return param.analytical_velocity_field and param.store_analytical_velocity_in_dof_vector;
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index() const
{
  return matrix_free_data->get_dof_index(get_dof_name());
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_dof_name_velocity() const
{
  if(needs_own_dof_handler_velocity())
  {
    return field + dof_index_velocity;
  }
  else // external velocity field not hosted by the convection-diffusion operator
{
    return dof_index_velocity_external;
  }
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index_velocity() const
{
  if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    return matrix_free_data->get_dof_index(get_dof_name_velocity());
  else
    return dealii::numbers::invalid_unsigned_int;
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_quad_index() const
{
  return matrix_free_data->get_quad_index(field + quad_index_std);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_quad_index_overintegration() const
{
  return matrix_free_data->get_quad_index(field + quad_index_overintegration);
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_quad_index_eddy_velocity() const
{
  return matrix_free_data->get_quad_index(field + quad_index_eddy_velocity);
}

template<int dim, typename Number>
std::shared_ptr<dealii::Mapping<dim> const>
Operator<dim, Number>::get_mapping() const
{
  return mapping;
}

template<int dim, typename Number>
dealii::AffineConstraints<Number> const &
Operator<dim, Number>::get_constraints() const
{
  return affine_constraints;
}

template<int dim, typename Number>
void
Operator<dim, Number>::update_eddy_viscosity(BlockVectorType const & src,
                                             VectorType const * velocity) const
{
  AssertThrow(param.diffusive_problem(),
              dealii::ExcMessage("Updating viscosity reasonable for diffusive problem"));

  if(param.turbulence_model_data.is_active)
  {
    if (velocity != nullptr) 
    {
      turbulence_model_ptr->set_velocity_ptr(*velocity);
    }
    turbulence_model_ptr->set_viscosity(src);
  }
}

template<int dim, typename Number>
void
Operator<dim, Number>::get_eddy_viscosity(VectorType & dst) const
{
  turbulence_model_ptr->get_eddy_viscosity(dst);
}

template<int dim, typename Number>
std::string
Operator<dim, Number>::get_dof_name_eddy_viscosity() const
{
  return dof_index_eddy_viscosity;
}

template<int dim, typename Number>
unsigned int
Operator<dim, Number>::get_dof_index_eddy_viscosity() const
{
  return matrix_free_data->get_dof_index(get_dof_name_eddy_viscosity());
}

template<int dim, typename Number>
void
Operator<dim, Number>::get_turbulent_kinetic_energy(VectorType & dst,
                                                    BlockVectorType const & solution) const
{
  turbulence_model_ptr->get_turbulent_kinetic_energy(dst, solution);
}

template<int dim, typename Number>
void
Operator<dim, Number>::update_time_step_size(double const dt,
                                             double const time,
                                             double const end_time,
                                             double const time_step_number) const
{
  turbulence_source_operator.set_time_step_size(dt, time, end_time, time_step_number);
}

template<int dim, typename Number>
void
Operator<dim, Number>::apply_modal_filter(BlockVectorType & solution) const
{
  for(unsigned int b = 0; b < n_blocks; ++b)
    modal_filter.apply_filter(solution.block(b));
}

template<int dim, typename Number>
void
Operator<dim, Number>::enforce_positivity(BlockVectorType & solution) const
{
  if(param.turbulence_model_data.is_active)
  {
    turbulence_model_ptr->enforce_positivity(solution);
  }
}

template class Operator<2, float>;
template class Operator<2, double>;

template class Operator<3, float>;
template class Operator<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
