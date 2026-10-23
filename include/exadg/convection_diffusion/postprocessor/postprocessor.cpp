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

#include <deal.II/base/exceptions.h>
#include <deal.II/base/smartpointer.h>
#include <exadg/convection_diffusion/postprocessor/postprocessor.h>
#include <exadg/convection_diffusion/spatial_discretization/operator.h>
#include <exadg/postprocessor/solution_field.h>
#include <exadg/convection_diffusion/user_interface/component_function.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
PostProcessor<dim, Number>::PostProcessor(PostProcessorData<dim> const & pp_data_in,
                                                        MPI_Comm const &               mpi_comm_in)
  : mpi_comm(mpi_comm_in),
  pp_data(pp_data_in),
  output_generator(mpi_comm_in),
  line_plot_calculator(std::make_shared<LinePlotCalculator<dim, Number>>(mpi_comm_in))
{
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup(Operator<dim, Number> const & pde_operator)
{
  conv_diff_operator = &pde_operator;
  unsigned int const n_blocks = pde_operator.get_n_blocks();

  error_calculators.resize(n_blocks);
  for(unsigned int b = 0; b < n_blocks; ++b)
  {
    ErrorCalculationData<dim> error_data_block = pp_data.error_data;

    error_data_block.analytical_solution = get_component_function(pp_data.error_data.analytical_solution, b);

    if(n_blocks > 1)
    {
      error_data_block.name = pp_data.error_data.name + " block " + std::to_string(b);
    }

    error_calculators[b] = std::make_shared<ErrorCalculator<dim, Number>>(mpi_comm);
    error_calculators[b]->setup(pde_operator.get_dof_handler(),
                                *pde_operator.get_mapping(),
                                error_data_block);
  }  

  if(pde_operator.get_turbulence_model_data().is_active)
    output_generator.setup(pde_operator.get_dof_handler(),
                           *pde_operator.get_mapping(),
                           pp_data.output_data,
                           pde_operator.get_turbulence_model_data());
  else
    output_generator.setup(pde_operator.get_dof_handler(),
                           *pde_operator.get_mapping(),
                           pp_data.output_data);

  // The eddy viscosity DoFHandler only exists if the turbulence model is active, so line plots and
  // eddy viscosity output are only available for the RANS scalar.
  if(pde_operator.get_turbulence_model_data().is_active)
  {
    line_plot_calculator->setup(pde_operator.get_dof_handler(),
                                pde_operator.get_dof_handler_eddy_viscosity(),
                                *pde_operator.get_mapping(),
                                pp_data.line_plot_data);

    initialize_additional_fields();

    conv_diff_operator->initialize_dof_vector_eddy_viscosity(nu_t);
  }
  else
  {
    AssertThrow(not pp_data.line_plot_data.time_control_data.is_active,
                dealii::ExcMessage("Line plots are only implemented for the turbulence model."));
    AssertThrow(not pp_data.output_data.write_eddy_viscosity,
                dealii::ExcMessage("write_eddy_viscosity requires an active turbulence model."));
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::setup_after_coarsening_and_refinement()
{
  // The `error_calculator` and `output_generator` do not require any additional setup after
  // coarsening and refinement.
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::do_postprocessing(BlockVectorType const &     solution,
                                              double const           time,
                                              types::time_step const time_step_number)
{
  invalidate_additional_fields();
  std::vector<dealii::SmartPointer<SolutionField<dim, Number>>> additional_fields_vtu;
  if (pp_data.output_data.write_eddy_viscosity) {
    eddy_viscosity.evaluate(nu_t);
    additional_fields_vtu.push_back(&eddy_viscosity);
  }

  for(unsigned int b = 0; b < error_calculators.size(); ++b)
  {
    if(error_calculators[b]->time_control.needs_evaluation(time,
                                                           time_step_number))
      error_calculators[b]->evaluate(solution.block(b),
                                     time,
                                     Utilities::is_unsteady_timestep(time_step_number));
  }

  if(output_generator.time_control.needs_evaluation(time, time_step_number))
    output_generator.evaluate(solution,
                              additional_fields_vtu,
                              time,
                              Utilities::is_unsteady_timestep(time_step_number));

  if(line_plot_calculator->time_control.needs_evaluation(time, time_step_number))
  {
    conv_diff_operator->get_eddy_viscosity(nu_t);
    
    line_plot_calculator->evaluate(solution, nu_t);
  }
}


template<int dim, typename Number>
void
PostProcessor<dim, Number>::initialize_additional_fields()
{
  // eddy_viscosity
  if(pp_data.output_data.write_eddy_viscosity )
  {
    eddy_viscosity.type              = SolutionFieldType::scalar;
    eddy_viscosity.name              = "eddy_viscosity";
    eddy_viscosity.dof_handler       = &conv_diff_operator->get_dof_handler_eddy_viscosity();
    eddy_viscosity.initialize_vector = [&](VectorType & dst) {
      conv_diff_operator->initialize_dof_vector_eddy_viscosity(dst);
    };
    eddy_viscosity.recompute_solution_field = [&](VectorType & dst, VectorType const & /*src*/) {
      conv_diff_operator->get_eddy_viscosity(dst);
    };

    eddy_viscosity.reinit();
  }
}

template<int dim, typename Number>
void
PostProcessor<dim, Number>::invalidate_additional_fields()
{
  eddy_viscosity.invalidate();
}


template class PostProcessor<2, float>;
template class PostProcessor<3, float>;

template class PostProcessor<2, double>;
template class PostProcessor<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
