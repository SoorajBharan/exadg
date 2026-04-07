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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SOLVER_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SOLVER_H_

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

// driver
#include <exadg/convection_diffusion/driver.h>

// utilities
#include <exadg/operators/resolution_parameters.h>
#include <exadg/time_integration/resolution_parameters.h>
#include <exadg/utilities/enum_patterns.h>
#include <exadg/utilities/general_parameters.h>

// application
#include <exadg/convection_diffusion/user_interface/declare_get_application.h>

namespace ExaDG
{
void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  GeneralParameters general;
  general.add_parameters(prm);

  SpatialResolutionParametersMinMax spatial;
  spatial.add_parameters(prm);

  TemporalResolutionParameters temporal;
  temporal.add_parameters(prm);

  // we have to assume a default dimension and default Number type
  // for the automatic generation of a default input file
  unsigned int const Dim = 2;
  typedef double     Number;
  unsigned int const N_Components = 1;
  ConvDiff::get_application<Dim, N_Components, Number>(input_file, MPI_COMM_WORLD)->add_parameters(prm);

  prm.print_parameters(input_file,
                       dealii::ParameterHandler::Short |
                         dealii::ParameterHandler::KeepDeclarationOrder);
}

template<int dim, int n_components, typename Number>
void
run(std::string const & input_file,
    unsigned int const  degree,
    unsigned int const  refine_space,
    unsigned int const  refine_time,
    MPI_Comm const &    mpi_comm,
    bool const          is_test)
{
  dealii::Timer timer;
  timer.restart();

  std::shared_ptr<ConvDiff::ApplicationBase<dim, n_components, Number>> application =
    ConvDiff::get_application<dim, n_components, Number>(input_file, mpi_comm);

  application->set_parameters_convergence_study(degree, refine_space, refine_time);

  std::shared_ptr<ConvDiff::Driver<dim, n_components, Number>> driver =
    std::make_shared<ConvDiff::Driver<dim, n_components, Number>>(mpi_comm, application, is_test, false);

  driver->setup();

  driver->solve();

  if(not(is_test))
    driver->print_performance_results(timer.wall_time());
}

} // namespace ExaDG

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;

  if(argc == 1)
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      // clang-format off
      std::cout << "To run the program, use:      ./solver input_file" << std::endl
                << "To setup the input file, use: ./solver input_file --help" << std::endl;
      // clang-format on
    }

    return 0;
  }
  else if(argc >= 2)
  {
    input_file = std::string(argv[1]);

    if(argc == 3 and std::string(argv[2]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        ExaDG::create_input_file(input_file);

      return 0;
    }
  }

  ExaDG::GeneralParameters                 general(input_file);
  ExaDG::SpatialResolutionParametersMinMax spatial(input_file);
  ExaDG::TemporalResolutionParameters      temporal(input_file);

  // k-refinement
  for(unsigned int degree = spatial.degree_min; degree <= spatial.degree_max; ++degree)
  {
    // h-refinement
    for(unsigned int refine_space = spatial.refine_space_min;
        refine_space <= spatial.refine_space_max;
        ++refine_space)
    {
      // dt-refinement
      for(unsigned int refine_time = temporal.refine_time_min;
          refine_time <= temporal.refine_time_max;
          ++refine_time)
      {
        // run the simulation
        if(general.dim == 2 and general.precision == "float" and general.n_components == 1)
        {
          ExaDG::run<2, 1, float>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        }
        else if(general.dim == 2 and general.precision == "double" and general.n_components == 2)
        {
          ExaDG::run<2, 2, float>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        }
        else if(general.dim == 2 and general.precision == "double" and general.n_components == 1)
        {
          ExaDG::run<2, 1, double>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        }
        else if(general.dim == 2 and general.precision == "double" and general.n_components == 2)
        {
          ExaDG::run<2, 2, double>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        }
        else if(general.dim == 3 and general.precision == "float" and general.n_components == 1)
        {
          ExaDG::run<3, 1, float>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        }
        else if(general.dim == 3 and general.precision == "float" and general.n_components == 2)
        {
          ExaDG::run<3, 2, float>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        }
        else if(general.dim == 3 and general.precision == "double" and general.n_components == 1) 
        {
          ExaDG::run<3, 1, double>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        }
        else if(general.dim == 3 and general.precision == "double" and general.n_components == 2)
        {
          ExaDG::run<3, 2, double>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        }
        else
        {
          AssertThrow(false,
                      dealii::ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
        }
      }
    }
  }

  return 0;
}

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SOLVER_H_ */
