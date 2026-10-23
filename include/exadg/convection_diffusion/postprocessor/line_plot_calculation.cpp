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

// C/C++
#include <fstream>

// ExaDG
#include <exadg/convection_diffusion/postprocessor/line_plot_calculation.h>
#include <exadg/postprocessor/solution_interpolation.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
LinePlotCalculator<dim, Number>::LinePlotCalculator(MPI_Comm const & comm)
  : mpi_comm(comm), clear_files(true)
{
}

template<int dim, typename Number>
void
LinePlotCalculator<dim, Number>::setup(dealii::DoFHandler<dim> const & dof_handler_in,
                                                     dealii::DoFHandler<dim> const & dof_handler_eddy_viscosity_in,
                                                     dealii::Mapping<dim> const &    mapping_in,
                                                     LinePlotData<dim> const &       line_plot_data_in)
{
  dof_handler = &dof_handler_in;
  dof_handler_eddy_viscosity = &dof_handler_eddy_viscosity_in;
  mapping              = &mapping_in;
  data                 = line_plot_data_in;

  time_control.setup(line_plot_data_in.time_control_data);

  if(line_plot_data_in.time_control_data.is_active)
    create_directories(line_plot_data_in.directory, mpi_comm);
}

template<int dim, typename Number>
void
LinePlotCalculator<dim, Number>::evaluate(
  BlockVectorType const & solution,
  VectorType const & eddy_viscosity) const
{
  // precision
  unsigned int const precision = data.precision;

  // loop over all lines
  for(typename std::vector<std::shared_ptr<Line<dim>>>::const_iterator line = data.lines.begin();
      line != data.lines.end();
      ++line)
  {
    // store all points along current line in a vector
    unsigned int                    n_points = (*line)->n_points;
    std::vector<dealii::Point<dim>> points(n_points);

    // we consider straight lines with an equidistant distribution of points along the line
    for(unsigned int i = 0; i < n_points; ++i)
      points[i] =
        (*line)->begin + double(i) / double(n_points - 1) * ((*line)->end - (*line)->begin);

    // filename prefix for current line
    std::string filename_prefix = data.directory + (*line)->name;

    // write output for all specified quantities
    for(std::vector<std::shared_ptr<Quantity>>::const_iterator quantity =
          (*line)->quantities.begin();
        quantity != (*line)->quantities.end();
        ++quantity)
    {
      if((*quantity)->type == QuantityType::RANSTransportVariables)
      {
        unsigned int const n_blocks = solution.n_blocks();

        std::vector<std::vector<Number>> solution_vector(n_points,
                                                         std::vector<Number>(n_blocks));

        for(unsigned int i = 0; i < n_points; ++i)
        {
          for(unsigned int b = 0; b < n_blocks; ++b)
          {
            Number value;
            evaluate_scalar_quantity_in_point(
              value, *dof_handler, *mapping, solution.block(b), points[i],
              mpi_comm);    
            solution_vector[i][b] = value;
          }
        }

        // write output to file
        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        {
          std::string filename = filename_prefix + "_rans_scalar" + ".txt";

          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          // headline
          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);

          for(unsigned int b = 0; b < n_blocks; ++b)
            f << std::setw(precision + 8) << std::left
              << "scalar_" + dealii::Utilities::int_to_string(b + 1);

          f << std::endl;

          // loop over all points
          for(unsigned int i = 0; i < n_points; ++i)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << points[i][d];

            for(unsigned int b = 0; b < n_blocks; ++b)
              f << std::setw(precision + 8) << std::left << solution_vector[i][b];

            f << std::endl;
          }
          f.close();
        }
      }
      else if((*quantity)->type == QuantityType::EddyViscosity)
      {
        std::vector<Number> solution_vector(n_points);

        // calculate pressure for all points along line
        for(unsigned int i = 0; i < n_points; ++i)
        {
          Number nu_t;
          evaluate_scalar_quantity_in_point(
            nu_t, *dof_handler_eddy_viscosity, *mapping, eddy_viscosity, points[i], mpi_comm);
          solution_vector[i] = nu_t;
        }

        // write output to file
        if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        {
          std::string filename = filename_prefix + "_eddy_viscosity" + ".txt";

          std::ofstream f;
          if(clear_files)
          {
            f.open(filename.c_str(), std::ios::trunc);
          }
          else
          {
            f.open(filename.c_str(), std::ios::app);
          }

          // headline
          for(unsigned int d = 0; d < dim; ++d)
            f << std::setw(precision + 8) << std::left
              << "x_" + dealii::Utilities::int_to_string(d + 1);
          f << std::setw(precision + 8) << std::left << "nu_t";
          f << std::endl;

          // loop over all points
          for(unsigned int i = 0; i < n_points; ++i)
          {
            f << std::scientific << std::setprecision(precision);

            // write data
            for(unsigned int d = 0; d < dim; ++d)
              f << std::setw(precision + 8) << std::left << points[i][d];
            f << std::setw(precision + 8) << std::left << solution_vector[i];
            f << std::endl;
          }
          f.close();
        }
      }
    } // loop over quantities
  }   // loop over lines
}
template class LinePlotCalculator<2, float>;
template class LinePlotCalculator<2, double>;
template class LinePlotCalculator<3, float>;
template class LinePlotCalculator<3, double>;

} // namespace IncNS
} // namespace ExaDG
