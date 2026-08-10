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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_POSTPROCESSOR_LINE_PLOT_DATA_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_POSTPROCESSOR_LINE_PLOT_DATA_H_

// deal.II
#include <deal.II/base/point.h>

// ExaDG
#include <exadg/postprocessor/time_control_statistics.h>
#include <exadg/utilities/print_functions.h>

#include <memory>
#include <vector>

namespace ExaDG
{
namespace ConvDiff
{
enum class QuantityType
{
  Undefined,
  RANSTransportVariables,
  EddyViscosity
};

struct Quantity
{
  Quantity() : type(QuantityType::Undefined)
  {
  }

  Quantity(QuantityType const & quantity_type) : type(quantity_type)
  {
  }

  virtual ~Quantity()
  {
  }

  QuantityType type;
};

template<int dim>
struct Line
{
  Line() : n_points(2), name("line")
  {
  }

  virtual ~Line()
  {
  }

  /*
   *  begin and end points of line
   */
  dealii::Point<dim> begin;
  dealii::Point<dim> end;

  /*
   *  number of data points written along a line
   */
  unsigned int n_points;

  /*
   *  name of line
   */
  std::string name;

  /*
   *  Specify for which fields/quantities to write output
   */
  std::vector<std::shared_ptr<Quantity>> quantities;
};

template<int dim>
struct LinePlotDataBase
{
  LinePlotDataBase() : directory("output/"), precision(10)
  {
  }

  void
  print_base(dealii::ConditionalOStream & pcout)
  {
    pcout << "  Line plot data:" << std::endl;
    print_parameter(pcout, "Directory", directory);
    print_parameter(pcout, "Precision", precision);
    print_parameter(pcout, "Line", lines.name);
    print_parameter(pcout, "  Quantity", lines.quantity);
  }

  /*
   *  output folder
   */
  std::string directory;

  /*
   *  precision (number of decimal places when writing to files)
   */
  unsigned int precision;

  /*
   *  a vector of lines along which we want to write output
   */
  std::vector<std::shared_ptr<Line<dim>>> lines;
};


template<int dim>
struct LinePlotData : public LinePlotDataBase<dim>
{
  TimeControlData time_control_data;

  void
  print(dealii::ConditionalOStream & pcout)
  {
    if(time_control_data.is_active)
    {
      this->print_base();
      // only makes sense in unsteady case
      time_control_data.print(pcout, true);
    }
  }
};

template<int dim>
struct LinePlotDataStatistics : public LinePlotDataBase<dim>
{
  TimeControlDataStatistics time_control_data_statistics;

  void
  print(dealii::ConditionalOStream & pcout)
  {
    if(time_control_data_statistics.time_control_data.is_active)
    {
      this->print_base();
      // only makes sense in unsteady case
      time_control_data_statistics.print(pcout, true);
    }
  }
};


} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_LINE_PLOT_DATA_H_ */
