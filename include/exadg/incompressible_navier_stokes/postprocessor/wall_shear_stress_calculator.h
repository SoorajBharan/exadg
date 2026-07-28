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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_

#include <exadg/matrix_free/integrators.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim>
struct WallShearStressCalculatorData
{
  WallShearStressCalculatorData()
    : calculate(false),
      write_to_file(false),
      kinematic_viscosity(1.0),
      direction(dealii::Tensor<1, dim, double>()),
      directory("output/"),
      filename("mean_velocity")
  {
  }

  void
  print(dealii::ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << "  Wall shear stress calculator:" << std::endl;

      print_parameter(pcout, "Calculate wall shear stress rate", calculate);
      print_parameter(pcout, "Kinematic viscosity", kinematic_viscosity);
      print_parameter(pcout, "Write results to file", write_to_file);
      if(write_to_file == true)
      {
        print_parameter(pcout, "Directory", directory);
        print_parameter(pcout, "Filename", filename);
      }
    }
  }

  // calculate mean velocity?
  bool calculate;

  // Set containing boundary ID's of the surface area
  // for which we want to calculate the mean velocity.
  // This parameter is only relevant for area-based computation.
  std::set<dealii::types::boundary_id> boundary_IDs;

  // write results to file?
  bool write_to_file;

  double kinematic_viscosity;

  // Direction in which we want to compute the flow rate
  // This parameter is only relevant for volume-based computation.
  dealii::Tensor<1, dim, double> direction;

  // directory and filename
  std::string directory;
  std::string filename;
};

template<int dim, typename Number>
class WallShearStressCalculator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

  typedef WallShearStressCalculator<dim, Number> This;

  typedef dealii::VectorizedArray<Number> scalar;

  WallShearStressCalculator(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                         unsigned int const                      dof_index_in,
                         unsigned int const                      quad_index_in,
                         WallShearStressCalculatorData<dim> const & data_in,
                         MPI_Comm const &                        comm_in);

  /*
   * Calculates the mean velocity through a given cross section of the domain by dividing
   * the flow rate through the cross section area. This function is more general than
   * calculate_mean_velocity_volume() and can be used for domains with varying cross-section
   * area in streamwise direction.
   */
  Number
  calculate_mean_wall_shear_stress(VectorType const & velocity, double const & time);

private:
  void
  write_output(Number const & value, double const & time, std::string const & name) const;

  void
  write_output(Number const & tau_w,
               Number const & u_tau,
               double const & time) const;

  Number
  calculate_area() const;

  Number
  do_calculate_wall_shear_stress(VectorType const & velocity) const;

  WallShearStressCalculatorData<dim> const   data;
  dealii::MatrixFree<dim, Number> const & matrix_free;
  unsigned int                            dof_index, quad_index;
  bool                                    area_has_been_initialized;
  double                                  area;
  mutable bool                            clear_files;

  MPI_Comm const mpi_comm;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_MEAN_VELOCITY_CALCULATOR_H_ */
