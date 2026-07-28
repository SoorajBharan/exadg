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
#include <exadg/incompressible_navier_stokes/postprocessor/wall_shear_stress_calculator.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
WallShearStressCalculator<dim, Number>::WallShearStressCalculator(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_in,
  unsigned int const                      quad_index_in,
  WallShearStressCalculatorData<dim> const & data_in,
  MPI_Comm const &                        comm_in)
  : data(data_in),
    matrix_free(matrix_free_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    area_has_been_initialized(false),
    area(0.0),
    clear_files(true),
    mpi_comm(comm_in)
{
  if(data.calculate and data.write_to_file)
    create_directories(data.directory, mpi_comm);
}

template<int dim, typename Number>
Number
WallShearStressCalculator<dim, Number>::calculate_mean_wall_shear_stress(VectorType const & velocity,
                                                                  double const &     time)
{
  if(data.calculate == true)
  {
    if(area_has_been_initialized == false)
    {
      this->area = calculate_area();

      area_has_been_initialized = true;
    }

    Number total_shear_force = do_calculate_wall_shear_stress(velocity);
    Number mean_tau_w = total_shear_force / this->area;
    Number u_tau = std::sqrt(mean_tau_w);

    AssertThrow(area_has_been_initialized == true,
                dealii::ExcMessage("Area has not been initialized."));
    AssertThrow(this->area != 0.0, dealii::ExcMessage("Area has not been initialized."));

    write_output(mean_tau_w, u_tau, time);
    return mean_tau_w;
  }
  else
  {
    return -1.0;
  }
}

template<int dim, typename Number>
void
WallShearStressCalculator<dim, Number>::write_output(Number const &      value,
                                                     double const &      time,
                                                     std::string const & name) const
{
  // write output file
  if(data.write_to_file == true and dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::string filename = data.directory + data.filename;

    std::ofstream f;
    if(clear_files == true)
    {
      f.open(filename.c_str(), std::ios::trunc);
      f << std::endl << "  Time                " + name << std::endl;

      clear_files = false;
    }
    else
    {
      f.open(filename.c_str(), std::ios::app);
    }

    unsigned int precision = 12;
    f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
      << std::setw(precision + 8) << value << std::endl;
  }
}


template<int dim, typename Number>
void
WallShearStressCalculator<dim, Number>::write_output(Number const & tau_w,
                                                     Number const & u_tau,
                                                     double const & time) const
{
  if(data.write_to_file == true && dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::string filename = data.directory + data.filename;
    std::ofstream f;
    if(clear_files == true)
    {
      f.open(filename.c_str(), std::ios::trunc);
      f << std::endl << "  Time                Tau_w                 u_tau" << std::endl;
      clear_files = false;
    }
    else
    {
      f.open(filename.c_str(), std::ios::app);
    }

    unsigned int precision = 12;
    f << std::scientific << std::setprecision(precision) 
      << std::setw(precision + 8) << time
      << std::setw(precision + 8) << tau_w 
      << std::setw(precision + 8) << u_tau << std::endl;
  }
}

template<int dim, typename Number>
Number
WallShearStressCalculator<dim, Number>::calculate_area() const
{
  FaceIntegratorU integrator(matrix_free, true, dof_index, quad_index);

  Number area = 0.0;

  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
      face++)
  {
    typename std::set<dealii::types::boundary_id>::iterator it;
    dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

    it = data.boundary_IDs.find(boundary_id);
    if(it != data.boundary_IDs.end())
    {
      integrator.reinit(face);

      scalar area_local = dealii::make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        area_local += integrator.JxW(q);
      }

      // sum over all entries of dealii::VectorizedArray
      for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
        area += area_local[n];
    }
  }

  area = dealii::Utilities::MPI::sum(area, mpi_comm);

  return area;
}

template<int dim, typename Number>
Number
WallShearStressCalculator<dim, Number>::do_calculate_wall_shear_stress(VectorType const & velocity) const
{
  // Set up the FaceIntegrator
  FaceIntegratorU integrator(matrix_free, true, dof_index, quad_index);

  // initialize with zero since we accumulate into this variable
  Number shear_force = 0.0;

  // Local face loop over boundary faces only
  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
      face++)
  {
    typename std::set<dealii::types::boundary_id>::iterator it;
    dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

    // If the face belongs to our designated walls
    if(data.boundary_IDs.find(boundary_id) != data.boundary_IDs.end())
    {
      integrator.reinit(face);
      integrator.read_dof_values(velocity);
      
      // We only need gradients to evaluate the normal derivative
      integrator.evaluate(dealii::EvaluationFlags::gradients);

      scalar local_force = dealii::make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // Compute dU/dn at the quadrature point
        auto du_dn = integrator.get_normal_derivative(q);

        // Magnitude squared: |dU/dn|^2
        scalar mag_du_dn_sq = 0.0;
        for(unsigned int c = 0; c < dim; ++c) 
        {
          mag_du_dn_sq += du_dn[c] * du_dn[c];
        }

        // tau_w = kinematic_viscosity * |dU/dn|
        scalar tau_mag = data.kinematic_viscosity * std::sqrt(mag_du_dn_sq);

        local_force += tau_mag * integrator.JxW(q);
      }

      // sum over active entries of the dealii::VectorizedArray
      for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
      {
        shear_force += local_force[n];
      }
    }
  }

  // Sum the integrated shear force across all MPI processes
  shear_force = dealii::Utilities::MPI::sum(shear_force, mpi_comm);

  return shear_force;
}

template class WallShearStressCalculator<2, float>;
template class WallShearStressCalculator<2, double>;

template class WallShearStressCalculator<3, float>;
template class WallShearStressCalculator<3, double>;

} // namespace IncNS
} // namespace ExaDG
