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

#ifndef CONV_DIFF_DIFFUSIVE_OPERATOR
#define CONV_DIFF_DIFFUSIVE_OPERATOR

#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>


namespace ExaDG
{
namespace ConvDiff
{
namespace Operators
{
struct DiffusiveKernelData
{
  DiffusiveKernelData() : IP_factor(1.0),
    diffusivity(1.0),
    rans_model(false),
    positivity_preserving_limiter(PositivityPreservingLimiter::Undefined)
  {
  }

  double IP_factor;
  double diffusivity;

  bool rans_model;
  PositivityPreservingLimiter positivity_preserving_limiter;

  TurbulenceModelData turbulence_model_data;
  unsigned int dof_index_eddy_viscosity;

  std::vector<double> inverse_sigma;
};

template<int dim, int n_components, typename Number>
class DiffusiveKernel
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number>                         scalar;

  typedef CellIntegrator<dim, n_components, Number> IntegratorCell;
  typedef FaceIntegrator<dim, n_components, Number> IntegratorFace;

  using value_type = typename IntegratorCell::value_type;
  using gradient_type = typename IntegratorCell::gradient_type;

  typedef CellIntegrator<dim, 1, Number> IntegratorCellScalar;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFaceScalar;
public:
  DiffusiveKernel() : degree(1), tau(dealii::make_vectorized_array<Number>(0.0))
  {
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         DiffusiveKernelData const &             data_in,
         unsigned int const                      dof_index,
         unsigned int const                      quad_index)
  {
    data = data_in;

    dealii::FiniteElement<dim> const & fe = matrix_free.get_dof_handler(dof_index).get_fe();
    degree                                = fe.degree;

    calculate_penalty_parameter(matrix_free, dof_index);

    AssertThrow(data.diffusivity > (0.0 - std::numeric_limits<double>::epsilon()),
                dealii::ExcMessage("Diffusivity is not set!"));

    if(data.rans_model)
    {
      integrator_cell_eddy_viscosity =
        std::make_shared<IntegratorCellScalar>(matrix_free, data.dof_index_eddy_viscosity, quad_index);
      integrator_face_eddy_viscosity_m = std::make_shared<IntegratorFaceScalar>(
        matrix_free, true, data.dof_index_eddy_viscosity, quad_index);
      integrator_face_eddy_viscosity_p = std::make_shared<IntegratorFaceScalar>(
        matrix_free, false, data.dof_index_eddy_viscosity, quad_index);
    }
  }

  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter, matrix_free, dof_index);
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::gradients;
    flags.cell_integrate = dealii::EvaluationFlags::gradients;

    flags.face_evaluate  = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
    flags.face_integrate = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

    return flags;
  }

  static MappingFlags
  get_mapping_flags(bool const compute_interior_face_integrals,
                    bool const compute_boundary_face_integrals)
  {
    MappingFlags flags;

    flags.cells = dealii::update_gradients | dealii::update_JxW_values;
    if(compute_interior_face_integrals)
      flags.inner_faces =
        dealii::update_gradients | dealii::update_JxW_values | dealii::update_normal_vectors;
    if(compute_boundary_face_integrals)
      flags.boundary_faces = dealii::update_gradients | dealii::update_JxW_values |
                             dealii::update_normal_vectors | dealii::update_quadrature_points;

    return flags;
  }

  void
  reinit_cell(unsigned const int cell) const
  {
    if(data.rans_model)
    {
      integrator_cell_eddy_viscosity->reinit(cell);
      integrator_cell_eddy_viscosity->gather_evaluate(*eddy_viscosity,
                                                      dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_face(IntegratorFace &   integrator_m,
              IntegratorFace &   integrator_p,
              unsigned int const dof_index,
              unsigned int const face) const
  {
    tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                   integrator_p.read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);

    if(data.rans_model)
    {
      integrator_face_eddy_viscosity_m->reinit(face);
      integrator_face_eddy_viscosity_p->reinit(face);
      integrator_face_eddy_viscosity_m->gather_evaluate(*eddy_viscosity,
                                                        dealii::EvaluationFlags::values);
      integrator_face_eddy_viscosity_p->gather_evaluate(*eddy_viscosity,
                                                        dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m,
                       unsigned int const dof_index,
                       unsigned int const face) const
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);

    if(data.rans_model)
    {
      integrator_face_eddy_viscosity_m->reinit(face);
      integrator_face_eddy_viscosity_m->gather_evaluate(*eddy_viscosity,
                                                        dealii::EvaluationFlags::values);
    }
  }

  void
  reinit_face_cell_based(dealii::types::boundary_id const boundary_id,
                         IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p,
                         unsigned int const               dof_index) const
  {
    if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
    {
      tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                     integrator_p.read_cell_data(array_penalty_parameter)) *
            IP::get_penalty_factor<dim, Number>(
              degree,
              get_element_type(
                integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
              data.IP_factor);
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(array_penalty_parameter) *
            IP::get_penalty_factor<dim, Number>(
              degree,
              get_element_type(
                integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
              data.IP_factor);
    }
  }

void
  set_eddy_viscosity_ptr(VectorType const & eddy_viscosity_in)
  {
    eddy_viscosity.own() = eddy_viscosity_in;
    eddy_viscosity->update_ghost_values();
  }

  inline DEAL_II_ALWAYS_INLINE //
    value_type
    calculate_gradient_flux(value_type const & value_m,
                            value_type const & value_p,
                            unsigned int const q,
                            bool boundary_face) const
  {
    value_type gradient_flux;
    value_type effective_viscosity;
    if(boundary_face)
    {
      effective_viscosity = get_int_face_eddy_viscosity(q);
    }
    else
    {
      effective_viscosity = 0.5 * (get_int_face_eddy_viscosity(q) + get_ext_face_eddy_viscosity(q));
    }

    if constexpr(n_components == 1)
    {
      gradient_flux = -0.5 * effective_viscosity * (value_m - value_p);
    }
    else
    {
      for(unsigned int c = 0; c < n_components; ++c)
        gradient_flux[c] = -0.5 * effective_viscosity[c] * (value_m[c] - value_p[c]);
    }

    return gradient_flux;
  }

  /*
   * Calculation of gradient flux. Strictly speaking, this value is not a numerical flux since the
   * flux is multiplied by the normal vector, i.e., "gradient_flux" = numerical_flux * normal, where
   * normal denotes the normal vector of element e⁻.
   */
  inline DEAL_II_ALWAYS_INLINE //
    value_type
    calculate_value_flux(value_type const & normal_gradient_m,
                         value_type const & normal_gradient_p,
                         value_type const & value_m,
                         value_type const & value_p,
                         unsigned int const q,
                         bool boundary_face) const
  {
    value_type value_flux;
    value_type effective_viscosity;
    if(boundary_face)
    {
      effective_viscosity = get_int_face_eddy_viscosity(q);
    }
    else
    {
      effective_viscosity = 0.5 * (get_int_face_eddy_viscosity(q) + get_ext_face_eddy_viscosity(q));
    }

    if constexpr(n_components == 1)
    {
      value_flux = effective_viscosity * (0.5 * (normal_gradient_m + normal_gradient_p) - tau * (value_m - value_p));
    }
    else
    {
      for(unsigned int c = 0; c < n_components; ++c)
        value_flux[c] =
          effective_viscosity[c] * (0.5 * (normal_gradient_m[c] + normal_gradient_p[c]) - tau * (value_m[c] - value_p[c]));
    }

    return value_flux;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    gradient_type
    get_volume_flux(IntegratorCell & integrator, unsigned int const q) const
  {
    value_type effective_viscosity = get_effective_cell_viscosity(q);
    gradient_type solution_gradient = integrator.get_gradient(q);

    gradient_type volume_flux;
    if constexpr(n_components == 1)
    {
      volume_flux = solution_gradient * effective_viscosity; 
    }
    else
  {
      for(unsigned int c = 0; c < n_components; ++c)
        volume_flux[c] = solution_gradient[c] * effective_viscosity[c];
    }

    return volume_flux;
  }

  value_type
  get_effective_cell_viscosity(unsigned const int q) const
  {
    value_type nu_eff;
    scalar nu_laminar = dealii::make_vectorized_array<Number>(data.diffusivity);
    scalar nu_t = dealii::make_vectorized_array<Number>(0.0);
    scalar coefficient = dealii::make_vectorized_array<Number>(0.0);

    if(data.rans_model)
    {
      nu_t = integrator_cell_eddy_viscosity->get_value(q);
      coefficient = dealii::make_vectorized_array<Number>(data.inverse_sigma[0]);
    }


    if constexpr(n_components == 1)
    {
      nu_eff = nu_laminar + nu_t * coefficient;
    }
    else
    {
      for(unsigned int c = 0; c < n_components; ++c)
      {
        nu_eff[c] = nu_laminar + nu_t * dealii::make_vectorized_array<Number>(data.inverse_sigma[c]);
      }
    }

    return nu_eff;
  }

  value_type
  get_int_face_eddy_viscosity(unsigned const int q) const
  {
    value_type nu_eff;
    scalar nu_laminar = dealii::make_vectorized_array<Number>(data.diffusivity);
    scalar nu_t = dealii::make_vectorized_array<Number>(0.0);
    scalar coefficient = dealii::make_vectorized_array<Number>(0.0);

    if(data.rans_model)
    {
      nu_t = integrator_face_eddy_viscosity_m->get_value(q);
      coefficient = dealii::make_vectorized_array<Number>(data.inverse_sigma[0]);
    }

    if constexpr(n_components == 1)
    {
      nu_eff = nu_laminar + nu_t * coefficient;
    }
    else
  {
      for(unsigned int c = 0; c < n_components; ++c)
      {
        nu_eff[c] = nu_laminar + nu_t * dealii::make_vectorized_array<Number>(data.inverse_sigma[c]);
      }
    }

    return nu_eff;
  }

  value_type
  get_ext_face_eddy_viscosity(unsigned const int q) const
  {
    value_type nu_eff;
    scalar nu_laminar = dealii::make_vectorized_array<Number>(data.diffusivity);
    scalar nu_t = dealii::make_vectorized_array<Number>(0.0);
    scalar coefficient = dealii::make_vectorized_array<Number>(0.0);

    if(data.rans_model)
    {
      nu_t = integrator_face_eddy_viscosity_p->get_value(q);
      coefficient = dealii::make_vectorized_array<Number>(data.inverse_sigma[0]);
    }

    if constexpr(n_components == 1)
    {
      nu_eff = nu_laminar + nu_t * coefficient;
    }
    else
    {
      for(unsigned int c = 0; c < n_components; ++c)
      {
        nu_eff[c] = nu_laminar + nu_t * dealii::make_vectorized_array<Number>(data.inverse_sigma[c]);
      }
    }

    return nu_eff;
  }

  void
  set_eddy_viscosity_ptr(VectorType const & eddy_viscosity_in) const
  {
    eddy_viscosity.own() = eddy_viscosity_in;
    eddy_viscosity->update_ghost_values();
  }

  mutable lazy_ptr<VectorType> eddy_viscosity;

private:
  DiffusiveKernelData data;

  unsigned int degree;

  dealii::AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;

  std::shared_ptr<IntegratorCellScalar> integrator_cell_eddy_viscosity;
  std::shared_ptr<IntegratorFaceScalar> integrator_face_eddy_viscosity_m;
  std::shared_ptr<IntegratorFaceScalar> integrator_face_eddy_viscosity_p;
};

} // namespace Operators


template<int dim>
struct DiffusiveOperatorData : public OperatorBaseData
{
  DiffusiveOperatorData() : OperatorBaseData()
  {
  }

  Operators::DiffusiveKernelData kernel_data;

  std::shared_ptr<BoundaryDescriptor<dim> const> bc;
};


template<int dim, int n_components, typename Number>
class DiffusiveOperator : public OperatorBase<dim, Number, n_components>
{
private:
  typedef OperatorBase<dim, Number, n_components> Base;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> vector;

public:
  void
  initialize(dealii::MatrixFree<dim, Number> const &                  matrix_free,
             dealii::AffineConstraints<Number> const &                affine_constraints,
             DiffusiveOperatorData<dim> const &                       data,
             std::shared_ptr<Operators::DiffusiveKernel<dim, n_components, Number>> kernel);

  void
  update();

  void
  set_eddy_viscosity_ptr(dealii::LinearAlgebra::distributed::Vector<Number> const & eddy_viscosity) const;

private:
  void
  reinit_face_derived(IntegratorFace &   integrator_m,
                      IntegratorFace &   integrator_p,
                      unsigned int const face) const final;

  void
  reinit_boundary_face_derived(IntegratorFace & integrator_m, unsigned int const face) const final;

  void
  reinit_face_cell_based_derived(IntegratorFace &                 integrator_m,
                                 IntegratorFace &                 integrator_p,
                                 unsigned int const               cell,
                                 unsigned int const               face,
                                 dealii::types::boundary_id const boundary_id) const final;

  void
  do_cell_integral(IntegratorCell & integrator) const final;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_boundary_integral(IntegratorFace &                   integrator_m,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const final;

  DiffusiveOperatorData<dim> operator_data;

  std::shared_ptr<Operators::DiffusiveKernel<dim, n_components, Number>> kernel;
};
} // namespace ConvDiff
} // namespace ExaDG

#endif
