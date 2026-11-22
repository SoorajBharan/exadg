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
 *  ______________________________________________________________________
 */

#include <exadg/operators/finite_element.h>
#include <exadg/operators/quadrature.h>
#include <exadg/operators/wall_modelling/fe_omega_enriched_evaluation.h>
#include <exadg/operators/wall_modelling/function_enrichment_rans.h>
#include "exadg/operators/wall_modelling/wall_law_evaluator.h"

#include <deal.II/base/conditional_ostream.h>

#include <array>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>

namespace ExaDG
{
template<int dim, typename Number>
FunctionEnrichmentRANS<dim, Number>::FunctionEnrichmentRANS(
  FunctionEnrichment<dim, Number> const & velocity_enrichment_in,
  double const                            kinematic_viscosity_in)
  : velocity_enrichment(velocity_enrichment_in), kinematic_viscosity(kinematic_viscosity_in)
{
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::add_variable(Variable const & variable)
{
  AssertThrow(matrix_free == nullptr,
              dealii::ExcMessage("FunctionEnrichmentRANS: add_variable() before initialize()."));
  variables.push_back(variable);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::initialize(
  dealii::Triangulation<dim> const &          triangulation,
  std::shared_ptr<dealii::Mapping<dim> const> mapping,
  dealii::DoFHandler<dim> const &             global_dof_handler_rans_in,
  dealii::DoFHandler<dim> const &             global_dof_handler_eddy_viscosity_in)
{
  AssertThrow(variables.size() > 0,
              dealii::ExcMessage("FunctionEnrichmentRANS: no enriched variable (add_variable())."));

  global_dof_handler_rans = &global_dof_handler_rans_in;
  global_dof_handler_eddy_viscosity = &global_dof_handler_eddy_viscosity_in;

  // (1) FE collections, active_fe_index as in FunctionEnrichment:
  //     0 not enriched, 1 enriched, 2 halo (bar part only)
  dealii::FiniteElement<dim> const & fe_rans = global_dof_handler_rans->get_fe();
  AssertThrow(fe_rans.n_components() == 1,
              dealii::ExcMessage("FunctionEnrichmentRANS: scalar RANS finite element expected."));

  // index 0 of the same kind as index 1 (FESystem only compares to FESystem)
  auto const nothing_like = [](dealii::FiniteElement<dim> const & fe) {
    if(dynamic_cast<dealii::FESystem<dim> const *>(&fe) != nullptr)
      return std::unique_ptr<dealii::FiniteElement<dim>>(
        new dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), 1));
    return std::unique_ptr<dealii::FiniteElement<dim>>(new dealii::FE_Nothing<dim>());
  };

  // bar part: RANS FE on enriched and halo cells
  fe_shadow_scalar_collection.push_back(*nothing_like(fe_rans)); // index 0
  fe_shadow_scalar_collection.push_back(fe_rans);                // index 1: enriched
  fe_shadow_scalar_collection.push_back(fe_rans);                // index 2: halo
  dof_handler_shadow_scalar.reinit(triangulation);

  // tilde part: DG scalar of the degree of U_tilde, enriched cells only
  unsigned int const degree_en = velocity_enrichment.get_dof_handler_en_vector().get_fe(1).degree;
  fe_en = create_finite_element<dim>(ElementType::Hypercube, true /* is_dg */, 1, degree_en);
  fe_en_collection.push_back(*nothing_like(*fe_en)); // index 0
  fe_en_collection.push_back(*fe_en);                // index 1: enriched
  fe_en_collection.push_back(*nothing_like(*fe_en)); // index 2: halo, no enrichment
  dof_handler_en.reinit(triangulation);

  // (2) same active_fe_index as FunctionEnrichment (same triangulation, lock-step)
  auto cell_v = velocity_enrichment.get_dof_handler_en_scalar().begin_active();
  auto cell_s = dof_handler_shadow_scalar.begin_active();
  auto cell_e = dof_handler_en.begin_active();
  for(; cell_s != dof_handler_shadow_scalar.end(); ++cell_v, ++cell_s, ++cell_e)
    if(cell_s->is_locally_owned())
    {
      cell_s->set_active_fe_index(cell_v->active_fe_index());
      cell_e->set_active_fe_index(cell_v->active_fe_index());
    }

  dof_handler_shadow_scalar.distribute_dofs(fe_shadow_scalar_collection);

  // locally owned and ghost enriched cells by global active cell index (is_enriched_cell; ghost
  // cells: faces between processes)
  enriched_cells.clear();
  for(auto const & cell : dof_handler_shadow_scalar.active_cell_iterators())
    if((cell->is_locally_owned() or cell->is_ghost()) and
       cell->active_fe_index() == active_fe_index)
      enriched_cells.insert(cell->global_active_cell_index());
  dof_handler_en.distribute_dofs(fe_en_collection);

  constraint_dg.clear();
  constraint_dg.close();

  // (3) MatrixFree: bar (0), tilde (1), tau_w (2), y (3), velocity U_bar (4), U_tilde (5) of the
  //     fluid (DoFHandlers of FunctionEnrichment, same hp layout); enrichment quadrature
  std::vector<dealii::DoFHandler<dim> const *> dof_handlers = {
    &dof_handler_shadow_scalar,
    &dof_handler_en,
    &velocity_enrichment.get_dof_handler_cg_scalar(),
    &velocity_enrichment.get_dof_handler_en_scalar(),
    &velocity_enrichment.get_dof_handler_shadow_vector(),
    &velocity_enrichment.get_dof_handler_en_vector()};

  std::vector<dealii::AffineConstraints<Number> const *> constraints = {
    &constraint_dg,
    &constraint_dg,
    &velocity_enrichment.get_constraint_cg_scalar(),
    &constraint_dg,
    &constraint_dg,
    &constraint_dg};

  // quad_index (0): enrichment quadrature of the velocity
  // quad_index_nodal (1): nodes of the RANS finite element (Gauss-Lobatto)
  std::vector<dealii::Quadrature<dim>> quadratures = {
    *create_quadrature<dim>(ElementType::Hypercube, velocity_enrichment.get_n_q_points_1d()),
    dealii::QGaussLobatto<dim>(fe_rans.degree + 1)};

  typename dealii::MatrixFree<dim, Number>::AdditionalData additional_data;
  dealii::UpdateFlags const flags = dealii::update_values | dealii::update_gradients |
                                    dealii::update_JxW_values | dealii::update_quadrature_points |
                                    dealii::update_normal_vectors;
  additional_data.mapping_update_flags                = flags;
  additional_data.mapping_update_flags_boundary_faces = flags;
  additional_data.mapping_update_flags_inner_faces    = flags;

  matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
  matrix_free->reinit(*mapping, dof_handlers, constraints, quadratures, additional_data);

  matrix_free->initialize_dof_vector(velocity_bar, dof_index_velocity_bar);
  matrix_free->initialize_dof_vector(velocity_tilde, dof_index_velocity_tilde);

  // point a y_c of the wall-function value (a = exp(-sum_{n=1}^p 1/n), p the RANS degree)
  double sum = 0.0;
  for(unsigned int n = 1; n <= fe_rans.degree; ++n)
    sum += 1.0 / double(n);
  a_wall_function = std::exp(-sum);

  // k+ = 0, y_c = 0 until set_turbulent_kinetic_energy()
  wall_cell_fields.reinit(2);
  matrix_free->initialize_dof_vector(wall_cell_fields.block(0), dof_index_dg);
  matrix_free->initialize_dof_vector(wall_cell_fields.block(1), dof_index_dg);
  wall_cell_fields.collect_sizes();
  wall_cell_fields.update_ghost_values();

  // (4) index maps global layout <-> shadow layout (RANS variables, eddy viscosity fields)
  n_scalar_index_enriched =
    build_index_map(*global_dof_handler_rans, scalar_index_global, scalar_index_shadow);

  AssertThrow(global_dof_handler_eddy_viscosity->get_fe().n_dofs_per_cell() ==
                fe_rans.n_dofs_per_cell(),
              dealii::ExcMessage("FunctionEnrichmentRANS: the eddy viscosity FE must be the RANS FE."));
  build_index_map(*global_dof_handler_eddy_viscosity,
                  eddy_viscosity_index_global,
                  eddy_viscosity_index_shadow);
}

template<int dim, typename Number>
unsigned int
FunctionEnrichmentRANS<dim, Number>::build_index_map(
  dealii::DoFHandler<dim> const & dof_handler_global,
  std::vector<unsigned int> &     index_global,
  std::vector<unsigned int> &     index_shadow) const
{
  index_global.clear();
  index_shadow.clear();

  dealii::IndexSet const & owned_global = dof_handler_global.locally_owned_dofs();
  dealii::IndexSet const & owned_shadow = dof_handler_shadow_scalar.locally_owned_dofs();

  unsigned int const dofs_per_cell = dof_handler_global.get_fe().n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> global_dofs(dofs_per_cell);
  std::vector<dealii::types::global_dof_index> shadow_dofs(dofs_per_cell);

  // position in the locally owned index set (= local_element index)
  unsigned int n_enriched = 0;
  for(unsigned int const fe_index : {1u, 2u})
  {
    auto cell_g = dof_handler_global.begin_active();
    auto cell_h = dof_handler_shadow_scalar.begin_active();
    for(; cell_g != dof_handler_global.end(); ++cell_g, ++cell_h)
      if(cell_h->is_locally_owned() and cell_h->active_fe_index() == fe_index)
      {
        cell_g->get_dof_indices(global_dofs);
        cell_h->get_dof_indices(shadow_dofs);
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          index_global.push_back(
            static_cast<unsigned int>(owned_global.index_within_set(global_dofs[i])));
          index_shadow.push_back(
            static_cast<unsigned int>(owned_shadow.index_within_set(shadow_dofs[i])));
        }
      }

    if(fe_index == 1)
      n_enriched = static_cast<unsigned int>(index_global.size());
  }

  return n_enriched;
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::set_velocity(BlockVectorType const & velocity_np_enriched)
{
  // U_bar: global DG layout -> shadow layout of FunctionEnrichment (same DoFHandler as
  // dof_index_velocity_bar here) -> this MatrixFree
  VectorType u_bar_shadow;
  velocity_enrichment.get_matrix_free()->initialize_dof_vector(
    u_bar_shadow, velocity_enrichment.get_dof_index_shadow_vector());
  velocity_enrichment.copy_global_dg_to_wall_layout(velocity_np_enriched.block(0), u_bar_shadow);

  velocity_bar.copy_locally_owned_data_from(u_bar_shadow);
  velocity_tilde.copy_locally_owned_data_from(velocity_np_enriched.block(1));

  // exact velocity gradient at the quadrature points of the enriched cells
  BlockVectorType src(4);
  matrix_free->initialize_dof_vector(src.block(0), dof_index_velocity_bar);
  matrix_free->initialize_dof_vector(src.block(1), dof_index_velocity_tilde);
  matrix_free->initialize_dof_vector(src.block(2), dof_index_wall_distance);
  matrix_free->initialize_dof_vector(src.block(3), dof_index_cg);
  src.block(0).copy_locally_owned_data_from(velocity_bar);
  src.block(1).copy_locally_owned_data_from(velocity_tilde);
  src.block(2).copy_locally_owned_data_from(velocity_enrichment.wall_distance);
  src.block(3).copy_locally_owned_data_from(velocity_enrichment.wall_shear_stress);
  src.collect_sizes();

  n_q_points_velocity_gradient = matrix_free->get_n_q_points(quad_index);
  velocity_gradient_cell.resize(matrix_free->n_cell_batches() * n_q_points_velocity_gradient,
                                tensor());
  velocity_value_cell.resize(matrix_free->n_cell_batches() * n_q_points_velocity_gradient,
                             vector());

  n_q_points_face_velocity_gradient = matrix_free->get_n_q_points_face(quad_index);
  unsigned int const n_faces =
    matrix_free->n_inner_face_batches() + matrix_free->n_boundary_face_batches();
  for(unsigned int side = 0; side < 2; ++side)
  {
    velocity_gradient_face[side].resize(n_faces * n_q_points_face_velocity_gradient, tensor());
    velocity_value_face[side].resize(n_faces * n_q_points_face_velocity_gradient, vector());
  }

  // no vector is written: dst is a placeholder of the cell_loop interface
  VectorType placeholder;
  matrix_free->initialize_dof_vector(placeholder, dof_index_dg);
  matrix_free->loop(&This::cell_loop_velocity_gradient,
                    &This::face_loop_velocity_gradient,
                    &This::boundary_face_loop_velocity_gradient,
                    this,
                    placeholder,
                    src);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_velocity_gradient(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range)
{
  (void)dst;

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number> CellIntegratorVector;
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number>   CellIntegratorScalar;

  // velocity enrichment only on enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  CellIntegratorVector u_bar(matrix_free_in, dof_index_velocity_bar, quad_index, 0, active_fe_index);
  CellIntegratorVector u_tilde(
    matrix_free_in, dof_index_velocity_tilde, quad_index, 0, active_fe_index);
  CellIntegratorScalar y_cell(matrix_free_in, dof_index_wall_distance, quad_index, 0, active_fe_index);
  CellIntegratorScalar tau_cell(matrix_free_in, dof_index_cg, quad_index, 0, active_fe_index);

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    u_bar.reinit(cell);
    u_tilde.reinit(cell);
    y_cell.reinit(cell);
    tau_cell.reinit(cell);
    u_bar.gather_evaluate(src.block(0), flags);
    u_tilde.gather_evaluate(src.block(1), flags);
    y_cell.gather_evaluate(src.block(2), flags);
    tau_cell.gather_evaluate(src.block(3), flags);

    for(unsigned int q = 0; q < u_bar.n_q_points; ++q)
    {
      // psi_u, grad psi_u of the velocity enrichment (Spalding)
      std::pair<scalar, vector> psi_u = velocity_enrichment.get_enrichment_function(
        y_cell.get_value(q), y_cell.get_gradient(q), tau_cell.get_value(q), tau_cell.get_gradient(q));

      // lanes switched off by the y+ criterion and empty lanes: polynomial part only
      for(unsigned int v = 0; v < scalar::size(); ++v)
        if(not is_enrichment_active_lane(cell, v))
        {
          psi_u.first[v] = 0.0;
          for(unsigned int d = 0; d < dim; ++d)
            psi_u.second[d][v] = 0.0;
        }

      // grad u_h = grad U_bar + psi_u grad U_tilde + U_tilde (x) grad psi_u
      velocity_gradient_cell[cell * n_q_points_velocity_gradient + q] =
        u_bar.get_gradient(q) + psi_u.first * u_tilde.get_gradient(q) +
        dealii::outer_product(u_tilde.get_value(q), psi_u.second);

      // u_h = U_bar + psi_u U_tilde
      velocity_value_cell[cell * n_q_points_velocity_gradient + q] =
        u_bar.get_value(q) + psi_u.first * u_tilde.get_value(q);
    }
  }
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::face_loop_velocity_gradient(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range)
{
  (void)dst;

  // inner faces: both sides
  fill_velocity_gradient_face(matrix_free_in, src, range, true);
  fill_velocity_gradient_face(matrix_free_in, src, range, false);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::boundary_face_loop_velocity_gradient(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range)
{
  (void)dst;

  // boundary faces: interior side only
  fill_velocity_gradient_face(matrix_free_in, src, range, true);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::fill_velocity_gradient_face(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range,
  bool const                                    interior)
{
  typedef dealii::FEFaceEvaluation<dim, -1, 0, dim, Number> FaceIntegratorVector;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number>   FaceIntegratorScalar;

  // enriched side: grad U_bar + psi_u grad U_tilde + U_tilde (x) grad psi_u; halo side: grad U_bar
  unsigned int const fe_index = matrix_free_in.get_face_active_fe_index(range, interior);
  if(fe_index != active_fe_index and fe_index != 2)
    return;

  bool const enriched = (fe_index == active_fe_index);

  FaceIntegratorVector u_bar(matrix_free_in, interior, dof_index_velocity_bar, quad_index, 0, fe_index);

  std::shared_ptr<FaceIntegratorVector> u_tilde;
  std::shared_ptr<FaceIntegratorScalar> y_face, tau_face;
  if(enriched)
  {
    u_tilde = std::make_shared<FaceIntegratorVector>(
      matrix_free_in, interior, dof_index_velocity_tilde, quad_index, 0, fe_index);
    y_face = std::make_shared<FaceIntegratorScalar>(
      matrix_free_in, interior, dof_index_wall_distance, quad_index, 0, fe_index);
    tau_face = std::make_shared<FaceIntegratorScalar>(
      matrix_free_in, interior, dof_index_cg, quad_index, 0, fe_index);
  }

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  unsigned int const side  = interior ? 0 : 1;
  unsigned int const lanes = scalar::size();

  for(unsigned int face = range.first; face < range.second; ++face)
  {
    u_bar.reinit(face);
    u_bar.gather_evaluate(src.block(0), flags);

    if(enriched)
    {
      u_tilde->reinit(face);
      y_face->reinit(face);
      tau_face->reinit(face);
      u_tilde->gather_evaluate(src.block(1), flags);
      y_face->gather_evaluate(src.block(2), flags);
      tau_face->gather_evaluate(src.block(3), flags);
    }

    auto const & info  = matrix_free_in.get_face_info(face);
    auto const & cells = interior ? info.cells_interior : info.cells_exterior;

    for(unsigned int q = 0; q < n_q_points_face_velocity_gradient; ++q)
    {
      tensor gradient = u_bar.get_gradient(q);
      vector value    = u_bar.get_value(q);

      if(enriched)
      {
        std::pair<scalar, vector> psi_u =
          velocity_enrichment.get_enrichment_function(y_face->get_value(q),
                                                      y_face->get_gradient(q),
                                                      tau_face->get_value(q),
                                                      tau_face->get_gradient(q));

        // lanes switched off by the y+ criterion and empty lanes: polynomial part only
        for(unsigned int v = 0; v < lanes; ++v)
          if(cells[v] == dealii::numbers::invalid_unsigned_int or
             not is_enrichment_active_lane(cells[v] / lanes, cells[v] % lanes))
          {
            psi_u.first[v] = 0.0;
            for(unsigned int d = 0; d < dim; ++d)
              psi_u.second[d][v] = 0.0;
          }

        gradient += psi_u.first * u_tilde->get_gradient(q) +
                    dealii::outer_product(u_tilde->get_value(q), psi_u.second);
        value += psi_u.first * u_tilde->get_value(q);
      }

      velocity_gradient_face[side][face * n_q_points_face_velocity_gradient + q] = gradient;
      velocity_value_face[side][face * n_q_points_face_velocity_gradient + q]    = value;
    }
  }
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::tensor const &
FunctionEnrichmentRANS<dim, Number>::get_velocity_gradient_face(unsigned int const face,
                                                                bool const         interior,
                                                                unsigned int const q) const
{
  unsigned int const side = interior ? 0 : 1;
  AssertIndexRange(face * n_q_points_face_velocity_gradient + q, velocity_gradient_face[side].size());
  return velocity_gradient_face[side][face * n_q_points_face_velocity_gradient + q];
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::vector const &
FunctionEnrichmentRANS<dim, Number>::get_velocity_value_cell(unsigned int const cell,
                                                             unsigned int const q) const
{
  AssertIndexRange(cell * n_q_points_velocity_gradient + q, velocity_value_cell.size());
  return velocity_value_cell[cell * n_q_points_velocity_gradient + q];
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::vector const &
FunctionEnrichmentRANS<dim, Number>::get_velocity_value_face(unsigned int const face,
                                                             bool const         interior,
                                                             unsigned int const q) const
{
  unsigned int const side = interior ? 0 : 1;
  AssertIndexRange(face * n_q_points_face_velocity_gradient + q, velocity_value_face[side].size());
  return velocity_value_face[side][face * n_q_points_face_velocity_gradient + q];
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::scalar const &
FunctionEnrichmentRANS<dim, Number>::get_wall_value_face(unsigned int const variable,
                                                         unsigned int const face,
                                                         unsigned int const q) const
{
  AssertIndexRange(face * n_q_points_face + q, wall_value_face[variable].size());
  return wall_value_face[variable][face * n_q_points_face + q];
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::scalar const &
FunctionEnrichmentRANS<dim, Number>::get_wall_value_off_face(unsigned int const variable,
                                                             unsigned int const face,
                                                             unsigned int const q) const
{
  AssertIndexRange(face * n_q_points_face + q, wall_value_off_face[variable].size());
  return wall_value_off_face[variable][face * n_q_points_face + q];
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::tensor const &
FunctionEnrichmentRANS<dim, Number>::get_velocity_gradient_cell(unsigned int const cell,
                                                                unsigned int const q) const
{
  AssertIndexRange(cell * n_q_points_velocity_gradient + q, velocity_gradient_cell.size());
  return velocity_gradient_cell[cell * n_q_points_velocity_gradient + q];
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::copy_eddy_viscosity_to_wall_layout(VectorType const & global,
                                                                        VectorType &       shadow) const
{
  shadow = 0.0;

  // locally owned enriched (1) and halo (2) cells
  for(unsigned int i = 0; i < eddy_viscosity_index_global.size(); ++i)
    shadow.local_element(eddy_viscosity_index_shadow[i]) =
      global.local_element(eddy_viscosity_index_global[i]);
}

template<int dim, typename Number>
std::pair<typename FunctionEnrichmentRANS<dim, Number>::scalar,
          typename FunctionEnrichmentRANS<dim, Number>::vector>
FunctionEnrichmentRANS<dim, Number>::evaluate_enrichment_function(Variable const & variable,
                                                                  scalar const &   y,
                                                                  vector const &   grad_y,
                                                                  scalar const &   tau_w,
                                                                  vector const &   grad_tau_w) const
{
  AssertThrow(variable.shape == Shape::Omega,
              dealii::ExcMessage("FunctionEnrichmentRANS: only the omega shape is implemented."));

  // as FunctionEnrichment::tau_w_absolute_minimum: finite y+ and grad psi before the first tau_w
  scalar const tau_min = dealii::make_vectorized_array<Number>(1.e-10);
  scalar const nu      = dealii::make_vectorized_array<Number>(kinematic_viscosity);
  scalar const zero    = dealii::make_vectorized_array<Number>(0.0);

  scalar const tau   = std::max(tau_w, tau_min);
  scalar const u_tau = std::sqrt(tau);

  // grad tau_w only where tau_w,h is not replaced by the minimum
  vector grad_tau;
  for(unsigned int d = 0; d < dim; ++d)
    grad_tau[d] = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
      tau_w, tau_min, grad_tau_w[d], zero);

  // y+ = y u_tau / nu,   grad y+ = (u_tau / nu) grad y + y / (2 nu u_tau) grad tau_w
  scalar const y_plus      = y * u_tau / nu;
  vector const grad_y_plus = (u_tau / nu) * grad_y + (y / (Number(2.0) * nu * u_tau)) * grad_tau;

  // psi_omega(y+) and d psi_omega / d y+ (capped blend)
  std::pair<scalar, scalar> const psi_omega = variable.omega_wall_law(y_plus);

  // ln omega: psi = ln(psi_omega),   grad psi = (psi_omega' / psi_omega) grad y+
  if(variable.log_form)
    return std::make_pair(std::log(psi_omega.first),
                          (psi_omega.second / psi_omega.first) * grad_y_plus);

  // omega: psi = psi_omega,   grad psi = psi_omega' grad y+
  return std::make_pair(psi_omega.first, psi_omega.second * grad_y_plus);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::update()
{
  // input of the loop: y (block 0) and tau_w (block 1) of FunctionEnrichment, copied into the
  // layout of this MatrixFree (same DoFHandlers, ghost values set by the loop)
  BlockVectorType src(2);
  matrix_free->initialize_dof_vector(src.block(0), dof_index_wall_distance);
  matrix_free->initialize_dof_vector(src.block(1), dof_index_cg);
  src.block(0).copy_locally_owned_data_from(velocity_enrichment.wall_distance);
  src.block(1).copy_locally_owned_data_from(velocity_enrichment.wall_shear_stress);
  src.collect_sizes();

  unsigned int const n_var     = variables.size();
  unsigned int const n_batches = matrix_free->n_cell_batches() + matrix_free->n_ghost_cell_batches();
  unsigned int const n_faces =
    matrix_free->n_inner_face_batches() + matrix_free->n_boundary_face_batches();
  scalar const zero = dealii::make_vectorized_array<Number>(0.0);

  n_q_points_cell = matrix_free->get_n_q_points(quad_index);
  n_q_points_face = matrix_free->get_n_q_points_face(quad_index);

  psi_cell.resize(n_var);
  grad_psi_cell.resize(n_var);
  psi_cell_old.resize(n_var);
  psi_face.resize(n_var);
  grad_psi_face.resize(n_var);
  wall_value_face.resize(n_var);
  wall_value_off_face.resize(n_var);

  for(unsigned int i = 0; i < n_var; ++i)
  {
    // psi of the space the current solution is expressed in (projection old -> new)
    psi_cell_old[i] = psi_cell[i];

    // zero outside the enriched cells; enriched cells are overwritten by the loop
    psi_cell[i].resize(n_batches * n_q_points_cell, zero);
    grad_psi_cell[i].resize(n_batches * n_q_points_cell, vector());
    for(unsigned int side = 0; side < 2; ++side)
    {
      psi_face[i][side].resize(n_faces * n_q_points_face, zero);
      grad_psi_face[i][side].resize(n_faces * n_q_points_face, vector());
    }
    wall_value_face[i].resize(n_faces * n_q_points_face, zero);
    wall_value_off_face[i].resize(n_faces * n_q_points_face, zero);
  }

  // no vector is written: dst is a placeholder of the loop interface
  VectorType placeholder;
  matrix_free->initialize_dof_vector(placeholder, dof_index_dg);

  matrix_free->loop(&This::cell_loop_update,
                    &This::face_loop_update,
                    &This::boundary_face_loop_update,
                    this,
                    placeholder,
                    src,
                    false /*zero_dst*/,
                    dealii::MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                    dealii::MatrixFree<dim, Number>::DataAccessOnFaces::gradients);

  // first call: no old space yet
  for(unsigned int i = 0; i < n_var; ++i)
    if(psi_cell_old[i].size() != psi_cell[i].size())
      psi_cell_old[i] = psi_cell[i];
}

template<int dim, typename Number>
bool
FunctionEnrichmentRANS<dim, Number>::is_enrichment_active_lane(unsigned int const cell,
                                                               unsigned int const lane) const
{
  // empty lanes
  if(lane >= matrix_free->n_active_entries_per_cell_batch(cell))
    return false;

  dealii::types::global_cell_index const index =
    matrix_free->get_cell_iterator(cell, lane)->global_active_cell_index();

  // y+ criterion of the velocity enrichment and, if set, the k criterion
  return velocity_enrichment.is_enrichment_active_cell(index) and
         (k_criterion_k_plus_on <= 0.0 or k_active_cells.count(index) > 0);
}

template<int dim, typename Number>
bool
FunctionEnrichmentRANS<dim, Number>::is_k_switched_off_lane(unsigned int const cell,
                                                            unsigned int const lane) const
{
  if(k_criterion_k_plus_on <= 0.0 or lane >= matrix_free->n_active_entries_per_cell_batch(cell))
    return false;

  dealii::types::global_cell_index const index =
    matrix_free->get_cell_iterator(cell, lane)->global_active_cell_index();

  // velocity enrichment active, k criterion off
  return velocity_enrichment.is_enrichment_active_cell(index) and k_active_cells.count(index) == 0;
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_update(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range)
{
  (void)dst;

  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number> CellIntegratorScalar;

  // psi only on enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  CellIntegratorScalar y_cell(matrix_free_in, dof_index_wall_distance, quad_index, 0, active_fe_index);
  CellIntegratorScalar tau_cell(matrix_free_in, dof_index_cg, quad_index, 0, active_fe_index);

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    y_cell.reinit(cell);
    tau_cell.reinit(cell);
    y_cell.gather_evaluate(src.block(0), flags);
    tau_cell.gather_evaluate(src.block(1), flags);

    for(unsigned int i = 0; i < variables.size(); ++i)
      for(unsigned int q = 0; q < n_q_points_cell; ++q)
      {
        auto const psi = evaluate_enrichment_function(variables[i],
                                                      y_cell.get_value(q),
                                                      y_cell.get_gradient(q),
                                                      tau_cell.get_value(q),
                                                      tau_cell.get_gradient(q));
        psi_cell[i][cell * n_q_points_cell + q]      = psi.first;
        grad_psi_cell[i][cell * n_q_points_cell + q] = psi.second;
      }

    // lanes switched off by the y+ criterion of the velocity and empty lanes: psi = 0
    for(unsigned int v = 0; v < scalar::size(); ++v)
    {
      if(is_enrichment_active_lane(cell, v))
        continue;

      for(unsigned int i = 0; i < variables.size(); ++i)
        for(unsigned int q = 0; q < n_q_points_cell; ++q)
        {
          psi_cell[i][cell * n_q_points_cell + q][v] = 0.0;
          for(unsigned int d = 0; d < dim; ++d)
            grad_psi_cell[i][cell * n_q_points_cell + q][d][v] = 0.0;
        }
    }
  }
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::face_loop_update(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range)
{
  (void)dst;

  // inner faces: both sides
  fill_psi_face(matrix_free_in, src, range, true);
  fill_psi_face(matrix_free_in, src, range, false);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::boundary_face_loop_update(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range)
{
  (void)dst;

  // boundary faces: interior side only
  fill_psi_face(matrix_free_in, src, range, true);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::fill_psi_face(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range,
  bool const                                    interior)
{
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number> FaceIntegratorScalar;

  // psi only on the enriched side of a face (hp: one active_fe_index per side and range)
  if(matrix_free_in.get_face_active_fe_index(range, interior) != active_fe_index)
    return;

  FaceIntegratorScalar y_face(
    matrix_free_in, interior, dof_index_wall_distance, quad_index, 0, active_fe_index);
  FaceIntegratorScalar tau_face(matrix_free_in, interior, dof_index_cg, quad_index, 0, active_fe_index);
  // mean wall distance y_c of the cell on this side (cell-constant field: DoF 0)
  FaceIntegratorScalar y_c_face(matrix_free_in, interior, dof_index_dg, quad_index, 0, active_fe_index);

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  unsigned int const side  = interior ? 0 : 1;
  unsigned int const lanes = scalar::size();

  for(unsigned int face = range.first; face < range.second; ++face)
  {
    y_face.reinit(face);
    tau_face.reinit(face);
    y_face.gather_evaluate(src.block(0), flags);
    tau_face.gather_evaluate(src.block(1), flags);

    y_c_face.reinit(face);
    y_c_face.read_dof_values(wall_cell_fields.block(1));
    scalar const y_c = y_c_face.get_dof_value(0);

    for(unsigned int i = 0; i < variables.size(); ++i)
      for(unsigned int q = 0; q < n_q_points_face; ++q)
      {
        auto const psi = evaluate_enrichment_function(variables[i],
                                                      y_face.get_value(q),
                                                      y_face.get_gradient(q),
                                                      tau_face.get_value(q),
                                                      tau_face.get_gradient(q));
        psi_face[i][side][face * n_q_points_face + q]      = psi.first;
        grad_psi_face[i][side][face * n_q_points_face + q] = psi.second;

        // boundary faces: wall value of the weak Dirichlet condition, omega = (u_tau^2 / nu)
        // psi_omega (psi = psi_omega) or ln omega = ln(u_tau^2 / nu) + ln psi_omega (psi = ln psi_omega)
        if(face >= matrix_free_in.n_inner_face_batches())
        {
          scalar const u_tau2_nu =
            std::max(tau_face.get_value(q), dealii::make_vectorized_array<Number>(1.e-10)) /
            Number(kinematic_viscosity);
          wall_value_face[i][face * n_q_points_face + q] =
            variables[i].log_form ? std::log(u_tau2_nu) + psi.first : u_tau2_nu * psi.first;

          // wall cells switched off by the k criterion (WallFunction): omega_wf at y = a y_c,
          // omega_wf = (u_tau^2 / nu) psi_omega(y+ = a y_c u_tau / nu) (blend of the viscous and the
          // log-layer omega, as the wall condition of the velocity wall function)
          scalar const u_tau =
            std::sqrt(std::max(tau_face.get_value(q), dealii::make_vectorized_array<Number>(1.e-10)));
          scalar const y_plus_c =
            Number(a_wall_function) * y_c * u_tau / Number(kinematic_viscosity);
          scalar const psi_omega_c = variables[i].omega_wall_law(y_plus_c).first;
          wall_value_off_face[i][face * n_q_points_face + q] =
            variables[i].log_form ? std::log(u_tau2_nu * psi_omega_c) : u_tau2_nu * psi_omega_c;
        }
      }

    // face sides of switched-off cells and empty lanes: psi = 0 (as on the cell)
    auto const & info  = matrix_free_in.get_face_info(face);
    auto const & cells = interior ? info.cells_interior : info.cells_exterior;
    for(unsigned int v = 0; v < lanes; ++v)
    {
      if(cells[v] != dealii::numbers::invalid_unsigned_int and
         is_enrichment_active_lane(cells[v] / lanes, cells[v] % lanes))
        continue;

      for(unsigned int i = 0; i < variables.size(); ++i)
        for(unsigned int q = 0; q < n_q_points_face; ++q)
        {
          psi_face[i][side][face * n_q_points_face + q][v] = 0.0;
          for(unsigned int d = 0; d < dim; ++d)
            grad_psi_face[i][side][face * n_q_points_face + q][d][v] = 0.0;
        }
    }
  }
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::copy_global_to_wall_layout(VectorType const & global,
                                                                VectorType &       shadow) const
{
#ifdef EXADG_DEBUG
  AssertThrow(global.locally_owned_size() == global_dof_handler_rans->n_locally_owned_dofs() and
                shadow.locally_owned_size() == dof_handler_shadow_scalar.n_locally_owned_dofs(),
              dealii::ExcMessage("copy_global_to_wall_layout: vector layouts do not match."));
#endif

  shadow = 0.0;

  // locally owned enriched (1) and halo (2) cells
  for(unsigned int i = 0; i < scalar_index_global.size(); ++i)
    shadow.local_element(scalar_index_shadow[i]) = global.local_element(scalar_index_global[i]);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::add_wall_layout_to_global(VectorType const & shadow,
                                                               VectorType &       global) const
{
#ifdef EXADG_DEBUG
  AssertThrow(global.locally_owned_size() == global_dof_handler_rans->n_locally_owned_dofs() and
                shadow.locally_owned_size() == dof_handler_shadow_scalar.n_locally_owned_dofs(),
              dealii::ExcMessage("add_wall_layout_to_global: vector layouts do not match."));
#endif

  // locally owned enriched (1) and halo (2) cells
  for(unsigned int i = 0; i < scalar_index_global.size(); ++i)
    global.local_element(scalar_index_global[i]) += shadow.local_element(scalar_index_shadow[i]);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::copy_wall_layout_to_global(VectorType const & shadow,
                                                                VectorType &       global) const
{
#ifdef EXADG_DEBUG
  AssertThrow(global.locally_owned_size() == global_dof_handler_rans->n_locally_owned_dofs() and
                shadow.locally_owned_size() == dof_handler_shadow_scalar.n_locally_owned_dofs(),
              dealii::ExcMessage("copy_wall_layout_to_global: vector layouts do not match."));
#endif

  // locally owned enriched cells (1) only: the first n_scalar_index_enriched pairs
  for(unsigned int i = 0; i < n_scalar_index_enriched; ++i)
    global.local_element(scalar_index_global[i]) = shadow.local_element(scalar_index_shadow[i]);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::apply_inverse_mass(unsigned int const variable,
                                                        VectorType &       dst_bar_shadow,
                                                        VectorType &       dst_tilde,
                                                        VectorType const & src_bar_shadow,
                                                        VectorType const & src_tilde) const
{
  AssertIndexRange(variable, variables.size());
  AssertThrow(inverse_mass_data.size() == variables.size(),
              dealii::ExcMessage("apply_inverse_mass: call precompute_inverse_mass() first."));
  variable_inverse_mass = variable;

  // input: right-hand side, bar part (shadow layout, block 0), tilde part (block 1)
  BlockVectorType src(2);
  matrix_free->initialize_dof_vector(src.block(0), dof_index_dg);
  matrix_free->initialize_dof_vector(src.block(1), dof_index_en);
  src.block(0).copy_locally_owned_data_from(src_bar_shadow);
  src.block(1).copy_locally_owned_data_from(src_tilde);
  src.collect_sizes();

  // output: W_bar (0), W_tilde (1)
  std::vector<VectorType> dst(2);
  matrix_free->initialize_dof_vector(dst[0], dof_index_dg);
  matrix_free->initialize_dof_vector(dst[1], dof_index_en);

  matrix_free->cell_loop(&This::cell_loop_inverse_mass_apply, this, dst, src, true /*zero_dst*/);

  dst_bar_shadow.copy_locally_owned_data_from(dst[0]);
  dst_tilde.copy_locally_owned_data_from(dst[1]);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_inverse_mass_apply(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  std::vector<VectorType> &                     dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number> CellIntegratorScalar;

  // block inverse only on enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  CellIntegratorScalar phi_bar(matrix_free_in, dof_index_dg, quad_index, 0, active_fe_index);
  CellIntegratorScalar phi_tilde(matrix_free_in, dof_index_en, quad_index, 0, active_fe_index);

  unsigned int const n_dg = phi_bar.dofs_per_cell;
  unsigned int const n_en = phi_tilde.dofs_per_cell;

  scalar const zero = dealii::make_vectorized_array<Number>(0.0);

  std::vector<scalar> R_bar(n_dg), M_inv_R_bar(n_dg), W_bar(n_dg);
  std::vector<scalar> R_tilde(n_en), coupling(n_en), W_tilde(n_en);

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    phi_bar.reinit(cell);
    phi_tilde.reinit(cell);
    phi_bar.read_dof_values(src.block(0));
    phi_tilde.read_dof_values(src.block(1));

    for(unsigned int i = 0; i < n_dg; ++i)
      R_bar[i] = phi_bar.get_dof_value(i);
    for(unsigned int i = 0; i < n_en; ++i)
      R_tilde[i] = phi_tilde.get_dof_value(i);

    InverseMassData const & data              = inverse_mass_data[variable_inverse_mass][cell];
    SimdMatrix const &      M_bar_bar_inverse = M_bar_bar_inverse_cell[cell];

    // Eq. (6.16): W_tilde = S^+ (R_tilde - M_bar_tilde^T M_bar_bar^{-1} R_bar)
    for(unsigned int i = 0; i < n_dg; ++i)
    {
      M_inv_R_bar[i] = zero;
      for(unsigned int j = 0; j < n_dg; ++j)
        M_inv_R_bar[i] += M_bar_bar_inverse[i][j] * R_bar[j];
    }

    for(unsigned int i = 0; i < n_en; ++i)
      coupling[i] = zero;
    for(unsigned int j = 0; j < n_dg; ++j)
      for(unsigned int i = 0; i < n_en; ++i)
        coupling[i] += data.M_bar_tilde[j][i] * M_inv_R_bar[j];

    for(unsigned int i = 0; i < n_en; ++i)
    {
      W_tilde[i] = zero;
      for(unsigned int j = 0; j < n_en; ++j)
        W_tilde[i] += data.schur_inverse[i][j] * (R_tilde[j] - coupling[j]);
    }

    // Eq. (6.17): W_bar = M_bar_bar^{-1} (R_bar - M_bar_tilde W_tilde)
    for(unsigned int i = 0; i < n_dg; ++i)
    {
      scalar r = R_bar[i];
      for(unsigned int j = 0; j < n_en; ++j)
        r -= data.M_bar_tilde[i][j] * W_tilde[j];
      M_inv_R_bar[i] = r;
    }
    for(unsigned int i = 0; i < n_dg; ++i)
    {
      W_bar[i] = zero;
      for(unsigned int j = 0; j < n_dg; ++j)
        W_bar[i] += M_bar_bar_inverse[i][j] * M_inv_R_bar[j];
    }

    for(unsigned int i = 0; i < n_dg; ++i)
      phi_bar.submit_dof_value(W_bar[i], i);
    for(unsigned int i = 0; i < n_en; ++i)
      phi_tilde.submit_dof_value(W_tilde[i], i);

    phi_bar.distribute_local_to_global(dst[0]);
    phi_tilde.distribute_local_to_global(dst[1]);
  }
}

template<int dim, typename Number>
unsigned int
FunctionEnrichmentRANS<dim, Number>::get_rans_index(unsigned int const variable) const
{
  AssertIndexRange(variable, variables.size());
  return variables[variable].rans_index;
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::apply_mass(unsigned int const variable,
                                                VectorType &       dst_bar_shadow,
                                                VectorType &       dst_tilde,
                                                VectorType const & src_bar_shadow,
                                                VectorType const & src_tilde) const
{
  AssertIndexRange(variable, variables.size());
  variable_mass = variable;

  // input: bar part (shadow layout, block 0), tilde part (block 1)
  BlockVectorType src(2);
  matrix_free->initialize_dof_vector(src.block(0), dof_index_dg);
  matrix_free->initialize_dof_vector(src.block(1), dof_index_en);
  src.block(0).copy_locally_owned_data_from(src_bar_shadow);
  src.block(1).copy_locally_owned_data_from(src_tilde);
  src.collect_sizes();

  // output: rows tested with v_bar (0) and psi v_tilde (1)
  std::vector<VectorType> dst(2);
  matrix_free->initialize_dof_vector(dst[0], dof_index_dg);
  matrix_free->initialize_dof_vector(dst[1], dof_index_en);

  matrix_free->cell_loop(&This::cell_loop_mass_apply, this, dst, src, true /*zero_dst*/);

  dst_bar_shadow.copy_locally_owned_data_from(dst[0]);
  dst_tilde.copy_locally_owned_data_from(dst[1]);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_mass_apply(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  std::vector<VectorType> &                     dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number> CellIntegratorScalar;

  // block mass only on enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  FEOmegaEnrichedEvaluation<dim, Number, CellIntegratorScalar> phi(matrix_free_in,
                                                                   *this,
                                                                   variable_mass);

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(src.block(0), src.block(1), dealii::EvaluationFlags::values);

    // W_h = W_bar + psi W_tilde, tested with v_bar and psi v_tilde
    for(unsigned int q = 0; q < phi.n_q_points; ++q)
      phi.submit_value(phi.get_value(q), q);

    phi.integrate_scatter(dealii::EvaluationFlags::values, dst[0], dst[1]);
  }
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::compute_enrichment_nodal(unsigned int const variable,
                                                              VectorType &       global_dst,
                                                              VectorType const & src_tilde) const
{
  AssertIndexRange(variable, variables.size());
  variable_enrichment_nodal = variable;

  // input: W_tilde (block 0), y (block 1), tau_w (block 2), layouts of this MatrixFree
  BlockVectorType src(3);
  matrix_free->initialize_dof_vector(src.block(0), dof_index_en);
  matrix_free->initialize_dof_vector(src.block(1), dof_index_wall_distance);
  matrix_free->initialize_dof_vector(src.block(2), dof_index_cg);
  src.block(0).copy_locally_owned_data_from(src_tilde);
  src.block(1).copy_locally_owned_data_from(velocity_enrichment.wall_distance);
  src.block(2).copy_locally_owned_data_from(velocity_enrichment.wall_shear_stress);
  src.collect_sizes();

  // output: nodal values of psi W_tilde (shadow layout, RANS finite element)
  VectorType dst;
  matrix_free->initialize_dof_vector(dst, dof_index_dg);

  matrix_free->cell_loop(&This::cell_loop_enrichment_nodal, this, dst, src, true /*zero_dst*/);

  // enriched cells -> global RANS layout, 0 elsewhere
  global_dst = 0.0;
  copy_wall_layout_to_global(dst, global_dst);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_enrichment_nodal(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number> CellIntegratorScalar;

  // only enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  // quadrature = nodes of the RANS finite element (Gauss-Lobatto): q-point q = DoF q
  CellIntegratorScalar phi_tilde(matrix_free_in, dof_index_en, quad_index_nodal, 0, active_fe_index);
  CellIntegratorScalar y_cell(
    matrix_free_in, dof_index_wall_distance, quad_index_nodal, 0, active_fe_index);
  CellIntegratorScalar tau_cell(matrix_free_in, dof_index_cg, quad_index_nodal, 0, active_fe_index);
  CellIntegratorScalar phi_out(matrix_free_in, dof_index_dg, quad_index_nodal, 0, active_fe_index);

  AssertDimension(phi_out.n_q_points, phi_out.dofs_per_cell);

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  Variable const & var = variables[variable_enrichment_nodal];

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    phi_tilde.reinit(cell);
    y_cell.reinit(cell);
    tau_cell.reinit(cell);
    phi_out.reinit(cell);

    phi_tilde.gather_evaluate(src.block(0), dealii::EvaluationFlags::values);
    y_cell.gather_evaluate(src.block(1), flags);
    tau_cell.gather_evaluate(src.block(2), flags);

    for(unsigned int q = 0; q < phi_out.n_q_points; ++q)
    {
      scalar psi = evaluate_enrichment_function(var,
                                                y_cell.get_value(q),
                                                y_cell.get_gradient(q),
                                                tau_cell.get_value(q),
                                                tau_cell.get_gradient(q))
                     .first;

      // lanes switched off by the y+ criterion of the velocity and empty lanes: psi = 0
      for(unsigned int v = 0; v < scalar::size(); ++v)
        if(not is_enrichment_active_lane(cell, v))
          psi[v] = 0.0;

      // nodal value of the enrichment part psi W_tilde
      phi_out.submit_dof_value(psi * phi_tilde.get_value(q), q);
    }

    phi_out.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::apply_mass_wall_law_target(unsigned int const variable,
                                                                VectorType &       dst_bar_shadow,
                                                                VectorType &       dst_tilde) const
{
  AssertIndexRange(variable, variables.size());
  variable_wall_law_target = variable;

  // input: y (block 0), tau_w (block 1) of FunctionEnrichment
  BlockVectorType src(2);
  matrix_free->initialize_dof_vector(src.block(0), dof_index_wall_distance);
  matrix_free->initialize_dof_vector(src.block(1), dof_index_cg);
  src.block(0).copy_locally_owned_data_from(velocity_enrichment.wall_distance);
  src.block(1).copy_locally_owned_data_from(velocity_enrichment.wall_shear_stress);
  src.collect_sizes();

  // output: tested with v_bar (0) and psi v_tilde (1)
  std::vector<VectorType> dst(2);
  matrix_free->initialize_dof_vector(dst[0], dof_index_dg);
  matrix_free->initialize_dof_vector(dst[1], dof_index_en);

  matrix_free->cell_loop(&This::cell_loop_wall_law_target, this, dst, src, true /*zero_dst*/);

  dst_bar_shadow.copy_locally_owned_data_from(dst[0]);
  dst_tilde.copy_locally_owned_data_from(dst[1]);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_wall_law_target(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  std::vector<VectorType> &                     dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number> CellIntegratorScalar;

  // only enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  unsigned int const variable = variable_wall_law_target;
  Variable const &   var      = variables[variable];

  CellIntegratorScalar y_cell(matrix_free_in, dof_index_wall_distance, quad_index, 0, active_fe_index);
  CellIntegratorScalar tau_cell(matrix_free_in, dof_index_cg, quad_index, 0, active_fe_index);
  CellIntegratorScalar phi_bar(matrix_free_in, dof_index_dg, quad_index, 0, active_fe_index);
  CellIntegratorScalar phi_tilde(matrix_free_in, dof_index_en, quad_index, 0, active_fe_index);

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  scalar const tau_min = dealii::make_vectorized_array<Number>(1.e-10);
  scalar const nu      = dealii::make_vectorized_array<Number>(kinematic_viscosity);

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    y_cell.reinit(cell);
    tau_cell.reinit(cell);
    phi_bar.reinit(cell);
    phi_tilde.reinit(cell);
    y_cell.gather_evaluate(src.block(0), flags);
    tau_cell.gather_evaluate(src.block(1), flags);

    for(unsigned int q = 0; q < phi_bar.n_q_points; ++q)
    {
      // wall-law shape at this point (psi_omega or ln psi_omega), not switched off by the y+
      // criterion: the target is the wall law in every wall cell
      scalar const psi_wall_law = evaluate_enrichment_function(var,
                                                               y_cell.get_value(q),
                                                               y_cell.get_gradient(q),
                                                               tau_cell.get_value(q),
                                                               tau_cell.get_gradient(q))
                                    .first;

      // omega = (u_tau^2 / nu) psi_omega,   ln omega = ln(u_tau^2 / nu) + ln psi_omega
      scalar const u_tau2_nu = std::max(tau_cell.get_value(q), tau_min) / nu;
      scalar const target =
        var.log_form ? std::log(u_tau2_nu) + psi_wall_law : u_tau2_nu * psi_wall_law;

      // tested with v_bar and psi v_tilde (cached psi: 0 on switched-off lanes, as the block mass)
      phi_bar.submit_value(target, q);
      phi_tilde.submit_value(get_psi_cell(variable, cell, q) * target, q);
    }

    phi_bar.integrate_scatter(dealii::EvaluationFlags::values, dst[0]);
    phi_tilde.integrate_scatter(dealii::EvaluationFlags::values, dst[1]);
  }
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::apply_mass_old_to_new(unsigned int const variable,
                                                           VectorType &       dst_bar_shadow,
                                                           VectorType &       dst_tilde,
                                                           VectorType const & src_bar_shadow,
                                                           VectorType const & src_tilde) const
{
  AssertIndexRange(variable, variables.size());
  variable_mass_old_to_new = variable;

  // input: bar part (shadow layout, block 0), tilde part (block 1)
  dealii::LinearAlgebra::distributed::BlockVector<Number> src(2);
  matrix_free->initialize_dof_vector(src.block(0), dof_index_dg);
  matrix_free->initialize_dof_vector(src.block(1), dof_index_en);
  src.block(0).copy_locally_owned_data_from(src_bar_shadow);
  src.block(1).copy_locally_owned_data_from(src_tilde);
  src.collect_sizes();

  // output: tested with v_bar (0) and psi_new v_tilde (1)
  std::vector<VectorType> dst(2);
  matrix_free->initialize_dof_vector(dst[0], dof_index_dg);
  matrix_free->initialize_dof_vector(dst[1], dof_index_en);

  matrix_free->cell_loop(&This::cell_loop_mass_old_to_new, this, dst, src, true /*zero_dst*/);

  dst_bar_shadow.copy_locally_owned_data_from(dst[0]);
  dst_tilde.copy_locally_owned_data_from(dst[1]);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_mass_old_to_new(
  dealii::MatrixFree<dim, Number> const &                         matrix_free_in,
  std::vector<VectorType> &                                       dst,
  dealii::LinearAlgebra::distributed::BlockVector<Number> const & src,
  std::pair<unsigned int, unsigned int> const &                   range) const
{
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number> CellIntegratorScalar;

  // psi only on enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  unsigned int const variable = variable_mass_old_to_new;

  CellIntegratorScalar phi_bar(matrix_free_in, dof_index_dg, quad_index, 0, active_fe_index);
  CellIntegratorScalar phi_tilde(matrix_free_in, dof_index_en, quad_index, 0, active_fe_index);

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    phi_bar.reinit(cell);
    phi_tilde.reinit(cell);
    phi_bar.gather_evaluate(src.block(0), dealii::EvaluationFlags::values);
    phi_tilde.gather_evaluate(src.block(1), dealii::EvaluationFlags::values);

    for(unsigned int q = 0; q < phi_bar.n_q_points; ++q)
    {
      // W_old = W_bar + psi_old W_tilde
      scalar const w_old =
        phi_bar.get_value(q) + get_psi_cell_old(variable, cell, q) * phi_tilde.get_value(q);

      // tested with v_bar and psi_new v_tilde
      phi_bar.submit_value(w_old, q);
      phi_tilde.submit_value(get_psi_cell(variable, cell, q) * w_old, q);
    }

    phi_bar.integrate_scatter(dealii::EvaluationFlags::values, dst[0]);
    phi_tilde.integrate_scatter(dealii::EvaluationFlags::values, dst[1]);
  }
}

#ifdef EXADG_DEBUG
template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::print_wall_diagnostics(unsigned int const variable,
                                                            VectorType const & global_bar,
                                                            VectorType const & src_tilde,
                                                            unsigned int const time_step_number) const
{
  AssertIndexRange(variable, variables.size());
  variable_wall_diagnostics = variable;
  wall_diagnostics          = WallDiagnostics();

  // input: W_bar (shadow layout, block 0), W_tilde (1), y (2), tau_w (3)
  BlockVectorType src(4);
  matrix_free->initialize_dof_vector(src.block(0), dof_index_dg);
  matrix_free->initialize_dof_vector(src.block(1), dof_index_en);
  matrix_free->initialize_dof_vector(src.block(2), dof_index_wall_distance);
  matrix_free->initialize_dof_vector(src.block(3), dof_index_cg);
  copy_global_to_wall_layout(global_bar, src.block(0));
  src.block(1).copy_locally_owned_data_from(src_tilde);
  src.block(2).copy_locally_owned_data_from(velocity_enrichment.wall_distance);
  src.block(3).copy_locally_owned_data_from(velocity_enrichment.wall_shear_stress);
  src.collect_sizes();

  // no vector is written: dst is a placeholder of the loop interface
  VectorType placeholder;
  matrix_free->initialize_dof_vector(placeholder, dof_index_dg);

  matrix_free->loop(&This::cell_loop_wall_diagnostics,
                    &This::face_loop_wall_diagnostics,
                    &This::boundary_face_loop_wall_diagnostics,
                    this,
                    placeholder,
                    src,
                    false /*zero_dst*/,
                    dealii::MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                    dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);

  // sums and extrema over all processes
  MPI_Comm const    comm = dof_handler_shadow_scalar.get_triangulation().get_communicator();
  WallDiagnostics & d    = wall_diagnostics;

  double const tilde_min           = dealii::Utilities::MPI::min(d.tilde_min, comm);
  double const tilde_max           = dealii::Utilities::MPI::max(d.tilde_max, comm);
  double const tilde_sum           = dealii::Utilities::MPI::sum(d.tilde_sum, comm);
  double const tilde_count         = dealii::Utilities::MPI::sum(d.tilde_count, comm);
  double const near_y_plus_sum     = dealii::Utilities::MPI::sum(d.near_y_plus_sum, comm);
  double const near_value_sum      = dealii::Utilities::MPI::sum(d.near_value_sum, comm);
  double const near_wall_law_sum   = dealii::Utilities::MPI::sum(d.near_wall_law_sum, comm);
  double const near_difference_max = dealii::Utilities::MPI::max(d.near_difference_max, comm);
  double const near_count          = dealii::Utilities::MPI::sum(d.near_count, comm);
  double const face_value_sum      = dealii::Utilities::MPI::sum(d.face_value_sum, comm);
  double const face_target_sum     = dealii::Utilities::MPI::sum(d.face_target_sum, comm);
  double const face_difference_max = dealii::Utilities::MPI::max(d.face_difference_max, comm);
  double const face_count          = dealii::Utilities::MPI::sum(d.face_count, comm);

  auto const mean = [](double const sum, double const count) {
    return count > 0.0 ? sum / count : 0.0;
  };

  // x of a maximum over all processes: x of the process with the largest value
  auto const x_of_max = [&](double const local_value, double const local_x) {
    std::vector<std::array<double, 2>> const all =
      dealii::Utilities::MPI::all_gather(comm, std::array<double, 2>{{local_value, local_x}});
    std::array<double, 2> best = all[0];
    for(auto const & a : all)
      if(a[0] > best[0])
        best = a;
    return best[1];
  };
  double const tilde_max_x           = x_of_max(d.tilde_max, d.tilde_max_x);
  double const near_difference_max_x = x_of_max(d.near_difference_max, d.near_difference_max_x);
  double const face_difference_max_x = x_of_max(d.face_difference_max, d.face_difference_max_x);

  dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);
  pcout << "  [enrichment wall] time step " << time_step_number << ", variable " << variable
        << (variables[variable].log_form ? " (W = ln omega)" : " (W = omega)") << ":" << std::endl
        << "    W_tilde in active wall cells:  min " << tilde_min << "  mean "
        << mean(tilde_sum, tilde_count) << "  max " << tilde_max << " at x = " << tilde_max_x
        << std::endl
        << "    point closest to the wall (mean y+ " << mean(near_y_plus_sum, near_count)
        << "):  W_h mean " << mean(near_value_sum, near_count) << "  wall law mean "
        << mean(near_wall_law_sum, near_count) << "  max |W_h - wall law| " << near_difference_max
        << " at x = " << near_difference_max_x << std::endl
        << "    wall faces:  W_h mean " << mean(face_value_sum, face_count) << "  g mean "
        << mean(face_target_sum, face_count) << "  max |W_h - g| " << face_difference_max
        << " at x = " << face_difference_max_x << std::endl;
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_wall_diagnostics(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  (void)dst;

  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number> CellIntegratorScalar;

  // only enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  unsigned int const variable = variable_wall_diagnostics;
  Variable const &   var      = variables[variable];

  CellIntegratorScalar w_bar(matrix_free_in, dof_index_dg, quad_index, 0, active_fe_index);
  CellIntegratorScalar w_tilde(matrix_free_in, dof_index_en, quad_index, 0, active_fe_index);
  CellIntegratorScalar y_cell(matrix_free_in, dof_index_wall_distance, quad_index, 0, active_fe_index);
  CellIntegratorScalar tau_cell(matrix_free_in, dof_index_cg, quad_index, 0, active_fe_index);

  dealii::EvaluationFlags::EvaluationFlags const flags =
    dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  scalar const tau_min = dealii::make_vectorized_array<Number>(1.e-10);
  scalar const nu      = dealii::make_vectorized_array<Number>(kinematic_viscosity);

  WallDiagnostics & d = wall_diagnostics;

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    w_bar.reinit(cell);
    w_tilde.reinit(cell);
    y_cell.reinit(cell);
    tau_cell.reinit(cell);
    w_bar.gather_evaluate(src.block(0), dealii::EvaluationFlags::values);
    w_tilde.gather_evaluate(src.block(1), dealii::EvaluationFlags::values);
    y_cell.gather_evaluate(src.block(2), flags);
    tau_cell.gather_evaluate(src.block(3), flags);

    for(unsigned int v = 0; v < scalar::size(); ++v)
    {
      if(not is_enrichment_active_lane(cell, v))
        continue;

      // W_tilde at all points; point closest to the wall
      unsigned int q_near = 0;
      for(unsigned int q = 0; q < w_tilde.n_q_points; ++q)
      {
        double const tilde = w_tilde.get_value(q)[v];
        d.tilde_min        = std::min(d.tilde_min, tilde);
        if(tilde > d.tilde_max)
        {
          d.tilde_max   = tilde;
          d.tilde_max_x = w_tilde.quadrature_point(q)[0][v];
        }
        d.tilde_sum += tilde;
        d.tilde_count += 1.0;

        if(y_cell.get_value(q)[v] < y_cell.get_value(q_near)[v])
          q_near = q;
      }

      // W_h and wall law at the point closest to the wall
      unsigned int const q = q_near;

      scalar const psi_wall_law = evaluate_enrichment_function(var,
                                                               y_cell.get_value(q),
                                                               y_cell.get_gradient(q),
                                                               tau_cell.get_value(q),
                                                               tau_cell.get_gradient(q))
                                    .first;
      scalar const u_tau2_nu = std::max(tau_cell.get_value(q), tau_min) / nu;
      scalar const wall_law =
        var.log_form ? std::log(u_tau2_nu) + psi_wall_law : u_tau2_nu * psi_wall_law;

      double const value =
        w_bar.get_value(q)[v] + get_psi_cell(variable, cell, q)[v] * w_tilde.get_value(q)[v];
      double const y_plus =
        y_cell.get_value(q)[v] * std::sqrt(std::max(double(tau_cell.get_value(q)[v]), 1.e-10)) /
        kinematic_viscosity;

      d.near_y_plus_sum += y_plus;
      d.near_value_sum += value;
      d.near_wall_law_sum += wall_law[v];
      if(std::abs(value - double(wall_law[v])) > d.near_difference_max)
      {
        d.near_difference_max   = std::abs(value - double(wall_law[v]));
        d.near_difference_max_x = y_cell.quadrature_point(q)[0][v];
      }
      d.near_count += 1.0;
    }
  }
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::face_loop_wall_diagnostics(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  // inner faces: nothing to check
  (void)matrix_free_in;
  (void)dst;
  (void)src;
  (void)range;
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::boundary_face_loop_wall_diagnostics(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  (void)dst;

  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number> FaceIntegratorScalar;

  // boundary faces of enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_face_active_fe_index(range, true) != active_fe_index)
    return;

  unsigned int const variable = variable_wall_diagnostics;
  unsigned int const lanes    = scalar::size();

  FaceIntegratorScalar w_bar(matrix_free_in, true, dof_index_dg, quad_index, 0, active_fe_index);
  FaceIntegratorScalar w_tilde(matrix_free_in, true, dof_index_en, quad_index, 0, active_fe_index);

  WallDiagnostics & d = wall_diagnostics;

  for(unsigned int face = range.first; face < range.second; ++face)
  {
    if(not is_wall_boundary(matrix_free_in.get_boundary_id(face)))
      continue;

    w_bar.reinit(face);
    w_tilde.reinit(face);
    w_bar.gather_evaluate(src.block(0), dealii::EvaluationFlags::values);
    w_tilde.gather_evaluate(src.block(1), dealii::EvaluationFlags::values);

    auto const & cells = matrix_free_in.get_face_info(face).cells_interior;

    for(unsigned int v = 0; v < lanes; ++v)
    {
      if(cells[v] == dealii::numbers::invalid_unsigned_int or
         not is_enrichment_active_lane(cells[v] / lanes, cells[v] % lanes))
        continue;

      for(unsigned int q = 0; q < w_bar.n_q_points; ++q)
      {
        double const value = w_bar.get_value(q)[v] +
                             get_psi_face(variable, face, true, q)[v] * w_tilde.get_value(q)[v];
        double const target = get_wall_value_face(variable, face, q)[v];

        d.face_value_sum += value;
        d.face_target_sum += target;
        if(std::abs(value - target) > d.face_difference_max)
        {
          d.face_difference_max   = std::abs(value - target);
          d.face_difference_max_x = w_bar.quadrature_point(q)[0][v];
        }
        d.face_count += 1.0;
      }
    }
  }
}
#endif

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::set_turbulent_kinetic_energy(VectorType const & k_global,
                                                                  bool const k_logarithmic)
{
  k_log_form = k_logarithmic;

  // input: k (shadow layout, block 0), tau_w (1), y (2)
  BlockVectorType src(3);
  matrix_free->initialize_dof_vector(src.block(0), dof_index_dg);
  matrix_free->initialize_dof_vector(src.block(1), dof_index_cg);
  matrix_free->initialize_dof_vector(src.block(2), dof_index_wall_distance);
  copy_global_to_wall_layout(k_global, src.block(0));
  src.block(1).copy_locally_owned_data_from(velocity_enrichment.wall_shear_stress);
  src.block(2).copy_locally_owned_data_from(velocity_enrichment.wall_distance);
  src.collect_sizes();

  // output: k+ (block 0), y_c (block 1), constant per wall cell; the destination of the loop must
  // not be ghosted
  wall_cell_fields.zero_out_ghost_values();
  matrix_free->cell_loop(&This::cell_loop_wall_cell_fields, this, wall_cell_fields, src, true);
  wall_cell_fields.update_ghost_values();

  // k criterion with hysteresis, locally owned and ghost wall cells: the same k+ (ghost values) and
  // the same previous state on every process, so the same decision
  if(k_criterion_k_plus_on <= 0.0)
    return;

  std::vector<dealii::types::global_dof_index> dof_indices;
  for(auto const & cell : dof_handler_shadow_scalar.active_cell_iterators())
  {
    if(not(cell->is_locally_owned() or cell->is_ghost()) or
       cell->active_fe_index() != active_fe_index)
      continue;

    // k+ is constant in the cell: first DoF. Ghost cells whose DoFs are not in the vector (only a
    // vertex in common with the locally owned cells) are never used by the MatrixFree loops: skipped
    dof_indices.resize(cell->get_fe().n_dofs_per_cell());
    cell->get_dof_indices(dof_indices);
    auto const & partitioner = *wall_cell_fields.block(0).get_partitioner();
    if(not partitioner.in_local_range(dof_indices[0]) and
       not partitioner.is_ghost_entry(dof_indices[0]))
      continue;
    double const k_plus = wall_cell_fields.block(0)(dof_indices[0]);

    dealii::types::global_cell_index const index = cell->global_active_cell_index();
    bool const was_on = k_active_cells.count(index) > 0;
    if(not was_on and k_plus >= k_criterion_k_plus_on)
      k_active_cells.insert(index);
    else if(was_on and k_plus < k_criterion_k_plus_off)
      k_active_cells.erase(index);
  }

#ifdef EXADG_DEBUG
  // number of wall cells with the enrichment on by the k criterion (locally owned)
  unsigned int n_on = 0, n_wall = 0;
  for(auto const & cell : dof_handler_shadow_scalar.active_cell_iterators())
    if(cell->is_locally_owned() and cell->active_fe_index() == active_fe_index)
    {
      ++n_wall;
      if(k_active_cells.count(cell->global_active_cell_index()) > 0)
        ++n_on;
    }
  MPI_Comm const comm = dof_handler_shadow_scalar.get_triangulation().get_communicator();
  n_on   = dealii::Utilities::MPI::sum(n_on, comm);
  n_wall = dealii::Utilities::MPI::sum(n_wall, comm);
  dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);
  pcout << "  [omega enrichment k criterion] on in " << n_on << " of " << n_wall
        << " wall cells (k+ on " << k_criterion_k_plus_on << ", off " << k_criterion_k_plus_off
        << ")" << std::endl;
#endif
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_wall_cell_fields(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  BlockVectorType &                             dst,
  BlockVectorType const &                       src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number> CellIntegratorScalar;

  // only wall cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  CellIntegratorScalar k_cell(matrix_free_in, dof_index_dg, quad_index, 0, active_fe_index);
  CellIntegratorScalar tau_cell(matrix_free_in, dof_index_cg, quad_index, 0, active_fe_index);
  CellIntegratorScalar y_cell(matrix_free_in, dof_index_wall_distance, quad_index, 0, active_fe_index);
  CellIntegratorScalar out(matrix_free_in, dof_index_dg, quad_index, 0, active_fe_index);

  scalar const zero    = dealii::make_vectorized_array<Number>(0.0);
  scalar const tau_min = dealii::make_vectorized_array<Number>(1.e-10);

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    k_cell.reinit(cell);
    tau_cell.reinit(cell);
    y_cell.reinit(cell);
    out.reinit(cell);
    k_cell.gather_evaluate(src.block(0), dealii::EvaluationFlags::values);
    tau_cell.gather_evaluate(src.block(1), dealii::EvaluationFlags::values);
    y_cell.gather_evaluate(src.block(2), dealii::EvaluationFlags::values);

    // cell averages of max(0, k) (ln k: exp(ln k)), tau_w and the wall distance
    scalar k_sum = zero, tau_sum = zero, y_sum = zero, volume = zero;
    for(unsigned int q = 0; q < k_cell.n_q_points; ++q)
    {
      scalar const k =
        k_log_form ? std::exp(k_cell.get_value(q)) : std::max(k_cell.get_value(q), zero);
      k_sum += k * k_cell.JxW(q);
      tau_sum += tau_cell.get_value(q) * k_cell.JxW(q);
      y_sum += y_cell.get_value(q) * k_cell.JxW(q);
      volume += k_cell.JxW(q);
    }

    // k+ = k_bar / u_tau^2 with the cell averages (block 0), y_c (block 1), in every DoF
    scalar const k_plus = (k_sum / volume) / std::max(tau_sum / volume, tau_min);
    for(unsigned int i = 0; i < out.dofs_per_cell; ++i)
      out.submit_dof_value(k_plus, i);
    out.set_dof_values(dst.block(0));
    for(unsigned int i = 0; i < out.dofs_per_cell; ++i)
      out.submit_dof_value(y_sum / volume, i);
    out.set_dof_values(dst.block(1));
  }
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::set_rank_tolerance(double const tolerance)
{
  AssertThrow(tolerance > 0.0 and tolerance < 1.0,
              dealii::ExcMessage("FunctionEnrichmentRANS: rank tolerance must be in (0, 1)."));
  rank_tolerance = tolerance;
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::apply_enrichment_projector(unsigned int const variable,
                                                                VectorType &       dst_tilde,
                                                                VectorType const & src_tilde) const
{
  AssertIndexRange(variable, variables.size());

  // no projector yet: identity
  if(inverse_mass_data.size() != variables.size())
  {
    if(&dst_tilde != &src_tilde)
      dst_tilde = src_tilde;
    return;
  }

  variable_projector = variable;

  // dst and src may be the same vector: result in a temporary vector
  VectorType result;
  matrix_free->initialize_dof_vector(result, dof_index_en);
  matrix_free->cell_loop(&This::cell_loop_projector, this, result, src_tilde, true /*zero_dst*/);
  dst_tilde.copy_locally_owned_data_from(result);
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_projector(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number> CellIntegratorScalar;

  // W_tilde only on enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  CellIntegratorScalar phi_tilde(matrix_free_in, dof_index_en, quad_index, 0, active_fe_index);
  unsigned int const   n_en = phi_tilde.dofs_per_cell;

  std::vector<scalar> values(n_en);

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    phi_tilde.reinit(cell);
    phi_tilde.read_dof_values(src);
    for(unsigned int j = 0; j < n_en; ++j)
      values[j] = phi_tilde.get_dof_value(j);

    // P of the cell batch (same DoF numbering as the block mass)
    SimdMatrix const & P = inverse_mass_data[variable_projector][cell].projector;
    for(unsigned int i = 0; i < n_en; ++i)
    {
      scalar sum = dealii::make_vectorized_array<Number>(0.0);
      for(unsigned int j = 0; j < n_en; ++j)
        sum += P[i][j] * values[j];
      phi_tilde.submit_dof_value(sum, i);
    }

    phi_tilde.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
unsigned int
FunctionEnrichmentRANS<dim, Number>::get_n_modes_dropped() const
{
  return n_modes_dropped;
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::precompute_inverse_mass()
{
  unsigned int const n_batches = matrix_free->n_cell_batches();

  // psi-dependent blocks: recomputed in every call
  inverse_mass_data.assign(variables.size(), std::vector<InverseMassData>(n_batches));

  // M_bar_bar^{-1} depends on the geometry only: computed in the first call
  if(M_bar_bar_inverse_cell.size() != n_batches)
    M_bar_bar_inverse_cell.assign(n_batches, SimdMatrix());

  n_modes_total   = 0;
  n_modes_dropped = 0;
  lambda_min_kept = std::numeric_limits<double>::max();

  // no vector is read or written: dst and src are placeholders of the cell_loop interface
  VectorType placeholder;
  matrix_free->initialize_dof_vector(placeholder, dof_index_dg);
  matrix_free->cell_loop(&This::cell_loop_inverse_mass, this, placeholder, placeholder);

  MPI_Comm const comm = dof_handler_shadow_scalar.get_triangulation().get_communicator();
  n_modes_total       = dealii::Utilities::MPI::sum(n_modes_total, comm);
  n_modes_dropped     = dealii::Utilities::MPI::sum(n_modes_dropped, comm);
  lambda_min_kept     = dealii::Utilities::MPI::min(lambda_min_kept, comm);

#ifdef EXADG_DEBUG
  // conditioning of the enrichment space (as the velocity enrichment)
  dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);
  pcout << "  [omega enrichment] removed " << n_modes_dropped << " of " << n_modes_total
        << " enrichment modes, smallest kept eigenvalue of D^{-1/2} S D^{-1/2} = "
        << lambda_min_kept << std::endl;
#endif
}

template<int dim, typename Number>
void
FunctionEnrichmentRANS<dim, Number>::cell_loop_inverse_mass(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  (void)dst;
  (void)src;

  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number> CellIntegratorScalar;

  // block mass only on enriched cells (hp: one active_fe_index per range)
  if(matrix_free_in.get_cell_range_category(range) != active_fe_index)
    return;

  CellIntegratorScalar phi_bar(matrix_free_in, dof_index_dg, quad_index, 0, active_fe_index);
  CellIntegratorScalar phi_tilde(matrix_free_in, dof_index_en, quad_index, 0, active_fe_index);

  unsigned int const n_dg = phi_bar.dofs_per_cell;
  unsigned int const n_en = phi_tilde.dofs_per_cell;

  scalar const zero = dealii::make_vectorized_array<Number>(0.0);
  scalar const one  = dealii::make_vectorized_array<Number>(1.0);

  for(unsigned int cell = range.first; cell < range.second; ++cell)
  {
    phi_bar.reinit(cell);
    phi_tilde.reinit(cell);

    // M_bar_bar(j, i) = (N_j, N_i)_K, column i = polynomial shape function i (first call only)
    if(M_bar_bar_inverse_cell[cell].empty())
    {
      SimdMatrix M_bar_bar(n_dg, std::vector<scalar>(n_dg, zero));
      for(unsigned int i = 0; i < n_dg; ++i)
      {
        for(unsigned int j = 0; j < n_dg; ++j)
          phi_bar.submit_dof_value(j == i ? one : zero, j);
        phi_bar.evaluate(dealii::EvaluationFlags::values);
        for(unsigned int q = 0; q < phi_bar.n_q_points; ++q)
          phi_bar.submit_value(phi_bar.get_value(q), q);
        phi_bar.integrate(dealii::EvaluationFlags::values);
        for(unsigned int j = 0; j < n_dg; ++j)
          M_bar_bar[j][i] = phi_bar.get_dof_value(j);
      }
      M_bar_bar_inverse_cell[cell] = invert_matrix(M_bar_bar);
    }
    SimdMatrix const & M_bar_bar_inverse = M_bar_bar_inverse_cell[cell];

    for(unsigned int variable = 0; variable < variables.size(); ++variable)
    {
      InverseMassData & data = inverse_mass_data[variable][cell];

      // psi-dependent blocks, column i = enrichment shape function i:
      //   M_bar_tilde(j, i) = (N_j, psi N~_i)_K,   M_tilde_tilde(j, i) = (psi N~_j, psi N~_i)_K
      data.M_bar_tilde.assign(n_dg, std::vector<scalar>(n_en, zero));
      SimdMatrix M_tilde_tilde(n_en, std::vector<scalar>(n_en, zero));
      for(unsigned int i = 0; i < n_en; ++i)
      {
        for(unsigned int j = 0; j < n_en; ++j)
          phi_tilde.submit_dof_value(j == i ? one : zero, j);
        phi_tilde.evaluate(dealii::EvaluationFlags::values);
        for(unsigned int q = 0; q < phi_tilde.n_q_points; ++q)
        {
          scalar const psi   = get_psi_cell(variable, cell, q);
          scalar const psi_N = psi * phi_tilde.get_value(q);
          phi_bar.submit_value(psi_N, q);
          phi_tilde.submit_value(psi * psi_N, q);
        }
        phi_bar.integrate(dealii::EvaluationFlags::values);
        phi_tilde.integrate(dealii::EvaluationFlags::values);
        for(unsigned int j = 0; j < n_dg; ++j)
          data.M_bar_tilde[j][i] = phi_bar.get_dof_value(j);
        for(unsigned int j = 0; j < n_en; ++j)
          M_tilde_tilde[j][i] = phi_tilde.get_dof_value(j);
      }

      // S = M_tilde_tilde - M_bar_tilde^T X,   X = M_bar_bar^{-1} M_bar_tilde
      SimdMatrix X(n_dg, std::vector<scalar>(n_en, zero));
      for(unsigned int k = 0; k < n_dg; ++k)
        for(unsigned int l = 0; l < n_dg; ++l)
          for(unsigned int j = 0; j < n_en; ++j)
            X[k][j] += M_bar_bar_inverse[k][l] * data.M_bar_tilde[l][j];

      SimdMatrix Schur = M_tilde_tilde;
      for(unsigned int i = 0; i < n_en; ++i)
        for(unsigned int j = 0; j < n_en; ++j)
          for(unsigned int k = 0; k < n_dg; ++k)
            Schur[i][j] -= data.M_bar_tilde[k][i] * X[k][j];

      // inverse of the Schur complement on the well-posed enrichment modes
      data.schur_inverse =
        compute_schur_pseudo_inverse(Schur,
                                     M_tilde_tilde,
                                     data.projector,
                                     matrix_free_in.n_active_entries_per_cell_batch(cell),
                                     cell);
    }
  }
}

/*
 * Inverse of the Schur complement of the block mass matrix of an enriched cell and the projector
 * onto the well-posed enrichment space, scalar version of
 * WallDGCoupler::compute_schur_pseudo_inverse (Krank, thesis Eqs. (6.16), (6.17)):
 *
 *   S = M_tt - M_tb M_bb^{-1} M_bt,   D = diag(M_tt),   \hat S = D^{-1/2} S D^{-1/2} = V Lambda V^T
 *   W   = D^{-1/2} [ v_m ]_{lambda_m > epsilon}
 *   S^+ = W Lambda^{-1} W^T,   P = W (W^T W)^{-1} W^T
 *
 * epsilon = max(rank_tolerance, 10^3 eps_machine). Unfilled lanes and lanes switched off by the
 * y+ criterion (psi = 0): S^+ = 0, P = 0, not counted.
 */
template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::SimdMatrix
FunctionEnrichmentRANS<dim, Number>::compute_schur_pseudo_inverse(
  SimdMatrix const & Schur,
  SimdMatrix const & M_tilde_tilde,
  SimdMatrix &       projector,
  unsigned int const n_active_lanes,
  unsigned int const cell) const
{
  unsigned int const n = Schur.size();

  double const epsilon =
    std::max(rank_tolerance, 1.e3 * double(std::numeric_limits<Number>::epsilon()));

  scalar const zero = dealii::make_vectorized_array<Number>(0.0);
  SimdMatrix   schur_inverse(n, std::vector<scalar>(n, zero));
  projector.assign(n, std::vector<scalar>(n, zero));

  for(unsigned int v = 0; v < n_active_lanes; ++v)
  {
    // switched off by the y+ criterion of the velocity: psi = 0, M_tt = 0, no enrichment
    if(not is_enrichment_active_lane(cell, v))
      continue;

    // d_i = (M_tt)_{ii}^{-1/2}
    std::vector<double> d(n);
    for(unsigned int i = 0; i < n; ++i)
    {
      double const m_ii = M_tilde_tilde[i][i][v];
      AssertThrow(m_ii > 0.0,
                  dealii::ExcMessage("Enrichment mass matrix with non-positive diagonal entry."));
      d[i] = 1.0 / std::sqrt(m_ii);
    }

    // \hat S = D^{-1/2} S D^{-1/2}, symmetrized (S is symmetric up to round-off)
    dealii::LAPACKFullMatrix<double> S_hat(n, n);
    for(unsigned int i = 0; i < n; ++i)
      for(unsigned int j = 0; j < n; ++j)
      {
        S_hat(i, j) = 0.5 * d[i] * (double(Schur[i][j][v]) + double(Schur[j][i][v])) * d[j];
        AssertThrow(std::isfinite(S_hat(i, j)),
                    dealii::ExcMessage("Schur complement of an enriched cell is not finite."));
      }

    // symmetric eigensolver (robust for clustered near-zero modes)
    S_hat.set_property(dealii::LAPACKSupport::symmetric);
    dealii::Vector<double>     eigenvalues;
    dealii::FullMatrix<double> eigenvectors;
    S_hat.compute_eigenvalues_symmetric(-1.0, 1.0e10, 0.0, eigenvalues, eigenvectors);

    n_modes_total += n;
    n_modes_dropped += n - eigenvalues.size();

    // kept directions w_m = D^{-1/2} v_m (columns of W)
    std::vector<std::vector<double>> W;
    for(unsigned int m = 0; m < eigenvalues.size(); ++m)
    {
      double const lambda = eigenvalues(m);
      if(lambda <= epsilon)
      {
        ++n_modes_dropped;
        continue;
      }
      lambda_min_kept = std::min(lambda_min_kept, lambda);

      std::vector<double> w(n);
      for(unsigned int i = 0; i < n; ++i)
        w[i] = d[i] * eigenvectors(i, m);
      W.push_back(w);

      // (S^+)_{ij} += d_i (v_m)_i lambda_m^{-1} (v_m)_j d_j
      for(unsigned int i = 0; i < n; ++i)
        for(unsigned int j = 0; j < n; ++j)
          schur_inverse[i][j][v] +=
            Number(d[i] * eigenvectors(i, m) / lambda * eigenvectors(j, m) * d[j]);
    }

    // P = W (W^T W)^{-1} W^T; exactly I if nothing is removed, 0 if everything is removed
    unsigned int const r = W.size();
    if(r == n)
    {
      for(unsigned int i = 0; i < n; ++i)
        projector[i][i][v] = 1.0;
    }
    else if(r > 0)
    {
      dealii::FullMatrix<double> G(r, r);
      for(unsigned int a = 0; a < r; ++a)
        for(unsigned int b = 0; b < r; ++b)
          for(unsigned int i = 0; i < n; ++i)
            G(a, b) += W[a][i] * W[b][i];
      G.gauss_jordan();

      for(unsigned int i = 0; i < n; ++i)
        for(unsigned int j = 0; j < n; ++j)
        {
          double p_ij = 0.0;
          for(unsigned int a = 0; a < r; ++a)
            for(unsigned int b = 0; b < r; ++b)
              p_ij += W[a][i] * G(a, b) * W[b][j];
          projector[i][j][v] = Number(p_ij);
        }
    }
  }
  return schur_inverse;
}

/*
 * Inverse of a symmetric positive definite matrix per SIMD lane (Gauss-Jordan without pivoting,
 * as WallDGCoupler::invert_matrix_simd).
 */
template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::SimdMatrix
FunctionEnrichmentRANS<dim, Number>::invert_matrix(SimdMatrix M)
{
  unsigned int const n    = M.size();
  scalar const       zero = dealii::make_vectorized_array<Number>(0.0);
  scalar const       one  = dealii::make_vectorized_array<Number>(1.0);

  SimdMatrix inverse(n, std::vector<scalar>(n, zero));
  for(unsigned int i = 0; i < n; ++i)
    inverse[i][i] = one;

  for(unsigned int i = 0; i < n; ++i)
  {
    scalar const pivot = one / M[i][i];
    for(unsigned int j = 0; j < n; ++j)
    {
      M[i][j] *= pivot;
      inverse[i][j] *= pivot;
    }
    for(unsigned int k = 0; k < n; ++k)
      if(k != i)
      {
        scalar const factor = M[k][i];
        for(unsigned int j = 0; j < n; ++j)
        {
          M[k][j] -= factor * M[i][j];
          inverse[k][j] -= factor * inverse[i][j];
        }
      }
  }

  return inverse;
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::scalar const &
FunctionEnrichmentRANS<dim, Number>::get_psi_cell(unsigned int const variable,
                                                  unsigned int const cell,
                                                  unsigned int const q) const
{
  AssertIndexRange(cell * n_q_points_cell + q, psi_cell[variable].size());
  return psi_cell[variable][cell * n_q_points_cell + q];
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::vector const &
FunctionEnrichmentRANS<dim, Number>::get_grad_psi_cell(unsigned int const variable,
                                                       unsigned int const cell,
                                                       unsigned int const q) const
{
  AssertIndexRange(cell * n_q_points_cell + q, grad_psi_cell[variable].size());
  return grad_psi_cell[variable][cell * n_q_points_cell + q];
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::scalar const &
FunctionEnrichmentRANS<dim, Number>::get_psi_cell_old(unsigned int const variable,
                                                      unsigned int const cell,
                                                      unsigned int const q) const
{
  AssertIndexRange(cell * n_q_points_cell + q, psi_cell_old[variable].size());
  return psi_cell_old[variable][cell * n_q_points_cell + q];
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::scalar const &
FunctionEnrichmentRANS<dim, Number>::get_psi_face(unsigned int const variable,
                                                  unsigned int const face,
                                                  bool const         interior,
                                                  unsigned int const q) const
{
  AssertIndexRange(face * n_q_points_face + q, psi_face[variable][interior ? 0 : 1].size());
  return psi_face[variable][interior ? 0 : 1][face * n_q_points_face + q];
}

template<int dim, typename Number>
typename FunctionEnrichmentRANS<dim, Number>::vector const &
FunctionEnrichmentRANS<dim, Number>::get_grad_psi_face(unsigned int const variable,
                                                       unsigned int const face,
                                                       bool const         interior,
                                                       unsigned int const q) const
{
  AssertIndexRange(face * n_q_points_face + q, grad_psi_face[variable][interior ? 0 : 1].size());
  return grad_psi_face[variable][interior ? 0 : 1][face * n_q_points_face + q];
}

template<int dim, typename Number>
std::shared_ptr<dealii::MatrixFree<dim, Number>>
FunctionEnrichmentRANS<dim, Number>::get_matrix_free() const
{
  return matrix_free;
}

template<int dim, typename Number>
unsigned int
FunctionEnrichmentRANS<dim, Number>::get_dof_index_dg() const
{
  return dof_index_dg;
}

template<int dim, typename Number>
unsigned int
FunctionEnrichmentRANS<dim, Number>::get_dof_index_en() const
{
  return dof_index_en;
}

template<int dim, typename Number>
unsigned int
FunctionEnrichmentRANS<dim, Number>::get_quad_index() const
{
  return quad_index;
}

template<int dim, typename Number>
unsigned int
FunctionEnrichmentRANS<dim, Number>::get_active_fe_index() const
{
  return active_fe_index;
}

template<int dim, typename Number>
bool
FunctionEnrichmentRANS<dim, Number>::is_enriched_cell(
  dealii::types::global_cell_index const cell_index) const
{
  return enriched_cells.find(cell_index) != enriched_cells.end();
}

template<int dim, typename Number>
bool
FunctionEnrichmentRANS<dim, Number>::is_wall_boundary(
  dealii::types::boundary_id const boundary_id) const
{
  return velocity_enrichment.is_wall_boundary(boundary_id);
}

template<int dim, typename Number>
OmegaEnrichmentWallCondition
FunctionEnrichmentRANS<dim, Number>::get_wall_condition(unsigned int const variable) const
{
  AssertIndexRange(variable, variables.size());
  return variables[variable].wall_condition;
}

template<int dim, typename Number>
OmegaEnrichmentFaceFormulation
FunctionEnrichmentRANS<dim, Number>::get_face_formulation(
  unsigned int const variable) const
{
  AssertIndexRange(variable, variables.size());
  return variables[variable].face_formulation;
}

template<int dim, typename Number>
unsigned int
FunctionEnrichmentRANS<dim, Number>::n_variables() const
{
  return static_cast<unsigned int>(variables.size());
}

template class FunctionEnrichmentRANS<2, float>;
template class FunctionEnrichmentRANS<2, double>;
template class FunctionEnrichmentRANS<3, float>;
template class FunctionEnrichmentRANS<3, double>;

} // namespace ExaDG
