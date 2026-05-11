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

#include <exadg/operators/wall_modelling/function_enrichment.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>   // covers FEValues, FEFaceValues, FESubfaceValues
#include <deal.II/fe/mapping.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/lac/vector_operation.h>

namespace ExaDG
{

// ============================================================================
//   Constructor
// ============================================================================

template<int dim, typename Number>
FunctionEnrichment<dim, Number>::FunctionEnrichment(double kinematic_viscosity_in)
  : matrix_free(nullptr)
  , dof_index_dg(0)
  , dof_index_en(0)
  , quad_index(0)
  , kinematic_viscosity(kinematic_viscosity_in)
  , kappa(0.41)
  , beta(5.2)
  , dof_handler_en_vector()
{}


// ============================================================================
//   initialize_dofs
// ============================================================================

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::initialize_dofs(
  dealii::Triangulation<dim> const & triangulation,
  unsigned int fe_degree_en)
{
  ExaDG::ElementType type = ExaDG::ElementType::Hypercube;

  // FE enrichment vector
  fe_en_vector = create_finite_element<dim>(type,
                                            true /* is_dg*/,
                                            dim /*n_components*/,
                                            fe_degree_en);
  dof_handler_en_vector.reinit(triangulation);
  dof_handler_en_vector.distribute_dofs(*fe_en_vector);

  // FE wall vector
  fe_cg_vector = create_finite_element<dim>(type,
                                            false /* is_dg*/,
                                            dim /*n_components*/,
                                            1);
  dof_handler_cg_vector.reinit(triangulation);
  dof_handler_cg_vector.distribute_dofs(*fe_cg_vector);

  // FE wall scalar
  fe_cg_scalar = create_finite_element<dim>(type,
                                            false /* is_dg*/,
                                            1 /*n_components*/,
                                            1);
  dof_handler_cg_scalar.reinit(triangulation);
  dof_handler_cg_scalar.distribute_dofs(*fe_cg_scalar);
}


// ============================================================================
//   initialize
// ============================================================================

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  std::shared_ptr<IncNS::BoundaryDescriptorU<dim> const> boundary_descriptor_in,
  unsigned int                            dof_index_dg_in,
  unsigned int                            dof_index_en_in,
  unsigned int                            dof_index_cg_vector_in,
  unsigned int                            dof_index_cg_scalar_in,
  unsigned int                            quad_index_in)
{
  matrix_free  = &matrix_free_in;

  this->boundary_descriptor = boundary_descriptor_in;

  dof_index_dg = dof_index_dg_in;
  dof_index_en = dof_index_en_in;
  dof_index_cg_vector = dof_index_cg_vector_in;
  dof_index_cg_scalar = dof_index_cg_scalar_in;
  quad_index   = quad_index_in;

  // Allocate CG vectors in the layout described by dof_index_en
  matrix_free->initialize_dof_vector(wall_distance,     dof_index_cg_scalar);
  matrix_free->initialize_dof_vector(friction_velocity, dof_index_cg_scalar);
  friction_velocity = 0.0;
  friction_velocity.update_ghost_values();

  matrix_free->initialize_dof_vector(wall_velocity,     dof_index_cg_vector);

  matrix_free->initialize_dof_vector(enrichment_velocity, dof_index_en);
  enrichment_velocity = 0.0;

  matrix_free->initialize_dof_vector(enrichment_residual, dof_index_en);
  enrichment_residual = 0.0;
}

// Helper lambda for general SIMD matrix inversion(Gauss Jordan Elimination)
template<int dim, typename Number>
std::vector<std::vector<dealii::VectorizedArray<Number>>>
FunctionEnrichment<dim, Number>::invert_matrix_simd(std::vector<std::vector<dealii::VectorizedArray<Number>>> M, unsigned int n) 
{
  std::vector<std::vector<scalar>> M_copy = M;
  std::vector<std::vector<dealii::VectorizedArray<Number>>> inv(n, std::vector<dealii::VectorizedArray<Number>>(n, dealii::make_vectorized_array<Number>(0.0)));
  for (unsigned int i = 0; i < n; ++i)
  {
    inv[i][i] = dealii::make_vectorized_array<Number>(1.0);
  }
  for (unsigned int i = 0; i < n; ++i) 
  {
    auto pivot = dealii::make_vectorized_array<Number>(1.0) / M_copy[i][i];
    for (unsigned int j = 0; j < n; ++j) 
    {
      M_copy[i][j] *= pivot; inv[i][j] *= pivot; 
    }
    for (unsigned int k = 0; k < n; ++k) 
    {
      if (k != i) 
      {
        auto factor = M_copy[k][i];
        for (unsigned int j = 0; j < n; ++j) 
        {
          M_copy[k][j] -= factor * M_copy[i][j];
          inv[k][j] -= factor * inv[i][j];
        }
      }
    }
  }
  return inv;
}

// ============================================================================
//   Precompute Schur Matrices
// ============================================================================
template<int dim, typename Number>
void 
FunctionEnrichment<dim, Number>::precompute_schur_matrices()
{
  const unsigned int n_batches = matrix_free->n_cell_batches();
  cell_schur_data.resize(n_batches);

  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_dg(*matrix_free, dof_index_dg, quad_index);
  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_en(*matrix_free, dof_index_en, quad_index);
  dealii::FEEvaluation<dim, -1, 0, 1, Number> y_eval(*matrix_free, dof_index_cg_scalar, quad_index);
  dealii::FEEvaluation<dim, -1, 0, 1, Number> utau_eval(*matrix_free, dof_index_cg_scalar, quad_index);

  const unsigned int n_dg = phi_dg.tensor_dofs_per_cell;
  const unsigned int n_en = phi_en.tensor_dofs_per_cell;

  for (unsigned int cell = 0; cell < n_batches; ++cell)
  {
    phi_dg.reinit(cell);
    phi_en.reinit(cell);
    y_eval.reinit(cell);
    utau_eval.reinit(cell);

    y_eval.read_dof_values_plain(wall_distance);
    utau_eval.read_dof_values_plain(friction_velocity);
    y_eval.evaluate(dealii::EvaluationFlags::values);
    utau_eval.evaluate(dealii::EvaluationFlags::values);

    std::vector<std::vector<scalar>> N_en(n_en, std::vector<scalar>(phi_en.n_q_points));
    for(unsigned int i=0; i<n_en; ++i) 
    {
      dealii::Tensor<1, dim, scalar> unit_tensor;
      for(unsigned int d=0; d<n_en; ++d)
      {
        unit_tensor[0]=dealii::make_vectorized_array<Number>((d==i)?1.0:0.0);
        phi_en.submit_dof_value(unit_tensor, d);
      }
      phi_en.evaluate(dealii::EvaluationFlags::values);
      for(unsigned int q=0; q<phi_en.n_q_points; ++q)
      {
        N_en[i][q] = phi_en.get_value(q)[0];
      }
    }

    std::vector<std::vector<dealii::VectorizedArray<Number>>> N_dg(n_dg, std::vector<dealii::VectorizedArray<Number>>(phi_dg.n_q_points));
    for(unsigned int i=0; i<n_dg; ++i) 
    {
      dealii::Tensor<1, dim, scalar> unit_tensor;
      for(unsigned int d=0; d<n_dg; ++d) 
      {
        unit_tensor[0]=dealii::make_vectorized_array<Number>((d==i)?1.0:0.0);
        phi_dg.submit_dof_value(unit_tensor, d);
      }
      phi_dg.evaluate(dealii::EvaluationFlags::values);
      for(unsigned int q=0; q<phi_dg.n_q_points; ++q)
      {
        N_dg[i][q] = phi_dg.get_value(q)[0];
      }
    }

    // Allocate SIMD Matrices
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_tilde_tilde(n_en, std::vector<dealii::VectorizedArray<Number>>(n_en, dealii::make_vectorized_array<Number>(0.0)));
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_bar_tilde(n_dg, std::vector<dealii::VectorizedArray<Number>>(n_en, dealii::make_vectorized_array<Number>(0.0)));
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_bar_bar(n_dg, std::vector<dealii::VectorizedArray<Number>>(n_dg, dealii::make_vectorized_array<Number>(0.0)));

    for (unsigned int q = 0; q < phi_dg.n_q_points; ++q)
    {
      auto y    = y_eval.get_value(q);
      auto utau = utau_eval.get_value(q);
      auto psi  = get_value(utau, y); 
      auto JxW  = phi_dg.JxW(q);

      for (unsigned int i = 0; i < n_en; ++i) 
      {
        for (unsigned int j = 0; j < n_en; ++j)
        {
          M_tilde_tilde[i][j] += (psi * N_en[i][q]) * (psi * N_en[j][q]) * JxW;
        }
        for (unsigned int j = 0; j < n_dg; ++j)
        {
          M_bar_tilde[j][i] += N_dg[j][q] * (psi * N_en[i][q]) * JxW;
        }
      }
      for (unsigned int i = 0; i < n_dg; ++i) 
      {
        for (unsigned int j = 0; j < n_dg; ++j)
        {
          M_bar_bar[i][j] += N_dg[i][q] * N_dg[j][q] * JxW;
        }
      }
    }

    auto M_bar_inverse = invert_matrix_simd(M_bar_bar, n_dg);

    // Compute S = M_tilde_tilde - M_tilde_bar * (M_bar_bar^-1 * M_bar_tilde)
    std::vector<std::vector<dealii::VectorizedArray<Number>>> S = M_tilde_tilde;
    for(unsigned int i=0; i<n_en; ++i) 
    {
      for(unsigned int j=0; j<n_en; ++j) 
      {
        dealii::VectorizedArray<Number> sum = dealii::make_vectorized_array<Number>(0.0);
        for(unsigned int k=0; k<n_dg; ++k) 
        {
          for(unsigned int l=0; l<n_dg; ++l) 
          {
            sum += M_bar_tilde[k][i] * M_bar_inverse[k][l] * M_bar_tilde[l][j]; // transpose handling
          }
        }
        S[i][j] -= sum;
      }
    }

    cell_schur_data[cell].M_Vbar_Utilde  = M_bar_tilde;
    cell_schur_data[cell].M_Vbar_inverse = M_bar_inverse;
    cell_schur_data[cell].Schur_inverse  = invert_matrix_simd(S, n_en);
  }
}

// ============================================================================
//   Apply Schur Inverse Mass
// ============================================================================
template<int dim, typename Number>
void 
FunctionEnrichment<dim, Number>::apply_schur_inverse_mass(
                                VectorType & dst_bar,
                                VectorType & dst_tilde,
                                VectorType const & src_bar,
                                VectorType const & src_tilde) const
{
  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_dg(*matrix_free, dof_index_dg, quad_index);
  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_en(*matrix_free, dof_index_en, quad_index);

  const unsigned int n_dg = phi_dg.tensor_dofs_per_cell;
  const unsigned int n_en = phi_en.tensor_dofs_per_cell;
  const unsigned int n_batches = matrix_free->n_cell_batches();

  // Create a zeroed-out Tensor to safely initialize our arrays
  using TensorType = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  TensorType zero_tensor;
  for (unsigned int d = 0; d < dim; ++d) {
    zero_tensor[d] = dealii::make_vectorized_array<Number>(0.0);
  }

  for (unsigned int cell = 0; cell < n_batches; ++cell)
  {
    phi_dg.reinit(cell);
    phi_en.reinit(cell);

    phi_dg.read_dof_values_plain(src_bar);
    phi_en.read_dof_values_plain(src_tilde);

    // Grab the right hand sides as Velocity Tensors
    std::vector<TensorType> R_bar(n_dg);
    for(unsigned int i=0; i<n_dg; ++i) R_bar[i] = phi_dg.get_dof_value(i);

    std::vector<TensorType> R_tilde(n_en);
    for(unsigned int i=0; i<n_en; ++i) R_tilde[i] = phi_en.get_dof_value(i);

    auto const & M_bar_tilde   = cell_schur_data[cell].M_Vbar_Utilde;
    auto const & M_bar_inverse = cell_schur_data[cell].M_Vbar_inverse;
    auto const & S_inv         = cell_schur_data[cell].Schur_inverse;

    // --- STEP 1: U_tilde ---
    std::vector<TensorType> M_inv_R_bar(n_dg, zero_tensor);
    for(unsigned int i=0; i<n_dg; ++i) {
      for(unsigned int j=0; j<n_dg; ++j) M_inv_R_bar[i] += M_bar_inverse[i][j] * R_bar[j];
    }

    std::vector<TensorType> coupling(n_en, zero_tensor);
    for(unsigned int i=0; i<n_en; ++i) {
      for(unsigned int j=0; j<n_dg; ++j) coupling[i] += M_bar_tilde[j][i] * M_inv_R_bar[j];
    }

    std::vector<TensorType> U_tilde(n_en, zero_tensor);
    for(unsigned int i=0; i<n_en; ++i) {
      auto diff = R_tilde[i] - coupling[i];
      for(unsigned int j=0; j<n_en; ++j) U_tilde[i] += S_inv[i][j] * diff;
    }

    // --- STEP 2: U_bar ---
    std::vector<TensorType> diff_bar(n_dg, zero_tensor);
    for(unsigned int i=0; i<n_dg; ++i) {
      diff_bar[i] = R_bar[i];
      for(unsigned int j=0; j<n_en; ++j) diff_bar[i] -= M_bar_tilde[i][j] * U_tilde[j];
    }

    std::vector<TensorType> U_bar(n_dg, zero_tensor);
    for(unsigned int i=0; i<n_dg; ++i) {
      for(unsigned int j=0; j<n_dg; ++j) U_bar[i] += M_bar_inverse[i][j] * diff_bar[j];
    }

    // --- STEP 3: Write back to memory ---
    for(unsigned int i=0; i<n_dg; ++i) phi_dg.submit_dof_value(U_bar[i], i);
    phi_dg.set_dof_values_plain(dst_bar);

    for(unsigned int i=0; i<n_en; ++i) phi_en.submit_dof_value(U_tilde[i], i);
    phi_en.set_dof_values_plain(dst_tilde);
  }
}

// ============================================================================
//  setup_wall_distance
//
//  Implements eq. (6.13): y_B = shortest distance from node B to any wall node.
//
//  Additional work compared to the original code:
//    • Gathers wall DoF coordinates from all MPI ranks (fixes parallel gap).
//    • Deduplicates wall DoFs that appear on partition boundaries.
//    • Builds node_to_wall_node: near-wall DoF → nearest wall DoF global index.
// ============================================================================

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::setup_wall_distance(
  dealii::Mapping<dim> const &                           mapping,
  unsigned int                                           layers)
{
  // Initialize to a large number so freestream cells don't default to 0.0
  wall_distance = 1e10; 

  // Maps the global DoF index to its physical coordinates for all CG DoFs (including wall and near-wall)
  std::map<dealii::types::global_dof_index, dealii::Point<dim>> support_points;
  dealii::DoFTools::map_dofs_to_support_points(mapping, dof_handler_cg_scalar, support_points);

  std::vector<std::pair<dealii::Point<dim>, dealii::types::global_dof_index>> local_wall_points;
  std::set<typename dealii::DoFHandler<dim>::active_cell_iterator> current_layer;
  std::set<typename dealii::DoFHandler<dim>::active_cell_iterator> all_near_wall_cells;

  auto const & fe = dof_handler_cg_scalar.get_fe();

  // Step 1: Find wall DoFs and seed BFS
  for(auto const & cell : dof_handler_cg_scalar.active_cell_iterators())
  {
    if(cell->is_locally_owned() && cell->at_boundary())
    {
      for(auto face : cell->face_indices())
      {
        // Collect all points on wall boundaries and their global DoF indices for distance computation
        if(cell->face(face)->at_boundary())
        {
          auto bid = cell->face(face)->boundary_id();
          // Check if the boundary is a wall boundary
          if(boundary_descriptor->get_boundary_type(bid) == IncNS::BoundaryTypeU::WallEnrichment)
          {
            current_layer.insert(cell);
            all_near_wall_cells.insert(cell);

            // Extract the DoF indices for this cell and check which ones are on the face
            std::vector<dealii::types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            // Loop over local DoFs and check if they have support on the current face
            for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i) 
            {
              if(fe.has_support_on_face(i, face)) 
              {
                auto global_dof = local_dof_indices[i];
                if(wall_distance.locally_owned_elements().is_element(global_dof))
                {
                  // This DoF is on the wall boundary, so we add its coordinates to the local list
                  local_wall_points.push_back({support_points[local_dof_indices[i]], local_dof_indices[i]});
                }
              }
            }
          }
        }
      }
    }
  }

  // Step 2: Expand layers (BFS)
  for(unsigned int l = 1; l < layers; ++l)
  {
    std::set<typename dealii::DoFHandler<dim>::active_cell_iterator> next_layer;
    for(auto const & cell : current_layer) 
    {
      for(auto face : cell->face_indices()) 
      {
        if(!cell->face(face)->at_boundary()) 
        {
          auto neighbor = cell->neighbor(face);
          // Only consider locally owned neighbors that haven't already been marked as near-wall
          if(neighbor->is_locally_owned() && all_near_wall_cells.find(neighbor) == all_near_wall_cells.end()) 
          {
            next_layer.insert(neighbor);
            all_near_wall_cells.insert(neighbor);
          }
        }
      }
    }
    current_layer = next_layer;
  }

  // Step 3: Compute minimum distances
  for(auto const & cell : all_near_wall_cells)
  {
    std::vector<dealii::types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
    cell->get_dof_indices(local_dof_indices);

    for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    {
     auto global_dof = local_dof_indices[i];

      if (!wall_distance.locally_owned_elements().is_element(global_dof))
      {
        continue;
      }

     auto dof_point  = support_points[global_dof];

     Number min_dist = std::numeric_limits<Number>::max();
     dealii::types::global_dof_index nearest_wall_dof = dealii::numbers::invalid_dof_index;

     for(auto const & wall_point : local_wall_points) 
     {
        Number distance = dof_point.distance(wall_point.first);
        if(distance < min_dist)
        {
          min_dist = distance;
          nearest_wall_dof = wall_point.second;
        }
     }
     if (min_dist < wall_distance[global_dof])
     {
      wall_distance[global_dof] = min_dist;

      node_to_wall_node[global_dof] = nearest_wall_dof; 
     }
    }
  }

  // CRITICAL: Resolves partition boundaries cleanly across MPI without Allgather
  wall_distance.compress(dealii::VectorOperation::min);

  wall_distance.update_ghost_values();
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::project_velocity_to_wall(VectorType const & src)
{
  VectorType rhs, lumped_mass;
  matrix_free->initialize_dof_vector(rhs,         dof_index_cg_vector);
  matrix_free->initialize_dof_vector(lumped_mass, dof_index_cg_vector);

  rhs = 0.0;
  lumped_mass = 0.0;

  matrix_free->cell_loop(&FunctionEnrichment::loop_project_velocity_to_wall,
                         this,
                         rhs,
                         src,
                         false);

  matrix_free->cell_loop(&FunctionEnrichment::loop_lumped_mass,
                         this,
                         lumped_mass,
                         src,
                         false);

  rhs.compress(dealii::VectorOperation::add);
  lumped_mass.compress(dealii::VectorOperation::add);

  // Apply lumped M^{-1}: elementwise division
  for (unsigned int i = 0; i < rhs.locally_owned_size(); ++i)
  {
   AssertThrow(lumped_mass.local_element(i) > 1e-12,
               dealii::ExcMessage("Lumped mass entry is too small, check quadrature and mesh quality."));

   wall_velocity.local_element(i) = rhs.local_element(i) / lumped_mass.local_element(i);
  }

  wall_velocity.update_ghost_values();
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::loop_lumped_mass(
  dealii::MatrixFree<dim, Number> const &       data,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  dealii::FEEvaluation<dim, -1, 0, dim, Number> dg_eval(data, dof_index_dg, quad_index);

  dealii::FEEvaluation<dim, -1, 0, dim, Number> cg_eval(data, dof_index_cg_vector, quad_index);

  dealii::Tensor<1, dim, scalar> one_vector;
  for (unsigned int d = 0; d < dim; ++d)
  {
    one_vector[d] = dealii::make_vectorized_array<Number>(1.0);
  }

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    dg_eval.reinit(cell);
    dg_eval.read_dof_values(src);
    dg_eval.evaluate(dealii::EvaluationFlags::values);

    // Denominator
    cg_eval.reinit(cell);
    for (unsigned int q = 0; q < cg_eval.n_q_points; ++q)
    {
      cg_eval.submit_value(one_vector, q);
    }
    cg_eval.integrate(dealii::EvaluationFlags::values);
    cg_eval.distribute_local_to_global(dst);
  }
}

// This function projects the DG solution over CG space
template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::loop_project_velocity_to_wall(
  dealii::MatrixFree<dim, Number> const & data,
  VectorType &                           dst,
  VectorType const &                     src,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  dealii::FEEvaluation<dim, -1, 0, dim, Number> dg_eval(data, dof_index_dg, quad_index);

  dealii::FEEvaluation<dim, -1, 0, dim, Number> cg_eval(data, dof_index_cg_vector, quad_index);

  dealii::Tensor<1, dim, scalar> one_vector;
  for (unsigned int d = 0; d < dim; ++d)
  {
    one_vector[d] = dealii::make_vectorized_array<Number>(1.0);
  }

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    dg_eval.reinit(cell);
    dg_eval.read_dof_values(src);
    dg_eval.evaluate(dealii::EvaluationFlags::values);

    cg_eval.reinit(cell);

    // Numerator
    for (unsigned int q = 0; q < cg_eval.n_q_points; ++q)
    {
      cg_eval.submit_value(dg_eval.get_value(q), q);
    }
    cg_eval.integrate(dealii::EvaluationFlags::values);
    cg_eval.distribute_local_to_global(dst);
  }
}


// ============================================================================
//  Private helper – integrate_wall_traction
//
//  Loops over all locally owned boundary cells and, for each wall face,
//  evaluates \tau_w = \nu \nabla u \cdot n at face quadrature points using the DG velocity
//  and accumulates
//
//    traction_num[d][B] += \int_{\partial \Omega} N_B^{cg}  \nu (\nabla u \cdot n)_d \, dA
//    traction_den[B]    += \int_{\partial \Omega} N_B^{cg} \, dA
//
//  by iterating CG and DG cell iterators in lock-step (same triangulation,
//  same active-cell ordering).
// ============================================================================

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::integrate_wall_traction(
  std::array<VectorType, dim> & traction_num,
  VectorType &                  traction_den) const
{
  // Pack the pointers into our struct to pass them safely into the MatrixFree loop
  TractionVectors dst_vectors;
  for (unsigned int d = 0; d < dim; ++d) {
    dst_vectors.num[d] = &traction_num[d];
  }
  dst_vectors.den = &traction_den;

  // Launch the ExaDG-style boundary loop!
  matrix_free->loop(&This::empty_loop,
                    &This::empty_loop,
                    &This::local_integrate_wall_traction, 
                    this,
                    dst_vectors,
                    wall_velocity,
                    false);
}

template<int dim, typename Number>
void FunctionEnrichment<dim, Number>::local_integrate_wall_traction(
  dealii::MatrixFree<dim, Number> const & matrix_free_data,
  TractionVectors &                       dst,
  VectorType const &                      src,
  std::pair<unsigned int, unsigned int> const & face_range) const
{
  // Face Evaluator for the high-order DG velocity
  dealii::FEFaceEvaluation<dim, -1, 0, dim, Number> u_wall(matrix_free_data, true, dof_index_cg_vector, quad_index);
  
  // Face Evaluator for the continuous linear CG shape functions
  dealii::FEFaceEvaluation<dim, -1, 0, 1, Number> wss(matrix_free_data, true, dof_index_cg_scalar, quad_index);

  scalar nu = dealii::make_vectorized_array<Number>(kinematic_viscosity);

  // Loop over the SIMD macro-faces
  for (unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    auto boundary_id = matrix_free_data.get_boundary_id(face);
    
    // Check if this macro-face is on our specified wall boundary
    if (boundary_descriptor->get_boundary_type(boundary_id) == IncNS::BoundaryTypeU::WallEnrichment)
    {
      // Evaluate the DG velocity and its gradients exactly on the face
      u_wall.reinit(face);
      u_wall.read_dof_values(src);
      u_wall.evaluate(dealii::EvaluationFlags::gradients);

      // Compute the Traction Vector at all quadrature points first (to save CPU cycles)
      std::vector<std::array<scalar, dim>> traction(u_wall.n_q_points); 
      for (unsigned int q = 0; q < u_wall.n_q_points; ++q)
      {
        auto grad_u = u_wall.get_gradient(q);
        auto n = u_wall.get_normal_vector(q);
        
        // \tau_w = \nu (\nabla u_h) * n
        auto tau_w = nu * (grad_u * n);
        for (unsigned int d = 0; d < dim; ++d) 
        {
          traction[q][d] = tau_w[d];
        }
      }

      // Assemble the Numerator for each dimension (\int N_B * \tau_w dA)
      for (unsigned int d = 0; d < dim; ++d)
      {
        wss.reinit(face);
        for (unsigned int q = 0; q < wss.n_q_points; ++q) 
        {
          wss.submit_value(traction[q][d], q);
        }
        wss.integrate(dealii::EvaluationFlags::values);
        
        // This handles thread-safe addition to the global CG vector automatically
        wss.distribute_local_to_global(*dst.num[d]);
      }

      // Assemble the Denominator Area Integral (\int N_B dA)
      wss.reinit(face);
      for (unsigned int q = 0; q < wss.n_q_points; ++q) 
      {
        wss.submit_value(dealii::make_vectorized_array<Number>(1.0), q);
      }
      wss.integrate(dealii::EvaluationFlags::values);
      wss.distribute_local_to_global(*dst.den);
    }
  }
}

template<int dim, typename Number>
void FunctionEnrichment<dim, Number>::evaluate_friction_velocity(VectorType const & velocity)
{
  // Project DG velocity onto the continuous P1 wall space (Spatial Coarsening)
  project_velocity_to_wall(velocity);

  // Setup temporary vectors for the boundary traction integration
  std::array<VectorType, dim> traction_num;
  for(unsigned int d = 0; d < dim; ++d) 
  {
    matrix_free->initialize_dof_vector(traction_num[d], dof_index_cg_scalar);
    traction_num[d] = 0.0;
  }

  VectorType traction_den;
  matrix_free->initialize_dof_vector(traction_den, dof_index_cg_scalar);
  traction_den = 0.0;

  // Run the boundary loop to compute tau_w
  integrate_wall_traction(traction_num, traction_den);

  // MPI Synchronization for the boundary integrals
  for(unsigned int d = 0; d < dim; ++d) {
    traction_num[d].compress(dealii::VectorOperation::add);
  }
  traction_den.compress(dealii::VectorOperation::add);

  // Compute Friction Velocity exactly at the Wall Nodes
  friction_velocity = 0.0;

  for (auto const & global_node : friction_velocity.locally_owned_elements())
  {
    double den = traction_den[global_node];

    // Only process actual wall boundary nodes (nodes with integrated area)
    if (den > 1e-14) 
    {
      dealii::Tensor<1, dim> tau_w_B;
      for (unsigned int d = 0; d < dim; ++d) 
      {
        tau_w_B[d] = traction_num[d][global_node] / den;
      }

      // u_tau = \sqrt(||\tau_w|| / \rho). (Assuming kinematic formulation, rho = 1)
      friction_velocity[global_node] = std::sqrt(tau_w_B.norm());
    }
  }

  friction_velocity.update_ghost_values();

  // Vertical Copying: Propagate u_tau to the off-wall fluid nodes
  for (auto const & global_fluid_node : friction_velocity.locally_owned_elements())
  {
    // If this is an off-wall node (no face area) AND we mapped it to a wall node
    if (traction_den[global_fluid_node] <= 1e-14 && node_to_wall_node.count(global_fluid_node))
    {
      auto wall_node = node_to_wall_node[global_fluid_node];

      // Direct copy from the paired wall node!
      friction_velocity[global_fluid_node] = friction_velocity[wall_node];
    }
  }

  // Final share of the completely populated u_tau vector to the downstream solver
  friction_velocity.update_ghost_values();
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::empty_loop(
dealii::MatrixFree<dim, Number> const & matrix_free_data,
TractionVectors &                       dst,
VectorType const &                      src,
std::pair<unsigned int, unsigned int> const & range) const
{
(void)matrix_free_data;
(void)dst;
(void)src;
(void)range;
}

// ============================================================================
//  Accessors
// ============================================================================

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
FunctionEnrichment<dim, Number>::get_dof_handler_en_vector() const
{
  return dof_handler_en_vector;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
FunctionEnrichment<dim, Number>::get_dof_handler_cg_scalar() const
{
  return dof_handler_cg_scalar;
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
FunctionEnrichment<dim, Number>::get_dof_handler_cg_vector() const
{
  return dof_handler_cg_vector;
}

template<int dim, typename Number>
typename FunctionEnrichment<dim, Number>::scalar
FunctionEnrichment<dim, Number>::get_value(
  scalar const u_tau,
  scalar const y) const
{
  scalar nu = dealii::make_vectorized_array<Number>(kinematic_viscosity);
  scalar y_plus = (y * u_tau) / nu;
  
  scalar u_plus = wall_law_eval.get_value(y_plus);
  scalar u_scalar = u_plus * u_tau;

  return u_scalar;
}

template<int dim, typename Number>
typename FunctionEnrichment<dim, Number>::scalar
FunctionEnrichment<dim, Number>::get_gradient(
  scalar const u_tau,
  scalar const y) const
{
  scalar nu = dealii::make_vectorized_array<Number>(kinematic_viscosity);
  scalar y_plus = (y * u_tau) / nu;
  
  scalar du_plus_dy_plus = wall_law_eval.get_gradient(y_plus);
  
  scalar du_dy_physical = (u_tau * u_tau / nu) * du_plus_dy_plus;

  return du_dy_physical;
}

template<int dim, typename Number>
unsigned int 
FunctionEnrichment<dim, Number>::get_dof_index_dg() const
{
  return dof_index_dg;
}

template<int dim, typename Number>
unsigned int 
FunctionEnrichment<dim, Number>::get_dof_index_en() const
{
  return dof_index_en;
}

template<int dim, typename Number>
unsigned int 
FunctionEnrichment<dim, Number>::get_dof_index_cg_scalar() const
{
  return dof_index_cg_scalar;
}

template<int dim, typename Number>
unsigned int 
FunctionEnrichment<dim, Number>::get_dof_index_cg_vector() const
{
  return dof_index_cg_vector;
}

template<int dim, typename Number>
unsigned int 
FunctionEnrichment<dim, Number>::get_quad_index() const
{
  return quad_index;
}

template class FunctionEnrichment<2, float>;
template class FunctionEnrichment<2, double>;
template class FunctionEnrichment<3, float>;
template class FunctionEnrichment<3, double>;

} // namespace ExaDG
