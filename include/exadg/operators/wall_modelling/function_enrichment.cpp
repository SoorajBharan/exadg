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

#include "function_enrichment.h"
#include <exadg/operators/wall_modelling/function_enrichment.h>
#include <exadg/operators/wall_modelling/fe_enriched_evaluation.h>
#include <exadg/operators/wall_modelling/fe_face_enriched_evaluation.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>   // covers FEValues, FEFaceValues, FESubfaceValues
#include <deal.II/fe/mapping.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/grid/reference_cell.h>

namespace ExaDG
{

// ============================================================================
//   Constructor
// ============================================================================

template<int dim, typename Number>
FunctionEnrichment<dim, Number>::FunctionEnrichment(double kinematic_viscosity_in)
  : dof_index_en(0)
  , quad_index(0)
  , kinematic_viscosity(kinematic_viscosity_in)
  , matrix_free_en(nullptr)
  , dof_handler_en_vector()
{}


// ============================================================================
//   initialize_dofs
// ============================================================================

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::initialize(
  dealii::Triangulation<dim> const &                     triangulation,
  std::shared_ptr<dealii::Mapping<dim> const>            mapping_in,
  dealii::DoFHandler<dim> const &                        global_dof_handler_in,
  unsigned int                                           fe_degree_dg,
  unsigned int                                           fe_degree_en,
  std::shared_ptr<IncNS::BoundaryDescriptorU<dim> const> boundary_descriptor_in,
  std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>> const & periodic_faces,
  unsigned int                                           quad_index_in,
  unsigned int                                           layers)
{
  this->boundary_descriptor = boundary_descriptor_in;
  quad_index   = 0;
  this->global_dof_handler = &global_dof_handler_in;

  ExaDG::ElementType type = ExaDG::ElementType::Hypercube;
  
  // ====================================================================
  // FE collections. MatrixFree requires the same active_fe_index in all
  // DoFHandlers of a cell:
  //   index 0 : not enriched       (no shadow, no enrichment)
  //   index 1 : enriched           (shadow U_bar + enrichment U_tilde)
  //   index 2 : halo               (shadow U_bar only, no enrichment)
  // ====================================================================

  // CG wall vector (wall velocity)
  fe_cg_vector = create_finite_element<dim>(type, false /* is_dg */, dim, 1);
  fe_cg_vector_collection.push_back(*fe_cg_vector); // index 0
  fe_cg_vector_collection.push_back(*fe_cg_vector); // index 1
  fe_cg_vector_collection.push_back(*fe_cg_vector); // index 2: halo
  dof_handler_cg_vector.reinit(triangulation);

  // CG wall scalar (friction velocity)
  fe_cg_scalar = create_finite_element<dim>(type, false /* is_dg */, 1, 1);
  fe_cg_scalar_collection.push_back(*fe_cg_scalar); // index 0
  fe_cg_scalar_collection.push_back(*fe_cg_scalar); // index 1
  fe_cg_scalar_collection.push_back(*fe_cg_scalar); // index 2: halo
  dof_handler_cg_scalar.reinit(triangulation);

  // DG enrichment velocity U_tilde
  fe_en_vector = create_finite_element<dim>(type, true /* is_dg */, dim, fe_degree_en);
  fe_en_collection.push_back(dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), dim)); // index 0
  fe_en_collection.push_back(*fe_en_vector);                                         // index 1
  fe_en_collection.push_back(dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), dim)); // index 2: halo, no enrichment
  dof_handler_en_vector.reinit(triangulation);

  // DG shadow velocity U_bar
  fe_shadow_vector = create_finite_element<dim>(type, true /* is_dg */, dim, fe_degree_dg);
  fe_shadow_collection.push_back(dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), dim)); // index 0
  fe_shadow_collection.push_back(*fe_shadow_vector);                                     // index 1
  fe_shadow_collection.push_back(*fe_shadow_vector);                                     // index 2: halo carries U_bar
  dof_handler_shadow_vector.reinit(triangulation);

  AssertThrow(global_dof_handler_pressure != nullptr,
              dealii::ExcMessage("Call set_global_pressure_dof_handler() before initialize()."));
  {
    dealii::FiniteElement<dim> const & fe_p = global_dof_handler_pressure->get_fe();

    // index 0 must be of the same kind as fe_p, otherwise deal.II cannot compare
    // neighbouring elements in distribute_dofs (FESystem only compares to FESystem)
    if(dynamic_cast<dealii::FESystem<dim> const *>(&fe_p) != nullptr)
      fe_shadow_pressure_collection.push_back(
        dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), fe_p.n_components())); // index 0
    else
      fe_shadow_pressure_collection.push_back(dealii::FE_Nothing<dim>());        // index 0

    fe_shadow_pressure_collection.push_back(fe_p); // index 1: enriched
    fe_shadow_pressure_collection.push_back(fe_p); // index 2: halo
  }
  dof_handler_shadow_pressure.reinit(triangulation);

  // DG scalar (wall distance)
  fe_en_scalar = create_finite_element<dim>(type, true /* is_dg */, 1, 1);
  fe_en_collection_scalar.push_back(dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), 1)); // index 0
  fe_en_collection_scalar.push_back(*fe_en_scalar);                                       // index 1
  fe_en_collection_scalar.push_back(dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), 1)); // index 2: halo
  dof_handler_en_scalar.reinit(triangulation);

  // ====================================================================
  // Topological BFS to locate the wall layer
  // ====================================================================
  std::set<typename dealii::DoFHandler<dim>::active_cell_iterator> current_layer;
  std::set<typename dealii::DoFHandler<dim>::active_cell_iterator> all_near_wall_cells;

  std::set<unsigned int> globally_enriched_set;
  std::vector<unsigned int> local_frontier;

  for(auto const & cell : dof_handler_en_vector.active_cell_iterators()) 
  {
    if(!cell->is_artificial() && cell->at_boundary()) 
    {
      for(auto face : cell->face_indices()) 
      {
        if(cell->face(face)->at_boundary() and not cell->has_periodic_neighbor(face)) 
        {
          if(boundary_descriptor->get_boundary_type(cell->face(face)->boundary_id()) == IncNS::BoundaryTypeU::WallEnrichment) 
          {
            current_layer.insert(cell);
            all_near_wall_cells.insert(cell);
            local_frontier.push_back(static_cast<unsigned int>(cell->global_active_cell_index()));
          }
        }
      }
    }
  }
  // Sync the Seed Layer across all processors
  std::vector<unsigned int> global_frontier = 
    dealii::Utilities::MPI::compute_set_union(local_frontier, triangulation.get_communicator());

  globally_enriched_set.insert(global_frontier.begin(), global_frontier.end());

  for(unsigned int l = 1; l < layers; ++l) 
  {
    local_frontier.clear();
    std::set<typename dealii::DoFHandler<dim>::active_cell_iterator> next_layer;
    for(auto const & cell : current_layer) 
    {
      for(auto face : cell->face_indices()) 
      {
        if(!cell->face(face)->at_boundary()) 
        {
          auto neighbor = cell->neighbor(face);
          if(neighbor->is_active() && !neighbor->is_artificial() && all_near_wall_cells.find(neighbor) == all_near_wall_cells.end()) 
          {
            next_layer.insert(neighbor);
            all_near_wall_cells.insert(neighbor);
            local_frontier.push_back(neighbor->global_active_cell_index());
          }
        }
      }
    }

    // Sync the newly discovered layer across all processors
    global_frontier = dealii::Utilities::MPI::compute_set_union(local_frontier, triangulation.get_communicator());
    globally_enriched_set.insert(global_frontier.begin(), global_frontier.end());

    // This allows the local processor to search the neighbors of these cells in the next loop.
    for(auto const & cell : dof_handler_en_vector.active_cell_iterators())
    {
      if(!cell->is_artificial() && 
        all_near_wall_cells.find(cell) == all_near_wall_cells.end() &&
        globally_enriched_set.count(static_cast<unsigned int>(cell->global_active_cell_index())) > 0)
      {
        next_layer.insert(cell);
        all_near_wall_cells.insert(cell);
      }
    }
    current_layer = next_layer;
  }

  // ====================================================================
  // Paint the Triangulation and Distribute Memory
  // ====================================================================
  auto cell_en = dof_handler_en_vector.begin_active();
  auto cell_shadow = dof_handler_shadow_vector.begin_active();

  auto cell_cg_vector = dof_handler_cg_vector.begin_active();
  auto cell_cg_scalar = dof_handler_cg_scalar.begin_active();

  auto cell_en_scalar = dof_handler_en_scalar.begin_active();
  auto cell_shadow_pressure = dof_handler_shadow_pressure.begin_active();
  for(; cell_en != dof_handler_en_vector.end(); ++cell_en, ++cell_shadow , ++cell_cg_vector, ++cell_cg_scalar, ++cell_en_scalar, ++cell_shadow_pressure)
  {
    if(cell_en->is_locally_owned()) 
    {
      bool is_wall = (all_near_wall_cells.find(cell_en) != all_near_wall_cells.end());
      bool is_ghost = (globally_enriched_set.count(static_cast<unsigned int>(cell_en->global_active_cell_index())) > 0);
      bool is_enriched = is_wall || is_ghost;
      is_cell_enriched[cell_en->global_active_cell_index()] = is_enriched;

      if (is_enriched)
      {
        cell_en->set_active_fe_index(1); // Enriched Space (Allocates RAM)
        cell_shadow->set_active_fe_index(1);
        cell_cg_vector->set_active_fe_index(1);
        cell_cg_scalar->set_active_fe_index(1);
        cell_en_scalar->set_active_fe_index(1);
        cell_shadow_pressure->set_active_fe_index(1);
      }

    }
  }

  auto const is_enriched_cell = [&](auto const & c) {
    return all_near_wall_cells.count(c) > 0 or
    globally_enriched_set.count(static_cast<unsigned int>(c->global_active_cell_index())) > 0;
  };

  auto c_en        = dof_handler_en_vector.begin_active();
  auto c_shadow    = dof_handler_shadow_vector.begin_active();
  auto c_cg_vector = dof_handler_cg_vector.begin_active();
  auto c_cg_scalar = dof_handler_cg_scalar.begin_active();
  auto c_en_scalar = dof_handler_en_scalar.begin_active();
  auto c_shadow_pressure = dof_handler_shadow_pressure.begin_active();

  for(; c_en != dof_handler_en_vector.end();
  ++c_en, ++c_shadow, ++c_cg_vector, ++c_cg_scalar, ++c_en_scalar, ++c_shadow_pressure)
  {
    if(not c_en->is_locally_owned() or is_enriched_cell(c_en))
      continue;

    bool touches_enriched = false;
    for(auto const f : c_en->face_indices())
    {
      if(c_en->at_boundary(f) and not c_en->has_periodic_neighbor(f))
        continue;

      auto const nb = c_en->at_boundary(f) ? c_en->periodic_neighbor(f) : c_en->neighbor(f);
      if(nb->is_active() and is_enriched_cell(nb))
        touches_enriched = true;
    }

    if(touches_enriched)
    {
      c_en->set_active_fe_index(2);
      c_shadow->set_active_fe_index(2);
      c_cg_vector->set_active_fe_index(2);
      c_cg_scalar->set_active_fe_index(2);
      c_en_scalar->set_active_fe_index(2);
      c_shadow_pressure->set_active_fe_index(2);
    }
  }

  dof_handler_en_vector.distribute_dofs(fe_en_collection);
  dof_handler_shadow_vector.distribute_dofs(fe_shadow_collection);

  dof_handler_cg_vector.distribute_dofs(fe_cg_vector_collection);
  dof_handler_cg_scalar.distribute_dofs(fe_cg_scalar_collection);

  dof_handler_en_scalar.distribute_dofs(fe_en_collection_scalar);
  dof_handler_shadow_pressure.distribute_dofs(fe_shadow_pressure_collection);
  // ====================================================================
  // Setup Isolated MatrixFree Object
  // ====================================================================
  dof_index_en            = 0;
  dof_index_cg_scalar     = 1;
  dof_index_cg_vector     = 2;
  dof_index_shadow_vector = 3;
  dof_index_en_scalar     = 4;
  dof_index_shadow_pressure = 5;

  // CG scalar (friction velocity, wall traction)
  constraint_cg_scalar.clear();
  constraint_cg_scalar.reinit(dof_handler_cg_scalar.locally_owned_dofs(),
                              dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_cg_scalar));

  make_periodicity_constraints_hp(dof_handler_cg_scalar,
                                  periodic_faces,
                                  constraint_cg_scalar);
  constraint_cg_scalar.close();

  // CG vector (wall velocity)
  constraint_cg_vector.clear();
  constraint_cg_vector.reinit(dof_handler_cg_vector.locally_owned_dofs(),
                              dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_cg_vector));
  make_periodicity_constraints_hp(dof_handler_cg_vector,
                                  periodic_faces,
                                  constraint_cg_vector);
  constraint_cg_vector.close();

  // DG spaces: no constraints
  constraint_dg.clear();
  constraint_dg.close();

  std::vector<const dealii::DoFHandler<dim> *> dof_handlers = {
    &dof_handler_en_vector, &dof_handler_cg_scalar, &dof_handler_cg_vector, &dof_handler_shadow_vector, &dof_handler_en_scalar, &dof_handler_shadow_pressure
  };

  // must be in the SAME order as the dof_handlers vector below/above
  std::vector<const dealii::AffineConstraints<Number> *> constraints = {
    &constraint_dg,        // dof_handler_en_vector     (dof_index_en)
    &constraint_cg_scalar, // dof_handler_cg_scalar     (dof_index_cg_scalar)
    &constraint_cg_vector, // dof_handler_cg_vector     (dof_index_cg_vector)
    &constraint_dg,        // dof_handler_shadow_vector (dof_index_shadow_vector)
    &constraint_dg,         // dof_handler_en_scalar     (dof_index_en_scalar)
    &constraint_dg         // dof_handler_shadow_pressure (dof_index_shadow_pressure)
  };

  unsigned int const n_q_1d = std::max({fe_degree_dg + 1, fe_degree_en + 1, 8u});
  std::shared_ptr<dealii::Quadrature<dim>> quad =
    create_quadrature<dim>(ExaDG::ElementType::Hypercube, n_q_1d);
  std::vector<dealii::Quadrature<dim>> quadratures = { *quad };

  typename dealii::MatrixFree<dim, Number>::AdditionalData additional_data;

  additional_data.mapping_update_flags = dealii::update_values |
    dealii::update_gradients |
    dealii::update_JxW_values |
    dealii::update_quadrature_points | 
    dealii::update_normal_vectors;

  additional_data.mapping_update_flags_boundary_faces = dealii::update_values |
    dealii::update_gradients |
    dealii::update_JxW_values |
    dealii::update_quadrature_points | 
    dealii::update_normal_vectors;

  additional_data.mapping_update_flags_inner_faces = dealii::update_values |
                                                     dealii::update_gradients |
                                                     dealii::update_JxW_values |
                                                     dealii::update_quadrature_points |
                                                     dealii::update_normal_vectors;
  
  matrix_free_en = std::make_shared<dealii::MatrixFree<dim, Number>>();
  matrix_free_en->reinit(*mapping_in, dof_handlers, constraints, quadratures, additional_data);

  // ====================================================================
  //  Allocate Vectors natively on the new MatrixFree object
  // ====================================================================
  matrix_free_en->initialize_dof_vector(enrichment_velocity, dof_index_en);
  matrix_free_en->initialize_dof_vector(enrichment_residual, dof_index_en);

  matrix_free_en->initialize_dof_vector(shadow_velocity, dof_index_shadow_vector);
  matrix_free_en->initialize_dof_vector(shadow_velocity_residual, dof_index_shadow_vector);

  matrix_free_en->initialize_dof_vector(friction_velocity, dof_index_cg_scalar);

  matrix_free_en->initialize_dof_vector(wall_shear_stress, dof_index_cg_scalar);

  matrix_free_en->initialize_dof_vector(wall_distance,     dof_index_en_scalar);

  matrix_free_en->initialize_dof_vector(wall_velocity, dof_index_cg_vector);

  matrix_free_en->initialize_dof_vector(shadow_pressure, dof_index_shadow_pressure);

  matrix_free_en->initialize_dof_vector(src, dof_index_shadow_vector);
  matrix_free_en->initialize_dof_vector(dst, dof_index_shadow_vector);

  matrix_free_en->initialize_dof_vector(friction_velocity_old, dof_index_cg_scalar);

  enrichment_velocity = 0.0;
  enrichment_residual = 0.0;
  shadow_velocity = 0.0;
  shadow_velocity_residual = 0.0;
  friction_velocity = 0.0;
  wall_distance = 1e10;
  wall_velocity = 0.0;
  friction_velocity_old = 0.0;
  wall_shear_stress = 0.0;
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::copy_global_dg_to_wall_layout(
  VectorType const & global_vec,
  VectorType &       wall_vec) const
{
  wall_vec = 0.0;
  const unsigned int dofs_per_cell = dof_handler_shadow_vector.get_fe(1).n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> global_dofs(dofs_per_cell);
  std::vector<dealii::types::global_dof_index> shadow_dofs(dofs_per_cell);

  // get the cell iterator to the first active cell using dof handler
  auto cell_g      = global_dof_handler->begin_active();
  auto cell_shadow = dof_handler_shadow_vector.begin_active();
  
  // Loop throught the triangulation using cell iterators
  for (; cell_g != global_dof_handler->end(); ++cell_g, ++cell_shadow)
  {
    if (cell_shadow->is_locally_owned() && cell_shadow->active_fe_index() >= 1)
    {
      cell_g->get_dof_indices(global_dofs);
      cell_shadow->get_dof_indices(shadow_dofs);
      for (unsigned int i = 0; i < dofs_per_cell; ++i) 
      {
        wall_vec[shadow_dofs[i]] = global_vec[global_dofs[i]];
      }
    }
  }
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::copy_wall_layout_to_global_dg(
  VectorType const & wall_vec,
  VectorType &       global_vec) const
{
  const unsigned int dofs_per_cell = dof_handler_shadow_vector.get_fe(1).n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> global_dofs(dofs_per_cell);
  std::vector<dealii::types::global_dof_index> shadow_dofs(dofs_per_cell);

  auto cell_g      = global_dof_handler->begin_active();
  auto cell_shadow = dof_handler_shadow_vector.begin_active();
  
  for (; cell_g != global_dof_handler->end(); ++cell_g, ++cell_shadow)
  {
    if (cell_shadow->is_locally_owned() && cell_shadow->active_fe_index() == 1)
    {
      cell_g->get_dof_indices(global_dofs);
      cell_shadow->get_dof_indices(shadow_dofs);
      for (unsigned int i = 0; i < dofs_per_cell; ++i) 
      {
        global_vec[global_dofs[i]] = wall_vec[shadow_dofs[i]];
      }
    }
  }
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::add_wall_layout_to_global_dg(VectorType const & wall_vec,
                                                              VectorType &       global_vec) const
{
  const unsigned int dofs_per_cell = dof_handler_shadow_vector.get_fe(1).n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> global_dofs(dofs_per_cell);
  std::vector<dealii::types::global_dof_index> shadow_dofs(dofs_per_cell);

  auto cell_g      = global_dof_handler->begin_active();
  auto cell_shadow = dof_handler_shadow_vector.begin_active();

  for(; cell_g != global_dof_handler->end(); ++cell_g, ++cell_shadow)
  {
    if(cell_shadow->is_locally_owned() && cell_shadow->active_fe_index() >= 1)
    {
      cell_g->get_dof_indices(global_dofs);
      cell_shadow->get_dof_indices(shadow_dofs);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        global_vec[global_dofs[i]] += wall_vec[shadow_dofs[i]];
    }
  }
}

/* ============================================================================
  setup_wall_distance

  Implements eq. (6.13): y_B = shortest distance from node B to any wall node.

  Additional work compared to the original code:
    • Gathers wall DoF coordinates from all MPI ranks (fixes parallel gap).
    • Deduplicates wall DoFs that appear on partition boundaries.
    • Builds node_to_wall_node: near-wall DoF → nearest wall DoF global index.
 ============================================================================
*/

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::setup_wall_distance(
  dealii::Mapping<dim> const &                           mapping,
  unsigned int                                           layers)
{
  (void)layers;
  wall_distance.zero_out_ghost_values();
  wall_distance = 1e10; 

  // ====================================================================
  // PART 1: Compute Exact DG Wall Distance (for y_dist_vec)
  // ====================================================================
  std::map<dealii::types::global_dof_index, dealii::Point<dim>> dg_support_points;
  
  // Use the DG Scalar Collection
  auto const & fe_dg_real = fe_en_collection_scalar[1]; 
  dealii::Quadrature<dim> dg_support_quad(fe_dg_real.get_unit_support_points());
  dealii::FEValues<dim> fe_values_dg(mapping, fe_dg_real, dg_support_quad, dealii::update_quadrature_points);

  std::vector<dealii::types::global_dof_index> local_dg_indices(fe_dg_real.dofs_per_cell);

  // Loop over the DG Handler to map coordinates
  for (auto const & cell : dof_handler_en_scalar.active_cell_iterators()) 
  {
    if (cell->is_locally_owned() && cell->active_fe_index() == 1) 
    {
      fe_values_dg.reinit(cell);
      cell->get_dof_indices(local_dg_indices);
      for (unsigned int i = 0; i < fe_dg_real.dofs_per_cell; ++i) 
      {
        dg_support_points[local_dg_indices[i]] = fe_values_dg.quadrature_point(i); 
      }
    }
  }

  std::vector<std::pair<dealii::Point<dim>, dealii::types::global_dof_index>> local_dg_wall_points;
  auto const & fe_dg = *fe_en_scalar; // DG Element

  for(auto const & cell : dof_handler_en_scalar.active_cell_iterators()) 
  {
    if(cell->is_locally_owned() && cell->at_boundary()) 
    {
      for(auto face : cell->face_indices()) 
      {
        if(cell->face(face)->at_boundary() and not cell->has_periodic_neighbor(face) and boundary_descriptor->get_boundary_type(cell->face(face)->boundary_id()) == IncNS::BoundaryTypeU::WallEnrichment) 
        {
          cell->get_dof_indices(local_dg_indices);
          for(unsigned int i = 0; i < fe_dg_real.dofs_per_cell; ++i) 
          {
            if(fe_dg.has_support_on_face(i, face)) 
            {
              auto global_dof = local_dg_indices[i];
              if(wall_distance.locally_owned_elements().is_element(global_dof)) 
              {
                local_dg_wall_points.push_back({dg_support_points[global_dof], global_dof});
              }
            }
          }
        }
      }
    }
  }

  std::vector<dealii::Point<dim>> local_dg_wall_coordinates;
  local_dg_wall_coordinates.reserve(local_dg_wall_points.size());
  for(auto const & wall_point : local_dg_wall_points)
  {
    local_dg_wall_coordinates.push_back(wall_point.first);
  }

  std::vector<std::vector<dealii::Point<dim>>> gathered_dg_wall_coordinates = dealii::Utilities::MPI::all_gather(dof_handler_en_scalar.get_triangulation().get_communicator(), local_dg_wall_coordinates);

  std::vector<dealii::Point<dim>> global_dg_wall_coordinates;
  for(auto const & rank_points : gathered_dg_wall_coordinates)
  {
    global_dg_wall_coordinates.insert(global_dg_wall_coordinates.end(), rank_points.begin(), rank_points.end());
  }

  for(auto const & cell : dof_handler_en_scalar.active_cell_iterators())
  {
    if(cell->is_locally_owned() && is_cell_enriched[cell->global_active_cell_index()])
    {
      cell->get_dof_indices(local_dg_indices);
      for(unsigned int i = 0; i < fe_dg_real.dofs_per_cell; ++i) 
      {
        auto global_dof = local_dg_indices[i];
        if (!wall_distance.locally_owned_elements().is_element(global_dof)) continue;

        auto dof_point  = dg_support_points[global_dof];
        Number min_dist = std::numeric_limits<Number>::max();

        for(auto const & wall_point : global_dg_wall_coordinates) 
        {
          Number distance = dof_point.distance(wall_point);
          if(distance < min_dist)
          {
            min_dist = distance;
          }
        }

        if (min_dist < wall_distance[global_dof]) 
        {
          wall_distance[global_dof] = min_dist;
        }
      }
    }
  }
  wall_distance.update_ghost_values();


  // ====================================================================
  // PART 2: Build CG node_to_wall_node map (for friction_velocity copy)
  // ====================================================================
  std::map<dealii::types::global_dof_index, dealii::Point<dim>> cg_support_points;
  
  // Safe to use standard DoFTools here because CG uses the Mirror Dummy (No FE_Nothing!)
  dealii::DoFTools::map_dofs_to_support_points(mapping, dof_handler_cg_scalar, cg_support_points);

  std::vector<std::pair<dealii::Point<dim>, dealii::types::global_dof_index>> local_cg_wall_points;
  auto const & fe_cg = *fe_cg_scalar;

  for(auto const & cell : dof_handler_cg_scalar.active_cell_iterators()) 
  {
    if(cell->is_locally_owned() && cell->at_boundary()) 
    {
      for(auto face : cell->face_indices()) 
      {
        if(cell->face(face)->at_boundary() and not cell->has_periodic_neighbor(face) and boundary_descriptor->get_boundary_type(cell->face(face)->boundary_id()) == IncNS::BoundaryTypeU::WallEnrichment) 
        {
          std::vector<dealii::types::global_dof_index> cg_dofs(fe_cg.n_dofs_per_cell());
          cell->get_dof_indices(cg_dofs);
          for(unsigned int i = 0; i < fe_cg.n_dofs_per_cell(); ++i) 
          {
            if(fe_cg.has_support_on_face(i, face)) 
            {
              auto global_dof = cg_dofs[i];
              if(friction_velocity.locally_owned_elements().is_element(global_dof)) 
              {
                local_cg_wall_points.push_back({cg_support_points[global_dof], global_dof});
              }
            }
          }
        }
      }
    }
  }

  std::vector<std::vector<std::pair<dealii::Point<dim>, dealii::types::global_dof_index>>> gathered_cg_wall_points = dealii::Utilities::MPI::all_gather(dof_handler_cg_scalar.get_triangulation().get_communicator(), local_cg_wall_points);

  std::vector<std::pair<dealii::Point<dim>, dealii::types::global_dof_index>> global_cg_wall_points;

  for(auto const & rank_points : gathered_cg_wall_points)
  {
    global_cg_wall_points.insert(global_cg_wall_points.end(), rank_points.begin(), rank_points.end());
  }

  node_to_wall_node.clear();

  for(auto const & cell : dof_handler_cg_scalar.active_cell_iterators())
  {
    if(cell->is_locally_owned() || cell->is_ghost() && cell->active_fe_index() == 1)
    {
      std::vector<dealii::types::global_dof_index> cg_dofs(fe_cg.n_dofs_per_cell());
      cell->get_dof_indices(cg_dofs);

      for(unsigned int i = 0; i < fe_cg.n_dofs_per_cell(); ++i) 
      {
        auto global_dof = cg_dofs[i];
        if (!friction_velocity.locally_owned_elements().is_element(global_dof)) continue;

        auto dof_point  = cg_support_points[global_dof];
        Number min_dist = std::numeric_limits<Number>::max();
        dealii::types::global_dof_index nearest_wall_dof = dealii::numbers::invalid_dof_index;

        for(auto const & wall_point : global_cg_wall_points) 
        {
          Number distance = dof_point.distance(wall_point.first);
          if(distance < min_dist) 
          {
            min_dist = distance;
            nearest_wall_dof = wall_point.second;
          }
        }
        
        // Save the node mapping for evaluate_friction_velocity
        node_to_wall_node[global_dof] = nearest_wall_dof; 
      }
    }
  }
  setup_corner_nodes();

  update_enrichment_function();
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::project_velocity_to_wall()
{
  VectorType rhs, lumped_mass;
  matrix_free_en->initialize_dof_vector(rhs,         dof_index_cg_vector);
  matrix_free_en->initialize_dof_vector(lumped_mass, dof_index_cg_vector);

  rhs = 0.0;
  lumped_mass = 0.0;

  matrix_free_en->cell_loop(&FunctionEnrichment::loop_project_velocity_to_wall,
                         this, rhs, shadow_velocity, false);

  matrix_free_en->cell_loop(&FunctionEnrichment::loop_lumped_mass,
                         this, lumped_mass, shadow_velocity, false);

  rhs.compress(dealii::VectorOperation::add);
  lumped_mass.compress(dealii::VectorOperation::add);

  for (unsigned int i = 0; i < rhs.locally_owned_size(); ++i)
  {
    if (lumped_mass.local_element(i) > 1e-14) 
    {
      wall_velocity.local_element(i) = rhs.local_element(i) / lumped_mass.local_element(i);
    }
    else 
  {
      // Freestream nodes were safely skipped, so we just zero them out
      wall_velocity.local_element(i) = 0.0;
    }
  }

  wall_velocity.update_ghost_values();
  constraint_cg_vector.distribute(wall_velocity);
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::loop_lumped_mass(
  dealii::MatrixFree<dim, Number> const &       data,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  dealii::FEEvaluation<dim, -1, 0, dim, Number> dg_eval(data, dof_index_shadow_vector, quad_index, 0, active_fe_index);

  dealii::FEEvaluation<dim, -1, 0, dim, Number> cg_eval(data, dof_index_cg_vector, quad_index, 0, active_fe_index);

  dealii::Tensor<1, dim, scalar> one_vector;
  for (unsigned int d = 0; d < dim; ++d)
  {
    one_vector[d] = dealii::make_vectorized_array<Number>(1.0);
  }

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    if (data.get_cell_iterator(cell, 0)->active_fe_index() != 1)
    {
      continue;
    }
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
  dealii::FEEvaluation<dim, -1, 0, dim, Number> dg_eval(data, dof_index_shadow_vector, quad_index, 0, active_fe_index);

  dealii::FEEvaluation<dim, -1, 0, dim, Number> cg_eval(data, dof_index_cg_vector, quad_index, 0, active_fe_index);

  dealii::Tensor<1, dim, scalar> one_vector;
  for (unsigned int d = 0; d < dim; ++d)
  {
    one_vector[d] = dealii::make_vectorized_array<Number>(1.0);
  }

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    if (data.get_cell_iterator(cell, 0)->active_fe_index() != 1)
    {
      continue;
    }
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
  matrix_free_en->loop(&This::empty_loop,
                    &This::empty_loop,
                    &This::local_integrate_wall_traction, 
                    this,
                    dst_vectors,
                    shadow_velocity,
                    false);
}

template<int dim, typename Number>
void FunctionEnrichment<dim, Number>::local_integrate_wall_traction(
  dealii::MatrixFree<dim, Number> const & matrix_free_data,
  TractionVectors &                       dst,
  VectorType const &                      src,
  std::pair<unsigned int, unsigned int> const & face_range) const
{
  if(matrix_free_data.get_face_active_fe_index(face_range, true) != 1)
    return;
  // dealii::FEFaceEvaluation<dim, -1, 0, dim, Number> u_wall(
  //     matrix_free_data, true, dof_index_shadow_vector, quad_index, 0, active_fe_index);
  FEFaceEnrichedEvaluation<dim, Number, dealii::FEFaceEvaluation<dim, -1, 0, dim, Number>> u_wall(
    matrix_free_data, *this, true, active_fe_index);
  
  dealii::FEFaceEvaluation<dim, -1, 0, 1, Number> wss(
      matrix_free_data, true, dof_index_cg_scalar, quad_index, 0, active_fe_index);

  scalar nu = dealii::make_vectorized_array<Number>(kinematic_viscosity);

  for (unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    if (boundary_descriptor->get_boundary_type(matrix_free_data.get_boundary_id(face)) != IncNS::BoundaryTypeU::WallEnrichment)
    {
      continue;
    }
    // Extract smoothed gradients directly from the Continuous P1 field!
    u_wall.reinit(face);
    u_wall.gather_evaluate(src, enrichment_velocity,
                           dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    std::vector<std::array<scalar, dim>> traction(u_wall.n_q_points); 
    for (unsigned int q = 0; q < u_wall.n_q_points; ++q)
    {
      auto grad_u = u_wall.get_gradient(q);
      auto n = u_wall.get_normal_vector(q);
      auto tau_w = nu * (grad_u * n);

      for (unsigned int d = 0; d < dim; ++d)
      {
        traction[q][d] = tau_w[d];
      }
    }

    for (unsigned int d = 0; d < dim; ++d)
    {
      wss.reinit(face);
      for (unsigned int q = 0; q < wss.n_q_points; ++q)
      {
        wss.submit_value(traction[q][d], q);
      }
      wss.integrate(dealii::EvaluationFlags::values);
      wss.distribute_local_to_global(*dst.num[d]);
    }

    wss.reinit(face);
    for (unsigned int q = 0; q < wss.n_q_points; ++q)
    {
      wss.submit_value(dealii::make_vectorized_array<Number>(1.0), q);
    }
    wss.integrate(dealii::EvaluationFlags::values);
    wss.distribute_local_to_global(*dst.den);
  }
}

template<int dim, typename Number>
void FunctionEnrichment<dim, Number>::evaluate_friction_velocity(VectorType const & global_velocity)
{
  copy_global_dg_to_wall_layout(global_velocity, shadow_velocity);
  shadow_velocity.update_ghost_values();
  // Project DG velocity onto the continuous P1 wall space (Spatial Coarsening)
  project_velocity_to_wall();
  enrichment_velocity.update_ghost_values();

  // Setup temporary vectors for the boundary traction integration
  std::array<VectorType, dim> traction_num;
  for(unsigned int d = 0; d < dim; ++d)
  {
    matrix_free_en->initialize_dof_vector(traction_num[d], dof_index_cg_scalar);
    traction_num[d] = 0.0;
  }

  VectorType traction_den;
  matrix_free_en->initialize_dof_vector(traction_den, dof_index_cg_scalar);
  traction_den = 0.0;

  // Run the boundary loop to compute tau_w
  integrate_wall_traction(traction_num, traction_den);

  finish_friction_velocity(traction_num, traction_den);
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::finish_friction_velocity(std::array<VectorType, dim> & traction_num,
                                                          VectorType &                  traction_den)
{
  // the current u_tau defines the space of the current solution; keep it for the projection
  friction_velocity_old = friction_velocity;
  friction_velocity_old.update_ghost_values();

  for(unsigned int d = 0; d < dim; ++d)
  {
    traction_num[d].compress(dealii::VectorOperation::add);
    constraint_cg_scalar.distribute(traction_num[d]);
  }
  traction_den.compress(dealii::VectorOperation::add);
  constraint_cg_scalar.distribute(traction_den);

  MPI_Comm const comm = dof_handler_cg_scalar.get_triangulation().get_communicator();

  // (1) Eq. (13) at the wall nodes: tau_w,B = |num_B| / A_B   (rho = 1)
  std::map<dealii::types::global_dof_index, Number> local_tau;
  double local_tau_times_area = 0.0;
  double local_area           = 0.0;

  for(auto const node : wall_shear_stress.locally_owned_elements())
  {
    double const area = traction_den[node];
    if(area <= 1e-14) // not a wall node
      continue;

    dealii::Tensor<1, dim> traction;
    for(unsigned int d = 0; d < dim; ++d)
      traction[d] = traction_num[d][node] / area;

    double const tau = traction.norm();
    local_tau[node]  = tau;

    // periodic copies have the value of their master: count every node once in the mean
    if(not constraint_cg_scalar.is_constrained(node))
    {
      local_tau_times_area += tau * area;
      local_area += area;
    }
  }

  // (2) minimum: tau_w,B >= fraction * area-weighted mean over all walls
  double const area_total = dealii::Utilities::MPI::sum(local_area, comm);
  double const tau_mean =
    (area_total > 0.0) ? dealii::Utilities::MPI::sum(local_tau_times_area, comm) / area_total : 0.0;
  Number const tau_min = minimum_wall_shear_stress_fraction * tau_mean;

  for(auto & entry : local_tau)
  entry.second = std::max(entry.second, tau_min);

  // (3) wall values on all ranks (corner and off-wall nodes may refer to remote wall nodes)
  std::vector<std::pair<dealii::types::global_dof_index, Number>> const local_list(local_tau.begin(),
                                                                                   local_tau.end());
  auto const gathered = dealii::Utilities::MPI::all_gather(comm, local_list);      

  std::map<dealii::types::global_dof_index, Number> wall_tau;
  for(auto const & rank_values : gathered)
  for(auto const & entry : rank_values)
  wall_tau[entry.first] = entry.second;

  // corner nodes (wall + inflow/outflow): value of the neighbouring wall node
  for(auto const & entry : corner_to_wall_node)
  {
    auto const it = wall_tau.find(entry.second);
    if(it != wall_tau.end())
      wall_tau[entry.first] = it->second;
  }

  // (4) tau_w on wall nodes, off-wall nodes take the value of their nearest wall node
  wall_shear_stress.zero_out_ghost_values();
  wall_shear_stress = 0.0;

  for(auto const node : wall_shear_stress.locally_owned_elements())
  {
    auto it = wall_tau.find(node);
    if(it == wall_tau.end())
    {
      auto const nearest = node_to_wall_node.find(node);
      if(nearest != node_to_wall_node.end())
        it = wall_tau.find(nearest->second);
    }
    if(it != wall_tau.end())
      wall_shear_stress[node] = it->second;
  }

  constraint_cg_scalar.distribute(wall_shear_stress);
  wall_shear_stress.update_ghost_values();

  // (5) nodal u_tau = sqrt(tau_w / rho), same layout as wall_shear_stress
  friction_velocity.zero_out_ghost_values();
  for(unsigned int i = 0; i < friction_velocity.locally_owned_size(); ++i)
    friction_velocity.local_element(i) = std::sqrt(wall_shear_stress.local_element(i));
  friction_velocity.update_ghost_values();

  update_enrichment_function();
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::integrate_wall_residual(std::array<VectorType, dim> & traction_num,
                                                         VectorType const &            u_tilde,
                                                         VectorType const &            u_tilde_hat,
                                                         double const                  mass_factor) const
{
  auto const & mf = *matrix_free_en;

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number>     CellIntU;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, dim, Number> FaceIntU;
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number>       CellIntS;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number>   FaceIntS;

  scalar const nu = dealii::make_vectorized_array<Number>(kinematic_viscosity);
  scalar const s  = dealii::make_vectorized_array<Number>(mass_factor);

  // ---- cell terms: s (N, u_h - u_hat_h) + nu (grad N, grad u_h) -------------
  {
    FEEnrichedEvaluation<dim, Number, CellIntU> u(mf, *this), u_hat(mf, *this);
    CellIntS                                    N(mf, dof_index_cg_scalar, quad_index, 0, active_fe_index);

    for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
    {
      if(mf.get_cell_iterator(cell, 0)->active_fe_index() != 1)
        continue;

      u.reinit(cell);
      u.gather_evaluate(shadow_velocity, u_tilde,
                        dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
      u_hat.reinit(cell);
      u_hat.gather_evaluate(shadow_velocity_residual, u_tilde_hat, dealii::EvaluationFlags::values);

      for(unsigned int d = 0; d < dim; ++d)
      {
        N.reinit(cell);
        for(unsigned int q = 0; q < N.n_q_points; ++q)
        {
          auto const value    = u.get_value(q);
          auto const value_h  = u_hat.get_value(q);
          auto const gradient = u.get_gradient(q);

          N.submit_value(s * (value[d] - value_h[d]), q);

          dealii::Tensor<1, dim, scalar> row;
          for(unsigned int e = 0; e < dim; ++e)
            row[e] = nu * gradient[d][e];
          N.submit_gradient(row, q);
        }
        N.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
        N.distribute_local_to_global(traction_num[d]);
      }
    }
  }

  // ---- interior faces: - ({nu grad N} . n, [u_h]) ---------------------------
  //      (N continuous -> consistency and penalty terms vanish)
  auto const values_on_side = [&](bool const interior, unsigned int const fe_index, unsigned int const face,
                                  std::vector<dealii::Tensor<1, dim, scalar>> & vals) {
    if(fe_index == 1)
    {
      FEFaceEnrichedEvaluation<dim, Number, FaceIntU> phi(mf, *this, interior, 1);
      phi.reinit(face);
      phi.gather_evaluate(shadow_velocity, u_tilde, dealii::EvaluationFlags::values);
      vals.resize(phi.n_q_points);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        vals[q] = phi.get_value(q);
    }
    else // halo: U_bar only
    {
      FaceIntU phi(mf, interior, dof_index_shadow_vector, quad_index, 0, fe_index);
      phi.reinit(face);
      phi.gather_evaluate(shadow_velocity, dealii::EvaluationFlags::values);
      vals.resize(phi.n_q_points);
      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        vals[q] = phi.get_value(q);
    }
  };

  std::vector<dealii::Tensor<1, dim, scalar>> u_m, u_p;

  for(unsigned int face = 0; face < mf.n_inner_face_batches(); ++face)
  {
    auto const         range = std::make_pair(face, face + 1);
    unsigned int const fe_m  = mf.get_face_active_fe_index(range, true);
    unsigned int const fe_p  = mf.get_face_active_fe_index(range, false);

    if(fe_m != 1 and fe_p != 1)
      continue;
    if(fe_m == 0 or fe_p == 0)
      continue; // no shadow velocity there (cannot happen with a halo layer)

    values_on_side(true, fe_m, face, u_m);
    values_on_side(false, fe_p, face, u_p);

    FaceIntS Nm(mf, true, dof_index_cg_scalar, quad_index, 0, fe_m);
    FaceIntS Np(mf, false, dof_index_cg_scalar, quad_index, 0, fe_p);

    for(unsigned int d = 0; d < dim; ++d)
    {
      Nm.reinit(face);
      Np.reinit(face);
      for(unsigned int q = 0; q < Nm.n_q_points; ++q)
      {
        scalar const flux = -0.5 * nu * (u_m[q][d] - u_p[q][d]);
        Nm.submit_normal_derivative(flux, q); // normals of both sides refer to n^-
        Np.submit_normal_derivative(flux, q);
      }
      Nm.integrate(dealii::EvaluationFlags::gradients);
      Nm.distribute_local_to_global(traction_num[d]);
      Np.integrate(dealii::EvaluationFlags::gradients);
      Np.distribute_local_to_global(traction_num[d]);
    }
  }
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::evaluate_friction_velocity_from_residual(VectorType const & u_bar,
                                                                          VectorType const & u_tilde,
                                                                          VectorType const & u_bar_hat,
                                                                          VectorType const & u_tilde_hat,
                                                                          double const       mass_factor)
{
  // U_bar and U_bar_hat in shadow layout (shadow_velocity_residual is used as scratch)
  copy_global_dg_to_wall_layout(u_bar, shadow_velocity);
  shadow_velocity.update_ghost_values();
  copy_global_dg_to_wall_layout(u_bar_hat, shadow_velocity_residual);
  shadow_velocity_residual.update_ghost_values();
  u_tilde.update_ghost_values();
  u_tilde_hat.update_ghost_values();
  enrichment_velocity.update_ghost_values();

  std::array<VectorType, dim> traction_num;
  for(unsigned int d = 0; d < dim; ++d)
  {
    matrix_free_en->initialize_dof_vector(traction_num[d], dof_index_cg_scalar);
    traction_num[d] = 0.0;
  }
  VectorType traction_den;
  matrix_free_en->initialize_dof_vector(traction_den, dof_index_cg_scalar);
  traction_den = 0.0;

  // nodal wall areas A_B (traction_num from this call is discarded)
  integrate_wall_traction(traction_num, traction_den);
  for(unsigned int d = 0; d < dim; ++d)
    traction_num[d] = 0.0;

  // residual-based wall force
  integrate_wall_residual(traction_num, u_tilde, u_tilde_hat, mass_factor);

  finish_friction_velocity(traction_num, traction_den);
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
FunctionEnrichment<dim, Number>::get_dof_handler_en_scalar() const
{
  return dof_handler_en_scalar;
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
dealii::DoFHandler<dim> const &
FunctionEnrichment<dim, Number>::get_dof_handler_shadow_vector() const
{
  return dof_handler_shadow_vector;
}

template<int dim, typename Number>
typename FunctionEnrichment<dim, Number>::scalar
FunctionEnrichment<dim, Number>::get_value(
  scalar const u_tau,
  scalar const y) const
{
  scalar const nu = dealii::make_vectorized_array<Number>(kinematic_viscosity);
  scalar const kappa = dealii::make_vectorized_array<Number>(SpaldingLaw().kappa);

  // u_tau only enters through y+; temporary floor until the 2% tau_w floor is in place
  scalar const u_tau_safe = std::max(u_tau, dealii::make_vectorized_array<Number>(1e-5));

  scalar y_plus = (y * u_tau_safe) / nu;
  scalar u_plus = wall_law_eval.get_value(y_plus);

  return kappa * u_plus;
}

template<int dim, typename Number>
typename FunctionEnrichment<dim, Number>::scalar
FunctionEnrichment<dim, Number>::get_gradient(
  scalar const u_tau,
  scalar const y) const
{
  scalar const nu = dealii::make_vectorized_array<Number>(kinematic_viscosity);
  scalar const kappa = dealii::make_vectorized_array<Number>(SpaldingLaw().kappa);

  // same floor as in get_value, so that psi and dpsi/dy belong to the same function
  scalar const u_tau_safe = std::max(u_tau, dealii::make_vectorized_array<Number>(1e-5));

  scalar const y_plus = (y * u_tau_safe) / nu;
  
  scalar du_plus_dy_plus = wall_law_eval.get_gradient(y_plus);
  
  return du_plus_dy_plus * (kappa * u_tau_safe / nu);
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
FunctionEnrichment<dim, Number>::get_dof_index_en_scalar() const
{
  return dof_index_en_scalar;
}

template<int dim, typename Number>
unsigned int 
FunctionEnrichment<dim, Number>::get_quad_index() const
{
  return quad_index;
}

template<int dim, typename Number>
std::shared_ptr<dealii::MatrixFree<dim, Number>>
FunctionEnrichment<dim, Number>::get_matrix_free() const
{
  return matrix_free_en;
}

template<int dim, typename Number>
unsigned int
FunctionEnrichment<dim, Number>::get_dof_index_shadow_vector() const
{
  return dof_index_shadow_vector;
}

template<int dim, typename Number>
unsigned int
FunctionEnrichment<dim, Number>::get_active_fe_index() const
{
  return active_fe_index;
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::set_global_pressure_dof_handler(
  dealii::DoFHandler<dim> const & dof_handler_p_in)
{
  global_dof_handler_pressure = &dof_handler_p_in;
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::copy_global_pressure_to_wall_layout(VectorType const & global_p,
                                                                     VectorType &       wall_p) const
{
  wall_p = 0.0;

  unsigned int const dofs_per_cell = dof_handler_shadow_pressure.get_fe(1).n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> global_dofs(dofs_per_cell);
  std::vector<dealii::types::global_dof_index> shadow_dofs(dofs_per_cell);

  auto cell_g = global_dof_handler_pressure->begin_active();
  auto cell_s = dof_handler_shadow_pressure.begin_active();

  for(; cell_g != global_dof_handler_pressure->end(); ++cell_g, ++cell_s)
  {
    // enriched (1) and halo (2) cells carry the shadow pressure
    if(cell_s->is_locally_owned() && cell_s->active_fe_index() >= 1)
    {
      cell_g->get_dof_indices(global_dofs);
      cell_s->get_dof_indices(shadow_dofs);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        wall_p[shadow_dofs[i]] = global_p[global_dofs[i]];
    }
  }
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::add_wall_layout_to_global_pressure(VectorType const & wall_p,
                                                                    VectorType &       global_p) const
{
  unsigned int const dofs_per_cell = dof_handler_shadow_pressure.get_fe(1).n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> global_dofs(dofs_per_cell);
  std::vector<dealii::types::global_dof_index> shadow_dofs(dofs_per_cell);

  auto cell_g = global_dof_handler_pressure->begin_active();
  auto cell_s = dof_handler_shadow_pressure.begin_active();

  for(; cell_g != global_dof_handler_pressure->end(); ++cell_g, ++cell_s)
  {
    if(cell_s->is_locally_owned() && cell_s->active_fe_index() >= 1)
    {
      cell_g->get_dof_indices(global_dofs);
      cell_s->get_dof_indices(shadow_dofs);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        global_p[global_dofs[i]] += wall_p[shadow_dofs[i]];
    }
  }
}

/*
 * Right-hand side of the L2 projection onto the new enrichment space (Krank et al. 2018,
 * Section 3.2). The solution u_h^{old} = \bar{u}_h + \psi^{old} \tilde{u}_h was computed with the
 * previous wall shear stress; its projection onto the space with \psi^{new} solves
 * M^{new} U^{new} = b with
 *   b_{\bar{v}}   = (\bar{v}_h, u_h^{old})_K,
 *   b_{\tilde{v}} = (\psi^{new} \tilde{v}_h, u_h^{old})_K.
 * Enriched cells only (shadow / enrichment layout); \psi^{old}, \psi^{new} from the cache.
 */
template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::apply_mass_old_to_new(VectorType &       dst_bar_shadow,
                                                       VectorType &       dst_tilde,
                                                       VectorType const & u_bar_shadow,
                                                       VectorType const & u_tilde) const
{
  auto const & mf = *matrix_free_en;

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number> CellIntegratorU;

  CellIntegratorU phi_bar(mf, dof_index_shadow_vector, quad_index, 0, active_fe_index);
  CellIntegratorU phi_tilde(mf, dof_index_en, quad_index, 0, active_fe_index);

  dst_bar_shadow = 0.0;
  dst_tilde      = 0.0;

  for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
  {
    if(mf.get_cell_iterator(cell, 0)->active_fe_index() != active_fe_index)
      continue;

    phi_bar.reinit(cell);
    phi_tilde.reinit(cell);

    phi_bar.gather_evaluate(u_bar_shadow, dealii::EvaluationFlags::values);
    phi_tilde.gather_evaluate(u_tilde, dealii::EvaluationFlags::values);

    for(unsigned int q = 0; q < phi_bar.n_q_points; ++q)
    {
      // u_h^{old} = \bar{u}_h + \psi^{old} \tilde{u}_h
      vector const u_old = phi_bar.get_value(q) + get_psi_cell_old(cell, q) * phi_tilde.get_value(q);

      // tested with \bar{v}_h and \psi^{new} \tilde{v}_h
      phi_bar.submit_value(u_old, q);
      phi_tilde.submit_value(get_psi_cell(cell, q) * u_old, q);
    }

    phi_bar.integrate(dealii::EvaluationFlags::values);
    phi_bar.distribute_local_to_global(dst_bar_shadow);
    phi_tilde.integrate(dealii::EvaluationFlags::values);
    phi_tilde.distribute_local_to_global(dst_tilde);
  }

  dst_bar_shadow.compress(dealii::VectorOperation::add);
  dst_tilde.compress(dealii::VectorOperation::add);
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::setup_corner_nodes()
{
  corner_to_wall_node.clear();

  std::vector<dealii::types::global_dof_index> wall_dofs, other_dofs;

  for(auto const & cell : dof_handler_cg_scalar.active_cell_iterators())
  {
    if(not(cell->is_locally_owned() or cell->is_ghost()) or cell->active_fe_index() != 1)
      continue;

    for(auto const f : cell->face_indices())
    {
      if(not cell->at_boundary(f) or cell->has_periodic_neighbor(f) or
         boundary_descriptor->get_boundary_type(cell->face(f)->boundary_id()) !=
           IncNS::BoundaryTypeU::WallEnrichment)
        continue;

      wall_dofs.resize(cell->get_fe().n_dofs_per_face());
      cell->face(f)->get_dof_indices(wall_dofs, cell->active_fe_index());

      // wall-face DoFs that also belong to a non-wall boundary face of the same cell
      for(auto const g : cell->face_indices())
      {
        if(g == f or not cell->at_boundary(g) or cell->has_periodic_neighbor(g) or
           boundary_descriptor->get_boundary_type(cell->face(g)->boundary_id()) ==
             IncNS::BoundaryTypeU::WallEnrichment)
          continue;

        other_dofs.resize(cell->get_fe().n_dofs_per_face());
        cell->face(g)->get_dof_indices(other_dofs, cell->active_fe_index());

        for(auto const corner : wall_dofs)
        {
          if(std::find(other_dofs.begin(), other_dofs.end(), corner) == other_dofs.end())
            continue;
          // partner: a wall-face DoF of the same face that is not on the other boundary
          for(auto const partner : wall_dofs)
            if(std::find(other_dofs.begin(), other_dofs.end(), partner) == other_dofs.end())
            {
              corner_to_wall_node[corner] = partner;
              break;
            }
        }
      }
    }
  }
}

template<int dim, typename Number>
dealii::DoFHandler<dim> const &
FunctionEnrichment<dim, Number>::get_dof_handler_shadow_pressure() const
{
  return dof_handler_shadow_pressure;
}

template<int dim, typename Number>
unsigned int
FunctionEnrichment<dim, Number>::get_dof_index_shadow_pressure() const
{
  return dof_index_shadow_pressure;
}

template<int dim, typename Number>
double
FunctionEnrichment<dim, Number>::get_max_y_plus() const
{
  // linfty_norm() is the global (MPI) maximum
  double const y_max     = wall_distance.linfty_norm();
  double const u_tau_max = friction_velocity.linfty_norm();
  return y_max * u_tau_max / kinematic_viscosity;
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::set_minimum_wall_shear_stress_fraction(double const fraction)
{
  minimum_wall_shear_stress_fraction = fraction;
}

/*
 * DoFTools::make_periodicity_constraints(dof_handler, b_id_1, b_id_2, direction, ...) does not
 * work for an hp DoFHandler on a parallel::distributed::Triangulation: while recursing from the
 * coarse periodic face pairs to the active faces, it asks faces of artificial cells for their FE,
 * but those faces have no active FE index. Same recursion here; active face pairs touching an
 * artificial cell are skipped (their DoFs are not locally relevant on this process).
 */
template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::make_periodicity_constraints_hp(
  dealii::DoFHandler<dim> const & dof_handler,
  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::Triangulation<dim>::cell_iterator>> const & periodic_faces,
  dealii::AffineConstraints<Number> &                            constraints) const
{
  for(auto const & face_pair : periodic_faces)
  {
    // triangulation iterators -> DoFHandler iterators (same cells)
    typename dealii::DoFHandler<dim>::cell_iterator const cell_1(&dof_handler.get_triangulation(),
                                                                 face_pair.cell[0]->level(),
                                                                 face_pair.cell[0]->index(),
                                                                 &dof_handler);
    typename dealii::DoFHandler<dim>::cell_iterator const cell_2(&dof_handler.get_triangulation(),
                                                                 face_pair.cell[1]->level(),
                                                                 face_pair.cell[1]->index(),
                                                                 &dof_handler);

    make_periodicity_constraints_hp(cell_1->face(face_pair.face_idx[0]),
                                    cell_2->face(face_pair.face_idx[1]),
                                    face_pair.orientation,
                                    constraints);
  }
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::make_periodicity_constraints_hp(
  typename dealii::DoFHandler<dim>::face_iterator const & face_1,
  typename dealii::DoFHandler<dim>::face_iterator const & face_2,
  unsigned char const                                     orientation,
  dealii::AffineConstraints<Number> &                     constraints) const
{
  auto const is_artificial_active_face = [](auto const & face) {
    return not face->has_children() and face->n_active_fe_indices() == 0;
  };

  if(is_artificial_active_face(face_1) or is_artificial_active_face(face_2))
    return;

  if(face_1->has_children() and face_2->has_children())
  {
    // pairing of the children as in deal.II (hypercube faces)
    unsigned int const face_no = (dim == 2) ? 2 : 4;
    for(unsigned int i = 0; i < dealii::GeometryInfo<dim>::max_children_per_face; ++i)
    {
      unsigned int const j =
        dealii::ReferenceCells::get_hypercube<dim>().child_cell_on_face(face_no, i, orientation);
      make_periodicity_constraints_hp(face_1->child(i), face_2->child(j), orientation, constraints);
    }
  }
  else
  {
    AssertThrow(not face_1->has_children() and not face_2->has_children(),
                dealii::ExcMessage("Periodic faces on different refinement levels are not supported by the wall enrichment."));

    dealii::DoFTools::make_periodicity_constraints(face_1, face_2, constraints, dealii::ComponentMask(), orientation);
  }
}

template<int dim, typename Number>
std::pair<typename FunctionEnrichment<dim, Number>::scalar,
          typename FunctionEnrichment<dim, Number>::vector>
FunctionEnrichment<dim, Number>::evaluate_enrichment_function(scalar const & y,
                                                              vector const & grad_y,
                                                              scalar const & tau_w,
                                                              vector const & grad_tau_w) const
{
  scalar const nu      = dealii::make_vectorized_array<Number>(kinematic_viscosity);
  scalar const kappa   = dealii::make_vectorized_array<Number>(SpaldingLaw().kappa);
  scalar const tau_min = dealii::make_vectorized_array<Number>(tau_w_absolute_minimum);
  scalar const zero    = dealii::make_vectorized_array<Number>(0.0);

  scalar const tau   = std::max(tau_w, tau_min);
  scalar const u_tau = std::sqrt(tau);

  // grad tau_w only where tau_w,h is not replaced by the absolute minimum
  vector grad_tau;
  for(unsigned int d = 0; d < dim; ++d)
    grad_tau[d] = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
      tau_w, tau_min, grad_tau_w[d], zero);

  // y+ = y u_tau / nu,   grad y+ = (u_tau / nu) grad y + y / (2 nu u_tau) grad tau_w
  scalar const y_plus      = y * u_tau / nu;
  vector const grad_y_plus = (u_tau / nu) * grad_y + (y / (Number(2.0) * nu * u_tau)) * grad_tau;

  // Spalding's law: u+ and du+/dy+
  std::pair<scalar, scalar> const u_plus = wall_law_eval.get_value_and_gradient(y_plus);

  // Eq. (12): psi = kappa u+,   grad psi = kappa du+/dy+ grad y+
  return std::make_pair(kappa * u_plus.first, (kappa * u_plus.second) * grad_y_plus);
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::update_enrichment_function()
{
  auto const & mf = *matrix_free_en;

  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number>     CellIntegratorScalar;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number> FaceIntegratorScalar;

  // the exterior side of a face reads ghost cells
  wall_distance.update_ghost_values();
  wall_shear_stress.update_ghost_values();

  dealii::EvaluationFlags::EvaluationFlags const flags = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;

  // ---- cells -------------------------------------------------------------------------
  {
    CellIntegratorScalar y_eval(mf, dof_index_en_scalar, quad_index, 0, active_fe_index);
    CellIntegratorScalar tau_eval(mf, dof_index_cg_scalar, quad_index, 0, active_fe_index);

    // locally owned and ghost cell batches: the interior penalty parameter of faces between
    // processes needs psi on the ghost side as well
    unsigned int const n_batches = mf.n_cell_batches() + mf.n_ghost_cell_batches();

    // psi of the space the current solution is expressed in, kept for the re-projection
    // \bar{u} + \psi^{old} \tilde{u} -> space with \psi^{new} (apply_mass_old_to_new)
    psi_cell_old = psi_cell;

    n_q_points_cell = y_eval.n_q_points;
    psi_cell.resize(n_batches * n_q_points_cell);
    grad_psi_cell.resize(n_batches * n_q_points_cell);

    for(unsigned int cell = 0; cell < n_batches; ++cell)
    {
      // psi only exists on enriched cells (hp batches have a single active_fe_index);
      // the evaluators are set up for active_fe_index and must not be used elsewhere
      if(mf.get_cell_iterator(cell, 0)->active_fe_index() != active_fe_index)
        continue;
      y_eval.reinit(cell);
      tau_eval.reinit(cell);
      y_eval.gather_evaluate(wall_distance, flags);
      tau_eval.gather_evaluate(wall_shear_stress, flags);

      for(unsigned int q = 0; q < n_q_points_cell; ++q)
      {
        auto const psi = evaluate_enrichment_function(y_eval.get_value(q),
                                                      y_eval.get_gradient(q),
                                                      tau_eval.get_value(q),
                                                      tau_eval.get_gradient(q));
        psi_cell[cell * n_q_points_cell + q]      = psi.first;
        grad_psi_cell[cell * n_q_points_cell + q] = psi.second;
      }
    }
    if(psi_cell_old.size() != psi_cell.size())
      psi_cell_old = psi_cell;
  }

  // ---- faces: every enriched side of inner and boundary faces ----------------------
  {
    unsigned int const n_inner_faces = mf.n_inner_face_batches();
    unsigned int const n_faces       = n_inner_faces + mf.n_boundary_face_batches();

    // side 0: interior, side 1: exterior
    std::array<FaceIntegratorScalar, 2> y_eval = {
      {FaceIntegratorScalar(mf, true, dof_index_en_scalar, quad_index, 0, active_fe_index),
       FaceIntegratorScalar(mf, false, dof_index_en_scalar, quad_index, 0, active_fe_index)}};
    std::array<FaceIntegratorScalar, 2> tau_eval = {
      {FaceIntegratorScalar(mf, true, dof_index_cg_scalar, quad_index, 0, active_fe_index),
       FaceIntegratorScalar(mf, false, dof_index_cg_scalar, quad_index, 0, active_fe_index)}};

    n_q_points_face = y_eval[0].n_q_points;
    for(unsigned int side = 0; side < 2; ++side)
    {
      psi_face[side].resize(n_faces * n_q_points_face);
      grad_psi_face[side].resize(n_faces * n_q_points_face);
    }

    for(unsigned int face = 0; face < n_faces; ++face)
    {
      for(unsigned int side = 0; side < 2; ++side)
      {
        bool const interior = (side == 0);

        // boundary faces have no exterior side
        if(not interior and face >= n_inner_faces)
          continue;
        if(mf.get_face_active_fe_index(std::make_pair(face, face + 1), interior) != active_fe_index)
          continue;

        y_eval[side].reinit(face);
        tau_eval[side].reinit(face);
        y_eval[side].gather_evaluate(wall_distance, flags);
        tau_eval[side].gather_evaluate(wall_shear_stress, flags);

        for(unsigned int q = 0; q < n_q_points_face; ++q)
        {
          auto const psi = evaluate_enrichment_function(y_eval[side].get_value(q),
                                     y_eval[side].get_gradient(q),
                                     tau_eval[side].get_value(q),
                                     tau_eval[side].get_gradient(q));
          psi_face[side][face * n_q_points_face + q]      = psi.first;
          grad_psi_face[side][face * n_q_points_face + q] = psi.second;
        }
      }
    }
  }
}


template<int dim, typename Number>
typename FunctionEnrichment<dim, Number>::scalar const &
FunctionEnrichment<dim, Number>::get_psi_cell(unsigned int const cell, unsigned int const q) const
{
  AssertIndexRange(cell * n_q_points_cell + q, psi_cell.size());
  return psi_cell[cell * n_q_points_cell + q];
}

template<int dim, typename Number>
typename FunctionEnrichment<dim, Number>::scalar const &
FunctionEnrichment<dim, Number>::get_psi_cell_old(unsigned int const cell,
                                                  unsigned int const q) const
{
  AssertIndexRange(cell * n_q_points_cell + q, psi_cell_old.size());
  return psi_cell_old[cell * n_q_points_cell + q];
}

template<int dim, typename Number>
typename FunctionEnrichment<dim, Number>::vector const &
FunctionEnrichment<dim, Number>::get_grad_psi_cell(unsigned int const cell, unsigned int const q) const
{
  AssertIndexRange(cell * n_q_points_cell + q, grad_psi_cell.size());
  return grad_psi_cell[cell * n_q_points_cell + q];
}

template<int dim, typename Number>
typename FunctionEnrichment<dim, Number>::scalar const &
FunctionEnrichment<dim, Number>::get_psi_face(unsigned int const face,
                                              bool const         interior,
                                              unsigned int const q) const
{
  unsigned int const side = interior ? 0 : 1;
  AssertIndexRange(face * n_q_points_face + q, psi_face[side].size());
  return psi_face[side][face * n_q_points_face + q];
}

template<int dim, typename Number>
typename FunctionEnrichment<dim, Number>::vector const &
FunctionEnrichment<dim, Number>::get_grad_psi_face(unsigned int const face,
                                                   bool const         interior,
                                                   unsigned int const q) const
{
  unsigned int const side = interior ? 0 : 1;
  AssertIndexRange(face * n_q_points_face + q, grad_psi_face[side].size());
  return grad_psi_face[side][face * n_q_points_face + q];
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::compute_enrichment_component(
  VectorType &                    dst,
  VectorType const &              u_tilde,
  dealii::DoFHandler<dim> const & dof_handler_out,
  dealii::Mapping<dim> const &    mapping) const
{
  dealii::FiniteElement<dim> const & fe_out = dof_handler_out.get_fe();
  AssertThrow(fe_out.n_components() == dim and fe_out.n_base_elements() == 1 and
                fe_out.base_element(0).has_support_points(),
              dealii::ExcMessage("compute_enrichment_component: dof_handler_out must be a vector-valued "
                                 "space of dim equal Lagrange base elements."));

  // evaluation points = support points of one component of the output space
  dealii::Quadrature<dim> const points(fe_out.base_element(0).get_unit_support_points());
  unsigned int const            n_points = points.size();

  dealii::FEValues<dim> fe_values_u(mapping,
                                    dof_handler_en_vector.get_fe(active_fe_index),
                                    points,
                                    dealii::update_values);
  dealii::FEValues<dim> fe_values_y(mapping,
                                    dof_handler_en_scalar.get_fe(active_fe_index),
                                    points,
                                    dealii::update_values | dealii::update_gradients);
  dealii::FEValues<dim> fe_values_tau(mapping,
                                      dof_handler_cg_scalar.get_fe(active_fe_index),
                                      points,
                                      dealii::update_values | dealii::update_gradients);

  std::vector<dealii::Vector<Number>>          u_tilde_values(n_points, dealii::Vector<Number>(dim));
  std::vector<Number>                          y_values(n_points), tau_values(n_points), psi(n_points);
  std::vector<dealii::Tensor<1, dim, Number>>  y_gradients(n_points), tau_gradients(n_points);
  std::vector<dealii::types::global_dof_index> dof_indices(fe_out.n_dofs_per_cell());

  u_tilde.update_ghost_values();
  wall_distance.update_ghost_values();
  wall_shear_stress.update_ghost_values();

  dst = 0.0;
  for(auto const & cell : dof_handler_out.active_cell_iterators())
  {
    if(not cell->is_locally_owned())
      continue;

    // the same cell in the (hp) DoFHandlers of the enrichment
    typename dealii::DoFHandler<dim>::active_cell_iterator const cell_u(&cell->get_triangulation(),
                                                                        cell->level(),
                                                                        cell->index(),
                                                                        &dof_handler_en_vector);
    if(cell_u->active_fe_index() != active_fe_index)
      continue; // \tilde u_h = 0 outside the enriched layer

    typename dealii::DoFHandler<dim>::active_cell_iterator const cell_y(&cell->get_triangulation(),
                                                                        cell->level(),
                                                                        cell->index(),
                                                                        &dof_handler_en_scalar);
    typename dealii::DoFHandler<dim>::active_cell_iterator const cell_tau(&cell->get_triangulation(),
                                                                          cell->level(),
                                                                          cell->index(),
                                                                          &dof_handler_cg_scalar);
    fe_values_u.reinit(cell_u);
    fe_values_y.reinit(cell_y);
    fe_values_tau.reinit(cell_tau);

    fe_values_u.get_function_values(u_tilde, u_tilde_values);
    fe_values_y.get_function_values(wall_distance, y_values);
    fe_values_y.get_function_gradients(wall_distance, y_gradients);
    fe_values_tau.get_function_values(wall_shear_stress, tau_values);
    fe_values_tau.get_function_gradients(wall_shear_stress, tau_gradients);

    // \psi(x_i) (all SIMD lanes equal, lane 0 is used)
    for(unsigned int q = 0; q < n_points; ++q)
    {
      vector grad_y, grad_tau;
      for(unsigned int d = 0; d < dim; ++d)
      {
        grad_y[d]   = dealii::make_vectorized_array<Number>(y_gradients[q][d]);
        grad_tau[d] = dealii::make_vectorized_array<Number>(tau_gradients[q][d]);
      }
      psi[q] = evaluate_enrichment_function(dealii::make_vectorized_array<Number>(y_values[q]),
                                            grad_y,
                                            dealii::make_vectorized_array<Number>(tau_values[q]),
                                            grad_tau).first[0];
    }

    // U^{out}_{i,c} = \psi(x_i) \til
    cell->get_dof_indices(dof_indices);
    for(unsigned int i = 0; i < fe_out.n_dofs_per_cell(); ++i)
    {
      auto const [component, point] = fe_out.system_to_component_index(i);
      dst(dof_indices[i])           = psi[point] * u_tilde_values[point][component];
    }
  }

  dst.update_ghost_values();
}

template class FunctionEnrichment<2, float>;
template class FunctionEnrichment<2, double>;
template class FunctionEnrichment<3, float>;
template class FunctionEnrichment<3, double>;

} // namespace ExaDG
