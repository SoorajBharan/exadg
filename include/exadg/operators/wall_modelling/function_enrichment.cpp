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
  : dof_index_en(0)
  , quad_index(0)
  , kinematic_viscosity(kinematic_viscosity_in)
  , kappa(0.41)
  , beta(5.2)
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
  quad_index   = quad_index_in;
  this->global_dof_handler = &global_dof_handler_in;

  ExaDG::ElementType type = ExaDG::ElementType::Hypercube;

  // CG Wall Variables (Standard Allocation)
  fe_cg_vector = create_finite_element<dim>(type, false /* is_dg*/, dim, 1);
  // MatrixFree requires all active FE index to have the same value in any given cell
  // Index 0: CG
  fe_cg_vector_collection.push_back(*fe_cg_vector);
  // Index 1: CG
  fe_cg_vector_collection.push_back(*fe_cg_vector);
  dof_handler_cg_vector.reinit(triangulation);

  fe_cg_scalar = create_finite_element<dim>(type, false /* is_dg*/, 1, 1);
  // fe_cg_scalar_collection.push_back(dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), 1));
  fe_cg_scalar_collection.push_back(*fe_cg_scalar);
  fe_cg_scalar_collection.push_back(*fe_cg_scalar);
  dof_handler_cg_scalar.reinit(triangulation);

  //  hp-FEM Enrichment Setup
  fe_en_vector = create_finite_element<dim>(type, true /* is_dg*/, dim, fe_degree_en);
  fe_en_collection.push_back(dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), dim)); // Index 0: Freestream
  fe_en_collection.push_back(*fe_en_vector);                                         // Index 1: Wall
  dof_handler_en_vector.reinit(triangulation);

  // hp-FEM for shadow vector
  fe_shadow_vector = create_finite_element<dim>(type, true, dim, fe_degree_dg);
  fe_shadow_collection.push_back(dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), dim));
  fe_shadow_collection.push_back(*fe_shadow_vector);
  dof_handler_shadow_vector.reinit(triangulation);

  // hp-FEM for scalars
  fe_en_scalar = create_finite_element<dim>(type, true, 1, 1);
  fe_en_collection_scalar.push_back(dealii::FESystem<dim>(dealii::FE_Nothing<dim>(), 1));
  fe_en_collection_scalar.push_back(*fe_en_scalar);
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
        if(cell->face(face)->at_boundary()) 
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
    current_layer = next_layer;

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
  }

  // ====================================================================
  // Paint the Triangulation and Distribute Memory
  // ====================================================================
  auto cell_en = dof_handler_en_vector.begin_active();
  auto cell_shadow = dof_handler_shadow_vector.begin_active();

  auto cell_cg_vector = dof_handler_cg_vector.begin_active();
  auto cell_cg_scalar = dof_handler_cg_scalar.begin_active();

  auto cell_en_scalar = dof_handler_en_scalar.begin_active();
  for(; cell_en != dof_handler_en_vector.end(); ++cell_en, ++cell_shadow , ++cell_cg_vector, ++cell_cg_scalar, ++cell_en_scalar)
  {
    if(cell_en->is_locally_owned()) 
    {
      bool is_wall = (all_near_wall_cells.find(cell_en) != all_near_wall_cells.end());
      is_cell_enriched[cell_en->global_active_cell_index()] = is_wall;
      bool is_ghost = (globally_enriched_set.count(static_cast<unsigned int>(cell_en->global_active_cell_index())) > 0);
      is_cell_enriched[cell_en->global_active_cell_index()] = is_ghost;

      if (is_wall || is_ghost)
      {
        cell_en->set_active_fe_index(1); // Enriched Space (Allocates RAM)
        cell_shadow->set_active_fe_index(1);
        cell_cg_vector->set_active_fe_index(1);
        cell_cg_scalar->set_active_fe_index(1);
        cell_en_scalar->set_active_fe_index(1);
      }

    }
  }

  dof_handler_en_vector.distribute_dofs(fe_en_collection);
  dof_handler_shadow_vector.distribute_dofs(fe_shadow_collection);

  dof_handler_cg_vector.distribute_dofs(fe_cg_vector_collection);
  dof_handler_cg_scalar.distribute_dofs(fe_cg_scalar_collection);

  dof_handler_en_scalar.distribute_dofs(fe_en_collection_scalar);
  // ====================================================================
  // Setup Isolated MatrixFree Object
  // ====================================================================
  dof_index_en            = 0;
  dof_index_cg_scalar     = 1;
  dof_index_cg_vector     = 2;
  dof_index_shadow_vector = 3;
  dof_index_en_scalar     = 4;

  constraint_wall.clear();

  for(auto const & face_pair : periodic_faces) 
  {
    // Glue the scalar field (friction velocity / wall distance)
    dealii::DoFTools::make_periodicity_constraints(
      dof_handler_cg_scalar,
      face_pair.cell[0]->face(face_pair.face_idx[0])->boundary_id(),
      face_pair.cell[1]->face(face_pair.face_idx[1])->boundary_id(),
      face_pair.face_idx[0] / 2,
      constraint_wall);

    // Glue the vector field (wall velocity)
    dealii::DoFTools::make_periodicity_constraints(
      dof_handler_cg_vector,
      face_pair.cell[0]->face(face_pair.face_idx[0])->boundary_id(),
      face_pair.cell[1]->face(face_pair.face_idx[1])->boundary_id(),
      face_pair.face_idx[0] / 2,
      constraint_wall);
  }

  constraint_wall.close();

  std::vector<const dealii::DoFHandler<dim>*> dof_handlers = { 
    &dof_handler_en_vector, &dof_handler_cg_scalar, &dof_handler_cg_vector, &dof_handler_shadow_vector , &dof_handler_en_scalar
  };
  
  std::vector<const dealii::AffineConstraints<Number>*> constraints = { 
    &constraint_wall, &constraint_wall, &constraint_wall, &constraint_wall, &constraint_wall

  };

  std::shared_ptr<dealii::Quadrature<dim>> quad = 
    create_quadrature<dim>(ExaDG::ElementType::Hypercube, fe_en_vector->degree + 1);
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

  matrix_free_en->initialize_dof_vector(wall_distance,     dof_index_en_scalar);

  matrix_free_en->initialize_dof_vector(wall_velocity, dof_index_cg_vector);

  enrichment_velocity = 0.0;
  enrichment_residual = 0.0;
  shadow_velocity = 0.0;
  shadow_velocity_residual = 0.0;
  friction_velocity = 0.0;
  wall_distance = 1e10;
  wall_velocity = 0.0;

}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::copy_global_dg_to_wall_layout(
  VectorType const & global_vec,
  VectorType &       wall_vec) const
{
  global_vec.update_ghost_values();
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
    if (cell_shadow->is_locally_owned() && cell_shadow->active_fe_index() == 1)
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
  wall_distance = 2.0; 

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
        if(cell->face(face)->at_boundary() && boundary_descriptor->get_boundary_type(cell->face(face)->boundary_id()) == IncNS::BoundaryTypeU::WallEnrichment) 
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

        for(auto const & wall_point : local_dg_wall_points) 
        {
          Number distance = dof_point.distance(wall_point.first);
          if(distance < min_dist) min_dist = distance;
        }

        if (min_dist < wall_distance[global_dof]) 
        {
          wall_distance[global_dof] = min_dist;
        }
      }
    }
  }
  wall_distance.compress(dealii::VectorOperation::min);
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
        if(cell->face(face)->at_boundary() && boundary_descriptor->get_boundary_type(cell->face(face)->boundary_id()) == IncNS::BoundaryTypeU::WallEnrichment) 
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

  for(auto const & cell : dof_handler_cg_scalar.active_cell_iterators())
  {
    if(cell->is_locally_owned() && is_cell_enriched[cell->global_active_cell_index()])
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

        for(auto const & wall_point : local_cg_wall_points) 
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
}

template<int dim, typename Number>
void
FunctionEnrichment<dim, Number>::setup_wall_distance_cg(
  dealii::Mapping<dim> const &                           mapping,
  unsigned int                                           layers)
{
  (void)layers;

  wall_distance = 0.0; 

  std::map<dealii::types::global_dof_index, dealii::Point<dim>> support_points;
  // FOR:CG
  dealii::DoFTools::map_dofs_to_support_points(mapping, dof_handler_cg_scalar, support_points);

  std::vector<std::pair<dealii::Point<dim>, dealii::types::global_dof_index>> local_wall_points;
  auto const & fe = *fe_cg_scalar;

  for(auto const & cell : dof_handler_cg_scalar.active_cell_iterators()) 
  {
    if(cell->is_locally_owned() && cell->at_boundary()) 
    {
      for(auto face : cell->face_indices()) 
      {
        if(cell->face(face)->at_boundary()) 
        {
          auto bid = cell->face(face)->boundary_id();
          if(boundary_descriptor->get_boundary_type(bid) == IncNS::BoundaryTypeU::WallEnrichment) 
          {
            std::vector<dealii::types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);
            for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i) 
            {
              // If the DoF actually touches the wall boundary face
              if(fe.has_support_on_face(i, face)) 
              {
                auto global_dof = local_dof_indices[i];
                if(wall_distance.locally_owned_elements().is_element(global_dof)) 
                {
                  local_wall_points.push_back({support_points[global_dof], global_dof});
                }
              }
            }
          }
        }
      }
    }
  }

  for(auto const & cell : dof_handler_cg_scalar.active_cell_iterators())
  {
    if(cell->is_locally_owned() && is_cell_enriched[cell->global_active_cell_index()])
    {
      std::vector<dealii::types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
      cell->get_dof_indices(local_dof_indices);

      for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i) 
      {
        auto global_dof = local_dof_indices[i];
        
        // Only calculate for locally owned DoFs
        if (!wall_distance.locally_owned_elements().is_element(global_dof)) continue;

        auto dof_point  = support_points[global_dof];
        Number min_dist = std::numeric_limits<Number>::max();
        dealii::types::global_dof_index nearest_wall_dof = dealii::numbers::invalid_dof_index;

        // Find the absolute closest wall point
        for(auto const & wall_point : local_wall_points) 
        {
          Number distance = dof_point.distance(wall_point.first);
          if(distance < min_dist) 
          {
            min_dist = distance;
            nearest_wall_dof = wall_point.second;
          }
        }

        // Save the distance and the node pairing
        if (min_dist < wall_distance[global_dof]) 
        {
          wall_distance[global_dof] = min_dist;
          node_to_wall_node[global_dof] = nearest_wall_dof; 
        }
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
    if (data.get_cell_iterator(cell, 0)->active_fe_index() == 0)
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
    if (data.get_cell_iterator(cell, 0)->active_fe_index() == 0)
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
  dealii::FEFaceEvaluation<dim, -1, 0, dim, Number> u_wall(
      matrix_free_data, true, dof_index_cg_vector, quad_index, 0, active_fe_index);
  
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
    u_wall.read_dof_values(src);
    u_wall.evaluate(dealii::EvaluationFlags::gradients);

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
  // Project DG velocity onto the continuous P1 wall space (Spatial Coarsening)
  project_velocity_to_wall(global_velocity);

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

  // MPI Synchronization for the boundary integrals
  for(unsigned int d = 0; d < dim; ++d) 
  {
    traction_num[d].compress(dealii::VectorOperation::add);
    constraint_wall.distribute(traction_num[d]);
  }
  traction_den.compress(dealii::VectorOperation::add);
  constraint_wall.distribute(traction_den);

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

  constraint_wall.distribute(friction_velocity);
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

  constraint_wall.distribute(friction_velocity);
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
  scalar nu = dealii::make_vectorized_array<Number>(kinematic_viscosity);

  scalar u_tau_safe = std::max(u_tau, dealii::make_vectorized_array<Number>(1e-5));
  
  scalar y_plus = (y * u_tau_safe) / nu;
  scalar u_plus = wall_law_eval.get_value(y_plus);
  scalar u_scalar = u_plus * u_tau_safe;

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

template class FunctionEnrichment<2, float>;
template class FunctionEnrichment<2, double>;
template class FunctionEnrichment<3, float>;
template class FunctionEnrichment<3, double>;

} // namespace ExaDG
