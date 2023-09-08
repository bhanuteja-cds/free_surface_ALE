
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
    #if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace dealii::LinearAlgebraPETSc;
    #  define USE_PETSC_LA
    #elif defined(DEAL_II_WITH_TRILINOS)
    using namespace dealii::LinearAlgebraTrilinos;
    #else
    #  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
    #endif
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/sparse_ilu.h>

#include <iostream>
#include <fstream>
#include <sstream>

namespace Navierstokes
{
    using namespace dealii;
    
    template <int dim>
    class StokesProblem
    {
        
    private:
        MPI_Comm                                  mpi_communicator;
        double deltat = 0.005;
        double viscosity = 0.089, density = 250.0;
        int meshrefinement = 0;
        int degree;
        parallel::distributed::Triangulation<dim> triangulation;
        LA::MPI::SparseMatrix                     system_matrix;
        LA::MPI::SparseMatrix                     move_system_matrix;
        DoFHandler<dim>                           dof_handler, move_dof_handler;
        FESystem<dim>                             fe, femove;
        LA::MPI::Vector                           lr_solution, lr_old_iterate_solution, lr_nonlinear_residue, lr_old_move_solution, lr_move_solution, lr_total_meshmove;
        LA::MPI::Vector                           lo_system_rhs, lo_mass_matrix_times_old_velocity, lo_move_system_rhs, lo_total_meshmove, lo_lhs_productvec;
        AffineConstraints<double>                 moveconstraints, stokesconstraints;
        IndexSet                                  owned_partitioning_stokes, owned_partitioning_movemesh;
        IndexSet                                  relevant_partitioning_stokes, relevant_partitioning_movemesh;
        ConditionalOStream                        pcout;
        //         TimerOutput                               computing_timer;
        
    public:
        void setup_stokessystem();
        void setup_movemeshsystem();
        void resetup_stokessystem();
        void resetup_movemeshsystem();
        void assemble_stokessystem();
        void assemble_stokessystem_nonlinear();
        void assemble_movemesh();
        void apply_boundary_conditions_and_rhs();
        void solve_stokes();
        double compute_errors();
        double compute_nonlinear_residue();
        double compute_l2norm_velocity();
        void solve_movemesh();
        void movemesh();
        void output_results (int);
        void timeloop();
        
        StokesProblem(int degreein)
        :
        mpi_communicator (MPI_COMM_WORLD),
        degree(degreein),
        triangulation (mpi_communicator),
        dof_handler(triangulation),
        move_dof_handler(triangulation),
        fe(FE_Q<dim>(degree+1), dim, FE_Q<dim>(degree), 1),
        femove(FE_Q<dim>(degree), dim),
        pcout (std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
        //         computing_timer (mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
        {      
            pcout << "stokes constructor success...."<< std::endl;
        }
    };
    //=====================================================  
    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
        RightHandSide () : Function<dim>(dim+1)
        {}
        virtual double value (const Point<dim> &p,  const unsigned int  component = 0) const;
        virtual void vector_value(const Point<dim> &p, Vector<double> &value) const;
    };
    
    template <int dim>
    double
    RightHandSide<dim>::value(const Point<dim>  &/*p*/,  const unsigned int /*component*/) const
    {
        return 0;
    }
    
    template <int dim>
    void
    RightHandSide<dim>::vector_value(const Point<dim> &p,  Vector<double> &values) const
    {
        values[0] = 0;
        values[1] = -9.8;
        values[2] = 0*p[0];
        values[3] = 0;
    }
    //==================================================  
    template <int dim>
    class TractionBoundaryValues : public Function<dim>
    {
    public:
        TractionBoundaryValues () : Function<dim>(dim)
        {}
        virtual void vector_value_list(const std::vector<Point<dim> > &points, std::vector<Vector<double>> &values) const;
    };
    
    template <int dim>
    void  TractionBoundaryValues<dim>::vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &values) const
    {
        //       const double time = this->get_time();
        for (unsigned int p=0; p<points.size(); ++p)
        {
            values[p][0] = 0.0;
            values[p][1] = 0.0;
            values[p][2] = 0.0;
        }    
    }  
    //===========================================================
    template <int dim>
    class MoveRightHandSide : public TensorFunction<1,dim>
    {
    public:
        MoveRightHandSide() : TensorFunction<1,dim>() {}
        
        virtual void value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1,dim>> &values) const;
    };
    
    template <int dim>
    void  MoveRightHandSide<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1,dim>> &values) const
    {
        Assert (points.size() == values.size(), ExcDimensionMismatch (points.size(), values.size()));
        
        for (unsigned int p=0; p<points.size(); ++p)
        {
            values[p].clear ();
            values[p][0] = 0.;
            values[p][1] = 0.;
            values[p][2] = 0.;
        }
    }    
    //==============================================================
    template <int dim>
    void StokesProblem<dim>::setup_movemeshsystem()
    { 
        //         TimerOutput::Scope t(computing_timer, "setup_stokessystem");
        pcout << "setup system " << std::endl;
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file("bulkfluid3d_transfinite.msh");
//         std::ifstream input_file("bulkfluid3d_unstructured_hex.msh");
        grid_in.read_msh(input_file);
        triangulation.refine_global (meshrefinement);        
        dof_handler.distribute_dofs(fe);
        pcout << "Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "Total number of cells: "
        << triangulation.n_cells()
        << std::endl;
        pcout << "Number of stokes degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl; 
        
        std::vector<unsigned int> block_component(dim+1,0);
        block_component[dim] = 1;
//         DoFRenumbering::component_wise(dof_handler, block_component);
//         std::vector<types::global_dof_index> dofs_per_block (2);
       const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
        const unsigned int n_u = dofs_per_block[0],  n_p = dofs_per_block[1];
        pcout << "n_u + n_p = " << n_u << " + " << n_p  << std::endl;
        pcout << "stokes_dofspercell "<< fe.dofs_per_cell << std::endl;
        
        owned_partitioning_stokes = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (dof_handler, relevant_partitioning_stokes);
        
        {
            stokesconstraints.reinit (relevant_partitioning_stokes);
            //             DoFTools::make_hanging_node_constraints (dof_handler, stokesconstraints);
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (101);
            no_normal_flux_boundaries.insert (102);
            no_normal_flux_boundaries.insert (103);
            //             no_normal_flux_boundaries.insert (104);
            no_normal_flux_boundaries.insert (105);            
            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim);        
            ComponentMask velocities_mask = fe.component_mask(velocities);
            ComponentMask pressure_mask = fe.component_mask(pressure);
            VectorTools::compute_no_normal_flux_constraints (dof_handler, 0, no_normal_flux_boundaries, stokesconstraints);
            VectorTools::interpolate_boundary_values (dof_handler, 104, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            stokesconstraints.close();
        }
        
        system_matrix.clear();
        
        DynamicSparsityPattern dsp (relevant_partitioning_stokes);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, stokesconstraints, false);
        SparsityTools::distribute_sparsity_pattern (dsp, dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_stokes);
        system_matrix.reinit (owned_partitioning_stokes, owned_partitioning_stokes, dsp, mpi_communicator);
        
        lr_solution.reinit(owned_partitioning_stokes, relevant_partitioning_stokes, mpi_communicator);
        lr_old_iterate_solution.reinit(owned_partitioning_stokes, relevant_partitioning_stokes, mpi_communicator);
        lr_nonlinear_residue.reinit(owned_partitioning_stokes, relevant_partitioning_stokes, mpi_communicator);
        lo_system_rhs.reinit(owned_partitioning_stokes, mpi_communicator);
        lo_lhs_productvec.reinit(owned_partitioning_stokes, mpi_communicator);
        
        move_dof_handler.distribute_dofs(femove);        
        owned_partitioning_movemesh = move_dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (move_dof_handler, relevant_partitioning_movemesh);        
        {
            moveconstraints.reinit(relevant_partitioning_movemesh);
            //             DoFTools::make_hanging_node_constraints (move_dof_handler, moveconstraints);
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (101);
            no_normal_flux_boundaries.insert (102);
            no_normal_flux_boundaries.insert (103);
            //             no_normal_flux_boundaries.insert (104);
            no_normal_flux_boundaries.insert (105);
            VectorTools::compute_no_normal_flux_constraints (move_dof_handler, 0, no_normal_flux_boundaries, moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 104, ZeroFunction<dim>(dim), moveconstraints);
            MappingQGeneric<dim> mapping(fe.degree); 
            Quadrature<dim - 1> face_quadrature(femove.get_unit_face_support_points());
            FEFaceValues<dim> fe_face_values_stokes(mapping, fe, face_quadrature, update_values | update_normal_vectors);            
            const unsigned int n_face_dofs = face_quadrature.size();            
            std::vector<Vector<double>> stokes_values(n_face_dofs,  Vector<double>(dim + 1));
            std::vector<types::global_dof_index> local_dof_indices(femove.dofs_per_face);
            auto cell_move = move_dof_handler.begin_active();
            auto endcell_move = move_dof_handler.end();
            auto cell_stokes = dof_handler.begin_active();
            
            Tensor<1, dim> normal_vector1;
            Tensor<1, dim> col_indices;
            Tensor<1, dim> stokes_value_atvertex;
//========================w=u============================    
//             for (; cell_move != endcell_move; ++cell_stokes, ++cell_move)
//             {
//                 if (!cell_move->is_artificial())
//                 {
//                     for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) 
//                     {
//                         if (cell_move->face(f)->boundary_id() == 150)
//                         {
//                             fe_face_values_stokes.reinit(cell_stokes, f);
//                             fe_face_values_stokes.get_function_values(lr_solution, stokes_values);
//                             cell_move->face(f)->get_dof_indices(local_dof_indices);
//                             for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
//                             {                                
//                                 const unsigned int component = femove.system_to_component_index(i).first;
//                                 moveconstraints.add_line(local_dof_indices[i]);
//                                 moveconstraints.set_inhomogeneity(local_dof_indices[i], stokes_values[i](component));
//                             }
//                         }
//                     }
//                 }
//             }
//======================w.n = u.n==========================
            int max_comp;
            std::vector<bool> vertex_touched(triangulation.n_vertices(), false);            
            for (; cell_move != endcell_move; ++cell_stokes, ++cell_move)
            {
                if (!cell_move->is_artificial())
                {
                    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) 
                    {
                        if (cell_move->face(f)->boundary_id() == 150)
                        {
                            fe_face_values_stokes.reinit(cell_stokes, f);
                            normal_vector1 = fe_face_values_stokes.normal_vector(1);
                            
                            for(int ii = 1; ii < dim; ++ii)
                            {
                                if(normal_vector1[ii-1] >= normal_vector1[ii])
                                {
                                    max_comp = ii-1;
                                    break;
                                }
                                max_comp = ii;
                            }
                            
                            fe_face_values_stokes.get_function_values(lr_solution, stokes_values);
                            
                            for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_face; ++v)
                            {
                                if(vertex_touched[cell_move->face(f)->vertex_index(v)] == false)
                                {
                                    vertex_touched[cell_move->face(f)->vertex_index(v)] = true;
                                    
                                    for(int j=0; j < dim; ++j)
                                    {
                                        col_indices[j] = cell_move->face(f)->vertex_dof_index(v, j);
                                        const unsigned int component = femove.system_to_component_index(v*dim+j).first;
                                        stokes_value_atvertex[j] = stokes_values[v*dim+j](component);
                                    }
                                    
                                    moveconstraints.add_line(col_indices[max_comp]);
                                    
                                    for(int j=0; j<dim; ++j)
                                    {
                                        if(j!=max_comp)
                                            moveconstraints.add_entry(col_indices[max_comp], col_indices[j], normal_vector1[j]/normal_vector1[max_comp]);
                                    }
                                    moveconstraints.set_inhomogeneity(col_indices[max_comp], stokes_value_atvertex*normal_vector1/normal_vector1[max_comp]); 
                                    //moveconstraints.set_inhomogeneity(col_indices[max_comp], 0.01);
                                }
                            }
                        }
                    }
                }
            }
            //===========w.n = u.n==========================
            moveconstraints.close();
        }
        
        pcout << "Number of move degrees of freedom: " << move_dof_handler.n_dofs() << std::endl; 
        move_system_matrix.clear();
        DynamicSparsityPattern move_dsp(relevant_partitioning_movemesh);
        DoFTools::make_sparsity_pattern(move_dof_handler, move_dsp, moveconstraints, false);
        SparsityTools::distribute_sparsity_pattern (move_dsp, move_dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_movemesh);        
        move_system_matrix.reinit(owned_partitioning_movemesh, owned_partitioning_movemesh, move_dsp, mpi_communicator);
        
        lr_move_solution.reinit(owned_partitioning_movemesh, relevant_partitioning_movemesh, mpi_communicator);
        lr_old_move_solution.reinit(owned_partitioning_movemesh, relevant_partitioning_movemesh, mpi_communicator);
        lr_total_meshmove.reinit(owned_partitioning_movemesh, relevant_partitioning_movemesh, mpi_communicator);
        lo_total_meshmove.reinit(owned_partitioning_movemesh, mpi_communicator);
        lo_move_system_rhs.reinit(owned_partitioning_movemesh, mpi_communicator);
//         pcout <<"end of setup_movemeshsystem "<<std::endl;
    } 
    //========================================================  
    template <int dim>
    void StokesProblem<dim>::resetup_movemeshsystem()
    { 
        pcout <<"resetup_movemesh_system "<<std::endl;            
        {
            moveconstraints.reinit(relevant_partitioning_movemesh);
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (101);
            no_normal_flux_boundaries.insert (102);
            no_normal_flux_boundaries.insert (103);
            //             no_normal_flux_boundaries.insert (104);
            no_normal_flux_boundaries.insert (105);
            VectorTools::compute_no_normal_flux_constraints (move_dof_handler, 0, no_normal_flux_boundaries, moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 104, ZeroFunction<dim>(dim), moveconstraints);
            
            MappingQGeneric<dim> mapping(fe.degree); 
            Quadrature<dim - 1> face_quadrature(femove.get_unit_face_support_points());
            FEFaceValues<dim> fe_face_values_stokes(mapping, fe, face_quadrature, update_values | update_normal_vectors);            
            const unsigned int n_face_dofs = face_quadrature.size();            
            std::vector<Vector<double>> stokes_values(n_face_dofs,  Vector<double>(dim + 1));
            std::vector<types::global_dof_index> local_dof_indices(femove.dofs_per_face);
            
            auto cell_move = move_dof_handler.begin_active();
            auto endcell_move = move_dof_handler.end();
            auto cell_stokes = dof_handler.begin_active();
            
            Tensor<1, dim> normal_vector1;
            Tensor<1, dim> col_indices;
            Tensor<1, dim> stokes_value_atvertex;
//========================w=u============================    
            for (; cell_move != endcell_move; ++cell_stokes, ++cell_move)
            {
                if (!cell_move->is_artificial())
                {
                    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) 
                    {
                        if (cell_move->face(f)->boundary_id() == 150)
                        {
                            fe_face_values_stokes.reinit(cell_stokes, f);
                            fe_face_values_stokes.get_function_values(lr_solution, stokes_values);
                            cell_move->face(f)->get_dof_indices(local_dof_indices);
                            for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
                            {                               
                                const unsigned int component = femove.system_to_component_index(i).first;
                                moveconstraints.add_line(local_dof_indices[i]);
                                moveconstraints.set_inhomogeneity(local_dof_indices[i], stokes_values[i](component));
                            }
                        }
                    }
                }
            }
//=========================w.n = u.n=============================================        
//             int max_comp=0;
//             std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
//             for (; cell_move != endcell_move; ++cell_stokes, ++cell_move)
//             {
//                 if (!cell_move->is_artificial())
//                 {
//                     for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) 
//                     {
//                         if (cell_move->face(f)->boundary_id() == 150)
//                         {
//                             fe_face_values_stokes.reinit(cell_stokes, f);
//                             normal_vector1 = fe_face_values_stokes.normal_vector(1);
//                             for(int ii = 1; ii < dim; ++ii)
//                             {
//                                 if(normal_vector1[ii-1] >= normal_vector1[ii])
//                                 {
//                                     max_comp = ii-1;
// //                                     break;
//                                 }
//                                 else
//                                 {
//                                 max_comp = ii;
//                                 }
//                             }
//                             max_comp = 1;
//                             fe_face_values_stokes.get_function_values(lr_solution, stokes_values);
//                             //  cell_move->face(f)->get_dof_indices(local_dof_indices);
//                             for (unsigned int v=0; v < GeometryInfo<dim>::vertices_per_face; ++v)
//                             {
//                                 if(vertex_touched[cell_move->face(f)->vertex_index(v)] == false)
//                                 {
//                                     vertex_touched[cell_move->face(f)->vertex_index(v)] = true;
//                                     for(unsigned int j=0; j < dim; ++j)
//                                     {
//                                         col_indices[j] = cell_move->face(f)->vertex_dof_index(v, j);
//                                         const unsigned int component = femove.system_to_component_index(v*dim+j).first;
//                                         stokes_value_atvertex[j] = stokes_values[v*dim+j](component);
//                                     }
//                                     
//                                     moveconstraints.add_line(col_indices[max_comp]);
//                                     
//                                     for(int j=0; j<dim; ++j)
//                                     {
//                                         if(j!=max_comp)
//                                             moveconstraints.add_entry(col_indices[max_comp], col_indices[j], normal_vector1[j]/normal_vector1[max_comp]);
//                                     }
//                                     moveconstraints.set_inhomogeneity(col_indices[max_comp], stokes_value_atvertex*normal_vector1/normal_vector1[max_comp]);
// //                                        moveconstraints.set_inhomogeneity(col_indices[max_comp], 0.01);
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//=========================w.n = u.n============================================= 
          moveconstraints.close();
        }
        move_system_matrix.clear();
        DynamicSparsityPattern move_dsp(relevant_partitioning_movemesh);
        DoFTools::make_sparsity_pattern(move_dof_handler, move_dsp, moveconstraints, false);
        SparsityTools::distribute_sparsity_pattern (move_dsp, move_dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_movemesh);        
        move_system_matrix.reinit(owned_partitioning_movemesh, owned_partitioning_movemesh, move_dsp, mpi_communicator);
    }
    //================================================
    template <int dim>
    void StokesProblem<dim>::resetup_stokessystem()
    {   
        pcout << "resetup_stokes_system " << std::endl;
        stokesconstraints.reinit(relevant_partitioning_stokes);
        DoFTools::make_hanging_node_constraints (dof_handler, stokesconstraints);
        std::set<types::boundary_id> no_normal_flux_boundaries;
        no_normal_flux_boundaries.insert (101);
        no_normal_flux_boundaries.insert (102);
        no_normal_flux_boundaries.insert (103);
        //         no_normal_flux_boundaries.insert (104);
        no_normal_flux_boundaries.insert (105);
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);        
        ComponentMask velocities_mask = fe.component_mask(velocities);
        ComponentMask pressure_mask = fe.component_mask(pressure);
        VectorTools::compute_no_normal_flux_constraints (dof_handler, 0, no_normal_flux_boundaries, stokesconstraints);
        VectorTools::interpolate_boundary_values (dof_handler, 104, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
        stokesconstraints.close();        
        system_matrix.clear ();
        
        DynamicSparsityPattern dsp (relevant_partitioning_stokes);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, stokesconstraints, false);
        SparsityTools::distribute_sparsity_pattern (dsp, dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_stokes);
        system_matrix.reinit (owned_partitioning_stokes, owned_partitioning_stokes, dsp, mpi_communicator);
    }
    //===========================================================
    template <int dim>
    void StokesProblem<dim>::assemble_stokessystem()
    {
        //         TimerOutput::Scope t(computing_timer, "assembly_stokessystem");
        pcout <<"assemble_stokes_system"<<std::endl;
        system_matrix=0;
        lo_system_rhs=0;
        
        QGauss<dim>   quadrature_formula(degree+2);
        QGauss<dim-1> face_quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);
        
        FEValues<dim> fe_move_values (femove, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);
        
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);
        
        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim); 
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        FullMatrix<double>   local_mass_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index>  local_dof_indices (dofs_per_cell);        
        const RightHandSide<dim>              right_hand_side;
        const TractionBoundaryValues<dim>     traction_boundary_values;
        std::vector<Vector<double>>           rhs_values(n_q_points, Vector<double>(dim+1));
        std::vector<Vector<double>>           neumann_boundary_values(n_face_q_points, Vector<double>(dim+1));        
        std::vector<Tensor<1, dim>>           value_phi_u (dofs_per_cell);
        std::vector<Tensor<2, dim>>           gradient_phi_u (dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>>  symgrad_phi_u (dofs_per_cell);
        std::vector<double>                   div_phi_u(dofs_per_cell);
        std::vector<double>                   phi_p(dofs_per_cell);
        std::vector<Tensor<2, dim> >          solution_gradients(n_q_points);
        std::vector<Tensor<1, dim> >          old_solution_values(n_q_points);
        std::vector<Tensor<1, dim> >          meshvelocity_values(n_q_points);
        int lostokes = 0, lomeshmove = 0;        
        
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator cell_move = move_dof_handler.begin_active();
        
        for (; cell!=endc; ++cell)
        { 
            if (cell->is_locally_owned())
            {
                lostokes++;
                fe_values.reinit (cell);
                fe_move_values.reinit(cell_move);
                fe_values[velocities].get_function_values(lr_old_iterate_solution, old_solution_values);
                
                lomeshmove++;
                fe_move_values[velocities].get_function_values(lr_old_move_solution, meshvelocity_values);
                local_matrix = 0;
                local_mass_matrix = 0;
                local_rhs = 0;
                right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {                        
                        value_phi_u[k]   = fe_values[velocities].value (k, q);
                        gradient_phi_u[k]= fe_values[velocities].gradient (k, q);
                        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                        div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                        phi_p[k]         = fe_values[pressure].value (k, q);
                    }
                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {                    
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {
                            local_matrix(i, j) += ((value_phi_u[i]*value_phi_u[j] + 
                            deltat * value_phi_u[i] * (gradient_phi_u[j] * (old_solution_values[q] - meshvelocity_values[q])) +
                            (2*deltat*viscosity/density)*scalar_product(symgrad_phi_u[i], symgrad_phi_u[j])) -
                            deltat * div_phi_u[i] * phi_p[j]/density - 
                            phi_p[i] * div_phi_u[j]) *
                            fe_values.JxW(q);
                        }
                        const unsigned int component_i = fe.system_to_component_index(i).first;
                        local_rhs(i) += (deltat*fe_values.shape_value(i,q) * rhs_values[q](component_i) + old_solution_values[q]*value_phi_u[i]) * fe_values.JxW(q);
                    } // end of i loop                
                }  // end of quadrature points loop
                
                for (unsigned int face_n=0;  face_n<GeometryInfo<dim>::faces_per_cell;  ++face_n)
                {
                    if (cell->face(face_n)->at_boundary() && (cell->face(face_n)->boundary_id() == 150))
                    {             
                        fe_face_values.reinit (cell, face_n);
                        traction_boundary_values.vector_value_list(fe_face_values.get_quadrature_points(), neumann_boundary_values);
                        for (unsigned int q=0; q<n_face_q_points; ++q)
                            for (unsigned int i=0; i<dofs_per_cell; ++i)
                            {                 
                                const unsigned int component_i = fe.system_to_component_index(i).first;
                                local_rhs(i) += (fe_face_values.shape_value(i, q) * neumann_boundary_values[q](component_i) * fe_face_values.JxW(q))*deltat/density;
                            }
                    } // end of face if
                } // end of face for        
                cell->get_dof_indices (local_dof_indices);         
                stokesconstraints.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, system_matrix, lo_system_rhs);
            } // end of if cell->locally owned
            cell_move++;
        } // end of cell loop
        system_matrix.compress (VectorOperation::add);
        lo_system_rhs.compress (VectorOperation::add);
//         pcout <<"end of assemble_stokessystem "<<std::endl;
    }
    //============================================================
    template <int dim>
    void StokesProblem<dim>::assemble_stokessystem_nonlinear()
    {
        //         TimerOutput::Scope t(computing_timer, "assembly_system_nonlinear");
        pcout <<"assemble_system_nonlinear"<<std::endl;
        system_matrix=0;
        
        QGauss<dim>   quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);
        
        FEValues<dim> fe_move_values (femove, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);
        
        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim); 
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        std::vector<types::global_dof_index>  local_dof_indices (dofs_per_cell);        
        
        std::vector<Tensor<1, dim>>           value_phi_u (dofs_per_cell);    
        std::vector<Tensor<2, dim>>           gradient_phi_u (dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>>  symgrad_phi_u (dofs_per_cell);
        std::vector<double>                   div_phi_u(dofs_per_cell);
        std::vector<double>                   phi_p(dofs_per_cell);        
        std::vector<Tensor<2, dim> >          solution_gradients(n_q_points);
        std::vector<Tensor<1, dim> >          old_solution_values(n_q_points);
        std::vector<Tensor<1, dim> >          meshvelocity_values(n_q_points);        
        
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator cell_move = move_dof_handler.begin_active();
        
        for (; cell!=endc; ++cell)
        {    
            if(cell->is_locally_owned())
            {
                fe_values.reinit (cell);
                fe_move_values.reinit(cell_move);
                fe_values[velocities].get_function_values(lr_old_iterate_solution, old_solution_values);
                fe_move_values[velocities].get_function_values(lr_old_move_solution, meshvelocity_values);
                local_matrix = 0;
                
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {
                        value_phi_u[k]   = fe_values[velocities].value (k, q);
                        gradient_phi_u[k]= fe_values[velocities].gradient (k, q);
                        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                        div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                        phi_p[k]         = fe_values[pressure].value (k, q);
                    }
                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {                    
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {
                            local_matrix(i, j) += ((value_phi_u[i]*value_phi_u[j] + 
                            deltat * value_phi_u[i] * (gradient_phi_u[j] * (old_solution_values[q] - meshvelocity_values[q])) +
                            (2*deltat*viscosity/density)*scalar_product(symgrad_phi_u[i], symgrad_phi_u[j])) -
                            deltat * div_phi_u[i] * phi_p[j]/density - 
                            phi_p[i] * div_phi_u[j]) *
                            fe_values.JxW(q);
                        }
                    } // end of i loop              
                }  // end of quadrature points loop                
                cell->get_dof_indices (local_dof_indices);
                stokesconstraints.distribute_local_to_global(local_matrix, local_dof_indices, system_matrix);
            }
            cell_move++;
        } // end of cell loop
        system_matrix.compress (VectorOperation::add);
//         pcout <<"end of assemble_system_nonlinear "<<std::endl;
    }
    //=======================================
    template <int dim>
    void StokesProblem<dim>::assemble_movemesh()
    {
        //         TimerOutput::Scope t(computing_timer, "assembly_movemesh");
        pcout << "assemble_movemesh" << std::endl;
        move_system_matrix=0;
        lo_move_system_rhs=0;
        QGauss<dim>   move_quadrature_formula(degree+2);
        
        FEValues<dim> move_fe_values (femove, move_quadrature_formula,
                                      update_values  |
                                      update_quadrature_points |
                                      update_JxW_values |
                                      update_gradients);        
        
        const unsigned int dofs_per_cell = femove.dofs_per_cell;
        const unsigned int move_n_q_points = move_quadrature_formula.size();
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);
        
        FullMatrix<double>                   move_local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>                       move_local_rhs(dofs_per_cell);        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        const MoveRightHandSide<dim>         move_right_hand_side;
        std::vector<Tensor<1,dim>>           move_rhs_values(move_n_q_points);        
        
        typename DoFHandler<dim>::active_cell_iterator cell = move_dof_handler.begin_active(), endc = move_dof_handler.end();        
        
        for (; cell!=endc; ++cell)
        {
            if (cell->is_locally_owned())
            {
                move_fe_values.reinit(cell);
                move_local_matrix = 0;
                move_local_rhs = 0;
                move_right_hand_side.value_list(move_fe_values.get_quadrature_points(), move_rhs_values);
                
                for (unsigned int q_index=0; q_index<move_n_q_points; ++q_index)
                {
                    for (unsigned int i=0; i<dofs_per_cell; ++i)            
                    {
                        const Tensor<2,dim> symmgrad_i_u = move_fe_values[velocities].symmetric_gradient(i, q_index);
                        const double div_i_u = move_fe_values[velocities].divergence(i, q_index); 
                        
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {  
                            const Tensor<2,dim> symmgrad_j_u = move_fe_values[velocities].symmetric_gradient(j, q_index);
                            const double div_j_u = move_fe_values[velocities].divergence(j, q_index);
                            
                            move_local_matrix(i,j) += (scalar_product(symmgrad_i_u, symmgrad_j_u) + div_i_u*div_j_u) * move_fe_values.JxW(q_index);
                        }
                        move_local_rhs(i) += 0;
                    }
                }
                    cell->get_dof_indices(local_dof_indices);                
                    moveconstraints.distribute_local_to_global (move_local_matrix, move_local_rhs, local_dof_indices, move_system_matrix, lo_move_system_rhs);
            } // end of if cell->is_locally_owned()
        } //  end of cell loop   
        move_system_matrix.compress (VectorOperation::add);
        lo_move_system_rhs.compress (VectorOperation::add);
//         pcout << "end of assemble_movemesh"<< std::endl;
    }
    //================================================================
    template <int dim>
    void StokesProblem<dim>::solve_stokes()
    {
        pcout <<"solve_stokes"<<std::endl;
        //         TimerOutput::Scope t(computing_timer, "solve");
        LA::MPI::Vector  distributed_solution_stokes (owned_partitioning_stokes, mpi_communicator);
        LA::MPI::Vector  distributed_solution_movemesh (owned_partitioning_movemesh, mpi_communicator);
        double nonlinear_error = 1e5;
        int nonlinear_iterations = 0;
        lo_total_meshmove = lr_move_solution;
        
        SolverControl solver_control_stokes (dof_handler.n_dofs(), 1e-12);
        dealii::PETScWrappers::SparseDirectMUMPS solver_stokes(solver_control_stokes, mpi_communicator);
        SolverControl solver_control_movemesh (move_dof_handler.n_dofs(), 1e-12);
        dealii::PETScWrappers::SparseDirectMUMPS solver_movemesh(solver_control_movemesh, mpi_communicator);
        
        while(nonlinear_error > 1)
        {
            solver_stokes.solve (system_matrix, distributed_solution_stokes, lo_system_rhs);
            stokesconstraints.distribute(distributed_solution_stokes);                 
            lr_solution = distributed_solution_stokes;            
//             distributed_solution_stokes -=lr_old_iterate_solution;
//             lr_nonlinear_residue = distributed_solution_stokes;
            resetup_movemeshsystem();
            assemble_movemesh();    
            solver_movemesh.solve (move_system_matrix, distributed_solution_movemesh, lo_move_system_rhs);
            moveconstraints.distribute(distributed_solution_movemesh);
            lr_move_solution = distributed_solution_movemesh;
            nonlinear_error = compute_nonlinear_residue();
            pcout <<"residual_norm " << nonlinear_error << std::endl;
            lr_old_iterate_solution = lr_solution;
            lr_old_move_solution = lr_move_solution; 
            pcout <<"end of nonlinear iteration "<< nonlinear_iterations << std::endl;
            nonlinear_iterations++;
            if(nonlinear_error < 1 || nonlinear_iterations > 3)
                break;
            assemble_stokessystem_nonlinear();            
        } // end of while
        lo_total_meshmove += lr_move_solution;
//         pcout <<"end of solve_stokes "<<std::endl;
    }
    //==========================================
    template <int dim>
    double StokesProblem<dim>::compute_errors()
    {        
        const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
        const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0,dim), dim+1);        
        Vector<double> cellwise_errors(triangulation.n_active_cells());
        QGauss<dim> quadrature(4);
        VectorTools::integrate_difference (dof_handler, lr_nonlinear_residue, ZeroFunction<dim>(dim+1), cellwise_errors, quadrature, VectorTools::L2_norm, &velocity_mask);
        const double u_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);
        return u_l2_error;
    }
    //==========================================
    template <int dim>
    double StokesProblem<dim>::compute_nonlinear_residue()
    {        
        const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
        const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0,dim), dim+1);
        system_matrix.vmult(lo_lhs_productvec, lr_old_iterate_solution);
        lo_lhs_productvec -= lo_system_rhs;
        lr_nonlinear_residue = lo_lhs_productvec;
        Vector<double> cellwise_errors(triangulation.n_active_cells());
        QGauss<dim> quadrature(4);
        VectorTools::integrate_difference (dof_handler, lr_nonlinear_residue, ZeroFunction<dim>(dim+1), cellwise_errors, quadrature, VectorTools::L2_norm, &velocity_mask);
        const double residue_norm = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);
        return residue_norm;
    }
    //==========================================
    template <int dim>
    double StokesProblem<dim>::compute_l2norm_velocity()
    {
        const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
        const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0,dim), dim+1);
        Vector<double> cellwise_errors(triangulation.n_active_cells());
        QGauss<dim> quadrature(4);
        VectorTools::integrate_difference (dof_handler, lr_old_iterate_solution, ZeroFunction<dim>(dim+1), cellwise_errors, quadrature, VectorTools::L2_norm, &velocity_mask);
        const double velocity_l2norm = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);
        return velocity_l2norm;
    }
    //==================================================================
    template <int dim>
    void StokesProblem<dim>::movemesh()
    {
        pcout << "moving mesh..." << std::endl;
        lo_total_meshmove *= 0.5*deltat;
        lr_total_meshmove = lo_total_meshmove;
        
        std::vector<bool> vertex_touched(triangulation.n_vertices(), false);        
        Point<dim> vertex_displacement;
        const std::vector<bool> vertex_locally_moved = GridTools::get_locally_owned_vertices(triangulation);
        
        for (typename DoFHandler<dim>::active_cell_iterator  cell = move_dof_handler.begin_active(); cell != move_dof_handler.end(); ++cell)
        {
            if(cell->is_locally_owned())
                for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                    if (vertex_touched[cell->vertex_index(v)] == false)
                    {
                        vertex_touched[cell->vertex_index(v)] = true;
                        //                     for (unsigned int d=0; d<dim; ++d)
                        //                         vertex_displacement[d] = lr_total_meshmove(cell->vertex_dof_index(v,d));
                        vertex_displacement[0] = lr_total_meshmove(cell->vertex_dof_index(v,0));
                        vertex_displacement[1] = lr_total_meshmove(cell->vertex_dof_index(v,1));
                        vertex_displacement[2] = lr_total_meshmove(cell->vertex_dof_index(v,2));
                        cell->vertex(v) += vertex_displacement;
                    }
        }
        triangulation.communicate_locally_moved_vertices(vertex_locally_moved);
        lo_total_meshmove = 0;
//         pcout << "end of moving mesh..." << std::endl;
    }
    //===================================================================
    template <int dim>
    void StokesProblem<dim>::output_results(int timestepnumber)
    {
        //         TimerOutput::Scope t(computing_timer, "output");
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.emplace_back ("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (lr_solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
        
        Vector<float> subdomain (triangulation.n_active_cells());
        for (unsigned int i=0; i<subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector (subdomain, "subdomain");
        data_out.build_patches ();
        
        std::string filenamebase = "zfs-unstructured-";
        const std::string filename = (filenamebase + Utilities::int_to_string (timestepnumber, 3) + "." +Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
        std::ofstream output ((filename + ".vtu").c_str());
        data_out.write_vtu (output);
        
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            std::vector<std::string> filenames;
            for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
                filenames.push_back (filenamebase + Utilities::int_to_string (timestepnumber, 3) + "." + Utilities::int_to_string (i, 4) + ".vtu");
            
            std::ofstream master_output ((filenamebase + Utilities::int_to_string (timestepnumber, 3) + ".pvtu").c_str());
            data_out.write_pvtu_record (master_output, filenames);
        }
    }
    //==================================================================  
    template <int dim>
    void StokesProblem<dim>::timeloop()
    {      
        double timet = deltat;
        double totaltime = 20;
        int timestepnumber=0;
        
        while(timet<totaltime)
        {  
            pcout << "=====================" << std::endl;
            pcout << "timestep " << timestepnumber << ", time " << timet << std::endl;
            output_results(timestepnumber);
            pcout << "volume " << GridTools::volume(triangulation) << std::endl;
            assemble_stokessystem();
            solve_stokes();
            pcout <<"velocity_norm " << compute_l2norm_velocity() << std::endl;
            movemesh();
//             pcout <<"timet "<<timet <<std::endl;    
            timet+=deltat;
            timestepnumber++;
        } 
        output_results(timestepnumber);
    }
}  // end of namespace
//====================================================
int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace Navierstokes;
        
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        
        StokesProblem<3> flow_problem(1);    
        flow_problem.setup_movemeshsystem();
//         flow_problem.setup_stokessystem();
        flow_problem.timeloop();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }    
    return 0;
}

