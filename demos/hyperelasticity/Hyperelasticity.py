# Import some useful modules.
import jax
import jax.numpy as np
import os
import meshio
import pandas as pd


# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh, rectangle_mesh


# Define constitutive relationship.
class HyperElasticity(Problem):
    def custom_init(self):
        # Override base class method.
        self.fe = self.fes[0]
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.
    def get_tensor_map(self):

        def psi(F):
            J = np.linalg.det(F)
            C = F.T @ F
            JC = J*C
            eigval = np.linalg.eigvalsh(JC)
            eigval = np.clip(eigval, 1e-8, None)
            lambdas = np.sqrt(eigval)
            lambdas = np.clip(lambdas, a_min=1e-12) 
            energy = (lambdas[0]**2+lambdas[1]**2+lambdas[2]**2 -3)
            # I1 = np.trace(C)
            # energy = (I1-3)
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress


# Specify mesh-related information (first-order hexahedron element).
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')

# read mesh
# meshio_mesh = meshio.read('test.inp')
# points = meshio_mesh.points[:, :2]  # Knotenpunkte
# cells = meshio_mesh.cells_dict  # Zelltypen und Indizes
# mesh = Mesh(points, meshio_mesh.cells_dict[cell_type])

ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh_gmsh(Nx=20,
                       Ny=20,
                       Nz=20,
                       Lx=Lx,
                       Ly=Ly,
                       Lz=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# Define boundary locations.
def bottom(point):
    return np.isclose(point[1], 0., atol=1e-2)


def top(point):
    return np.isclose(point[1], 100, atol=1e-2)


# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.


def get_dirichlet_top(disp):
    def val_fn(point):
        return disp
    return val_fn


disps = np.hstack(np.linspace(10, 200, 20))

location_fns = [top, top, top, bottom, bottom, bottom]
vec = [0, 1, 2, 0, 1, 2]
value_fns = [zero_dirichlet_val, get_dirichlet_top(disps[0]), zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]

dirichlet_bc_info = [location_fns,
                     vec,
                     value_fns]


# Create an instance of the problem.
problem = HyperElasticity(mesh,
                          vec=3,
                          dim=3,
                          ele_type=ele_type,
                          dirichlet_bc_info=dirichlet_bc_info)

dofs_previous = np.zeros(problem.fes[0].num_total_dofs)

sol_list = list()
for i, disp in enumerate(disps):
    print(f"\nStep {i} in {len(disps)}, disp = {disp}")
    dirichlet_bc_info[-1][1] = get_dirichlet_top(disp)
    problem.fe.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    sol, dofs_converged = solver(problem, solver_options = {'petsc_solver': {}, 'inital_guess':dofs_previous})
    sol_list.append(sol)
    vtk_path = os.path.join(data_dir, f'vtk/u_{i:03d}.vtu')
    save_sol(problem.fe, sol[0], vtk_path)
    dofs_previous = dofs_converged

u_list = list(np.zeros(3*3*2))
for sol in sol_list:
    u_list.append(np.hstack([sol[0][6], sol[0][7], sol[0][8], sol[0][11], sol[0][12], sol[0][13], sol[0][16], sol[0][17], sol[0][18]]))
u = np.hstack(u_list)
u_DIC_df = pd.DataFrame(u)
u_DIC_df.to_csv('u_DIC.csv', index=None, header=False)
# Solve the defined problem.    
# sol_list = solver(problem, solver_options={'petsc_solver': {}})

#data_dir = os.path.join(os.path.dirname(__file__), 'data')
#vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
#save_sol(problem.fes[0], sol_list[0], vtk_path)