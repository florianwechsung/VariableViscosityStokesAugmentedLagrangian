from firedrake import *
from bary import BaryMeshHierarchy
from transfer import SchoeberlProlongation, NullTransfer

from functools import reduce

mesh = UnitSquareMesh(100, 100)
tdim = 2

nref = 1
gamma = Constant(1e3)

mh = BaryMeshHierarchy(mesh, nref)
mesh = mh[-1]


V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
Z = V * Q

X = SpatialCoordinate(mesh)
bcs = [DirichletBC(Z.sub(0), 0, "on_boundary")]

z = Function(Z)
u, p = split(z)
v, q = TestFunctions(Z)


omega = 0.1
delta = 200
dr = 1e8
mu_min = dr**-0.5
mu_max = dr**0.5

def Max(a, b): return (a+b+abs(a-b))/Constant(2)

def indi(ci):
    return 1-exp(-delta * Max(0, sqrt(inner(ci-X, ci-X))-omega/2)**2)

if tdim == 2:
    indis = [indi(Constant(((cx+1)/2, (cy+1)/2))) for cx in range(1) for cy in range(1)]
else:
    indis = [indi(Constant(((cx+1)/3, (cy+1)/3, (cz+1)/3))) for cx in range(2) for cy in range(2) for cz in range(2)]

chi_n = reduce(lambda x, y : x*y, indis, Constant(1.0))
mu = (mu_max-mu_min)*(1-chi_n) + mu_min
nu = Function(Q).interpolate(mu)


F = (
    nu * inner(grad(u), grad(v))*dx
    + gamma * inner(div(u), div(v))*dx
    - p * div(v) * dx
    - div(u) * q * dx
    - 10 * (chi_n-1)*v[tdim-1] * dx
)

appctx = {"nu": nu, "gamma": gamma}

fieldsplit_1 = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "schurcomplement.DGMassInv"
}

fieldsplit_0_lu = {
    "ksp_type": "preonly",
    "ksp_max_it": 1,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

fieldsplit_0_hypre = {
    "ksp_type": "richardson",
    "ksp_monitor": None,
    "ksp_max_it": 2,
    "pc_type": "hypre",
}

fieldsplit_0_mg = {
    "ksp_type": "cg",
    "ksp_max_it": 10,
    "pc_type": "mg",
}

outer = {
    "snes_type": "ksponly",
    "mat_type": "nest",
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-6,
    "ksp_atol": 1.0e-6,
    # "ksp_max_it": 500,
    "ksp_max_it": 10,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_0": fieldsplit_0_lu,
    # "fieldsplit_0": fieldsplit_0_mg,
    # "fieldsplit_0": fieldsplit_0_hypre,
    "fieldsplit_1": fieldsplit_1,
}
params = outer

problem = NonlinearVariationalProblem(F, z, bcs=bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters=params, options_prefix="ns_",
                                    appctx=appctx)

solver.solve()

File("u.pvd").write(z.split()[0])
File("p.pvd").write(z.split()[1])
File("nu.pvd").write(nu)
