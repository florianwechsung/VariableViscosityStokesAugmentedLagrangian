from firedrake import *
from bary import BaryMeshHierarchy
from transfer import SchoeberlProlongation, NullTransfer

from functools import reduce
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--k", type=int, default=4)
parser.add_argument("--solver-type", type=str, default="almg")
parser.add_argument("--gamma", type=float, default=1e4)
parser.add_argument("--dr", type=float, default=1e8)
args, _ = parser.parse_known_args()


nref = args.nref
dr = args.dr
k = args.k
gamma = Constant(args.gamma)

mesh = RectangleMesh(20, 20, 4, 4)
def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)

mh = BaryMeshHierarchy(mesh, nref, callbacks=(before, after), reorder=True,
                       distribution_parameters={"partition": True,
                                                "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)})
mesh = mh[-1]


V = VectorFunctionSpace(mesh, "CG", k)
Q = FunctionSpace(mesh, "DG", k-1)
Z = V * Q
z = Function(Z)
u, p = split(z)
v, q = TestFunctions(Z)
bcs = [DirichletBC(Z.sub(0), 0, "on_boundary")]


omega = 0.1
delta = 200
mu_min = dr**-0.5
mu_max = dr**0.5

def Max(a, b): return (a+b+abs(a-b))/Constant(2)

def chi_n(mesh):
    X = SpatialCoordinate(mesh)
    def indi(ci):
        return 1-exp(-delta * Max(0, sqrt(inner(ci-X, ci-X))-omega/2)**2)

    indis = [indi(Constant((4*(cx+1)/3, 4*(cy+1)/3))) for cx in range(2) for cy in range(2)]

    return reduce(lambda x, y : x*y, indis, Constant(1.0))

def mu_expr(mesh):
    return (mu_max-mu_min)*(1-chi_n(mesh)) + mu_min

def mu(mesh):
    Q = FunctionSpace(mesh, "DG", k-1)
    return Function(Q).interpolate(mu_expr(mesh))

mus = [mu(m) for m in mh]
mu = mus[-1]

F = (
    mu_expr(mesh) * inner(grad(u), grad(v))*dx
    + gamma * inner(div(u), div(v))*dx
    - p * div(v) * dx
    - div(u) * q * dx
    - 10 * (chi_n(mesh)-1)*v[1] * dx
)

appctx = {"nu": mu, "gamma": gamma}

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

mg_levels_solver = {
    "ksp_type": "fgmres",
    "ksp_norm_type": "unpreconditioned",
    "ksp_max_it": 5,
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": False,
    "patch_pc_patch_sub_mat_type": "aij",
    "patch_pc_patch_local_type": "multiplicative",
    "patch_pc_patch_statistics": False,
    "patch_pc_patch_symmetrise_sweep": True,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
    "patch_sub_pc_factor_mat_solver_type": "mumps",
    "patch_pc_patch_construct_type": "python",
    "patch_pc_patch_construct_python_type": "relaxation.MacroStar",
}

fieldsplit_0_mg = {
    "ksp_type": "richardson",
    "ksp_richardson_self_scale": False,
    "ksp_max_it": 2,
    "ksp_norm_type": "unpreconditioned",
    "ksp_convergence_test": "skip",
    "pc_type": "mg",
    "pc_mg_type": "full",
    "mg_levels": mg_levels_solver,
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
}

outer = {
    "snes_type": "ksponly",
    "mat_type": "nest",
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-6,
    "ksp_atol": 1.0e-10,
    "ksp_max_it": 100,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_1": fieldsplit_1,
}
if args.solver_type == "almg":
    outer["fieldsplit_0"] = fieldsplit_0_mg
elif args.solver_type == "allu":
    outer["fieldsplit_0"] = fieldsplit_0_lu
elif args.solver_type == "alamg":
    outer["fieldsplit_0"] = fieldsplit_0_hypre
else:
    raise ValueError("please specify almg, allu or alamg for --solver-type")
params = outer

MVSB = MixedVectorSpaceBasis
nsp = MVSB(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
problem = NonlinearVariationalProblem(F, z, bcs=bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters=params, options_prefix="ns_",
                                    appctx=appctx, nullspace=nsp)

prolongation = SchoeberlProlongation(mus, gamma, 2)
injection = NullTransfer()
solver.set_transfer_operators(dmhooks.transfer_operators(V, prolong=prolongation.prolong),
                              dmhooks.transfer_operators(Q, inject=injection.inject))

solver.solve()

File("u.pvd").write(z.split()[0])
File("p.pvd").write(z.split()[1])
File("nu.pvd").write(mu)
