from firedrake import *
from firedrake.petsc import *
import weakref
from numpy import unique
import numpy
from pyop2.datatypes import IntType
from firedrake.mg.utils import *
from pyop2.profiling import timed_function

class CoarseCellMacroPatches(object):
    def __call__(self, pc):
        from firedrake.mg.utils import get_level
        from firedrake.mg.impl import get_entity_renumbering

        dmf = pc.getDM()
        ctx = pc.getAttr("ctx")

        mf = ctx.a.ufl_domain()
        (mh, level) = get_level(mf)

        coarse_to_fine_cell_map = mh.coarse_to_fine_cells[level-1]
        (_, firedrake_to_plex) = get_entity_renumbering(dmf, mf._cell_numbering, "cell")
        mc = mh[level-1]
        (_, coarse_firedrake_to_plex) = get_entity_renumbering(mc._plex, mc._cell_numbering, "cell")


        patches = []

        tdim = mf.topological_dimension()
        for i, fine_firedrake in enumerate(coarse_to_fine_cell_map):
            # there are d+1 many coarse cells that all map to the same fine cells.
            # We only want to build the patch once, so skip repitions
            if coarse_firedrake_to_plex[i]%(tdim+1) != 0:
                continue
            # we need to convert firedrake cell numbering to plex cell numbering
            fine_plex = [firedrake_to_plex[ff] for ff in fine_firedrake]
            entities = []
            for fp in fine_plex:
                (pts, _) = dmf.getTransitiveClosure(fp, True)
                for pt in pts:
                    value = dmf.getLabelValue("prolongation", pt)
                    if not (value > -1 and value <= level):
                        entities.append(pt)

            iset = PETSc.IS().createGeneral(unique(entities), comm=PETSc.COMM_SELF)
            patches.append(iset)

        piterset = PETSc.IS().createStride(size=len(patches), first=0, step=1, comm=PETSc.COMM_SELF)
        return (patches, piterset)


class SchoeberlProlongation(object):
    def __init__(self, nus, gamma, tdim):
        self.solver = {}
        self.bcs = {}
        self.rhs = {}
        self.tensors = {}
        self.nus = nus
        self.gamma = gamma

        patchparams = {"snes_type": "ksponly",
                       "ksp_type": "richardson",
                       "ksp_norm_type": "unpreconditioned",
                       "mat_type": "matfree",
                       "pc_type": "python",
                       "pc_python_type": "firedrake.PatchPC",
                       "patch_pc_patch_save_operators": "true",
                       "patch_pc_patch_partition_of_unity": False,
                       "patch_pc_patch_multiplicative": False,
                       "patch_pc_patch_sub_mat_type": "seqaij" if tdim > 2 else "seqdense",
                       "patch_pc_patch_construct_type": "python",
                       "patch_pc_patch_construct_python_type": "transfer.CoarseCellMacroPatches",
                       "patch_sub_ksp_type": "preonly",
                       "patch_sub_pc_type": "lu"}
        self.patchparams = patchparams

    def break_ref_cycles(self):
        for attr in ["solver", "bcs", "rhs", "tensors", "Re", "prev_Re",
                     "prev_gamma", "gamma"]:
            if hasattr(self, attr):
                delattr(self, attr)

    @staticmethod
    def fix_coarse_boundaries(V):
        hierarchy, level = get_level(V.mesh())
        dm = V.mesh()._plex

        section = V.dm.getDefaultSection()
        indices = []
        fStart, fEnd = dm.getHeightStratum(1)
        # Spin over faces, if the face is marked with a magic label
        # value, it means it was in the coarse mesh.
        for p in range(fStart, fEnd):
            value = dm.getLabelValue("prolongation", p)
            if value > -1 and value <= level:
                # OK, so this is a coarse mesh face.
                # Grab all the points in the closure.
                closure, _ = dm.getTransitiveClosure(p)
                for c in closure:
                    # Now add all the dofs on that point to the list
                    # of boundary nodes.
                    dof = section.getDof(c)
                    off = section.getOffset(c)
                    for d in range(dof):
                        indices.append(off + d)
        nodelist = unique(indices).astype(IntType)

        class FixedDirichletBC(DirichletBC):
            def __init__(self, V, g, nodelist):
                self.nodelist = nodelist
                DirichletBC.__init__(self, V, g, "on_boundary")

            @utils.cached_property
            def nodes(self):
                return self.nodelist

        dim = V.mesh().topological_dimension()
        bc = FixedDirichletBC(V,  dim * (0, ), nodelist)

        return bc
    @timed_function("SchoeberlProlong")
    def prolong(self, coarse, fine):
        # Rebuild without any indices
        V = FunctionSpace(fine.ufl_domain(), fine.function_space().ufl_element())
        key = V.dim()

        firsttime = self.bcs.get(key, None) is None

        # prev_gamma = self.prev_gamma.get(key, None)

        gamma = self.gamma

        _, level = get_level(fine.function_space().mesh())
        nu = self.nus[level]
        if firsttime:
            bcs = self.fix_coarse_boundaries(V)
            u = TrialFunction(V)
            v = TestFunction(V)
            A = assemble(nu * inner(grad(u), grad(v))*dx + gamma*inner(div(u), div(v))*dx, bcs=bcs, mat_type=self.patchparams["mat_type"])

            tildeu, rhs = Function(V), Function(V)

            bform = nu * inner(grad(rhs), grad(v))*dx + gamma*inner(div(rhs), div(v))*dx
            b = Function(V)

            solver = LinearSolver(A, solver_parameters=self.patchparams,
                                  options_prefix="prolongation")
            self.bcs[key] = bcs
            self.solver[key] = solver
            self.rhs[key] = tildeu, rhs
            self.tensors[key] = A, b, bform
        else:
            bcs = self.bcs[key]
            solver = self.solver[key]
            A, b, bform = self.tensors[key]
            tildeu, rhs = self.rhs[key]

        # # Update operator if parameters have changed.
        # if float(self.Re) != prev_Re or float(self.gamma) != prev_gamma:
        #     A = solver.A
        #     A = assemble(A.a, bcs=bcs, mat_type=self.patchparams["mat_type"], tensor=A)
        #     A.force_evaluation()
        #     self.prev_Re[key] = float(self.Re)
        #     self.prev_gamma[key] = float(self.gamma)

        prolong(coarse, rhs)

        b = assemble(bform, bcs=bcs, tensor=b)
        # # Could do
        # #solver.solve(tildeu, b)
        # # but that calls a lot of SNES and KSP overhead.
        # # We know we just want to apply the PC:
        with solver.inserted_options():
            with b.dat.vec_ro as rhsv:
                with tildeu.dat.vec_wo as x:
                    solver.ksp.pc.apply(rhsv, x)
        fine.assign(rhs - tildeu)

        # def energy_norm(u):
        #     return assemble(nu * inner(grad(u), grad(u)) * dx + gamma * inner(div(u), div(u)) * dx)
        # def H1_norm(u):
        #     return assemble(nu * inner(grad(u), grad(u)) * dx)

        # warning("\|coarse\| %.10f " % energy_norm(coarse))
        # warning("\|coarse\|_1 %f " % H1_norm(coarse))
        # warning("\|fine\| %.10f" % energy_norm(fine))
        # warning("\|fine\|_1 %f" % H1_norm(fine))
        # warning("\|tildeu\| %.10f" % energy_norm(tildeu))
        # warning("\|rhs\| %.10f" % energy_norm(rhs))
        # import sys; sys.exit(1)


class NullTransfer(object):
    def transfer(self, src, dest):
        with dest.dat.vec_wo as x:
            x.set(numpy.nan)

    inject = transfer
    prolong = transfer
    restrict = transfer
