# An Augmented Lagrangian preconditioner for variable viscosity stokes

We solve a similar problem as _Weighted BFBT Preconditioner for Stokes Flow Problems with Highly Heterogeneous Viscosity_ by Johann Rudi, Georg Stadler, and Omar Ghattas.
The test case implemented considers 4 sinkers in a 2D domain with homogeneous boundary terms on the side. The sinkers are pulled down by gravity, resulting in this picture:

_Viscosity_

![Viscosity](https://i.imgur.com/6XdbJF5.png)

_Velocity_

![Velocity](https://i.imgur.com/lNnwHCb.png)


# Installation and execution

Install firedrake via 

    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    python3 firedrake-install

and then run 

    mpirun -n 4 python3 stokes.py --k 4 --nref 1 --gamma 1e4 --dr 1e6 --solver-type almg

The options are

+ `--k` order of velocity approximation. We use a `CG(k)-DG(k-1)` Scott-Vogelius element. We need k>=2.
+ `--nref` number of multigrid refinements
+ `--gamma` value of gamma for augmentation
+ `--dr` orders of magnitude between high and low viscosity
+ `--solver-type`: use `allu` to solve top-left using LU, `alamg` to solve top-left block using hypre, and `almg` to solve top-left using custom geometric multigrid scheme.


