#!/bin/bash
for K in 2 4; do
    for NREF in 1 2; do
        for DR in 1e2 1e4 1e6 1e8; do
            for GAMMA in 1 100 10000; do
                CMD="python3 stokes.py --k $K --nref $NREF --gamma $GAMMA --dr $DR --solver-type almg --element sv"
                echo $CMD
                mpiexec -n 4 $CMD
            done
        done
    done
done

for K in 2; do
    for NREF in 1 2; do
        for DR in 1e2 1e4 1e6 1e8; do
            for GAMMA in 1 100 10000; do
                CMD="python3 stokes.py --k $K --nref $NREF --gamma $GAMMA --dr $DR --solver-type almg --element p2p0"
                echo $CMD
                mpiexec -n 4 $CMD
            done
        done
    done
done
