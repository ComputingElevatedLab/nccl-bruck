#!/bin/bash

module load nvhpc/23.1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/11.8/lib64:$LD_LIBRARY_PATH
# module use /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/comm_libs/hpcx/hpcx-2.13/modulefiles
# module load hpcx-ompi

export NCCL_PLUGIN_P2P=ucx

NODES=`wc -l < $PBS_NODEFILE`
let RANKS=4*$NODES
let MIN_BYTES=16*$RANKS
let MAX_BYTES=2048*$RANKS

export btl_base_verbose=100

echo ------NCCL Bruck------
mpiexec -n $RANKS -ppn 4 ./build/modbruck_perf -a 3 -d int8 -c 0 -b $MIN_BYTES -e $MAX_BYTES -f 2 -g 1 -n 500 -w 100

