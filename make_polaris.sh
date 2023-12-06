#!/bin/bash
module load nvhpc/23.1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/11.8/lib64:$LD_LIBRARY_PATH
# module use /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/comm_libs/hpcx/hpcx-2.13/modulefiles
# module load hpcx-ompi
make
