/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "common.h"
#include "cuda_runtime.h"
#include <mpi.h>

void MpispreadoutGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks)
{
    *sendcount = (count / nranks) * nranks;
    *recvcount = (count / nranks) * nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = count / nranks;
}

testResult_t MpispreadoutInitData(struct threadArgs *args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place)
{
    size_t sendcount = args->sendBytes / wordSize(type);
    size_t recvcount = args->expectedBytes / wordSize(type);
    int nranks = args->nProcs * args->nThreads * args->nGpus;

    for (int i = 0; i < args->nGpus; i++)
    {
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        int rank = ((args->proc * args->nThreads + args->thread) * args->nGpus + i);
        CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
        void *data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
        TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33 * rep + rank, 1, 0));
        for (int j = 0; j < nranks; j++)
        {
            size_t partcount = sendcount / nranks;
            TESTCHECK(InitData((char *)args->expected[i] + j * partcount * wordSize(type), partcount, rank * partcount, type, ncclSum, 33 * rep + j, 1, 0));
        }
        CUDACHECK(cudaDeviceSynchronize());
    }
    args->reportErrors = in_place ? 0 : 1;
    return testSuccess;
}

void MpispreadoutGetBw(size_t count, int typesize, double sec, double *algBw, double *busBw, int nranks)
{
    double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

    *algBw = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)(nranks));
    *busBw = baseBw * factor;
}

testResult_t MpispreadoutRunColl(void *sendbuff, void *recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
    printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#else
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int buffer_size = count * wordSize(type);

    char *h_sendbuff = new char[buffer_size * nprocs];
    char *h_recvbuff = new char[buffer_size * nprocs];

    CUDACHECK(cudaMemcpy(h_sendbuff, sendbuff, buffer_size, cudaMemcpyDeviceToHost));

    MPI_Request *req = (MPI_Request *)malloc(2 * nprocs * sizeof(MPI_Request));
    MPI_Status *stat = (MPI_Status *)malloc(2 * nprocs * sizeof(MPI_Status));
    for (int i = 0; i < nprocs; i++)
    {
        int src = (rank + i) % nprocs; // avoid always to reach first master node
        MPI_Irecv(&h_recvbuff[src * buffer_size], buffer_size, MPI_BYTE, src, 0, MPI_COMM_WORLD, &req[i]);
    }

    for (int i = 0; i < nprocs; i++)
    {
        int dst = (rank - i + nprocs) % nprocs;
        MPI_Isend(&h_sendbuff[dst * buffer_size], buffer_size, MPI_BYTE, dst, 0, MPI_COMM_WORLD, &req[i + nprocs]);
    }

    MPI_Waitall(2 * nprocs, req, stat);

    CUDACHECK(cudaMemcpy(recvbuff, h_recvbuff, buffer_size * nprocs, cudaMemcpyHostToDevice));

    free(req);
    free(stat);
    delete[] h_sendbuff;
    delete[] h_recvbuff;

    return testSuccess;
#endif
}

struct testColl mpispreadoutTest = {
    "Mpispreadout",
    MpispreadoutGetCollByteCount,
    MpispreadoutInitData,
    MpispreadoutGetBw,
    MpispreadoutRunColl};

void MpispreadoutGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks)
{
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    MpispreadoutGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t MpispreadoutRunTest(struct threadArgs *args, int root, ncclDataType_t type, const char *typeName, ncclRedOp_t op, const char *opName)
{
    args->collTest = &mpispreadoutTest;
    ncclDataType_t *run_types;
    const char **run_typenames;
    int type_count;

    if ((int)type != -1)
    {
        type_count = 1;
        run_types = &type;
        run_typenames = &typeName;
    }
    else
    {
        type_count = test_typenum;
        run_types = test_types;
        run_typenames = test_typenames;
    }

    for (int i = 0; i < type_count; i++)
    {
        TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
    }
    return testSuccess;
}

struct testEngine mpispreadoutEngine = {
    MpispreadoutGetBuffSize,
    MpispreadoutRunTest};

#pragma weak ncclTestEngine = mpispreadoutEngine
