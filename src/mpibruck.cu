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

void MpibruckGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks)
{
    *sendcount = (count / nranks) * nranks;
    *recvcount = (count / nranks) * nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = count / nranks;
}

testResult_t MpibruckInitData(struct threadArgs *args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place)
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

void MpibruckGetBw(size_t count, int typesize, double sec, double *algBw, double *busBw, int nranks)
{
    double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

    *algBw = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)(nranks));
    *busBw = baseBw * factor;
}

int myPow(int x, unsigned int p)
{
    if (p == 0)
        return 1;
    if (p == 1)
        return x;

    int tmp = myPow(x, p / 2);
    if (p % 2 == 0)
        return tmp * tmp;
    else
        return x * tmp * tmp;
}

testResult_t MpibruckRunColl(void *sendbuff, void *recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
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

    int r = std::ceil(std::sqrt(nprocs));

    int w = ceil(log(nprocs) / log(r)); // calculate the number of digits when using r-representation
    int nlpow = pow(r, w - 1);
    int d = (pow(r, w) - nprocs) / nlpow; // calculate the number of highest digits

    for (int i = 0; i < nprocs; i++)
    {
        int index = (2 * rank - i + nprocs) % nprocs;
        memcpy(h_recvbuff + (index * buffer_size), h_sendbuff + (i * buffer_size), buffer_size);
    }

    int sent_blocks[nlpow];
    int di = 0;
    int ci = 0;

    char *temp_buffer = (char *)malloc(nlpow * buffer_size); // temporary buffer
    int spoint = 1, distance = myPow(r, w - 1), next_distance = distance * r;
    for (int x = w - 1; x > -1; x--)
    {
        int ze = (x == w - 1) ? r - d : r;
        for (int z = ze - 1; z > 0; z--)
        {
            // get the sent data-blocks
            // copy blocks which need to be sent at this step
            di = 0;
            ci = 0;
            spoint = z * distance;
            for (int i = spoint; i < nprocs; i += next_distance)
            {
                for (int j = i; j < (i + distance); j++)
                {
                    if (j > nprocs - 1)
                    {
                        break;
                    }
                    int id = (j + rank) % nprocs;
                    sent_blocks[di++] = id;
                    memcpy(&temp_buffer[buffer_size * ci++], &h_recvbuff[id * buffer_size], buffer_size);
                }
            }

            // send and receive
            int recv_proc = (rank + spoint) % nprocs;          // receive data from rank - 2^step process
            int send_proc = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process
            long long comm_size = di * buffer_size;
            MPI_Sendrecv(temp_buffer, comm_size, MPI_BYTE, send_proc, 0, h_sendbuff, comm_size, MPI_BYTE, recv_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // replace with received data
            for (int i = 0; i < di; i++)
            {
                long long offset = sent_blocks[i] * buffer_size;
                memcpy(h_recvbuff + offset, h_sendbuff + (i * buffer_size), buffer_size);
            }
        }
        distance /= r;
        next_distance /= r;
    }

    CUDACHECK(cudaMemcpy(recvbuff, h_recvbuff, buffer_size * nprocs, cudaMemcpyHostToDevice));

    free(temp_buffer);
    delete[] h_sendbuff;
    delete[] h_recvbuff;

    return testSuccess;
#endif
}

struct testColl mpibruckTest = {
    "Mpibruck",
    MpibruckGetCollByteCount,
    MpibruckInitData,
    MpibruckGetBw,
    MpibruckRunColl};

void MpibruckGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks)
{
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    MpibruckGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t MpibruckRunTest(struct threadArgs *args, int root, ncclDataType_t type, const char *typeName, ncclRedOp_t op, const char *opName)
{
    args->collTest = &mpibruckTest;
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

struct testEngine mpibruckEngine = {
    MpibruckGetBuffSize,
    MpibruckRunTest};

#pragma weak ncclTestEngine = mpibruckEngine
