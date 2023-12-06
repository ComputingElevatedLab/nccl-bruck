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

void ModBruckGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks)
{
    *sendcount = (count / nranks) * nranks;
    *recvcount = (count / nranks) * nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = count / nranks;
}

testResult_t ModBruckInitData(struct threadArgs *args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place)
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
    // We don't support in-place modified bruck
    args->reportErrors = in_place ? 0 : 1;
    return testSuccess;
}

void ModBruckGetBw(size_t count, int typesize, double sec, double *algBw, double *busBw, int nranks)
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
    {
        return tmp * tmp;
    }
    else
    {
        return x * tmp * tmp;
    }
}

std::vector<int> convert10tob(int w, int N, int b)
{
    std::vector<int> v(w);
    int i = 0;
    while (N)
    {
        v[i++] = (N % b);
        N /= b;
    }
    return v;
}

testResult_t ModBruckRunColl(void *sendbuff, void *recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
    printf("NCCL 2.7 or later is needed for modified bruck. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#else
    char *c_sendbuff = (char *)sendbuff;
    char *c_recvbuff = (char *)recvbuff;

    int nprocs, rank;
    NCCLCHECK(ncclCommCount(comm, &nprocs));
    NCCLCHECK(ncclCommUserRank(comm, &rank));

    int radix = std::ceil(std::sqrt(nprocs));

    size_t unit_size = count * wordSize(type);
    int w = std::ceil(std::log(nprocs) / std::log(radix));
    int nlpow = std::pow(radix, w - 1);
    int d = (std::pow(radix, w) - nprocs) / nlpow;

    for (int i = 0; i < nprocs; i++)
    {
        int index = (2 * rank - i + nprocs) % nprocs;
        CUDACHECK(cudaMemcpyAsync(c_recvbuff + (index * unit_size), c_sendbuff + (i * unit_size), unit_size, cudaMemcpyDeviceToDevice, stream));
    }

    // CUDACHECK(cudaMemcpyAsync(&c_sendbuff[(nprocs - rank) * count], c_recvbuff, rank * count, cudaMemcpyDeviceToDevice, stream));

    int sent_blocks[nlpow];
    int di = 0;
    int ci = 0;

    char *tempbuff;
    CUDACHECK(cudaMallocAsync((void **)&tempbuff, nlpow * unit_size, stream));
    int spoint = 1;
    int distance = myPow(radix, w - 1);
    int next_distance = distance * radix;
    for (int x = w - 1; x > -1; x--)
    {
        int ze = (x == w - 1) ? radix - d : radix;
        for (int z = ze - 1; z > 0; z--)
        {
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
                    CUDACHECK(cudaMemcpyAsync(&tempbuff[unit_size * ci++], &c_recvbuff[id * unit_size], unit_size, cudaMemcpyDeviceToDevice, stream));
                }
            }

            int recv_proc = (rank + spoint) % nprocs;
            int send_proc = (rank - spoint + nprocs) % nprocs;
            long long comm_size = di * unit_size;
            NCCLCHECK(ncclGroupStart());
            NCCLCHECK(ncclSend(tempbuff, comm_size, ncclChar, send_proc, comm, stream));
            NCCLCHECK(ncclRecv(c_sendbuff, comm_size, ncclChar, recv_proc, comm, stream));
            NCCLCHECK(ncclGroupEnd());

            for (int i = 0; i < di; i++)
            {
                long long offset = sent_blocks[i] * unit_size;
                CUDACHECK(cudaMemcpyAsync(c_recvbuff + offset, c_sendbuff + (unit_size * i), unit_size, cudaMemcpyDeviceToDevice, stream));
            }
        }
        distance /= radix;
        next_distance /= radix;
    }

    CUDACHECK(cudaFreeAsync(tempbuff, stream));

    return testSuccess;
#endif
}

struct testColl modbruckTest = {
    "ModifiedBruck",
    ModBruckGetCollByteCount,
    ModBruckInitData,
    ModBruckGetBw,
    ModBruckRunColl};

void ModBruckGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks)
{
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    ModBruckGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t ModBruckRunTest(struct threadArgs *args, int root, ncclDataType_t type, const char *typeName, ncclRedOp_t op, const char *opName)
{
    args->collTest = &modbruckTest;
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

struct testEngine modbruckEngine = {
    ModBruckGetBuffSize,
    ModBruckRunTest};

#pragma weak ncclTestEngine = modbruckEngine
