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

void BrucksqrtGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks)
{
    *sendcount = (count / nranks) * nranks;
    *recvcount = (count / nranks) * nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = count / nranks;
}

testResult_t BrucksqrtInitData(struct threadArgs *args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place)
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
    // We don't support in-place brucksqrt
    args->reportErrors = in_place ? 0 : 1;
    return testSuccess;
}

void BrucksqrtGetBw(size_t count, int typesize, double sec, double *algBw, double *busBw, int nranks)
{
    double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

    *algBw = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)(nranks));
    *busBw = baseBw * factor;
}

int myPow(int x, unsigned int p)
{
    if (p == 0)
    {
        return 1;
    }
    else if (p == 1)
    {
        return x;
    }

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

testResult_t BrucksqrtRunColl(void *sendbuff, void *recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
    printf("NCCL 2.7 or later is needed for brucksqrt. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#else
    int nprocs, rank;
    NCCLCHECK(ncclCommCount(comm, &nprocs));
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    size_t unit_size = count * wordSize(type);

    char *c_sendbuff = (char *)sendbuff;
    char *c_recvbuff = (char *)recvbuff;
    int radix = std::ceil(std::sqrt(nprocs));
    int w = std::ceil(std::log(nprocs) / std::log(radix));

    int pows[w];
    for (int i = 0; i <= w; i++)
    {
        pows[i] = myPow(radix, i);
    }

    int nlpow = pows[w - 1];
    int d = (pows[w] - nprocs) / nlpow;

    CUDACHECK(cudaMemcpyAsync(c_recvbuff, c_sendbuff, nprocs * count, cudaMemcpyDeviceToDevice, stream));
    CUDACHECK(cudaMemcpyAsync(&c_sendbuff[(nprocs - rank) * count], c_recvbuff, rank * count, cudaMemcpyDeviceToDevice, stream));
    CUDACHECK(cudaMemcpyAsync(c_sendbuff, &c_recvbuff[rank * count], (nprocs - rank) * count, cudaMemcpyDeviceToDevice, stream));

    int *rank_r_reps = new int[nprocs * w * sizeof(int)];
    for (int i = 0; i < nprocs; i++)
    {
        std::vector<int> r_rep = convert10tob(w, i, radix);
        std::memcpy(&rank_r_reps[i * w], r_rep.data(), w * sizeof(int));
    }

    int sent_blocks[nlpow];
    int di = 0;
    int ci = 0;

    char *tempbuff;
    CUDACHECK(cudaMallocAsync((void **)&tempbuff, nlpow * count, stream));
    for (int x = 0; x < w; x++)
    {
        int ze = (x == w - 1) ? radix - d : radix;
        for (int z = 1; z < ze; z++)
        {
            di = 0;
            ci = 0;
            for (int i = 0; i < nprocs; i++)
            {
                if (rank_r_reps[i * w + x] == z)
                {
                    sent_blocks[di++] = i;
                    CUDACHECK(cudaMemcpyAsync(&tempbuff[count * ci++], &c_sendbuff[count * i], count, cudaMemcpyDeviceToDevice, stream));
                }
            }

            int distance = z * pows[x];
            int recv_proc = (rank - distance + nprocs) % nprocs;
            int send_proc = (rank + distance) % nprocs;
            long long comm_size = di * count;
            NCCLCHECK(ncclGroupStart());
            NCCLCHECK(ncclSend(tempbuff, comm_size, ncclChar, send_proc, comm, stream));
            NCCLCHECK(ncclRecv(c_recvbuff, comm_size, ncclChar, recv_proc, comm, stream));
            NCCLCHECK(ncclGroupEnd());

            for (int i = 0; i < di; i++)
            {
                long long offset = sent_blocks[i] * count;
                CUDACHECK(cudaMemcpyAsync(c_sendbuff + offset, c_recvbuff + (count * i), count, cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    for (int i = 0; i < nprocs; i++)
    {
        int index = (rank - i + nprocs) % nprocs;
        CUDACHECK(cudaMemcpyAsync(&c_recvbuff[count * index], &c_sendbuff[count * i], count, cudaMemcpyDeviceToDevice, stream));
    }

    CUDACHECK(cudaFreeAsync(tempbuff, stream));

    return testSuccess;
#endif
}

struct testColl brucksqrtTest = {
    "Brucksqrt",
    BrucksqrtGetCollByteCount,
    BrucksqrtInitData,
    BrucksqrtGetBw,
    BrucksqrtRunColl};

void BrucksqrtGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks)
{
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    BrucksqrtGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t BrucksqrtRunTest(struct threadArgs *args, int root, ncclDataType_t type, const char *typeName, ncclRedOp_t op, const char *opName)
{
    args->collTest = &brucksqrtTest;
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

struct testEngine brucksqrtEngine = {
    BrucksqrtGetBuffSize,
    BrucksqrtRunTest};

#pragma weak ncclTestEngine = brucksqrtEngine
