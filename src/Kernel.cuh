#pragma once
#include "HelpFunction.cuh"

#define checkCudaErrors(x) check((x), #x, __FILE__, __LINE__)
template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    }
}

__global__ void BaseKernel(uint8_t * d_query , uint8_t *  d_key , float *  d_LUT , float * d_store , DebugStruct * d_debug , int offset);


__global__ void StreamKernel(uint8_t * d_query , uint8_t *  d_key , float *  d_LUT , float * d_store , DebugStruct * d_debug , int offset , int BlockOffset);


__global__ void Fancykernel(uint8_t * d_query , uint8_t *  d_key , float *  d_LUT , float * d_store , DebugStruct * d_debug , int offset , int BlockOffset);

__global__ void MoreFancykernel(StreamStruct * d_IntergralStream , DebugStruct *  debugTool ,  int offset, int BlockOffset);