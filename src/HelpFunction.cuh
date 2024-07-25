#pragma once

#include "type.cuh"

__device__ void  CopyMap(float * d_LUT , float * SharedLUT , int BlockIdx);

__device__ void  CopyQ_K(u_int8_t * query, u_int8_t * key , u_int8_t * d_query , u_int8_t * d_key , int QVisitPoint , int KisitPoint);

__device__ void  Encoding(u_int8_t * query , u_int8_t * key , uint8_t * EncodingNumberQ ,uint8_t * EncodingNumberK);
    
__device__ void  LooKUpAndStore(float * d_store , u_int8_t EncodingNumberQ , uint8_t EncodingNumberK ,float * SharedLUT , int StorePosition);



__device__ void  KernelMapDebug(DebugStruct * d_debug , float * SharedLUT);
__device__ void  KernelqueryDebug(DebugStruct * d_debug, u_int8_t * query);
__device__ void  KernelkeyDebug(DebugStruct * d_debug, u_int8_t * key);

__device__ void KernelEncodingDebug( DebugStruct *d_debug ,u_int8_t  EncodingNumberQ ,u_int8_t EncodingNumberK);


__device__ void  FancyCopyMap(float * d_LUT , float * SharedLUT );


__device__ void  CorrectLookUpAndStore(float * d_store , uint8_t * d_query ,uint8_t * d_key , int StorePosition , int QVisitPoint , int KVisitPoint , float * d_LUT , int blockIdx); 









