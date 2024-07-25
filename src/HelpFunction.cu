#include "HelpFunction.cuh"

__device__ void  CopyMap(float * d_LUT , float * SharedLUT , int BlockIdx)
{
    int MapId = BlockIdx % nh;
    for(int i = 0 ; i < K * K ; i ++)
    {
        SharedLUT[i] = d_LUT[MapId * K * K + i];
    }
}



__device__ void  CopyQ_K(u_int8_t * query, u_int8_t * key , u_int8_t * d_query , u_int8_t * d_key , int QVisitPoint , int KisitPoint)
{
    for(int i = 0 ; i < dim ; i++)
    {
        query[i] = d_query[QVisitPoint + i];
    }
    for(int i = 0 ; i < dim ; i++)
    {
        key[i] = d_key[KisitPoint + i];
    }
    
}

__device__ void  Encoding(u_int8_t * query , u_int8_t * key , uint8_t * EncodingNumberQ ,uint8_t * EncodingNumberK)
{
    for(int i = 0 ; i < dim ; i ++)
    {
        *EncodingNumberQ += query[i];
    }
    *EncodingNumberQ /= 5;


    for(int i = 0 ; i < dim ; i ++)
    {
        *EncodingNumberK  += key[i];
    }
    *EncodingNumberK  /= 5 ; 

}




__device__ void  KernelMapDebug(DebugStruct * d_debug , float * SharedLUT)
{
    for(int i = 0 ;i < K * K ; i ++)
    {
        d_debug->LUT[i] = SharedLUT[i];
    }
}

__device__ void  KernelqueryDebug(DebugStruct * d_debug, u_int8_t * query)
{
    for(int i = 0 ;i < dim ; i ++)
    {
        d_debug->query[i] = query[i];
    }
}

__device__ void  KernelkeyDebug(DebugStruct * d_debug, u_int8_t * key)
{
     for(int i = 0 ;i < dim ; i ++)
    {
        d_debug->key[i] = key[i];
    }
}


 __device__ void KernelEncodingDebug( DebugStruct *d_debug ,u_int8_t  EncodingNumberQ ,u_int8_t EncodingNumberK)
 {
    d_debug->EncodingK = EncodingNumberK;
    d_debug->EncodingQ = EncodingNumberQ;
 }

 __device__ void  LooKUpAndStore(float * d_store , u_int8_t EncodingNumberQ , uint8_t EncodingNumberK ,float * SharedLUT , int StorePosition)
 {
    float temp = SharedLUT[EncodingNumberQ * K + EncodingNumberK];
    d_store[StorePosition] = temp;
    
 }


  __device__ void  FancyCopyMap(float * d_LUT , float * SharedLUT )
{
    for(int i = 0 ; i < K * K ; i ++)
    {
        SharedLUT[i] = d_LUT[i];
    }
}