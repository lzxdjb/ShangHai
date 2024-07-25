#include "Kernel.cuh"
#include "HelpFunction.cuh"



__global__ void BaseKernel(uint8_t * d_query , uint8_t *  d_key , float *  d_LUT , float * d_store , DebugStruct *d_debug , int offset )
{
    int BlockIdx = blockIdx.x;
    int ThreadIdx = threadIdx.x;
    // int TotalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

    int Qposition = (ThreadIdx + offset) / klen;
    int Kposition = (ThreadIdx + offset) % klen;

 
    int QVisitPoint = BlockIdx * qlen * dim  + (Qposition) * dim;
    int KVisitPoint = BlockIdx * klen * dim  + (Kposition) * dim;

    int StorePosition =  BlockIdx * qlen * klen + (ThreadIdx + offset);

    // __shared__ float SharedLUT[K * K];
    // uint8_t query[dim];
    // uint8_t key[dim];

    // uint8_t EncodingNumberQ = 0;
    // uint8_t EncodingNumberK = 0;


    // printf("asdfasdfasdf\n");
    // CopyMap(d_LUT , SharedLUT , BlockIdx);
    // CopyQ_K(query , key , d_query , d_key , QVisitPoint , KVisitPoint);
    // Encoding(query , key , &EncodingNumberQ , &EncodingNumberK);

    // LooKUpAndStore(d_store , EncodingNumberQ , EncodingNumberK , SharedLUT , StorePosition);

    CorrectLookUpAndStore(d_store , d_query , d_key , StorePosition ,  QVisitPoint ,  KVisitPoint , d_LUT , BlockIdx);
    
    

    // KernelMapDebug(d_debug , SharedLUT);
    // KernelqueryDebug(d_debug , query);
    // KernelkeyDebug(d_debug , key);
    // KernelEncodingDebug(d_debug , EncodingNumberQ , EncodingNumberK);
}


__global__ void StreamKernel(uint8_t * d_query , uint8_t *  d_key , float *  d_LUT , float * d_store , DebugStruct *d_debug , int offset  , int BlockOffset)
{
    int BlockIdx = blockIdx.x + BlockOffset;
    int ThreadIdx = threadIdx.x;

    int Qposition = (ThreadIdx + offset) / klen;
    int Kposition = (ThreadIdx + offset) % klen;

 
    int QVisitPoint = BlockIdx * qlen * dim  + (Qposition) * dim;
    int KVisitPoint = BlockIdx * klen * dim  + (Kposition) * dim;

    int StorePosition =  BlockIdx * qlen * klen + (ThreadIdx + offset);

    // __shared__ float SharedLUT[K * K];
    // uint8_t query[dim];
    // uint8_t key[dim];

    // uint8_t EncodingNumberQ = 0;
    // uint8_t EncodingNumberK = 0;


    // // printf("asdfasdfasdf\n");
    // CopyMap(d_LUT , SharedLUT , BlockIdx);
    // CopyQ_K(query , key , d_query , d_key , QVisitPoint , KVisitPoint);
    // Encoding(query , key , &EncodingNumberQ , &EncodingNumberK);

    // LooKUpAndStore(d_store , EncodingNumberQ , EncodingNumberK , SharedLUT , StorePosition);
    

    CorrectLookUpAndStore(d_store , d_query , d_key , StorePosition ,  QVisitPoint ,  KVisitPoint , d_LUT , BlockIdx);
    
    

    // KernelMapDebug(d_debug , SharedLUT);
    // KernelqueryDebug(d_debug , query);
    // KernelkeyDebug(d_debug , key);
    // KernelEncodingDebug(d_debug , EncodingNumberQ , EncodingNumberK);
}




__global__ void Fancykernel(uint8_t *d_query, uint8_t *d_key, float *d_LUT, float *d_store, DebugStruct *d_debug, int offset, int BlockOffset)

{
    for (int i = 0; i < Fancyworkload; i++)
    {
        int BlockIdx = blockIdx.x * nh + BlockOffset;
        int ThreadIdx = threadIdx.x;
        // int TotalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

        int Qposition = (ThreadIdx + offset) / klen;
        int Kposition = (ThreadIdx + offset) % klen;

        int QVisitPoint = BlockIdx * qlen * dim + (Qposition)*dim;
        int KVisitPoint = BlockIdx * klen * dim + (Kposition)*dim;

        int StorePosition = BlockIdx * qlen * klen + (ThreadIdx + offset);

        __shared__ float SharedLUT[K * K];
        uint8_t query[dim];
        uint8_t key[dim];

        uint8_t EncodingNumberQ = 0;
        uint8_t EncodingNumberK = 0;

        // printf("asdfasdfasdf\n");
        FancyCopyMap(d_LUT, SharedLUT);
        CopyQ_K(query, key, d_query, d_key, QVisitPoint, KVisitPoint);
        Encoding(query, key, &EncodingNumberQ, &EncodingNumberK);

        LooKUpAndStore(d_store, EncodingNumberQ, EncodingNumberK, SharedLUT, StorePosition);

        KernelMapDebug(d_debug, SharedLUT);
        KernelqueryDebug(d_debug, query);
        KernelkeyDebug(d_debug, key);
        KernelEncodingDebug(d_debug, EncodingNumberQ, EncodingNumberK);
    }
}


__global__ void MoreFancykernel(StreamStruct * d_IntergralStream , DebugStruct *debugTool ,  int offset, int BlockOffset)
{
   
        int BlockIdx = blockIdx.x + BlockOffset;
        int ThreadIdx = threadIdx.x;
    

        int Qposition = (ThreadIdx + offset) / klen;
        int Kposition = (ThreadIdx + offset) % klen;

        int QVisitPoint = BlockIdx * qlen * dim + (Qposition)*dim;
        int KVisitPoint = BlockIdx * klen * dim + (Kposition)*dim;

        int StorePosition = BlockIdx * qlen * klen + (ThreadIdx + offset);

        // __shared__ float SharedLUT[K * K];
        uint8_t query[dim];
        uint8_t key[dim];

        uint8_t EncodingNumberQ = 0;
        uint8_t EncodingNumberK = 0;

        // printf("asdfasdfasdf\n");
    
        CopyQ_K(query, key, d_IntergralStream->query, d_IntergralStream->key, QVisitPoint, KVisitPoint);
        Encoding(query, key, &EncodingNumberQ, &EncodingNumberK);

        LooKUpAndStore(d_IntergralStream->Store, EncodingNumberQ, EncodingNumberK, d_IntergralStream->LUT, StorePosition);


        // KernelMapDebug(debugTool, d_IntergralStream->LUT);
        // KernelqueryDebug(d_debug, query);
        // KernelkeyDebug(d_debug, key);
        // KernelEncodingDebug(d_debug, EncodingNumberQ, EncodingNumberK);
    
}
