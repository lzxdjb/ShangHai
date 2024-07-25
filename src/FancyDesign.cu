#include "FancyDesign.cuh"

void FancyAllocation(uint8_t *query, uint8_t *key, float *LUT, float *store)
{

    u_int8_t *d_query;
    u_int8_t *d_key;
    float *d_LUT;
    float *d_store;

    DebugStruct *h_debugTool;
    DebugStruct *d_debugTool;

    ///// Debug
    h_debugTool = (DebugStruct *)malloc(sizeof(DebugStruct));
    checkCudaErrors(cudaMalloc((void **)&d_debugTool, sizeof(DebugStruct)));

    checkCudaErrors(cudaMemcpy(d_debugTool, h_debugTool, sizeof(DebugStruct), cudaMemcpyHostToDevice));
    //////

    //// init
    checkCudaErrors(cudaMalloc((void **)&d_query, sizeof(uint8_t) * bs * nh * qlen * dim));
    checkCudaErrors(cudaMalloc((void **)&d_key, sizeof(uint8_t) * bs * nh * klen * dim));

    ///// change here
    checkCudaErrors(cudaMalloc((void **)&d_LUT, sizeof(float) * K * K));
    /////

    checkCudaErrors(cudaMalloc((void **)&d_store, sizeof(float) * bs * nh * qlen * klen));

    float ms; // elapsed time in milliseconds
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    checkCudaErrors(cudaEventCreate(&startEvent));
    checkCudaErrors(cudaEventCreate(&stopEvent));
    checkCudaErrors(cudaEventCreate(&dummyEvent));

    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
    {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

    //// begin
    checkCudaErrors(cudaEventRecord(startEvent, 0));

    for (int streamNumber = 0; streamNumber < nStreams; streamNumber++)
    {
        int LUTOffset = streamNumber * K * K;
        checkCudaErrors(cudaMemcpyAsync(d_LUT, &LUT[LUTOffset], (K * K) * sizeof(float), cudaMemcpyHostToDevice, stream[streamNumber]));

        int OutOffset = 0;

        for (int b = 0; b < bs; b += Fancyworkload)
        {
            for (int w = 0; w < Fancyworkload; w++)
            {
                int QOffset = ((b + w) * nh + streamNumber) * qlen * dim;
                checkCudaErrors(cudaMemcpyAsync(&d_query[QOffset], &query[QOffset], (qlen * dim) * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[streamNumber]));

                int KOffset = ((b + w) * nh + streamNumber) * klen * dim;
                checkCudaErrors(cudaMemcpyAsync(&d_key[KOffset], &key[KOffset], (klen * dim) * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[streamNumber]));
            }

            int BestThread = 1024;
            int j = 0;
            int offset = 0;

            OutOffset = b * nh + streamNumber;

            for (j = qlen * klen; j > BestThread; j -= BestThread)
            {
                // cout<<"j = "<<j<<endl;
                // std::cout<<"CNddddM"<<endl;
                Fancykernel<<<Fancyworkload, BestThread, 0, stream[streamNumber]>>>(d_query, d_key, d_LUT, d_store, d_debugTool, offset, OutOffset);
                offset += BestThread;
            }
            Fancykernel<<<Fancyworkload, j, 0, stream[streamNumber]>>>(d_query, d_key, d_LUT, d_store, d_debugTool, offset, OutOffset);

            //// @@@@@

            for (int w = 0; w < Fancyworkload; w++)
            {
                int StoreOffset = ((b + w) * nh + streamNumber) * klen * qlen;

                checkCudaErrors(cudaMemcpyAsync(&store[StoreOffset], &d_store[StoreOffset], (qlen * klen) * sizeof(float), cudaMemcpyDeviceToHost, stream[streamNumber]));
            }
        }
    }

    checkCudaErrors(cudaEventRecord(stopEvent, 0));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    ///// End Time Record
    printf("Time for Fancy execute time (ms): %f\n", ms);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    ////// Debug
    // checkCudaErrors(cudaMemcpy(h_debugTool, d_debugTool, sizeof(DebugStruct), cudaMemcpyDeviceToHost));

    // checkCudaErrors(cudaDeviceSynchronize());
    // debug(h_debugTool);

    // Debug2K(key);
    // Debug2Q(query);
    // Debug2LUT(LUT);
    // Debug2Store(store);

    ///// DebugEnd
}

void MoreFancy(uint8_t *query, uint8_t *key, float *LUT, float *store)
{

    // StreamStruct *IntegralStream;
    // checkCudaErrors(cudaMallocHost((void **)&IntegralStream, sizeof(DebugStruct) * 4));
    StreamStruct *d_IntergralStream;
    checkCudaErrors(cudaMalloc((void **)&d_IntergralStream, sizeof(StreamStruct) * nStreams));

    // std::cout<<"sizeof debug = "<<sizeof(DebugStruct)<<endl;
    // exit(0);
    // checkCudaErrors( cudaMallocHost((void**)&IntegralStream, sizeof(StreamStruct) * nStreams) ) ;
    // checkCudaErrors(cudaMalloc((void **)&IntegralStream, sizeof(DebugStruct) * nStreams));

    // u_int8_t *d_query;
    // u_int8_t *d_key;
    // float *d_LUT;
    ///// change here
    // checkCudaErrors(cudaMalloc((void **)&d_LUT, sizeof(float) * K * K));
    /////
    // float *d_store;

    DebugStruct *h_debugTool;
    DebugStruct *d_debugTool;

    //// Debug
    h_debugTool = (DebugStruct *)malloc(sizeof(DebugStruct));
    checkCudaErrors(cudaMalloc((void **)&d_debugTool, sizeof(DebugStruct)));

    checkCudaErrors(cudaMemcpy(d_debugTool, h_debugTool, sizeof(DebugStruct), cudaMemcpyHostToDevice));
    ////

    //// init
    // checkCudaErrors(cudaMalloc((void **)&d_query, sizeof(uint8_t) * bs * nh * qlen * dim));
    // checkCudaErrors(cudaMalloc((void **)&d_key, sizeof(uint8_t) * bs * nh * klen * dim));

  

    // checkCudaErrors(cudaMalloc((void **)&d_store, sizeof(float) * bs * nh * qlen * klen));

    float ms; // elapsed time in milliseconds
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    checkCudaErrors(cudaEventCreate(&startEvent));
    checkCudaErrors(cudaEventCreate(&stopEvent));
    checkCudaErrors(cudaEventCreate(&dummyEvent));
    checkCudaErrors( cudaEventRecord(startEvent,0) );

    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
    {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

    //// begin
    // checkCudaErrors(cudaEventRecord(startEvent, 0));

    for (int streamNumber = 0; streamNumber < nStreams; streamNumber++)
    {
        // int LUTOffset = streamNumber * K * K;
        // checkCudaErrors(cudaMemcpyAsync(d_LUT, &LUT[LUTOffset], (K * K) * sizeof(float), cudaMemcpyHostToDevice, stream[streamNumber]));

        int LUTOffset = streamNumber * K * K;
        checkCudaErrors(cudaMemcpyAsync(d_IntergralStream[streamNumber].LUT, &LUT[LUTOffset], (K * K) * sizeof(float), cudaMemcpyHostToDevice, stream[streamNumber]));


        for (int b = 0; b < bs; b++)
        {
            int QOffset = (b * nh + streamNumber) * qlen * dim;
            int queryOffset = b * qlen * dim;
            checkCudaErrors(cudaMemcpyAsync(&d_IntergralStream[streamNumber].query[queryOffset], &query[QOffset], (qlen * dim) * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[streamNumber]));

            int KOffset = (b * nh + streamNumber) * klen * dim;
            int keyOffset = b * klen * dim;
            checkCudaErrors(cudaMemcpyAsync(&d_IntergralStream[streamNumber].key[keyOffset], &key[KOffset], (klen * dim) * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[streamNumber]));

            // exit(0);
        }
    }

    for (int streamNumber = 0; streamNumber < nStreams; streamNumber++)
    {

        for (int b = 0; b < bs; b += Fancyworkload)
        {
            int BestThread = 1024;
            int j = 0;
            int offset = 0;
            // int OutOffset = b * nh + streamNumber;

            for (j = qlen * klen; j > BestThread; j -= BestThread)
            {
                // cout<<"j = "<<j<<endl;
                // std::cout<<"CNddddM"<<endl;
                MoreFancykernel<<<Fancyworkload, BestThread, 0, stream[streamNumber]>>>(&d_IntergralStream[streamNumber] , d_debugTool , offset, b);
                offset += BestThread;
            }
            MoreFancykernel<<<Fancyworkload, j, 0, stream[streamNumber]>>>(&d_IntergralStream[streamNumber] , d_debugTool  , offset, b);
        }
    }


    // exit(0);
    for (int streamNumber = 0; streamNumber < nStreams; streamNumber++)
    {
        for (int b = 0; b < bs; b++)
        {
         
            int SOffset = (b * nh + streamNumber) * klen * qlen;
            int storeOffset = b * klen * qlen;

/////!!!!!!!
            checkCudaErrors(cudaMemcpyAsync(&store[SOffset], &d_IntergralStream[streamNumber].Store[storeOffset], (qlen * klen) * sizeof(float), cudaMemcpyDeviceToHost, stream[streamNumber]));

       
        }
    }

    checkCudaErrors(cudaEventRecord(stopEvent, 0));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    // ///// End Time Record
    printf("Time for MoreFancy execute time (ms): %f\n", ms);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaFree(d_debugTool);
    cudaFree(d_IntergralStream);
  
    // ////// Debug
    // checkCudaErrors(cudaMemcpy(h_debugTool, d_debugTool, sizeof(DebugStruct), cudaMemcpyDeviceToHost));

    // checkCudaErrors(cudaDeviceSynchronize());
    // debug(h_debugTool);

    // Debug2K(key);
    // Debug2Q(query);
    // Debug2LUT(LUT);
    // Debug2Store(store);

    ///// DebugEnd
}