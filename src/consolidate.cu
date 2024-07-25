#include "consolidate.cuh"

void debug(DebugStruct *h_debugTool)
{

    // for (int k1 = 0; k1 < K; k1++)
    // {
    //     for (int k2 = 0; k2 < K; k2++)
    //     {
    //         std::cout << h_debugTool->LUT[k1 * K + k2] << " ";
    //     }
    //     std::cout << endl;
    // }

     for(int k1 = 0 ; k1 < dim ; k1 ++)
        {
           cout<<static_cast<int>(h_debugTool->query[k1])<<" ";
        }
    cout<<endl;

    // cout<<"sdfasdfasdf"<<endl;

    for(int k1 = 0 ; k1 < dim ; k1 ++)
        {
            cout<<static_cast<int>(h_debugTool->key[k1])<<" ";
        }
    cout<<endl;

    // cout << static_cast<int>(h_debugTool->EncodingK) << " ";

    // cout << static_cast<int>(h_debugTool->EncodingQ) << " ";
}

void Debug2Store(float *store)
{
    for (int b = 0; b < bs; b++)
    {
        for (int n = 0; n < nh; n++)
        {
            std::cout << endl;
            for (int q = 0; q < qlen; q++)
            {
                for (int k = 0; k < klen; k++)
                {
                    std::cout << static_cast<int>(store[b * nh * qlen * klen + n * qlen * klen + q * klen + k]) << " ";
                }
            }
        }
    }
}

void Debug2Q(uint8_t *query)
{
    for (int b = 0; b < bs; b++)
    {
        for (int n = 0; n < nh; n++)
        {
            std::cout << endl;
            for (int q = 0; q < qlen; q++)
            {
                for (int d = 0; d < dim; d++)
                {
                    std::cout << static_cast<int>(query[b * nh * qlen * dim + n * qlen * dim + q * dim + d]) << " ";
                    // std::cout<<static_cast<int>(uint8_t(1));
                }
                std::cout << endl;
            }
        }
    }
}

void Debug2K(uint8_t *query)
{
    for (int b = 0; b < bs; b++)
    {
        for (int n = 0; n < nh; n++)
        {
            std::cout << endl;
            for (int k = 0; k < klen; k++)
            {
                for (int d = 0; d < dim; d++)
                {
                    std::cout << static_cast<int>(query[b * nh * klen * dim + n * klen * dim + k * dim + d]) << " ";
                    // std::cout<<static_cast<int>(uint8_t(1));
                }
                std::cout << endl;
            }
        }
    }
}

void Debug2LUT(float *LUT)
{
    for (int n = 0; n < nh; n++)
    {
        std::cout << endl;
        for (int k1 = 0; k1 < K; k1++)
        {
            for (int k2 = 0; k2 < K; k2++)
            {
                std::cout << static_cast<int>(LUT[n * K * K + k1 * K + k2]) << " ";
            }
            std::cout << endl;
        }
    }
}

void solve_base_kernel(uint8_t *query, uint8_t *key, float *LUT, float *store)
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
    checkCudaErrors(cudaMalloc((void **)&d_LUT, sizeof(float) * nh * K * K));
    checkCudaErrors(cudaMalloc((void **)&d_store, sizeof(float) * bs * nh * qlen * klen));

    float ms; // elapsed time in milliseconds
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    checkCudaErrors(cudaEventCreate(&startEvent));
    checkCudaErrors(cudaEventCreate(&stopEvent));
    checkCudaErrors(cudaEventCreate(&dummyEvent));

    //// begin
    checkCudaErrors(cudaEventRecord(startEvent, 0));

    checkCudaErrors(cudaMemcpy(d_query, query, sizeof(uint8_t) * bs * nh * qlen * dim, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_key, key, sizeof(uint8_t) * bs * nh * klen * dim, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_LUT, LUT, sizeof(float) * nh * K * K, cudaMemcpyHostToDevice));

    int BestThread = 1024;
    int j = 0;
    int offset = 0;
    for (j = qlen * klen; j > BestThread; j -= BestThread)
    {
        // cout<<"j = "<<j<<endl;
        // std::cout<<"CNddddM"<<endl;
        BaseKernel<<<block, BestThread>>>(d_query, d_key, d_LUT, d_store, d_debugTool, offset);
        offset += BestThread;
    }
    // cout<<"j = "<<j<<endl;
    // exit(0);
    // std::cout<<"CNM"<<endl;
    // cout<<j << endl;
    // cout<<block<<endl;
    // offset -= BestThread;
    BaseKernel<<<block, j>>>(d_query, d_key, d_LUT, d_store, d_debugTool, offset);
    // checkCudaErrors(cudaDeviceSynchronize());
    // }

    checkCudaErrors(cudaMemcpy(store, d_store, sizeof(float) * bs * nh * qlen * klen, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventRecord(stopEvent, 0));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    ///// End Time Record
    printf("Time for sequential transfer and execute (ms): %f\n", ms);

    ///// Debug

    checkCudaErrors(cudaMemcpy(h_debugTool, d_debugTool, sizeof(DebugStruct), cudaMemcpyDeviceToHost));

    // Debug2Store(store);



    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);



    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_LUT);
    cudaFree(d_store);
    cudaFree(d_debugTool);

    // debug(h_debugTool);

}

void solve_stream_kernel(uint8_t *query, uint8_t *key, float *LUT, float *store)
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
    checkCudaErrors(cudaMalloc((void **)&d_LUT, sizeof(float) * nh * K * K));
    checkCudaErrors(cudaMalloc((void **)&d_store, sizeof(float) * bs * nh * qlen * klen));

    float ms; // elapsed time in milliseconds
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    checkCudaErrors(cudaEventCreate(&startEvent));
    checkCudaErrors(cudaEventCreate(&stopEvent));
    checkCudaErrors(cudaEventCreate(&dummyEvent));

    cudaStream_t stream[StreamModulnStreams];
    for (int i = 0; i < StreamModulnStreams; ++i)
    {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }



    int workload = nh ; // workload must be the factor of nh
    int i = 0;
    int OutOffset = 0;
    int StreamNumber = 0;


    //// begin
    checkCudaErrors(cudaEventRecord(startEvent, 0));

    for (i = bs * nh; i > workload; i -= workload)
    // for(int i = 0 ; i < 1 ; i ++)
    {

        int QOffset = OutOffset * qlen * dim;
        checkCudaErrors(cudaMemcpyAsync(&d_query[QOffset], &query[QOffset], (qlen * dim) * workload  * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[StreamNumber % StreamModulnStreams]));

        int KOffset = OutOffset * klen * dim;
        checkCudaErrors(cudaMemcpyAsync(&d_key[KOffset], &key[KOffset], (klen * dim) * workload * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[StreamNumber % StreamModulnStreams]));

        int LUTOffset = (OutOffset % nh) * K * K;
        checkCudaErrors(cudaMemcpyAsync(&d_LUT[LUTOffset], &LUT[LUTOffset], (K * K) * workload * sizeof(float), cudaMemcpyHostToDevice, stream[StreamNumber % StreamModulnStreams]));


        ////@@@@@@ Launch Kernel

        int BestThread = 1024;
        int j = 0;
        int offset = 0;
        for (j = qlen * klen; j > BestThread; j -= BestThread)
        {
            // cout<<"j = "<<j<<endl;
            // std::cout<<"CNddddM"<<endl;
            StreamKernel<<<workload, BestThread , 0 , stream[StreamNumber % StreamModulnStreams]>>>(d_query, d_key, d_LUT, d_store, d_debugTool, offset, OutOffset);
            offset += BestThread;
        }
        StreamKernel<<<workload, j , 0 , stream[StreamNumber % StreamModulnStreams]>>>(d_query, d_key, d_LUT, d_store, d_debugTool, offset, OutOffset);

        //// @@@@@

        int StoreOffset = OutOffset * qlen * klen;
        
///// !!!!!!
        checkCudaErrors(cudaMemcpyAsync(&store[StoreOffset], &d_store[StoreOffset], (qlen * klen) * workload * sizeof(float), cudaMemcpyDeviceToHost, stream[StreamNumber % StreamModulnStreams]));


/////// Debug
        // checkCudaErrors(cudaMemcpyAsync(&d_store[StoreOffset], &store[StoreOffset], (qlen * klen) * workload * sizeof(float), cudaMemcpyHostToDevice, stream[StreamNumber % nStreams]));
///////


        OutOffset += workload;
        StreamNumber++;

        

    }


    int QOffset = OutOffset * qlen * dim;
    checkCudaErrors(cudaMemcpyAsync(&d_query[QOffset], &query[QOffset], (qlen * dim) * i * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[StreamNumber % StreamModulnStreams]));

    int KOffset = OutOffset * klen * dim;
    checkCudaErrors(cudaMemcpyAsync(&d_key[KOffset], &key[KOffset], (klen * dim) * i * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[StreamNumber % StreamModulnStreams]));

    int LUTOffset = (OutOffset % nh) * K * K;
    checkCudaErrors(cudaMemcpyAsync(&d_LUT[LUTOffset], &LUT[LUTOffset], (K * K) * i * sizeof(float), cudaMemcpyHostToDevice, stream[StreamNumber % StreamModulnStreams]));

    ////@@@@@@ Launch Kernel

    int BestThread = 1024;
    int j = 0;
    int offset = 0;
    for (j = qlen * klen; j > BestThread; j -= BestThread)
    {
        // cout<<"j = "<<j<<endl;
        // std::cout<<"CNddddM"<<endl;
        StreamKernel<<<i, BestThread , 0 , stream[StreamNumber % StreamModulnStreams]>>>(d_query, d_key, d_LUT, d_store, d_debugTool, offset, OutOffset);
        offset += BestThread;
    }
    StreamKernel<<<i, j , 0 , stream[StreamNumber % StreamModulnStreams]>>>(d_query, d_key, d_LUT, d_store, d_debugTool, offset, OutOffset);

////@@@@@ Store

    int StoreOffset = OutOffset * qlen * klen;
    checkCudaErrors(cudaMemcpyAsync(&store[StoreOffset], &d_store[StoreOffset], (qlen * klen) * i * sizeof(float), cudaMemcpyDeviceToHost, stream[StreamNumber % StreamModulnStreams]));

   

////Debug

    // int StoreOffset = OutOffset * qlen * klen;
    // checkCudaErrors(cudaMemcpyAsync(&d_store[StoreOffset], &store[StoreOffset], (qlen * klen) * i * sizeof(float), cudaMemcpyHostToDevice, stream[StreamNumber % nStreams]));



    checkCudaErrors(cudaEventRecord(stopEvent, 0));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    ///// End Time Record
    printf("Time for stream execute time (ms): %f\n", ms);

    // Debug2Store(store);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    
    cudaFree(d_debugTool);
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_LUT);
    cudaFree(d_store);




////// Debug
    // checkCudaErrors(cudaMemcpy(h_debugTool, d_debugTool, sizeof(DebugStruct), cudaMemcpyDeviceToHost));


    // checkCudaErrors(cudaDeviceSynchronize());
    // debug(h_debugTool);

    // Debug2K(key);
    // Debug2Q(query);
    // Debug2LUT(LUT);

///// DebugEnd


}

