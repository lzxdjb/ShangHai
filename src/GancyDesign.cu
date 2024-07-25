#include "FancyDesign.cuh"


void Debug3Store(float *store)
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

void debug3(DebugStruct *h_debugTool)
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

void FancyStream(uint8_t *query, uint8_t *key, float *LUT, float *store)
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



    // int workload = nh ; // workload must be the factor of nh
    // int i = 0;
    // int OutOffset = 0;
    // int StreamNumber = 0;


    checkCudaErrors(cudaEventRecord(startEvent, 0));


    // int LUTWorkload = nh; 
    for (int streamNumber = 0; streamNumber < nh; streamNumber++)
    {
        int LUTOffset = streamNumber * K * K;
        checkCudaErrors(cudaMemcpyAsync(&d_LUT[LUTOffset], &LUT[LUTOffset], (K * K) * sizeof(float), cudaMemcpyHostToDevice, stream[streamNumber % StreamModulnStreams]));

    }

    for (int streamNumber = 0; streamNumber < bs * nh; streamNumber++)
    {
        int QOffset = streamNumber * qlen * dim;
        int KOffset = streamNumber * klen * dim;

        checkCudaErrors(cudaMemcpyAsync(&d_query[QOffset], &query[QOffset], (qlen * dim) * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[streamNumber % StreamModulnStreams]));

        // int KOffset = OutOffset * klen * dim;
        checkCudaErrors(cudaMemcpyAsync(&d_key[KOffset], &key[KOffset], (klen * dim)  * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[streamNumber % StreamModulnStreams]));


    }

    int i = 0 ;
    int StreamNumber = 0;
    int OutOffset = 0;
    for (i = bs * nh; i > FancyStreamWorkLoad; i -= FancyStreamWorkLoad)
    // for(int i = 0 ; i < 1 ; i ++)
    {

        int BestThread = 1024;
        int j = 0;
        int offset = 0;
        for (j = qlen * klen; j > BestThread; j -= BestThread)
        {
            // cout<<"j = "<<j<<endl;
            // std::cout<<"CNddddM"<<endl;
            StreamKernel<<<FancyStreamWorkLoad, BestThread , 0 , stream[StreamNumber % StreamModulnStreams]>>>(d_query, d_key, d_LUT, d_store, d_debugTool, offset, OutOffset);
            offset += BestThread;
        }
        StreamKernel<<<FancyStreamWorkLoad, j , 0 , stream[StreamNumber % StreamModulnStreams]>>>(d_query, d_key, d_LUT, d_store, d_debugTool, offset, OutOffset);

        StreamNumber ++;
        OutOffset += FancyStreamWorkLoad;

    }


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





    for (int streamNumber = 0; streamNumber < bs * nh; streamNumber++)
    {
        int StoreOffset = streamNumber * qlen * klen;
        checkCudaErrors(cudaMemcpyAsync(&store[StoreOffset], &d_store[StoreOffset], (qlen * klen)*sizeof(float), cudaMemcpyDeviceToHost, stream[streamNumber % StreamModulnStreams]));
    }

    // Debug3Store(store);

    // checkCudaErrors(cudaMemcpy(h_debugTool, d_debugTool, sizeof(DebugStruct), cudaMemcpyDeviceToHost));


    // checkCudaErrors(cudaDeviceSynchronize());
    // debug3(h_debugTool);
    // Debug3Store(store);
    // exit(0);

    //// begin
 
    checkCudaErrors(cudaEventRecord(stopEvent, 0));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    ///// End Time Record
    printf("Time for Ultra execute time (ms): %f\n", ms);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaFree(d_debugTool);
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_LUT);
    cudaFree(d_store);

    




////// Debug
   

    // Debug2K(key);
    // Debug2Q(query);
    // Debug2LUT(LUT);

///// DebugEnd

}
