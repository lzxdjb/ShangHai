#include "src/consolidate.cuh"
using namespace std;


/*
SPECIFIED BY USER INPUT WAVEFORM (TODO):
    cycleCnt
    srcBts

HANDLED BY COLUMNIZATION:
    colIdxs
    colIDs

LEFT UNSPECIFIED (no action needed):
    regBuf
    outBts
*/



int main() {
    int cycleCnt;
    int ndCnt;
    int edgCnt;
    int colCnt;
    int lutCnt;
    int regCnt;
    int srcCnt;

    // These fields are sent to the device and copied as pointers
    // Note that these are host-accessible only
    int* host_srcIDs;
    int* host_srcBts;
    int* host_offs;
    int* host_csr;
    int* host_typeIDs;
    int* host_regInd;
    int* host_regBuf;
    int* host_lutOffs;
    unsigned int* host_lutBts;
    int* host_outBts;
    int* host_colIdxs;
    int* host_colIDs;

    // Declare containers to fill in reader func
    int* numNodes = new int(0);
    int* numEdges = new int(0);
    int* numLuts = new int(0);
    int* numRegs = new int(0);
    int* numSrcs = new int(0);

    // Vectors to fill in reader func
    std::vector<int> srcIDs_vec;
    std::vector<int> offs_vec;
    std::vector<int> csr_vec;
    std::vector<int> lutOffs_vec;
    std::vector<unsigned int> lutBts_vec;
    std::vector<int> regInd_vec;
    std::vector<int> typeIDs_vec;

    // Call reader function; this fills all num<field> vars and popualates vectors

    // fool();
    if (reader(numNodes, numEdges, numLuts, numRegs, numSrcs,
                regInd_vec, typeIDs_vec,
                srcIDs_vec, offs_vec, csr_vec, lutOffs_vec, lutBts_vec)) {
        cout<<"cnm"<<endl;
        // return 1;
    }

    // Transfer int info to correct containers
    ndCnt = *numNodes;
    edgCnt = *numEdges;
    lutCnt = *numLuts;
    regCnt = *numRegs;
    srcCnt = *numSrcs;








    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // // Init nontrivial info
    // //                      <TODO><TODO><TODO><TODO>MUST AUTOMATE
    // /*
    // cycleCnt = 5;
    // host_srcBts = new int[srcCnt*cycleCnt]{0,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,1, 0,1,1,0};
    // colCnt = 0;
    // */

    // /*
    // cycleCnt = 9;
    // host_srcBts = new int[srcCnt*cycleCnt]{0,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,1, 0,1,1,0, 1,0,1,1, 0,0,0,0, 1,1,1,1, 1,0,1,0};
    // colCnt = 0;
    // */

    // /*
    // cycleCnt = 10000000;
    // host_srcBts = new int[srcCnt*cycleCnt]();                       //<TODO> grab these as arguments instead of hardset to 0
    // colCnt = 0;
    // */

    // // simple initialization (every cycle same values)
    // cycleCnt = 2;
    // for (int x=0; x<srcCnt*cycleCnt; x++) {
    //     host_srcBts[x] = 1;
    // }
    // colCnt=0;


    // /*
    // // complex initialization (every cycle different values)
    // for (int x=1; x<cycleCnt; x++) {
    //     int chunk = x*srcCnt;
    //     for (int y=0; y<srcCnt; y+=x) {
    //         host_srcBts[chunk+y] = 1;
    //     }
    // }
    // */
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////









    // // Alloc space for int arrays
    // host_srcIDs = new int[srcCnt];
    // host_offs = new int[ndCnt+1];
    // host_csr = new int[edgCnt];
    // host_typeIDs = new int[ndCnt];
    // host_regInd = new int[ndCnt];
    // host_regBuf = new int[regCnt]();
    // host_lutOffs = new int[lutCnt+1];
    // host_lutBts = new unsigned int[lutCnt];
    // host_outBts = new int[ndCnt*cycleCnt]();
    // host_colIdxs = new int[ndCnt+1]();
    // host_colIDs = new int[ndCnt]();

    // // Transfer vector info to arrays
    // memcpy(host_srcIDs, srcIDs_vec.data(), sizeof(int)*srcCnt);
    // memcpy(host_offs, offs_vec.data(), sizeof(int)*(1+ndCnt));
    // memcpy(host_csr, csr_vec.data(), sizeof(int)*edgCnt);
    // memcpy(host_lutOffs, lutOffs_vec.data(), sizeof(int)*(1+lutCnt));
    // memcpy(host_lutBts, lutBts_vec.data(), sizeof(int)*lutCnt);
    // memcpy(host_regInd, regInd_vec.data(), sizeof(int)*ndCnt);
    // memcpy(host_typeIDs, typeIDs_vec.data(), sizeof(int)*ndCnt);

    // // One chief_net to rule them all, one chief_net to find them.
    // Network chief_net = Network(srcCnt, host_srcIDs, host_srcBts, ndCnt, edgCnt, colCnt, lutCnt, regCnt,
    //         host_offs, host_csr, host_typeIDs, host_regInd, host_regBuf, host_lutOffs, host_lutBts, host_outBts, host_colIdxs, host_colIDs);

    // ///////////////////////////////////////////////
    // /////paste printfs from txts dir if needed/////
    // ///////////////////////////////////////////////

    // // Columnize and set number of columns
    // chief_net.colCnt = columnize(chief_net);
    // colCnt = chief_net.colCnt;

    // // These fields exist on the device
    // int* dev_srcIDs;
    // int* dev_srcBts;
    // int* dev_offs;
    // int* dev_csr;
    // int* dev_typeIDs;
    // int* dev_regInd;
    // int* dev_regBuf;
    // int* dev_lutOffs;
    // unsigned int* dev_lutBts;
    // int* dev_outBts;
    // int* dev_colIdxs;
    // int* dev_colIDs;

    // // Allocate space for all pointers we're copying
    // cudaMalloc((void**) &(dev_srcIDs), sizeof(int)*srcCnt);
    // cudaMalloc((void**) &(dev_srcBts), sizeof(int)*srcCnt*cycleCnt);
    // cudaMalloc((void**) &(dev_offs), sizeof(int)*(ndCnt+1));
    // cudaMalloc((void**) &(dev_csr), sizeof(int)*edgCnt);
    // cudaMalloc((void**) &(dev_typeIDs), sizeof(int)*ndCnt);
    // cudaMalloc((void**) &(dev_regInd), sizeof(int)*ndCnt);
    // cudaMalloc((void**) &(dev_regBuf), sizeof(int)*regCnt);
    // cudaMalloc((void**) &(dev_lutOffs), sizeof(int)*(lutCnt+1));
    // cudaMalloc((void**) &(dev_lutBts), sizeof(int)*lutCnt);
    // cudaMalloc((void**) &(dev_outBts), sizeof(int)*ndCnt*cycleCnt);
    // cudaMalloc((void**) &(dev_colIdxs), sizeof(int)*(colCnt+1));
    // cudaMalloc((void**) &(dev_colIDs), sizeof(int)*ndCnt);

    // // Copy contents from host to device
    // cudaMemcpy(dev_srcIDs, host_srcIDs, sizeof(int)*srcCnt, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_srcBts, host_srcBts, sizeof(int)*srcCnt*cycleCnt, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_offs, host_offs, sizeof(int)*(ndCnt+1), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_csr, host_csr, sizeof(int)*edgCnt, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_typeIDs, host_typeIDs, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_regInd, host_regInd, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_regBuf, host_regBuf, sizeof(int)*regCnt, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_lutOffs, host_lutOffs, sizeof(int)*(lutCnt+1), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_lutBts, host_lutBts, sizeof(int)*lutCnt, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_outBts, host_outBts, sizeof(int)*ndCnt*cycleCnt, cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_colIdxs, host_colIdxs, sizeof(int)*(colCnt+1), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_colIDs, host_colIDs, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);

    // // Point to device pointers from chief_net
    // chief_net.sourceIDs = dev_srcIDs;
    // chief_net.sourceBits = dev_srcBts;
    // chief_net.offsets = dev_offs;
    // chief_net.csr = dev_csr;
    // chief_net.typeIDs = dev_typeIDs;
    // chief_net.regIndicator = dev_regInd;
    // chief_net.regBuffer = dev_regBuf;
    // chief_net.lutOffsets = dev_lutOffs;
    // chief_net.lutBits = dev_lutBts;
    // chief_net.outputBits = dev_outBts;
    // chief_net.colIdxs = dev_colIdxs;
    // chief_net.colIDs = dev_colIDs;

    // // Run cycle simulation on the kernels                          <TODO> possible optimization: try merging the kernels and calling one kernel on colCnt many synchronized blocks
    //                                                                     // instead, since this will minimize CPU-GPU communication latency and make the bulk of the simulation internal on the GPU.
    // for (int k=0; k<cycleCnt; k++) {
    //     printf("\n<<<<<<<<<entering cycle [%d]>>>>>>>>", k);
    //     // Invoke srcKernel once using srcCnt many threads
    //     srcKernel<<<1, srcCnt>>>(chief_net, k);
    //     cudaDeviceSynchronize();

    //     // Invoke genKernel once per non-source column (in-order), using as many threads as there are nodes in the column
    //     for (int i=1; i<colCnt; i++) {
    //         int columnHeight = host_colIdxs[i+1]-host_colIdxs[i];
    //         genKernel<<<1, columnHeight>>>(chief_net, i, k);
    //         cudaDeviceSynchronize();
    //     }
    // }

    // // Copy results from device to host
    // cudaMemcpy(host_outBts, dev_outBts, sizeof(int)*ndCnt*cycleCnt, cudaMemcpyDeviceToHost);

    // // Print simulation results
    // printf("\n");
    // for (int k=0; k<cycleCnt; k++) {
    //     printf(" _______________________________\n");
    //     printf("|        CYCLE[%d] RESULTS       |\n", k);
    //     printf("|-------------------------------|\n");
    //     for (int j=0; j<ndCnt; j++) {
    //         printf("|    Node[%d]'s output is [%d]    |\n", j, host_outBts[k*ndCnt + j]);
    //     }
    //     printf("|_______________________________|\n");
    // }

    // // Free device pointers
    // cudaFree(dev_srcIDs);
    // cudaFree(dev_srcBts);
    // cudaFree(dev_offs);
    // cudaFree(dev_csr);
    // cudaFree(dev_typeIDs);
    // cudaFree(dev_regInd);
    // cudaFree(dev_regBuf);
    // cudaFree(dev_lutOffs);
    // cudaFree(dev_lutBts);
    // cudaFree(dev_outBts);
    // cudaFree(dev_colIdxs);
    // cudaFree(dev_colIDs);

    // // Free host pointers
    // free(host_srcIDs);
    // free(host_srcBts);
    // free(host_offs);
    // free(host_csr);
    // free(host_typeIDs);
    // free(host_regInd);
    // free(host_regBuf);
    // free(host_lutOffs);
    // free(host_lutBts);
    // free(host_outBts);
    // free(host_colIdxs);
    // free(host_colIDs);

    // return 0;
}