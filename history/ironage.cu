#include <stdio.h>





struct Network {
    // Handled by srcKernel
    int sourceCnt;              // number of source nodes
    int* sourceIDs;             // sourceIDs[i] = ID of the i-th source node
    int* sourceBits;            // decimal-formatted source vals for k-th cycle are found through indices: k*sourceCnt -> (k+1)*sourceCnt-1

    // Handled by genKernel
    int nodeCnt;                // number of nodes
    int edgeCnt;                // number of edges
    int colCnt;                 // number of columns
    int lutCnt;                 // number of LUTs
    int regCnt;                 // number of registers
    int* offsets;               // offsets[i] = prefix sum of #prevNodes from node 0 to i-1, inclusive 
    int* csr;                   // i-th node's prevNode IDs are found:     csr[offs[i]] -> csr[offs[i+1]-1]
    int* typeIDs;               // typeIDs[i] = n where node with ID i is the n-th input, LUT, or register, 0-indexed (to get INP/LUT/REG-ID given nodeID)
    int* regIndicator;          // regIndicator[i] = 1 if node @ ID i is a register, 0 otherwise.
    int* regBuffer;             // regBuffer[i] = regID i register's oldGen val before being touched by genKernel, newGen val after being touched. 
    int* lutOffsets;            // lutOffs[i] = prefix sum of #ints-needed-to-represent-LUT-control-bits from LUT 0 to i-1, inclusive (offsets for the lutBits-CSR).
    unsigned int* lutBits;      // i-th LUT's lutbits (decimal-formatted) are found:         lutBits[lutOffsets[i]] -> lutBits[lutOffsets[i+1]-1]
    int* outputBits;            // decimal-formatted output vals for k-th cycle are found through indices: k*nodeCnt -> (k+1)*nodeCnt-1
    int* colIdxs;               // colIdxs[i] = prefix sum of column-nodecounts from col 0 to i-1, inclusive 
    int* colIDs;                // i-th column's nodeIDs are found:        colIDs[colIdxs[i]] -> colIDs[colIdxs[i+1]-1]

    Network(int srcCnt, int* srcIDs, int* srcBts, int ndCnt, int edgCnt, int colCnt, int lutCnt, int regCnt, int* offs, int* csr,
            int* tpIDs, int* regInd, int* regBuf, int* lutOffs, unsigned int* lutBts, int* outBts, int* colIdxs, int* colIDs)
       : sourceCnt(srcCnt),
         sourceIDs(srcIDs),
         sourceBits(srcBts),
         nodeCnt(ndCnt),
         edgeCnt(edgCnt),
         colCnt(colCnt),
         lutCnt(lutCnt),
         regCnt(regCnt),
         offsets(offs),
         csr(csr),
         typeIDs(tpIDs),
         regIndicator(regInd),
         regBuffer(regBuf),
         lutOffsets(lutOffs),
         lutBits(lutBts),
         outputBits(outBts),
         colIdxs(colIdxs),
         colIDs(colIDs) {} 
};





__global__ void srcKernel(Network ntwk, int cycleID) {
    int i = threadIdx.x;

    // Get the ID of the i-th source node
    int nodeID = ntwk.sourceIDs[i];

    // Set output as the source bit from the right cycle-set
    ntwk.outputBits[cycleID*ntwk.nodeCnt + nodeID] = ntwk.sourceBits[cycleID*ntwk.sourceCnt + i];
    //printf("Column [0], Thread [%d], NodeID [%d], Type(Lut=0,Reg=1) [%d]\n", i, nodeID, ntwk.regIndicator[nodeID]);
}





__global__ void genKernel(Network ntwk, int colNum, int cycleID) {
    int i = threadIdx.x;

    // Get the ID of the i-th node in the column
    int nodeID = ntwk.colIDs[ntwk.colIdxs[colNum] + i];
    int cycleSet = cycleID*ntwk.nodeCnt;

    // Get the IDs of its previous nodes by looking up in offsets and csr
    int prvNdStartIdx = ntwk.offsets[nodeID];
    int width = ntwk.offsets[nodeID+1]-ntwk.offsets[nodeID];                            //<TODO> if this quantity is 1 then is object for sure a register? If so can do away with regIndicator.
    int lookupIdx = 0;

    // Get the outputs from the previous nodes (these are the inputs to the i-th node)
    for (int j=0; j<width; j++) {
        int prvNdID = ntwk.csr[j + prvNdStartIdx];
        int prvOutput = ntwk.outputBits[cycleSet + prvNdID];
        if (prvOutput) {
            lookupIdx |= (1 << width-1-j);
        }
    }

    if (ntwk.regIndicator[nodeID]) {
        // Node is a register
        int regID = ntwk.typeIDs[nodeID];
        //Send old-gen val to outputBits 
        ntwk.outputBits[cycleSet + nodeID] = ntwk.regBuffer[regID]; 
        //Set new-gen val in regBuffer
        ntwk.regBuffer[regID] = lookupIdx;
    } else {
        // Node is a LUT
        int lutID = ntwk.typeIDs[nodeID];
        // Find the correct lutBit to bitmask with lookupIdx
        int intSizeInBits = 8*sizeof(int);
        int quotient = lookupIdx / intSizeInBits;
        int residue = lookupIdx % intSizeInBits;
        int mask = 1 << residue;
        int sectionIdx = ntwk.lutOffsets[lutID] + quotient;
        int sectionBits = ntwk.lutBits[sectionIdx];

        // Get lut output by bitmasking the residue of lookupIdx against the section from lutBits
        if (mask & sectionBits) {
            ntwk.outputBits[cycleSet + nodeID] = 1;
        } else {
            ntwk.outputBits[cycleSet + nodeID] = 0;
        }
    }
    //printf("Column [%d], Thread [%d], NodeID [%d], Type(Lut=0,Reg=1) [%d]\n", colNum, i, nodeID, ntwk.regIndicator[nodeID]);
}





int main() {

    // For how many cycles will the simulation run?
    int cycleCnt;

    // These fields are sent to the device, but not copied as pointers
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

    //=====================================================================//
    //===========================5-cycle sim test==========================//
    //=====================================================================//
    cycleCnt = 5;

    ndCnt = 8;
    edgCnt = 8;
    colCnt = 4;
    lutCnt = 3;
    regCnt = 1;
    srcCnt = 4;

    host_srcIDs = new int[srcCnt]{0,1,2,3};
    host_srcBts = new int[srcCnt*cycleCnt]{0,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,1,1, 0,1,1,0};
    host_offs = new int[ndCnt+1]{0,0,0,0,0,2,4,5,7};
    host_csr = new int[edgCnt]{0,1,2,3,4,6,5};
    host_typeIDs = new int[ndCnt]{0,1,2,3,0,1,0,2};
    host_regInd = new int[ndCnt]{0,0,0,0,0,0,1,0};
    host_regBuf = new int[regCnt];
    host_lutOffs = new int[lutCnt+1]{0,1,2,3};
    host_lutBts = new unsigned int[lutCnt]{4,14,1};
    host_outBts = new int[ndCnt*cycleCnt];
    host_colIdxs = new int[ndCnt+1]{0,4,6,7,8};
    host_colIDs = new int[ndCnt]{0,1,2,3,4,5,6,7};
    //=====================================================================//
    //=====================================================================//


    // These fields exist on the device
    int* dev_srcIDs;
    int* dev_srcBts;
    int* dev_offs;
    int* dev_csr;
    int* dev_typeIDs;
    int* dev_regInd;
    int* dev_regBuf;
    int* dev_lutOffs;
    unsigned int* dev_lutBts;
    int* dev_outBts;
    int* dev_colIdxs;
    int* dev_colIDs;

    // One chief_net to rule them all, one chief_net to find them.
    Network chief_net = Network(srcCnt, host_srcIDs, host_srcBts, ndCnt, edgCnt, colCnt, lutCnt, regCnt,
            host_offs, host_csr, host_typeIDs, host_regInd, host_regBuf, host_lutOffs, host_lutBts, host_outBts, host_colIdxs, host_colIDs);
    
    // Allocate space for all pointers we're copying 
    cudaMalloc((void**) &(dev_srcIDs), sizeof(int)*srcCnt);
    cudaMalloc((void**) &(dev_srcBts), sizeof(int)*srcCnt*cycleCnt);
    cudaMalloc((void**) &(dev_offs), sizeof(int)*(ndCnt+1));
    cudaMalloc((void**) &(dev_csr), sizeof(int)*edgCnt);
    cudaMalloc((void**) &(dev_typeIDs), sizeof(int)*ndCnt);
    cudaMalloc((void**) &(dev_regInd), sizeof(int)*ndCnt);
    cudaMalloc((void**) &(dev_regBuf), sizeof(int)*regCnt);
    cudaMalloc((void**) &(dev_lutOffs), sizeof(int)*(lutCnt+1));
    cudaMalloc((void**) &(dev_lutBts), sizeof(int)*lutCnt);
    cudaMalloc((void**) &(dev_outBts), sizeof(int)*ndCnt*cycleCnt);
    cudaMalloc((void**) &(dev_colIdxs), sizeof(int)*(colCnt+1));
    cudaMalloc((void**) &(dev_colIDs), sizeof(int)*ndCnt);

    // Copy contents from host to device
    cudaMemcpy(dev_srcIDs, host_srcIDs, sizeof(int)*srcCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_srcBts, host_srcBts, sizeof(int)*srcCnt*cycleCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_offs, host_offs, sizeof(int)*(ndCnt+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_csr, host_csr, sizeof(int)*edgCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_typeIDs, host_typeIDs, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_regInd, host_regInd, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_regBuf, host_regBuf, sizeof(int)*regCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lutOffs, host_lutOffs, sizeof(int)*(lutCnt+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lutBts, host_lutBts, sizeof(int)*lutCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_outBts, host_outBts, sizeof(int)*ndCnt*cycleCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colIdxs, host_colIdxs, sizeof(int)*(colCnt+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colIDs, host_colIDs, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);

    // Point to device pointers from chief_net
    chief_net.sourceIDs = dev_srcIDs;
    chief_net.sourceBits = dev_srcBts;
    chief_net.offsets = dev_offs;
    chief_net.csr = dev_csr;
    chief_net.typeIDs = dev_typeIDs;
    chief_net.regIndicator = dev_regInd;
    chief_net.regBuffer = dev_regBuf;
    chief_net.lutOffsets = dev_lutOffs;
    chief_net.lutBits = dev_lutBts;
    chief_net.outputBits = dev_outBts;
    chief_net.colIdxs = dev_colIdxs;
    chief_net.colIDs = dev_colIDs;

    // Run cycle simulation on the kernels
    for (int k=0; k<cycleCnt; k++) {
        printf("\n<<<<<<<<<entering cycle [%d]>>>>>>>>", k); 
        // Invoke srcKernel once using srcCnt many threads
        srcKernel<<<1, srcCnt>>>(chief_net, k);
        cudaDeviceSynchronize();

        // Invoke genKernel once per non-source column (in-order), using as many threads as there are nodes in the column
        for (int i=1; i<colCnt; i++) {
            int columnHeight = host_colIdxs[i+1]-host_colIdxs[i];
            genKernel<<<1, columnHeight>>>(chief_net, i, k);
            cudaDeviceSynchronize();
        }
    }

    // Copy results from device to host
    cudaMemcpy(host_outBts, dev_outBts, sizeof(int)*ndCnt*cycleCnt, cudaMemcpyDeviceToHost);

    // Print simulation outputs
    printf("\n");
    for (int k=0; k<cycleCnt; k++) {
        printf(" _______________________________\n");
        printf("|        CYCLE[%d] RESULTS       |\n", k); 
        printf("|-------------------------------|\n");
        for (int j=0; j<ndCnt; j++) {
            printf("|    Node[%d]'s output is [%d]    |\n", j, host_outBts[k*ndCnt + j]);
        }
        printf("|_______________________________|\n");
    }

    // Free device pointers
    cudaFree(dev_srcIDs);
    cudaFree(dev_srcBts);
    cudaFree(dev_offs);
    cudaFree(dev_csr);
    cudaFree(dev_typeIDs);
    cudaFree(dev_regInd);
    cudaFree(dev_regBuf);
    cudaFree(dev_lutOffs);
    cudaFree(dev_lutBts);
    cudaFree(dev_outBts);
    cudaFree(dev_colIdxs);
    cudaFree(dev_colIDs);

    // Free host pointers
    free(host_srcIDs);
    free(host_srcBts);
    free(host_offs);
    free(host_csr);
    free(host_typeIDs);
    free(host_regInd);
    free(host_regBuf);
    free(host_lutOffs);
    free(host_lutBts);
    free(host_outBts);
    free(host_colIdxs);
    free(host_colIDs);

    return 0;
}


