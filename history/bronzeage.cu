#include <stdio.h>





struct Network {
    // Handled by srcKernel
    int sourceCnt;              // number of source nodes
    int* sourceIDs;             // sourceIDs[i] = ID of the i-th source node
    int* sourceBits;            // sourceBits[i] = i-th source node's decimal-formatted input bitvalue 

    // Handled by genKernel
    int nodeCnt;                // number of nodes
    int edgeCnt;                // number of edges
    int colCnt;                 // number of columns
    int lutCnt;                 // number of LUTs
    int regCnt;                 // number of registers
    int cycleParity;            // parity of the cycle; dictates which of the two entries per register in regBuffer contains the old-generation value to be used
    int* offsets;               // offsets[i] = prefix sum of #prevNodes from node 0 to i-1, inclusive 
    int* csr;                   // i-th node's prevNode IDs are found:     csr[offs[i]] -> csr[offs[i+1]-1]
    int* typeIDs;               // typeIDs[i] = n where node with ID i is the n-th input, LUT, or register, 0-indexed (to get INP/LUT/REG-ID given nodeID)
    int* regIndicator;          // regIndicator[i] = 1 if node @ ID i is a register, 0 otherwise.
    int* regBuffer;             // the old/new-gen values from register @ regID i are found:        regBuffer[2*i] -> regBuffer[2*i + 1]
    int* lutOffsets;            // lutOffs[i] = prefix sum of #ints-needed-to-represent-LUT-control-bits from LUT 0 to i-1, inclusive (offsets for the lutBits-CSR).
    unsigned int* lutBits;      // i-th LUT's lutbits (decimal-formatted) are found:         lutBits[lutOffsets[i]] -> lutBits[lutOffsets[i+1]-1]
    int* outputBits;            // outputBits[i] = i-th node's decimal-formatted output bitvalue
    int* colIdxs;               // colIdxs[i] = prefix sum of column-nodecounts from col 0 to i-1, inclusive 
    int* colIDs;                // i-th column's nodeIDs are found:        colIDs[colIdxs[i]] -> colIDs[colIdxs[i+1]-1]

    Network(int srcCnt, int* srcIDs, int* srcBts, int ndCnt, int edgCnt, int colCnt, int lutCnt, int regCnt, int cycPar, int* offs, int* csr,
            int* tpIDs, int* regInd, int* regBuf, int* lutOffs, unsigned int* lutBts, int* outBts, int* colIdxs, int* colIDs)
       : sourceCnt(srcCnt),
         sourceIDs(srcIDs),
         sourceBits(srcBts),
         nodeCnt(ndCnt),
         edgeCnt(edgCnt),
         colCnt(colCnt),
         lutCnt(lutCnt),
         regCnt(regCnt),
         cycleParity(cycPar),
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





__global__ void srcKernel(Network ntwk) {
    // Looking at the i-th node in the source column
    int i = threadIdx.x;

    // Get the ID of the i-th source node
    int nodeID = ntwk.sourceIDs[i];
    printf("Thread [%d], NodeID [%d]\n", i, nodeID);

    // Set output as the source bit
    ntwk.outputBits[nodeID] = ntwk.sourceBits[i];
}




__global__ void genKernel(Network ntwk, int colNum) {
    // Looking at the i-th node in the column
    int i = threadIdx.x;

    // Get the ID of the i-th node in the column
    int nodeID = ntwk.colIDs[ntwk.colIdxs[colNum] + i];

    // Get the IDs of its previous nodes by looking up in offsets and csr
    int prvNdStartIdx = ntwk.offsets[nodeID];
    int width = ntwk.offsets[nodeID+1]-ntwk.offsets[nodeID];                            //<TODO> if this quantity is 1 then is object for sure a register? If so can do away with regIndicator.
    int lookupIdx = 0;

    // Get the outputs from the previous nodes (these are the inputs to the i-th node)
    for (int j=0; j<width; j++) {
        int prvNdID = ntwk.csr[j + prvNdStartIdx];
        int prvOutput = ntwk.outputBits[prvNdID];
        if (prvOutput) {
            lookupIdx |= (1 << width-1-j);
        }
    }

    if (ntwk.regIndicator[nodeID]) {
        // Node is a register
        int regID = ntwk.typeIDs[nodeID];
        int parity = ntwk.cycleParity;
        //Send old-gen val to outputBits 
        ntwk.outputBits[nodeID] = ntwk.regBuffer[2*regID + parity]; 
        //Set new-gen val in regBuf according to flipped parity
        int flipped = 1 - parity;
        ntwk.regBuffer[2*regID + flipped] = lookupIdx;
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
            ntwk.outputBits[nodeID] = 1;
        } else {
            ntwk.outputBits[nodeID] = 0;
        }
    }
    printf("Column [%d], Thread [%d], NodeID [%d]\n", colNum, i, nodeID);
}



int main() {
    // These fields are sent to the device, but not copied as pointers
    int ndCnt;
    int edgCnt;
    int colCnt;
    int lutCnt;
    int regCnt;
    int srcCnt;
    int cycPar;
    
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
    //============================Test Suite 1=============================//
    //=====================================================================//
    ndCnt = 12;
    edgCnt = 12;
    colCnt = 4;
    lutCnt = 6;
    regCnt = 0;
    srcCnt = 6;
    cycPar = 0;

    host_srcIDs = new int[srcCnt]{0,1,2,3,4,5};
    host_srcBts = new int[srcCnt]{0,0,1,0,0,0};
    host_offs = new int[ndCnt+1]{0,0,0,0,0,0,0,2,4,6,8,10,12};
    host_csr = new int[edgCnt]{0,1,2,3,4,5,6,7,7,8,9,10};
    host_typeIDs = new int[ndCnt]{0,1,2,3,4,5,0,1,2,3,4,5};
    host_regInd = new int[ndCnt]{0,0,0,0,0,0,0,0,0,0,0,0};
    host_regBuf = new int[regCnt];
    host_lutOffs = new int[lutCnt+1]{0,1,2,3,4,5,6};
    host_lutBts = new unsigned int[lutCnt]{2,4,8,2,4,8};
    host_outBts = new int[ndCnt];
    host_colIdxs = new int[ndCnt+1]{0,6,9,11,12};
    host_colIDs = new int[ndCnt]{0,1,2,3,4,5,6,7,8,9,10,11};
    //=====================================================================//
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

    // Create a single chief_net to hold all fields
    Network chief_net = Network(srcCnt, host_srcIDs, host_srcBts, ndCnt, edgCnt, colCnt, lutCnt, regCnt, cycPar,
            host_offs, host_csr, host_typeIDs, host_regInd, host_regBuf, host_lutOffs, host_lutBts, host_outBts, host_colIdxs, host_colIDs);
    
    // Now allocate space for all pointers which must be copied
    cudaMalloc((void**) &(dev_srcIDs), sizeof(int)*srcCnt);
    cudaMalloc((void**) &(dev_srcBts), sizeof(int)*srcCnt);
    cudaMalloc((void**) &(dev_offs), sizeof(int)*(ndCnt+1));
    cudaMalloc((void**) &(dev_csr), sizeof(int)*edgCnt);
    cudaMalloc((void**) &(dev_typeIDs), sizeof(int)*ndCnt);
    cudaMalloc((void**) &(dev_regInd), sizeof(int)*ndCnt);
    cudaMalloc((void**) &(dev_regBuf), sizeof(int)*regCnt);
    cudaMalloc((void**) &(dev_lutOffs), sizeof(int)*(lutCnt+1));
    cudaMalloc((void**) &(dev_lutBts), sizeof(int)*lutCnt);
    cudaMalloc((void**) &(dev_outBts), sizeof(int)*ndCnt);
    cudaMalloc((void**) &(dev_colIdxs), sizeof(int)*(colCnt+1));
    cudaMalloc((void**) &(dev_colIDs), sizeof(int)*ndCnt);

    // Now copy contents from host to device
    cudaMemcpy(dev_srcIDs, host_srcIDs, sizeof(int)*srcCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_srcBts, host_srcBts, sizeof(int)*srcCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_offs, host_offs, sizeof(int)*(ndCnt+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_csr, host_csr, sizeof(int)*edgCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_typeIDs, host_typeIDs, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_regInd, host_regInd, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_regBuf, host_regBuf, sizeof(int)*regCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lutOffs, host_lutOffs, sizeof(int)*(lutCnt+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lutBts, host_lutBts, sizeof(int)*lutCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_outBts, host_outBts, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);
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
 
    // Invoke srcKernel once using srcCnt many threads
    srcKernel<<<1, srcCnt>>>(chief_net);

    // Invoke genKernel once per non-source column (in-order), using as many threads as there are nodes in the column
    for (int i=1; i<colCnt; i++) {
        int columnHeight = host_colIdxs[i+1]-host_colIdxs[i];
        genKernel<<<1, columnHeight>>>(chief_net, i);
        cudaDeviceSynchronize();
    }

    // Now copy what's changed from device to host
    cudaMemcpy(host_outBts, dev_outBts, sizeof(int)*ndCnt, cudaMemcpyDeviceToHost);

    // Point once again to host pointers from chief_net                                 //<TODO> I think we can delete this all. We just want outputs...
    chief_net.sourceIDs = host_srcIDs;
    chief_net.sourceBits = host_srcBts;
    chief_net.offsets = host_offs;
    chief_net.csr = host_csr;
    chief_net.typeIDs = host_typeIDs;
    chief_net.regIndicator = host_regInd;
    chief_net.regBuffer = host_regBuf;
    chief_net.lutOffsets = host_lutOffs;
    chief_net.lutBits = host_lutBts;
    chief_net.outputBits = host_outBts;
    chief_net.colIdxs = host_colIdxs;
    chief_net.colIDs = host_colIDs;

    // Free the device pointers no longer in use
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

    // Now print simulation outputs or whatever else is desired
    printf("\n\n");
    for (int k=0; k<ndCnt; k++) {
        printf("Node [%d]'s output value is [%d]\n", k, chief_net.outputBits[k]);
    }

    // Finally, free the host pointers once everything is done
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


