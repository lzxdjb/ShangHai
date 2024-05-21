#include <stdio.h>





struct Network {
    // Handled by srcKernel
    int sourceCnt;              // number of source nodes
    int* sourceIDs;             // sourceIDs[i] = ID of the i-th source node
    int* sourceBits;            // sourceBits[i] = i-th source node's integer-formatted input bitvalue 

    // Handled by genKernel
    int nodeCnt;                // number of nodes
    int edgeCnt;                // number of edges
    int colCnt;                 // number of columns
    int* offsets;               // i-th node's prevNode IDs are found:     csr[offs[i]] -> csr[offs[i+1]-1]
    int* csr;                   // contains prevNode IDs as shown above
    int lutWidth;               // standard bitwidth of a LUT (assuming constant)
    int* lutBits;               // lutBits[i] = i-th node's lutbits, integer-formatted bitvalues
    int* outputBits;            // outputBits[i] = i-th node's integer-formatted output bitvalue
    int* colIdxs;               // colIdxs[i] = prefix sum of column nodecounts from col 0 to i-1, inclusive (so colIdxs[0]=0, colIdxs[1]=#ndsCol0, colIdxs[n]=sum(#ndsCol0->n-1)) 
    int* colIDs;                // i-th column's nodeIDs are found:        colIDs[colIdxs[i]] -> colIDs[colIdxs[i+1]-1]

    Network(int srcCnt,int* srcIDs,int* srcBts,int ndCnt,int edgCnt,int colCnt,int* offs,int* csr,int lutWdth,int* lutBts,int* outBts,int* colIdxs,int* colIDs)
       : sourceCnt(srcCnt),
         sourceIDs(srcIDs),
         sourceBits(srcBts),
         nodeCnt(ndCnt),
         edgeCnt(edgCnt),
         colCnt(colCnt),
         offsets(offs),
         csr(csr),
         lutWidth(lutWdth),
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

    // Set output by bitmasking sourceBits against lutBits
    int mask = 1 << ntwk.sourceBits[i];

    if (mask & ntwk.lutBits[nodeID]) {
        ntwk.outputBits[nodeID] = 1;
    } else {
        ntwk.outputBits[nodeID] = 0;
    }
}




__global__ void genKernel(Network ntwk, int colNum) {
    // Looking at the i-th node in the column
    int i = threadIdx.x;

    // Get the ID of the i-th node in the column
    int nodeID = ntwk.colIDs[ntwk.colIdxs[colNum] + i];

    // Get the IDs of his previous nodes by looking up in offsets and csr
    int prvNdStartIdx = ntwk.offsets[nodeID];
    int width = ntwk.lutWidth;
    int lutLookupIdx = 0;

    // Get the outputs from the previous nodes (these are the inputs to the i-th node)
    for (int j=0; j<width; j++) {
        int prvNdID = ntwk.csr[j + prvNdStartIdx];
        int prvOutput = ntwk.outputBits[prvNdID];
        if (prvOutput) {
            lutLookupIdx |= (1 << width-1-j);
        }
    }

    // Get lut output by bitmasking lutLookupIdx against lutBits
    int mask = 1 << lutLookupIdx;

    if (mask & ntwk.lutBits[nodeID]) {
        ntwk.outputBits[nodeID] = 1;
    } else {
        ntwk.outputBits[nodeID] = 0;
    }

    printf("Column [%d], Thread [%d], NodeID [%d], LutLookupIdx [%d]\n", colNum, i, nodeID, lutLookupIdx);
}





int main() {
    // These fields are sent to the device, but not copied as pointers
    int ndCnt;
    int edgCnt;
    int colCnt;
    int lutWdth;
    int srcCnt;
    
    // These fields are sent to the device and copied as pointers
    // Note that these are host-accessible only
    int* host_srcIDs;
    int* host_srcBts;
    int* host_offs;
    int* host_csr;
    int* host_lutBts;
    int* host_outBts;
    int* host_colIdxs;
    int* host_colIDs;


    //=====================================================================//
    //============================Test Suite 1=============================//
    //=====================================================================//
    ndCnt = 6;
    edgCnt = 6;
    colCnt = 3;
    lutWdth = 2;
    srcCnt = 3;

    host_srcIDs = new int[srcCnt]{0,1,2};
    host_srcBts = new int[srcCnt]{0,2,0};
    host_offs = new int[ndCnt+1]{0,0,0,0,2,4,6};
    host_csr = new int[edgCnt]{0,1,1,2,3,4};
    host_lutBts = new int[ndCnt]{2,4,8,2,4,8};
    host_outBts = new int[ndCnt];
    host_colIdxs = new int[ndCnt+1]{0,3,5,6};
    host_colIDs = new int[ndCnt]{0,1,2,3,4,5};
    //=====================================================================//
    //=====================================================================//
    //=====================================================================//


    // These fields exist on the device
    int* dev_srcIDs;
    int* dev_srcBts;
    int* dev_offs;
    int* dev_csr;
    int* dev_lutBts;
    int* dev_outBts;
    int* dev_colIdxs;
    int* dev_colIDs;

    // Create a single chief_net to hold all fields
    Network chief_net = Network(srcCnt, host_srcIDs, host_srcBts, ndCnt, edgCnt, colCnt, host_offs, host_csr, lutWdth, host_lutBts, host_outBts, host_colIdxs, host_colIDs);
    
    // Now allocate space for all pointers which must be copied
    cudaMalloc((void**) &(dev_srcIDs), sizeof(int)*srcCnt);
    cudaMalloc((void**) &(dev_srcBts), sizeof(int)*srcCnt);
    cudaMalloc((void**) &(dev_offs), sizeof(int)*(ndCnt+1));
    cudaMalloc((void**) &(dev_csr), sizeof(int)*edgCnt);
    cudaMalloc((void**) &(dev_lutBts), sizeof(int)*ndCnt);
    cudaMalloc((void**) &(dev_outBts), sizeof(int)*ndCnt);
    cudaMalloc((void**) &(dev_colIdxs), sizeof(int)*(colCnt+1));
    cudaMalloc((void**) &(dev_colIDs), sizeof(int)*ndCnt);

    // Now copy contents from host to device
    cudaMemcpy(dev_srcIDs, host_srcIDs, sizeof(int)*srcCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_srcBts, host_srcBts, sizeof(int)*srcCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_offs, host_offs, sizeof(int)*(ndCnt+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_csr, host_csr, sizeof(int)*edgCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lutBts, host_lutBts, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_outBts, host_outBts, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colIdxs, host_colIdxs, sizeof(int)*(colCnt+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colIDs, host_colIDs, sizeof(int)*ndCnt, cudaMemcpyHostToDevice);

    // Point to device pointers from chief_net
    chief_net.sourceIDs = dev_srcIDs;
    chief_net.sourceBits = dev_srcBts;
    chief_net.offsets = dev_offs;
    chief_net.csr = dev_csr;
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
        // Force sequential col-by-col execution.
        cudaDeviceSynchronize();
    }

    // Now copy what's changed from device to host
    cudaMemcpy(host_outBts, dev_outBts, sizeof(int)*ndCnt, cudaMemcpyDeviceToHost);

    // Point once again to host pointers from chief_net
    chief_net.sourceIDs = host_srcIDs;
    chief_net.sourceBits = host_srcBts;
    chief_net.offsets = host_offs;
    chief_net.csr = host_csr;
    chief_net.lutBits = host_lutBts;
    chief_net.outputBits = host_outBts;
    chief_net.colIdxs = host_colIdxs;
    chief_net.colIDs = host_colIDs;

    // Free the device pointers no longer in use
    cudaFree(dev_srcIDs);
    cudaFree(dev_srcBts);
    cudaFree(dev_offs);
    cudaFree(dev_csr);
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
    free(host_lutBts);
    free(host_outBts);
    free(host_colIdxs);
    free(host_colIDs);

    return 0;
}


