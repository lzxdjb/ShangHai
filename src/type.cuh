#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include<iostream>
using namespace std;

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
