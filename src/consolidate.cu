#include "consolidate.cuh"



// void fool()
// {
//     cout<<"afasdf"<<endl;
// }


// int comparator(const void *a, const void *b) {
//     return ((const int (*)[2])a)[0][0] - ((const int (*)[2])b)[0][0];
// }

// // This csr-reversal function WILL NOT preserve the order of dependencies (matters to LUT computation)
// void gen_outgoing_csr(Network ntwk, int* out_offs, int* out_csr) {
//     // Form tuples of all (srcID, dstID) connexions from the graph
//     int tupleCnt = 0;
//     int tuples[ntwk.edgeCnt][2];                      // k-th tuple located at: (tuples[k][0], tuples[k][1]])
//     int* outDegrees = new int[ntwk.nodeCnt]();        // outDegrees[nodeID] = out-degree of corresponding node

//     for (int dstID=0; dstID<ntwk.nodeCnt; dstID++) {
//         int numDependencies = ntwk.offsets[dstID+1]-ntwk.offsets[dstID];
//         for (int i=0; i<numDependencies; i++) {
//             int srcID = ntwk.csr[ntwk.offsets[dstID]+i];
//             tuples[tupleCnt][0] = srcID;
//             tuples[tupleCnt][1] = dstID;
//             outDegrees[srcID]++;
//             tupleCnt++;
//         }
//     }

//     // Sort tuples by srcID (ascending) and configure offsets for outgoing-csr
//     qsort((void*)tuples, ntwk.edgeCnt, sizeof(int*), comparator);

//     // Configure offsets for outgoing-csr
//     out_offs[0] = 0;
//     int appendCntr = 0;

//     for (int source=0; source<ntwk.nodeCnt; source++) {
//         out_offs[source+1] = out_offs[source] + outDegrees[source];
//         for (int i=0; i<outDegrees[source]; i++) {
//             out_csr[out_offs[source]+i] = tuples[appendCntr][1];
//             appendCntr++;
//         }
//     }
//     free(outDegrees);
// }

// // Columnizes a network. Assumes that all fields are set except for colIdxs and colIDs
// int columnize(Network ntwk) {
//     int colCounter = 0;
//     int nCnt = ntwk.nodeCnt;
//     int sCnt = ntwk.sourceCnt;
//     int curSourcesLength = sCnt;
//     int* inDegrees = new int[nCnt];
//     int* curSources = new int[nCnt];
//     int* frontier = new int[nCnt]();
//     int* frwrd_offsets = new int[ntwk.nodeCnt+1];
//     int* frwrd_csr = new int[ntwk.edgeCnt];

//     // Generate forward-edge (outgoing) graph representation
//     gen_outgoing_csr(ntwk, frwrd_offsets, frwrd_csr);

//     // Populate inDegrees
//     for (int k=0; k<nCnt; k++) {
//         inDegrees[k] = ntwk.offsets[k+1]-ntwk.offsets[k];
//     }

//     // Populate curSources with source IDs to begin with
//     memcpy(curSources, ntwk.sourceIDs, sCnt*sizeof(int));
//     int curColIDsLength = 0;

//     // Loop until all nodes have been placed into a column
//     while (curSourcesLength != 0) {
//         // Initialize next entry in colIdxs to continue the prefix sum
//         ntwk.colIdxs[colCounter+1] = ntwk.colIdxs[colCounter];
//         int frontierLength = 0;

//         for (int i=0; i<curSourcesLength; i++) {
//             // Flush each curSource to a column and decrement its inDegree so it isn't picked up again
//             int nodeID = curSources[i];
//             ntwk.colIDs[curColIDsLength] = nodeID;
//             curColIDsLength++;
//             inDegrees[nodeID]--;

//             // Bump the current column's offset
//             ntwk.colIdxs[colCounter+1]++;

//             // Decrement inDegree count for each node outgoing from a curSource
//             int outDeg = frwrd_offsets[nodeID+1]-frwrd_offsets[nodeID];
//             for (int k=0; k<outDeg; k++) {
//                 inDegrees[frwrd_csr[frwrd_offsets[nodeID]+k]]--;
//             }
//         }

//         // Append all nodes that have exactly 0 inDegree to the frontier
//         for (int j=0; j<nCnt; j++) {
//             if (inDegrees[j] == 0) {
//                 frontier[frontierLength] = j;
//                 frontierLength++;
//             }
//         }

//         // Flush frontier to curSources
//         // I think this can all be optimized by actually dual-buffering here. Too much memcpy...
//         memset(curSources, 0, nCnt*sizeof(int));
//         memcpy(curSources, frontier, frontierLength*sizeof(int));
//         memset(frontier, 0, nCnt*sizeof(int));
//         curSourcesLength = frontierLength;
//         colCounter++;
//     }
//     free(inDegrees);
//     free(curSources);
//     free(frontier);
//     free(frwrd_offsets);
//     free(frwrd_csr);
//     return colCounter;
// }




// __global__ void srcKernel(Network ntwk, int cycleID) {
//     // Get the ID of the i-th source node, and set output as the
//     // source bit from the correct cycle-set
//     int i = threadIdx.x;
//     int nodeID = ntwk.sourceIDs[i];
//     ntwk.outputBits[cycleID*ntwk.nodeCnt + nodeID] = ntwk.sourceBits[cycleID*ntwk.sourceCnt + i];
// }

// __global__ void genKernel(Network ntwk, int colNum, int cycleID) {
//     // Get the ID of the i-th node in the column
//     int i = threadIdx.x;
//     int nodeID = ntwk.colIDs[ntwk.colIdxs[colNum] + i];
//     int cycleSet = cycleID*ntwk.nodeCnt;

//     // Get the IDs of its previous nodes by looking up in offsets and csr
//     int prvNdStartIdx = ntwk.offsets[nodeID];
//     int width = ntwk.offsets[nodeID+1]-ntwk.offsets[nodeID];
//     int lookupIdx = 0;

//     // Get the outputs from the previous nodes (these are the inputs to the i-th node)
//     for (int j=0; j<width; j++) {
//         int prvNdID = ntwk.csr[j + prvNdStartIdx];
//         int prvOutput = ntwk.outputBits[cycleSet + prvNdID];
//         if (prvOutput) {
//             lookupIdx |= (1 << width-1-j);
//         }
//     }

//     if (ntwk.regIndicator[nodeID]) {
//         // Node is a register, send old-gen-val to outputBits and new-gen-val to regBuffer
//         lookupIdx = ntwk.outputBits[cycleSet+ntwk.csr[prvNdStartIdx]];
//         int regID = ntwk.typeIDs[nodeID];
//         ntwk.outputBits[cycleSet + nodeID] = ntwk.regBuffer[regID];
//         ntwk.regBuffer[regID] = lookupIdx;
//     } else {
//         // Node is a LUT, find the correct lutBit to bitmask with lookupIdx
//         int lutID = ntwk.typeIDs[nodeID];
//         int intSizeInBits = 8*sizeof(int);
//         int quotient = lookupIdx / intSizeInBits;
//         int residue = lookupIdx % intSizeInBits;
//         int mask = 1 << residue;
//         int sectionIdx = ntwk.lutOffsets[lutID] + quotient;
//         int sectionBits = ntwk.lutBits[sectionIdx];

//         // Get LUT output by bitmasking the residue of lookupIdx against the section from lutBits
//         if (mask & sectionBits) {
//             ntwk.outputBits[cycleSet + nodeID] = 1;
//         } else {
//             ntwk.outputBits[cycleSet + nodeID] = 0;
//         }
//     }
// }


// Reader function for populating network fields from text file "graph.txt"
int reader(int* ndCnt, int* edgCnt, int* lutCnt, int* regCnt, int*srcCnt,
        std::vector<int>& regInd_vec,
        std::vector<int>& typeIDs_vec,
        std::vector<int>& srcIDs_vec,
        std::vector<int>& offs_vec,
        std::vector<int>& csr_vec,
        std::vector<int>& lutOffs_vec,
        std::vector<unsigned int>& lutBts_vec) {

    // Define max lengths and container for lines
    int maxLineChars = 1000;
    int maxDigits = 6;
    char line[maxLineChars];

    // Open target file
    // FILE *file = fopen("/media/lzx/lzx/UCSB/gpuRTLsim/consolidate.cu", "r");
    FILE *file = fopen("/media/lzx/lzx/UCSB/MyOwnRTLsim/example/graph/smallgraph.txt", "r");

    if (file == NULL) {
        fprintf(stderr, "Error opening graph.txt\n");
        return 1;
    }


    // Read line 0 for nodecount and clear afterwards
    if (fgets(line, maxLineChars, file) != NULL) {
        *ndCnt = atoi(line);
    } else {
        fprintf(stderr, "Error parsing line: %s\n", line);
    }
    memset(line, 0, maxLineChars*sizeof(char));
    std::cout<<"line = " <<line<<endl;


    // // Initialize certain arrays
    // std::vector<int> indctr(*ndCnt, 0);
    // std::vector<int> typIDs(*ndCnt, 0);
    // lutOffs_vec.push_back(0);
    // offs_vec.push_back(0);

    // // Read file line by line
    // int lnCntr = 0;
    // int ltOffsCntr = 0;
    // int offsCntr = 0;
    // while (fgets(line, maxLineChars, file) != NULL) {
    //     lnCntr++;
    //     char nodeID[maxDigits], lutWidth[maxDigits], lutInfo[maxLineChars], regDependencies[maxLineChars];
    //     if (sscanf(line, "%s INPUT %s", nodeID, nodeID) == 2) {                         // Process INPUT
    //         int id = atoi(nodeID);
    //         srcIDs_vec.push_back(id);
    //         typIDs[id] = *srcCnt;
    //         offs_vec.push_back(offsCntr);
    //         (*srcCnt)++;
    //     } else if (sscanf(line, "%s REG %s", nodeID, regDependencies) == 2) {           // Process REG
    //         int id = atoi(nodeID);
    //         indctr[id] = 1;
    //         typIDs[id] = *regCnt;
    //         offsCntr++;
    //         offs_vec.push_back(offsCntr);
    //         std::string dep(regDependencies);
    //         int depID = std::stoi(dep.substr(1, dep.length()-1));
    //         csr_vec.push_back(depID);

    //         (*edgCnt)++;
    //         (*regCnt)++;
    //     } else if (sscanf(line, "%s LUT %s %[^\n]", nodeID, lutWidth, lutInfo) == 3) {  // Process LUT
    //         int id = atoi(nodeID);
    //         int width = atoi(lutWidth);
    //         int intSize = 8*sizeof(int);
    //         int numIntsNeeded = 1;
    //         if (width > intSize) {
    //             numIntsNeeded = width / intSize;
    //             if (width % intSize != 0) {
    //                 numIntsNeeded++;
    //             }
    //         }
    //         // Update lut offsets
    //         ltOffsCntr += numIntsNeeded;
    //         lutOffs_vec.push_back(ltOffsCntr);

    //         // Consolidate all lut bits into one number
    //         unsigned int curSum = 0;
    //         int bitCounter = 0;
    //         int numFlushed = 0;
    //         for (int j=0; j<width; j++) {
    //             char curBit;
    //             curBit = lutInfo[2*j];
    //             curSum += (atoi(&curBit) << (width-j-1));
    //             bitCounter++;
    //             if (bitCounter >= 32) {
    //                 // Flush current int to lutBits and reset curSum/bitCounter
    //                 lutBts_vec.push_back(curSum);
    //                 bitCounter = 0;
    //                 curSum = 0;
    //                 numFlushed++;
    //             }
    //         }

    //         // Flush curSum to lutBits if within numIntsNeeded
    //         if (numFlushed < numIntsNeeded) {
    //             lutBts_vec.push_back(curSum);
    //         }

    //         // Handle dependencies
    //         std::string info(lutInfo);
    //         std::string deps = info.substr(2*width, info.length()-1);
    //         std::string dep;
    //         std::stringstream strStrm(deps);
    //         while (std::getline(strStrm, dep, ' ')) {
    //             // Test for validity of dependency
    //             if (dep.find('D' != std::string::npos)) {
    //                 (*edgCnt)++;
    //                 offsCntr++;
    //                 std::string depID = dep.substr(1, dep.length()-1);
    //                 csr_vec.push_back(std::stoi(depID));
    //             }
    //         }

    //         offs_vec.push_back(offsCntr);
    //         typIDs[id] = *lutCnt;
    //         (*lutCnt)++;
    //     } else {
    //         fprintf(stderr, "Error parsing line: %s\n", line);
    //     }
    // }
    // regInd_vec = indctr;
    // typeIDs_vec = typIDs;
    // fclose(file);
 
    // return 0;
}