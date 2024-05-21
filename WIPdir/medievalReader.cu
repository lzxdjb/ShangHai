#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <vector>




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
    FILE *file = fopen("graph.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening graph.txt\n");
        return 1;
    }

    // Read line 0 for nodecount
    if (fgets(line, maxLineChars, file) != NULL) {
        *ndCnt = atoi(line);
    } else {
        fprintf(stderr, "Error parsing line: %s\n", line);
    }

    // Clear line
    memset(line, 0, maxLineChars*sizeof(char));

    // Initialize certain arrays
    std::vector<int> indctr(*ndCnt, 0);
    std::vector<int> typIDs(*ndCnt, 0);
    lutOffs_vec.push_back(0);
    offs_vec.push_back(0);

    // Read file line by line
    int lnCntr = 0;
    int ltOffsCntr = 0;
    int offsCntr = 0;
    while (fgets(line, maxLineChars, file) != NULL) {
        lnCntr++;
        char nodeID[maxDigits], lutWidth[maxDigits], lutInfo[maxLineChars], regDependencies[maxLineChars];
        if (sscanf(line, "%s INPUT %s", nodeID, nodeID) == 2) {                         // Process INPUT
            int id = atoi(nodeID);
            srcIDs_vec.push_back(id);
            typIDs[id] = *srcCnt;
            offs_vec.push_back(offsCntr);
            (*srcCnt)++;
        } else if (sscanf(line, "%s REG %s", nodeID, regDependencies) == 2) {           // Process REG
            int id = atoi(nodeID);
            indctr[id] = 1;
            typIDs[id] = *regCnt;
            offsCntr++;
            offs_vec.push_back(offsCntr);
            std::string dep(regDependencies);
            int depID = std::stoi(dep.substr(1, dep.length()-1));
            csr_vec.push_back(depID);

            (*edgCnt)++;
            (*regCnt)++;
        } else if (sscanf(line, "%s LUT %s %[^\n]", nodeID, lutWidth, lutInfo) == 3) {  // Process LUT
            int id = atoi(nodeID);
            int width = atoi(lutWidth);
            int intSize = 8*sizeof(int);
            int numIntsNeeded = 1;
            if (width > intSize) {
                numIntsNeeded = width / intSize;
                if (width % intSize != 0) {
                    numIntsNeeded++;
                }
            }
            // update lutOffs
            ltOffsCntr += numIntsNeeded;
            lutOffs_vec.push_back(ltOffsCntr);

            // retrieve lut bits and dependencies
            unsigned int curSum = 0;
            int bitCounter = 0;

            // consolidate all lut bits into one number
            int numFlushed = 0;
            for (int j=0; j<width; j++) {
                char curBit;
                curBit = lutInfo[2*j];
                curSum += (atoi(&curBit) << (width-j-1));
                bitCounter++;
                if (bitCounter >= 32) {
                    // flush current int to lutBits and reset curSum and bitCounter
                    lutBts_vec.push_back(curSum);
                    bitCounter = 0;
                    curSum = 0;
                    numFlushed++;
                }
            }

            // flush curSum to lutBits if within numIntsNeeded
            if (numFlushed < numIntsNeeded) {
                lutBts_vec.push_back(curSum);
            }

            // handle dependencies
            std::string info(lutInfo);
            std::string deps = info.substr(2*width, info.length()-1);
            std::string dep;
            std::stringstream strStrm(deps);
            while (std::getline(strStrm, dep, ' ')) {
                // test for validity of dependency
                if (dep.find('D' != std::string::npos)) {
                    // increment edge and offset counts
                    (*edgCnt)++;
                    offsCntr++;
                    std::string depID = dep.substr(1, dep.length()-1);
                    // append to csr
                    csr_vec.push_back(std::stoi(depID));
                }
            }

            // update offsets vector
            offs_vec.push_back(offsCntr);

            typIDs[id] = *lutCnt;
            (*lutCnt)++;
        } else {
            fprintf(stderr, "Error parsing line: %s\n", line);
        }
    }
    regInd_vec = indctr;
    typeIDs_vec = typIDs;
    fclose(file);
    return 0;
}





int main() {
    int* numNodes = new int;
    int* numEdges = new int;
    int* numLuts = new int;
    int* numRegs = new int;
    int* numSrcs = new int;
    *numNodes = 0;
    *numEdges = 0;
    *numLuts = 0;
    *numRegs = 0;
    *numSrcs = 0;

    // Vectors to fill in reader func
    std::vector<int> srcIDs_vec;
    std::vector<int> offs_vec;
    std::vector<int> csr_vec;
    std::vector<int> lutOffs_vec;
    std::vector<unsigned int> lutBts_vec;
    std::vector<int> regInd_vec;
    std::vector<int> typeIDs_vec;

    // Arrays to initialize from vectors
    int* host_srcIDs;
    int* host_offs;
    int* host_csr;
    int* host_lutOffs;
    unsigned int* host_lutBits;
    int* host_regInd;
    int* host_typeIDs;

    // Call reader function; this fills all num<field> vars and manipulates vectors
    if (reader(numNodes, numEdges, numLuts, numRegs, numSrcs,
                regInd_vec, typeIDs_vec, srcIDs_vec, offs_vec, csr_vec, lutOffs_vec, lutBts_vec)) {
        return 1;
    }

    // Transfer vector info to arrays
    host_srcIDs = srcIDs_vec.data();
    host_offs = offs_vec.data();
    host_csr = csr_vec.data();
    host_lutOffs = lutOffs_vec.data();
    host_lutBits = lutBts_vec.data();
    host_regInd = regInd_vec.data();
    host_typeIDs = typeIDs_vec.data();

    free(numNodes);
    free(numEdges);
    free(numLuts);
    free(numRegs);
    free(numSrcs);
}





