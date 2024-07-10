#include "type.cuh"

int reader(int* ndCnt, int* edgCnt, int* lutCnt, int* regCnt, int*srcCnt,
        std::vector<int>& regInd_vec,
        std::vector<int>& typeIDs_vec,
        std::vector<int>& srcIDs_vec,
        std::vector<int>& offs_vec,
        std::vector<int>& csr_vec,
        std::vector<int>& lutOffs_vec,
        std::vector<unsigned int>& lutBts_vec);

// void fool();
