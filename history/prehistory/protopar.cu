#include <stdio.h>





//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------START LIBRARY-----------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------


struct Node {
    int valid;                  // Valid=0 => output not set   :::   valid=1 => output set
    int nodeID;                 // ID of the node in the adjacency list
    int output;                 // Computed output bit
    int inDegree;               // In-degree of the node
    int* inputs;                // Array of input bits
    int* lut;                   // The LUT (which is an int array, indexed in decimal with binary entries)
    Node** prevNodes;            // Array of previous nodes, used to source input bits

    // Call as Node(nodeID=i, inDegree=x, lut=y, prevNodes=z)
    Node(int id, int indg=0, int* l=nullptr, Node** p=nullptr, int* inps=nullptr, int v=0, int o=0)
        : nodeID(id), inDegree(indg), lut(l), prevNodes(p), inputs(inps), valid(v), output(o) {}

    // Call this on non-source nodes, and only after you've set prevNodes.
    // Purpose: populates the 'inputs' field.
    void inputsFetch() {
        this->inputs = new int[this->inDegree];
        for (int i=0; i<this->inDegree; i++) {
            this->inputs[i] = this->prevNodes[i]->output;
        }
    }

    // Call this only after calling inputsFetch if node is non-source.
    // If node is a source node, then call after setting inputs field.
    void outputCompute() {
        // Set valid bit
        this->valid = 1;
        // Get input bits into a single decimal
        int decInput = 0;
        for (int i=0; i<this->inDegree; i++) {
            if (this->inputs[i]) {
                decInput |= (1 << (inDegree-1-i));
            }
        }
        // Send decimal as key to get val from LUT
        // <<TODO>> needs byte-ization optimizing.
        this->output = this->lut[decInput];
    }
};


struct Col {
    Node** nodes;                // All nodes in the given col, top-down
    Col(Node** ns = nullptr) : nodes(ns) {}
};


// Note: when ColNet is created, we are assuming that nodes and columns have already been declared
struct ColNet {
    Node*** adjList;            // Adjacency list: [[n0|nx,ny,nz],[n1|nx,ny,nz],...,[]]. indexed by nodeID field.
    Col** cols;                 // All cols in the network, left-to-right. cols[0] contains all srcNodes (which have no prevNodes)
    int** srcBits;              // Arr-of-arrs containing src bits with which to populate srcNodes: [[001],[01000],[1101],....,[10]]
    int numCols;                // The number of columns in the colnetwork
    int maxThreadDepth;         // Max number of threads allowed; calculated from #cores on a Turing-arch GPU; run nvidia-smi -q 

    // Call as ColNet(cols, scrBts)
    ColNet(Node*** adj=nullptr, Col** cls=nullptr, int clsCnt=0, int** srcBts=nullptr, int maxThrds=1024)
        : adjList(adj), cols(cls), numCols(clsCnt), srcBits(srcBts), maxThreadDepth(maxThrds) {} 

    //void propagate() {
        // Initialize the source nodes in Col[0] (which is a ptr to a Col) with the bits in srcBits
        // Run node->ouputCompute() on all the source nodes in Col[0]
        // Now, for every node in Col[i] where i:1->#cols, 
            // run node->inputsFetch()
            // run node->outputCompute()
        // Finally, collect the outputs from every node in Col[numCols] using node->output
        // print them all
};


//----------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------END LIBRARY------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------





// Note: kernel can only be called once source-output computation has been handled.
// There is no functionality for source-handling from this kernel.
__global__ void kernelCompute(Col* colInp) {
    //blockIdx.x * blockDim.x + threadIdx.x;
    printf("|GPUGPUGPUGPUGPUGPU|");
    printf("On thread %d, node ID %d", threadIdx.x, colInp->nodes[threadIdx.x]->nodeID); 
}




int main() {
    int srcBitLen = 2;
    int* srcInp0 = new int[srcBitLen]{0,0};
    int* srcInp1 = new int[srcBitLen]{0,1};
    int* srcInp2 = new int[srcBitLen]{1,0};
    int* srcInp3 = new int[srcBitLen]{1,1};
    int* lut1 = new int[4]{0,1,0,0};
    int* lut2 = new int[4]{1,0,0,1};
    int* lut3 = new int[4]{1,0,1,1};
    int* lut4 = new int[4]{1,0,0,0};
    int* lutn = new int[16]{0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0};

    Node* src1 = new Node(1, srcBitLen, lut1, nullptr, srcInp1);
    Node* src2 = new Node(2, srcBitLen, lut2, nullptr, srcInp2);
    Node* src3 = new Node(3, srcBitLen, lut3, nullptr, srcInp3);
    Node* src4 = new Node(4, srcBitLen, lut4, nullptr, srcInp0);
    Node* finNode = new Node(5, 4, lutn, new Node*[4]{src1,src2,src3,src4});
 
    Col* testCol = new Col(new Node*[4]{src1,src3,src2,src4});
    Col* lastCol = new Col(new Node*[1]{finNode});

    kernelCompute<<<1,4>>>(testCol);
    cudaDeviceSynchronize();

    printf("\nnodeIDs in testCol: %d, %d, %d, %d", testCol->nodes[0]->nodeID, testCol->nodes[1]->nodeID, testCol->nodes[2]->nodeID, testCol->nodes[3]->nodeID); 
    printf("\nnodeIDs in lastCol: %d",lastCol->nodes[0]->nodeID);

    src1->outputCompute();
    src2->outputCompute();
    src3->outputCompute();
    src4->outputCompute();

    finNode->inputsFetch();
    finNode->outputCompute();
    
    printf("\n\n\nsrc1 out is: %d\n", src1->output);
    printf("src2 out is: %d\n", src2->output);
    printf("src3 out is: %d\n", src3->output);
    printf("src4 out is: %d\n", src4->output);
    printf("finNode out is: %d\n", finNode->output);
    printf("reached end successfully!\n");

    return 0;
}






