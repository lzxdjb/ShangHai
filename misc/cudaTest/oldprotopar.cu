#include <stdio.h>

//<TODO> determine: can we optimize this by storing bits instead of whole ints? we only need bits
struct Node {
    int valid;                  // Valid=0 => output not set   :::   valid=1 => output set
    int output;                 // Computed output bit
    int inDegree;               // In-degree of the node
    int* inputs;                // Array of input bits
    int* lut;                   // The LUT (which is an int array, indexed in decimal with binary entries)
    Node** prevNodes;            // Array of previous nodes, used to source input bits       <<TODO>> this should be an array of node-ptrs, so Node** type. FIX

    // Call using:              n = new Node(inDegree=x, lut=y, prevNodes=z);
    Node(int indg=0, int* l=nullptr, Node** p=nullptr, int* inps=nullptr, int v=0, int o=0)
        : inDegree(indg), lut(l), prevNodes(p), inputs(inps), valid(v), output(o) {}

    void inputsFetch() {
        this->inputs = new int[this->inDegree];
        for (int i=0; i<this->inDegree; i++) {
            this->inputs[i] = this->prevNodes[i]->output;
        }
    }

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
        this->output = this->lut[decInput];                     // <<TODO>> needs byte-ization optimizing
    }
};


int main(int argc, char** argv) {
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

    Node* src1 = new Node(srcBitLen, lut1, nullptr, srcInp1);
    Node* src2 = new Node(srcBitLen, lut2, nullptr, srcInp2);
    Node* src3 = new Node(srcBitLen, lut3, nullptr, srcInp3);
    Node* src4 = new Node(srcBitLen, lut4, nullptr, srcInp0);
    Node* finNode = new Node(4, lutn, new Node*[4]{src1,src2,src3,src4});
    
    src1->outputCompute();
    src2->outputCompute();
    src3->outputCompute();
    src4->outputCompute();

    finNode->inputsFetch();
    finNode->outputCompute();

    printf("src1 out is: %d\n", src1->output);
    printf("src2 out is: %d\n", src2->output);
    printf("src3 out is: %d\n", src3->output);
    printf("src4 out is: %d\n", src4->output);
    printf("finNode out is: %d\n", finNode->output);
    printf("reached end successfully!\n");

    return 0;
}






