/** @file mnist_mlp.cu
 *  @brief A 4-layer MLP for MNIST.
 *  @author Ang Li (PNNL)
 *
*/

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "utility.h"
#include "sbnn32_param.h"
#include "sbnn64_param.h"
#include "sbnn32.cuh"
#include "sbnn64.cuh"
#include "data.h"

using namespace cooperative_groups;
using namespace std;


int main32();
int main64();

int main()
{
    //main32();
    main64();
}

__global__ void mnist_mlp32(In32LayerParam* bin, Fc32LayerParam* fc1, Fc32LayerParam* fc2, 
        Fc32LayerParam* fc3, Out32LayerParam* bout)
{
    grid_group grid = this_grid();
    //========= Input ============
    //In32LayerBatched(bin);
    In32Layer(bin);
    grid.sync();
    //========== FC1 ============
    //Fc32Layer(fc1);
    Fc32LayerBatched(fc1);
    grid.sync();
    //========== FC2 ============
    //Fc32Layer(fc2);
    Fc32LayerBatched(fc2);
    grid.sync();
    ////========== FC3 ============
    //Fc32Layer(fc3);
    Fc32LayerBatched(fc3);
    grid.sync();
    ////========== Output ===========
    //Out32Layer(bout);
    Out32LayerBatched(bout);
}

//==========================================================================
__global__ void mnist_mlp64(In64LayerParam* bin, Fc64LayerParam* fc1, Fc64LayerParam* fc2, 
        Fc64LayerParam* fc3, Out64LayerParam* bout)
{
    grid_group grid = this_grid();
    SET_KERNEL_TIMER;
    //========= Input ============
    //In64LayerBatched(bin);
    In64Layer(bin);
    grid.sync();
    TICK_KERNEL_TIMER(bin);
    //========== FC1 ============
    Fc64Layer(fc1);
    //Fc64LayerBatched(fc1);
    grid.sync();
    TICK_KERNEL_TIMER(fc1);
    //========== FC2 ============
    //Fc64Layer(fc2);
    Fc64LayerBatched(fc2);
    grid.sync();
    TICK_KERNEL_TIMER(fc2);
    ////========== FC3 ============
    //Fc64Layer(fc3);
    Fc64LayerBatched(fc3);
    grid.sync();
    TICK_KERNEL_TIMER(fc3);
    ////========== Output ===========
    Out64Layer(bout);
    //Out64LayerBatched(bout);
    TICK_KERNEL_TIMER(bout);
}


int main32()
{
    //=============== Configuration =================
    int dev = 7;
    cudaSetDevice(dev);
    const unsigned batch = 1024;
    const unsigned output_size = 10;
    const unsigned n_hidden = 1024;
    const unsigned image_height = 28;
    const unsigned image_width = 28;
    const unsigned image_size = image_height*image_width;

    //=============== Get Input and Label =================
    string mnist_dir = "/home/lian599/data/mnist/t10k-images-idx3-ubyte";
    float* images = (float*)malloc(image_height*image_width*batch*sizeof(float));
    string mnist_label = "/home/lian599/data/mnist/t10k-labels-idx1-ubyte";
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
    read_MNIST_normalized(mnist_dir, mnist_label, images, image_labels, batch);

    //================ Get Weight =================
    //FILE* config_file = fopen("./mnist_mlp.config","r");
    FILE* config_file = fopen("../pytorch/BinaryNet/mlp_mnist.csv","r");

    //================ Set Network =================
    //Input Layer
    In32LayerParam* bin = new In32LayerParam("Fin", batch, image_size);
    In32LayerParam* bin_gpu = bin->initialize(images);
    //Fc1 Layer
    Fc32LayerParam* bfc1 = new Fc32LayerParam("Fc1", batch, image_size, n_hidden); 
    Fc32LayerParam* bfc1_gpu = bfc1->initialize(config_file, bin->get_output_gpu());
    //Fc2 Layer
    Fc32LayerParam* bfc2 = new Fc32LayerParam("Fc2", batch, n_hidden, n_hidden); 
    Fc32LayerParam* bfc2_gpu = bfc2->initialize(config_file, bfc1->get_output_gpu());
    //Fc3 Layer
    Fc32LayerParam* bfc3 = new Fc32LayerParam("Fc3", batch, n_hidden, n_hidden); 
    Fc32LayerParam* bfc3_gpu = bfc3->initialize(config_file, bfc2->get_output_gpu());
    //Out Layer
    Out32LayerParam* bout = new Out32LayerParam("Fout", batch, n_hidden, output_size, true);
    Out32LayerParam* bout_gpu = bout->initialize(config_file, bfc3->get_output_gpu());

    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, mnist_mlp32, numThreads, 0);
    void* args[] = {&bin_gpu, &bfc1_gpu, &bfc2_gpu, &bfc3_gpu, &bout_gpu};

    START_TIMER;
    cudaLaunchCooperativeKernel((void*)mnist_mlp32, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args);
    //mnist_mlp32<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(bin_gpu, 
    //bfc1_gpu, bfc2_gpu, bfc3_gpu, bout_gpu);
    STOP_TIMER;

    //================ Output =================
    float* output = bout->download_output();
    validate_prediction(output, image_labels, output_size, batch);

    //================ Release =================
    delete bin;
    delete bfc1;
    delete bfc2;
    delete bfc3;
    delete bout;

    return 0;
}


int main64()
{
    //=============== Configuration =================
    int dev = 5;
    cudaSetDevice(dev);
    const unsigned batch = 1024;
    const unsigned output_size = 10;
    const unsigned n_hidden = 1024;
    const unsigned image_height = 28;
    const unsigned image_width = 28;
    const unsigned image_size = image_height*image_width;

    //=============== Get Input and Label =================
    string mnist_dir = "/home/lian599/data/mnist/t10k-images-idx3-ubyte";
    float* images = (float*)malloc(image_height*image_width*batch*sizeof(float));
    string mnist_label = "/home/lian599/data/mnist/t10k-labels-idx1-ubyte";
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
    read_MNIST_normalized(mnist_dir, mnist_label, images, image_labels, batch);

    //================ Get Weight =================
    //FILE* config_file = fopen("./mnist_mlp.config","r");
    FILE* config_file = fopen("./pytorch_training/mlp_mnist.csv","r");

    //================ Set Network =================
    //Input Layer
    In64LayerParam* bin = new In64LayerParam("Fin", batch, image_size);
    In64LayerParam* bin_gpu = bin->initialize(images);
    //Fc1 Layer
    Fc64LayerParam* bfc1 = new Fc64LayerParam("Fc1", batch, image_size, n_hidden); 
    Fc64LayerParam* bfc1_gpu = bfc1->initialize(config_file, bin->get_output_gpu());
    //Fc2 Layer
    Fc64LayerParam* bfc2 = new Fc64LayerParam("Fc2", batch, n_hidden, n_hidden); 
    Fc64LayerParam* bfc2_gpu = bfc2->initialize(config_file, bfc1->get_output_gpu());
    //Fc3 Layer
    Fc64LayerParam* bfc3 = new Fc64LayerParam("Fc3", batch, n_hidden, n_hidden); 
    Fc64LayerParam* bfc3_gpu = bfc3->initialize(config_file, bfc2->get_output_gpu());
    //Out Layer
    Out64LayerParam* bout = new Out64LayerParam("Fout", batch, n_hidden, output_size);
    Out64LayerParam* bout_gpu = bout->initialize(config_file, bfc3->get_output_gpu());

    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, mnist_mlp64, numThreads, 0);
    void* args[] = {&bin_gpu, &bfc1_gpu, &bfc2_gpu, &bfc3_gpu, &bout_gpu};

    START_TIMER;

    cudaLaunchCooperativeKernel((void*)mnist_mlp64, numBlocksPerSm*deviceProp.multiProcessorCount, numThreads, args);

    //mnist_mlp64<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads >>> (
    //bin_gpu, bfc1_gpu, bfc2_gpu, bfc3_gpu, bout_gpu);

    STOP_TIMER;
    
    //================ Output =================
    float* output = bout->download_output();
    validate_prediction(output, image_labels, output_size, batch);

    //================ Release =================
    delete bin;
    delete bfc1;
    delete bfc2;
    delete bfc3;
    delete bout;
    return 0;
}





