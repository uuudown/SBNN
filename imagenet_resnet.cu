/** @file imagenet_vgg.cu
 *  @brief A ResNet-18 network for ImageNet.
 *  @author Ang Li (PNNL)
 *
*/


#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
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

__global__ void resnet32(
        In32Conv32LayerParam* bconv1, 
        Conv32LayerParam* l1b1c1, 
        Conv32LayerParam* l1b1c2,
        Conv32LayerParam* l1b2c1, 
        Conv32LayerParam* l1b2c2,
        Conv32LayerParam* l2b1c1, 
        Conv32LayerParam* l2b1c2,
        Conv32LayerParam* l2b2c1, 
        Conv32LayerParam* l2b2c2,
        Conv32LayerParam* l3b1c1, 
        Conv32LayerParam* l3b1c2,
        Conv32LayerParam* l3b2c1, 
        Conv32LayerParam* l3b2c2,
        Conv32LayerParam* l4b1c1, 
        Conv32LayerParam* l4b1c2,
        Conv32LayerParam* l4b2c1, 
        Conv32LayerParam* l4b2c2,
        Fc32LayerParam* bfc1, 
        Out32LayerParam* bout)
{
    grid_group grid = this_grid();
    //========= Conv1 ============
    In32Conv32Layer(bconv1);
    grid.sync();
    //========= L1B1 ============
    Conv32Layer(l1b1c1);
    grid.sync();
    Conv32Layer(l1b1c2);
    grid.sync();
    //========= L1B2 ============
    Conv32Layer(l1b2c1);
    grid.sync();
    Conv32Layer(l1b2c2);
    grid.sync();

    //========= L2B1 ============
    Conv32Layer(l2b1c1);
    grid.sync();
    Conv32Layer(l2b1c2);
    grid.sync();
    //========= L2B2 ============
    Conv32Layer(l2b2c1);
    grid.sync();
    Conv32Layer(l2b2c2);
    grid.sync();
    //========= L3B1 ============
    Conv32Layer(l3b1c1);
    grid.sync();
    Conv32Layer(l3b1c2);
    grid.sync();
    //========= L4B2 ============
    Conv32Layer(l3b2c1);
    grid.sync();
    Conv32Layer(l3b2c2);
    grid.sync();
    //========= L4B1 ============
    Conv32Layer(l4b1c1);
    grid.sync();
    Conv32Layer(l4b1c2);
    grid.sync();
    //========= L4B2 ============
    Conv32Layer(l4b2c1);
    grid.sync();
    Conv32Layer(l4b2c2);
    grid.sync();
    //========= Fc1 ============
    //Fc32Layer(bfc1);
    Fc32LayerBatched(bfc1);
    grid.sync();
    ////========== Output ===========
    //Out32Layer(bout);
    Out32LayerBatched(bout);
}




__global__ void resnet64(
        In32Conv64LayerParam* bconv1, 
        Conv64LayerParam* l1b1c1, 
        Conv64LayerParam* l1b1c2,
        Conv64LayerParam* l1b2c1, 
        Conv64LayerParam* l1b2c2,
        Conv64LayerParam* l2b1c1, 
        Conv64LayerParam* l2b1c2,
        Conv64LayerParam* l2b2c1, 
        Conv64LayerParam* l2b2c2,
        Conv64LayerParam* l3b1c1, 
        Conv64LayerParam* l3b1c2,
        Conv64LayerParam* l3b2c1, 
        Conv64LayerParam* l3b2c2,
        Conv64LayerParam* l4b1c1, 
        Conv64LayerParam* l4b1c2,
        Conv64LayerParam* l4b2c1, 
        Conv64LayerParam* l4b2c2,
        Fc64LayerParam* bfc1, 
        Out64LayerParam* bout)
{
    grid_group grid = this_grid();
    //========= Conv1 ============
    In32Conv64Layer(bconv1);
    grid.sync();
    //========= L1B1 ============
    Conv64Layer(l1b1c1);
    grid.sync();
    Conv64Layer(l1b1c2);
    grid.sync();
    //========= L1B2 ============
    Conv64Layer(l1b2c1);
    grid.sync();
    Conv64Layer(l1b2c2);
    grid.sync();
    //========= L2B1 ============
    Conv64Layer(l2b1c1);
    grid.sync();
    Conv64Layer(l2b1c2);
    grid.sync();
    //========= L2B2 ============
    Conv64Layer(l2b2c1);
    grid.sync();
    Conv64Layer(l2b2c2);
    grid.sync();
    //========= L3B1 ============
    Conv64Layer(l3b1c1);
    grid.sync();
    Conv64Layer(l3b1c2);
    grid.sync();
    //========= L3B2 ============
    Conv64Layer(l3b2c1);
    grid.sync();
    Conv64Layer(l3b2c2);
    grid.sync();
    //========= L4B1 ============
    Conv64Layer(l4b1c1);
    grid.sync();
    Conv64Layer(l4b1c2);
    grid.sync();
    //========= L4B2 ============
    Conv64Layer(l4b2c1);
    grid.sync();
    Conv64Layer(l4b2c2);
    grid.sync();
    //========= Fc1 ============
    //Fc64Layer(bfc1);
    Fc64LayerBatched(bfc1);
    grid.sync();
    ////========== Output ===========
    //Out64Layer(bout);
    Out64LayerBatched(bout);
}



//Out64LayerBatched(bout);
//TICK_KERNEL_TIMER(bout);
 
int main32()
{
    int dev = 6;
    cudaSetDevice(dev);

    const unsigned batch = 32;
    const unsigned output_size = 1000;
    const unsigned image_height = 224;
    const unsigned image_width = 224;
    const unsigned image_channel = 3;

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
    read_ImageNet_normalized("./imagenet_files.txt", images, image_labels, batch);

    //================ Get Weight =================
    FILE* config_file = fopen("./pytorch_training/resnet_imagenet.csv","r");
    //FILE* config_file = fopen("./pytorch_training/alexnet_imagenet.csv","r");

    //================ Set Network =================
    //Layer-0
    In32Conv32LayerParam* bconv1 = new In32Conv32LayerParam("Conv1", image_height, image_width, 
            7, 7, 3, 64, batch,4,4,true,0,0,false,true);//save residual 
    In32Conv32LayerParam* bconv1_gpu = bconv1->initialize(images, config_file);



    //Layer-1, basic-block-1, conv1
    Conv32LayerParam* l1b1c1 = new Conv32LayerParam("L1B1C1", bconv1->output_height, 
            bconv1->output_width, 3, 3, 64, 64, batch);
    Conv32LayerParam* l1b1c1_gpu = l1b1c1->initialize(config_file, bconv1->get_output_gpu());
    //Layer-1, basic-block-1, conv2
    Conv32LayerParam* l1b1c2 = new Conv32LayerParam("L1B1C2", l1b1c1->output_height, 
            l1b1c1->output_width, 3, 3, 64, 64, batch,1,1,true,0,0,false,true,true,64);
    Conv32LayerParam* l1b1c2_gpu = l1b1c2->initialize(config_file, l1b1c1->get_output_gpu(),
            bconv1->get_residual_gpu());


    
    //Layer-1, basic-block-2, conv1
    Conv32LayerParam* l1b2c1 = new Conv32LayerParam("L1B2C1", l1b1c2->output_height, 
            l1b1c2->output_width, 3, 3, 64, 64, batch);
    Conv32LayerParam* l1b2c1_gpu = l1b2c1->initialize(config_file, l1b1c2->get_output_gpu());
    //Layer-1, basic-block-2, conv2
    Conv32LayerParam* l1b2c2 = new Conv32LayerParam("L1B2C2", l1b2c1->output_height, 
            l1b2c1->output_width, 3, 3, 64, 64, batch,1,1,true,0,0,false,true,true,64);
    Conv32LayerParam* l1b2c2_gpu = l1b2c2->initialize(config_file, l1b2c1->get_output_gpu(),
            l1b1c2->get_residual_gpu());



    //=============
    //Layer-2, basic-block-1, conv1
    Conv32LayerParam* l2b1c1 = new Conv32LayerParam("L2B1C1", l1b2c2->output_height, 
            l1b2c2->output_width, 3, 3, 64, 128, batch, 2, 2);
    Conv32LayerParam* l2b1c1_gpu = l2b1c1->initialize(config_file, l1b2c2->get_output_gpu());
    //Layer-2, basic-block-1, conv2
    Conv32LayerParam* l2b1c2 = new Conv32LayerParam("L2B1C2", l2b1c1->output_height, 
            l2b1c1->output_width, 3, 3, 128, 128, batch,1,1,true,0,0,false,true,true,64,true);
    Conv32LayerParam* l2b1c2_gpu = l2b1c2->initialize(config_file, l2b1c1->get_output_gpu(),
            l1b2c2->get_residual_gpu());



    //Layer-2, basic-block-2, conv1
    Conv32LayerParam* l2b2c1 = new Conv32LayerParam("L2B2C1", l2b1c2->output_height, 
            l2b1c2->output_width, 3, 3, 128, 128, batch, 1, 1);
    Conv32LayerParam* l2b2c1_gpu = l2b2c1->initialize(config_file, l2b1c2->get_output_gpu());
    //Layer-2, basic-block-2, conv2
    Conv32LayerParam* l2b2c2 = new Conv32LayerParam("L2B2C2", l2b2c1->output_height, 
            l2b2c1->output_width, 3, 3, 128, 128, batch,1,1,true,0,0,false,true,true,128);
    Conv32LayerParam* l2b2c2_gpu = l2b2c2->initialize(config_file, l2b2c1->get_output_gpu(),
            l2b1c2->get_residual_gpu());



    //=============
    //Layer-3, basic-block-1, conv1
    Conv32LayerParam* l3b1c1 = new Conv32LayerParam("L3B1C1", l2b2c2->output_height, 
            l2b2c2->output_width, 3, 3, 128, 256, batch, 2, 2);
    Conv32LayerParam* l3b1c1_gpu = l3b1c1->initialize(config_file, l2b2c2->get_output_gpu());
    //Layer-3, basic-block-1, conv2
    Conv32LayerParam* l3b1c2 = new Conv32LayerParam("L3B1C2", l3b1c1->output_height, 
            l3b1c1->output_width, 3, 3, 256, 256, batch,1,1,true,0,0,false,true,true,128,true);
    Conv32LayerParam* l3b1c2_gpu = l3b1c2->initialize(config_file, l3b1c1->get_output_gpu(),
            l2b2c2->get_residual_gpu());

    //Layer-3, basic-block-2, conv1
    Conv32LayerParam* l3b2c1 = new Conv32LayerParam("L3B2C1", l3b1c2->output_height, 
            l3b1c2->output_width, 3, 3, 256, 256, batch, 1, 1);
    Conv32LayerParam* l3b2c1_gpu = l3b2c1->initialize(config_file, l3b1c2->get_output_gpu());
    //Layer-3, basic-block-2, conv2
    Conv32LayerParam* l3b2c2 = new Conv32LayerParam("L3B2C2", l3b2c1->output_height, 
            l3b2c1->output_width, 3, 3, 256, 256, batch,1,1,true,0,0,false,true,true,256);
    Conv32LayerParam* l3b2c2_gpu = l3b2c2->initialize(config_file, l3b2c1->get_output_gpu(),
            l3b1c2->get_residual_gpu());

    //=============
    //Layer-4, basic-block-1, conv1
    Conv32LayerParam* l4b1c1 = new Conv32LayerParam("L4B1C1", l3b2c2->output_height, 
            l3b2c2->output_width, 3, 3, 256, 512, batch, 2, 2);
    Conv32LayerParam* l4b1c1_gpu = l4b1c1->initialize(config_file, l3b2c2->get_output_gpu());
    //Layer-4, basic-block-1, conv2
    Conv32LayerParam* l4b1c2 = new Conv32LayerParam("L4B1C2", l4b1c1->output_height, 
            l4b1c1->output_width, 3, 3, 512, 512, batch,1,1,true,0,0,false,true,true,256,true);
    Conv32LayerParam* l4b1c2_gpu = l4b1c2->initialize(config_file, l4b1c1->get_output_gpu(),
            l3b2c2->get_residual_gpu());

    //Layer-4, basic-block-2, conv1
    Conv32LayerParam* l4b2c1 = new Conv32LayerParam("L4B2C1", l4b1c2->output_height, 
            l4b1c2->output_width, 3, 3, 512, 512, batch, 1, 1);
    Conv32LayerParam* l4b2c1_gpu = l4b2c1->initialize(config_file, l4b1c2->get_output_gpu());
    //Layer-4, basic-block-2, conv2
    Conv32LayerParam* l4b2c2 = new Conv32LayerParam("L4B2C2", l4b2c1->output_height, 
            l4b2c1->output_width, 3, 3, 512, 512, batch,1,1,true,0,0,true,false,true,512);
    Conv32LayerParam* l4b2c2_gpu = l4b2c2->initialize(config_file, l4b2c1->get_output_gpu(),
            l4b1c2->get_residual_gpu());

    //=============
    //Layer-5
    Fc32LayerParam* bfc1 = new Fc32LayerParam("Fc1", batch, (l4b2c2->output_height)
            *(l4b2c2->output_width)*512, 512); 
    Fc32LayerParam* bfc1_gpu = bfc1->initialize(config_file, l4b2c2->get_output_gpu());
    //Out Layer
    Out32LayerParam* bout = new Out32LayerParam("Fout", batch, 512, output_size, true);
    Out32LayerParam* bout_gpu = bout->initialize(config_file, bfc1->get_output_gpu());  



    //float aaa = 0;
    //fscanf(config_file, "%f", &aaa);
    //printf("\n------%f----\n",aaa);

    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    int shared_memory = 512*sizeof(int)*32;
    cudaFuncSetAttribute(resnet32, cudaFuncAttributeMaxDynamicSharedMemorySize,98304);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, resnet32, numThreads, shared_memory);
    //cudaFuncSetAttribute(resnet32, cudaFuncAttributePreferredSharedMemoryCarveout,0);

    void* args[] = {&bconv1_gpu, 
        &l1b1c1_gpu, 
        &l1b1c2_gpu,
        &l1b2c1_gpu,
        &l1b2c2_gpu,
        &l2b1c1_gpu, 
        &l2b1c2_gpu,
        &l2b2c1_gpu,
        &l2b2c2_gpu,
        &l3b1c1_gpu, 
        &l3b1c2_gpu,
        &l3b2c1_gpu,
        &l3b2c2_gpu,
        &l4b1c1_gpu, 
        &l4b1c2_gpu,
        &l4b2c1_gpu,
        &l4b2c2_gpu,
        &bfc1_gpu,
        &bout_gpu};




    START_TIMER;

    cudaLaunchCooperativeKernel((void*)resnet32, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);

    STOP_TIMER;

    //================ Output =================
    float* output = bout->download_output();
    validate_prediction(output, image_labels, output_size, batch);

    delete bconv1;
    delete l1b1c1;
    delete l1b1c2;
    delete l1b2c1;
    delete l1b2c2;

    delete l2b1c1;
    delete l2b1c2;
    delete l2b2c1;
    delete l2b2c2;

    delete l3b1c1;
    delete l3b1c2;
    delete l3b2c1;
    delete l3b2c2;

    delete l4b1c1;
    delete l4b1c2;
    delete l4b2c1;
    delete l4b2c2;

    delete bfc1;
    delete bout;



    return 0;
}

int main64()
{
    int dev = 6;
    cudaSetDevice(dev);
    const unsigned batch = 32;
    const unsigned output_size = 1000;
    const unsigned image_height = 224;
    const unsigned image_width = 224;
    const unsigned image_channel = 3;

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
    read_ImageNet_normalized("./imagenet_files.txt", images, image_labels, batch);

    //================ Get Weight =================
    FILE* config_file = fopen("./pytorch_training/resnet_imagenet.csv","r");

    //================ Set Network =================
    //Layer-0
    In32Conv64LayerParam* bconv1 = new In32Conv64LayerParam("Conv1", image_height, image_width, 
            7, 7, 3, 64, batch,4,4,true,0,0,false,true);//save residual 
    In32Conv64LayerParam* bconv1_gpu = bconv1->initialize(images, config_file);

    //Layer-1, basic-block-1, conv1
    Conv64LayerParam* l1b1c1 = new Conv64LayerParam("L1B1C1", bconv1->output_height, 
            bconv1->output_width, 3, 3, 64, 64, batch);
    Conv64LayerParam* l1b1c1_gpu = l1b1c1->initialize(config_file, bconv1->get_output_gpu());
    //Layer-1, basic-block-1, conv2
    Conv64LayerParam* l1b1c2 = new Conv64LayerParam("L1B1C2", l1b1c1->output_height, 
            l1b1c1->output_width, 3, 3, 64, 64, batch,1,1,true,0,0,false,true,true,64);
    Conv64LayerParam* l1b1c2_gpu = l1b1c2->initialize(config_file, l1b1c1->get_output_gpu(),
            bconv1->get_residual_gpu());

    //Layer-1, basic-block-2, conv1
    Conv64LayerParam* l1b2c1 = new Conv64LayerParam("L1B2C1", l1b1c2->output_height, 
            l1b1c2->output_width, 3, 3, 64, 64, batch);
    Conv64LayerParam* l1b2c1_gpu = l1b2c1->initialize(config_file, l1b1c2->get_output_gpu());
    //Layer-1, basic-block-2, conv2
    Conv64LayerParam* l1b2c2 = new Conv64LayerParam("L1B2C2", l1b2c1->output_height, 
            l1b2c1->output_width, 3, 3, 64, 64, batch,1,1,true,0,0,false,true,true,64);
    Conv64LayerParam* l1b2c2_gpu = l1b2c2->initialize(config_file, l1b2c1->get_output_gpu(),
            l1b1c2->get_residual_gpu());

    //=============
    //Layer-2, basic-block-1, conv1
    Conv64LayerParam* l2b1c1 = new Conv64LayerParam("L2B1C1", l1b2c2->output_height, 
            l1b2c2->output_width, 3, 3, 64, 128, batch, 2, 2);
    Conv64LayerParam* l2b1c1_gpu = l2b1c1->initialize(config_file, l1b2c2->get_output_gpu());
    //Layer-2, basic-block-1, conv2
    Conv64LayerParam* l2b1c2 = new Conv64LayerParam("L2B1C2", l2b1c1->output_height, 
            l2b1c1->output_width, 3, 3, 128, 128, batch,1,1,true,0,0,false,true,true,64,true);
    Conv64LayerParam* l2b1c2_gpu = l2b1c2->initialize(config_file, l2b1c1->get_output_gpu(),
            l1b2c2->get_residual_gpu());


    //Layer-2, basic-block-2, conv1
    Conv64LayerParam* l2b2c1 = new Conv64LayerParam("L2B2C1", l2b1c2->output_height, 
            l2b1c2->output_width, 3, 3, 128, 128, batch, 1, 1);
    Conv64LayerParam* l2b2c1_gpu = l2b2c1->initialize(config_file, l2b1c2->get_output_gpu());
    //Layer-2, basic-block-2, conv2
    Conv64LayerParam* l2b2c2 = new Conv64LayerParam("L2B2C2", l2b2c1->output_height, 
            l2b2c1->output_width, 3, 3, 128, 128, batch,1,1,true,0,0,false,true,true,128);
    Conv64LayerParam* l2b2c2_gpu = l2b2c2->initialize(config_file, l2b2c1->get_output_gpu(),
            l2b1c2->get_residual_gpu());

    //=============
    //Layer-3, basic-block-1, conv1
    Conv64LayerParam* l3b1c1 = new Conv64LayerParam("L3B1C1", l2b2c2->output_height, 
            l2b2c2->output_width, 3, 3, 128, 256, batch, 2, 2);
    Conv64LayerParam* l3b1c1_gpu = l3b1c1->initialize(config_file, l2b2c2->get_output_gpu());
    //Layer-3, basic-block-1, conv2
    Conv64LayerParam* l3b1c2 = new Conv64LayerParam("L3B1C2", l3b1c1->output_height, 
            l3b1c1->output_width, 3, 3, 256, 256, batch,1,1,true,0,0,false,true,true,128,true);
    Conv64LayerParam* l3b1c2_gpu = l3b1c2->initialize(config_file, l3b1c1->get_output_gpu(),
            l2b2c2->get_residual_gpu());

    //Layer-3, basic-block-2, conv1
    Conv64LayerParam* l3b2c1 = new Conv64LayerParam("L3B2C1", l3b1c2->output_height, 
            l3b1c2->output_width, 3, 3, 256, 256, batch, 1, 1);
    Conv64LayerParam* l3b2c1_gpu = l3b2c1->initialize(config_file, l3b1c2->get_output_gpu());
    //Layer-3, basic-block-2, conv2
    Conv64LayerParam* l3b2c2 = new Conv64LayerParam("L3B2C2", l3b2c1->output_height, 
            l3b2c1->output_width, 3, 3, 256, 256, batch,1,1,true,0,0,false,true,true,256);
    Conv64LayerParam* l3b2c2_gpu = l3b2c2->initialize(config_file, l3b2c1->get_output_gpu(),
            l3b1c2->get_residual_gpu());


    //=============
    //Layer-4, basic-block-1, conv1
    Conv64LayerParam* l4b1c1 = new Conv64LayerParam("L4B1C1", l3b2c2->output_height, 
            l3b2c2->output_width, 3, 3, 256, 512, batch, 2, 2);
    Conv64LayerParam* l4b1c1_gpu = l4b1c1->initialize(config_file, l3b2c2->get_output_gpu());
    //Layer-4, basic-block-1, conv2
    Conv64LayerParam* l4b1c2 = new Conv64LayerParam("L4B1C2", l4b1c1->output_height, 
            l4b1c1->output_width, 3, 3, 512, 512, batch,1,1,true,0,0,false,true,true,256,true);
    Conv64LayerParam* l4b1c2_gpu = l4b1c2->initialize(config_file, l4b1c1->get_output_gpu(),
            l3b2c2->get_residual_gpu());

    //Layer-4, basic-block-2, conv1
    Conv64LayerParam* l4b2c1 = new Conv64LayerParam("L4B2C1", l4b1c2->output_height, 
            l4b1c2->output_width, 3, 3, 512, 512, batch, 1, 1);
    Conv64LayerParam* l4b2c1_gpu = l4b2c1->initialize(config_file, l4b1c2->get_output_gpu());
    //Layer-4, basic-block-2, conv2
    Conv64LayerParam* l4b2c2 = new Conv64LayerParam("L4B2C2", l4b2c1->output_height, 
            l4b2c1->output_width, 3, 3, 512, 512, batch,1,1,true,0,0,true,false,true,512);
    Conv64LayerParam* l4b2c2_gpu = l4b2c2->initialize(config_file, l4b2c1->get_output_gpu(),
            l4b1c2->get_residual_gpu());

    //=============
    //Layer-5
    Fc64LayerParam* bfc1 = new Fc64LayerParam("Fc1", batch, (l4b2c2->output_height)
            *(l4b2c2->output_width)*512, 512); 
    Fc64LayerParam* bfc1_gpu = bfc1->initialize(config_file, l4b2c2->get_output_gpu());
    //Out Layer
    Out64LayerParam* bout = new Out64LayerParam("Fout", batch, 512, output_size, true);
    Out64LayerParam* bout_gpu = bout->initialize(config_file, bfc1->get_output_gpu());  

    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    int shared_memory = 512*sizeof(int)*32;
    cudaFuncSetAttribute(resnet64, cudaFuncAttributeMaxDynamicSharedMemorySize,98304);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, resnet64, numThreads, shared_memory);
    //cudaFuncSetAttribute(resnet64, cudaFuncAttributePreferredSharedMemoryCarveout,0);

    void* args[] = {&bconv1_gpu, 
        &l1b1c1_gpu, 
        &l1b1c2_gpu,
        &l1b2c1_gpu,
        &l1b2c2_gpu,
        &l2b1c1_gpu, 
        &l2b1c2_gpu,
        &l2b2c1_gpu,
        &l2b2c2_gpu,
        &l3b1c1_gpu, 
        &l3b1c2_gpu,
        &l3b2c1_gpu,
        &l3b2c2_gpu,
        &l4b1c1_gpu, 
        &l4b1c2_gpu,
        &l4b2c1_gpu,
        &l4b2c2_gpu,
        &bfc1_gpu,
        &bout_gpu};

    START_TIMER;

    cudaLaunchCooperativeKernel((void*)resnet64, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);

    STOP_TIMER;

    //================ Output =================
    float* output = bout->download_output();
    validate_prediction(output, image_labels, output_size, batch);

    delete bconv1;
    delete l1b1c1;
    delete l1b1c2;
    delete l1b2c1;
    delete l1b2c2;

    delete l2b1c1;
    delete l2b1c2;
    delete l2b2c1;
    delete l2b2c2;

    delete l3b1c1;
    delete l3b1c2;
    delete l3b2c1;
    delete l3b2c2;

    delete l4b1c1;
    delete l4b1c2;
    delete l4b2c1;
    delete l4b2c2;

    delete bfc1;
    delete bout;

    return 0;
}























