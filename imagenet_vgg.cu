/** @file imagenet_vgg.cu
 *  @brief A VGG-16 network for ImageNet.
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

__global__ void vggnet32(
        In32Conv32LayerParam* bconv1, 
        Conv32LayerParam* bconv2, 
        Conv32LayerParam* bconv3,
        Conv32LayerParam* bconv4, 
        Conv32LayerParam* bconv5, 
        Conv32LayerParam* bconv6,
        Conv32LayerParam* bconv7,
        Conv32LayerParam* bconv8,
        Conv32LayerParam* bconv9,
        Conv32LayerParam* bconv10,
        Conv32LayerParam* bconv11,
        Conv32LayerParam* bconv12,
        Conv32LayerParam* bconv13,
        Fc32LayerParam* bfc1, 
        Fc32LayerParam* bfc2, 
        Out32LayerParam* bout)
{
    grid_group grid = this_grid();
    //========= Conv1 ============
    In32Conv32Layer(bconv1);
    grid.sync();
    //========= Conv2 ============
    ConvPool32Layer(bconv2);
    grid.sync();
    //========= Conv3 ============
    Conv32Layer(bconv3);
    grid.sync();
    //========= Conv4 ============
    ConvPool32Layer(bconv4);
    grid.sync();
    //========= Conv5 ============
    Conv32Layer(bconv5);
    grid.sync();
    //========= Conv6 ============
    Conv32Layer(bconv6);
    grid.sync();
    //========= Conv7 ============
    ConvPool32Layer(bconv7);
    grid.sync();
    //========= Conv8 ============
    Conv32Layer(bconv8);
    grid.sync();
    //========= Conv9 ============
    Conv32Layer(bconv9);
    grid.sync();
    //========= Conv10 ============
    ConvPool32Layer(bconv10);
    grid.sync();
    //========= Conv11 ============
    Conv32Layer(bconv11);
    grid.sync();
    //========= Conv12 ============
    Conv32Layer(bconv12);
    grid.sync();
    //========= Conv13 ============
    ConvPool32Layer(bconv13);
    grid.sync();
    //========= Fc1 ============
    Fc32Layer(bfc1);
    //Fc32LayerBatched(bfc1);
    grid.sync();
    //========= Fc2 ============
    Fc32Layer(bfc2);
    //Fc32LayerBatched(bfc2);
    grid.sync();
    ////========== Output ===========
    Out32Layer(bout);
    //Out32LayerBatched(bout);
}


__global__ void vggnet64(
        In32Conv64LayerParam* bconv1, 
        Conv64LayerParam* bconv2, 
        Conv64LayerParam* bconv3,
        Conv64LayerParam* bconv4, 
        Conv64LayerParam* bconv5, 
        Conv64LayerParam* bconv6,
        Conv64LayerParam* bconv7, 
        Conv64LayerParam* bconv8,
        Conv64LayerParam* bconv9, 
        Conv64LayerParam* bconv10, 
        Conv64LayerParam* bconv11,
        Conv64LayerParam* bconv12,
        Conv64LayerParam* bconv13,
        Fc64LayerParam* bfc1, 
        Fc64LayerParam* bfc2, 
        Out64LayerParam* bout)
{
    grid_group grid = this_grid();
    SET_KERNEL_TIMER;
    
    //========= Conv1 ============
    In32Conv64Layer(bconv1);
    grid.sync();
    TICK_KERNEL_TIMER(bconv1);
    //========= Conv2 ============
    ConvPool64Layer(bconv2);
    grid.sync();
    TICK_KERNEL_TIMER(bconv2);
    //========= Conv3 ============
    Conv64Layer(bconv3);
    grid.sync();
    TICK_KERNEL_TIMER(bconv3);
    //========= Conv4 ============
    ConvPool64Layer(bconv4);
    grid.sync();
    TICK_KERNEL_TIMER(bconv4);
    //========= Conv5 ============
    Conv64Layer(bconv5);
    grid.sync();
    TICK_KERNEL_TIMER(bconv5);
    //========= Conv6 ============
    Conv64Layer(bconv6);
    grid.sync();
    TICK_KERNEL_TIMER(bconv6);
    //========= Conv7 ============
    ConvPool64Layer(bconv7);
    grid.sync();
    TICK_KERNEL_TIMER(bconv7);
    //========= Conv8 ============
    Conv64Layer(bconv8);
    grid.sync();
    TICK_KERNEL_TIMER(bconv8);
    //========= Conv9 ============
    Conv64Layer(bconv9);
    grid.sync();
    TICK_KERNEL_TIMER(bconv9);
    //========= Conv10 ============
    ConvPool64Layer(bconv10);
    grid.sync();
    TICK_KERNEL_TIMER(bconv10);
    //========= Conv11 ============
    Conv64Layer(bconv11);
    grid.sync();
    TICK_KERNEL_TIMER(bconv11);
    //========= Conv12 ============
    Conv64Layer(bconv12);
    grid.sync();
    TICK_KERNEL_TIMER(bconv12);
    //========= Conv13 ============
    ConvPool64Layer(bconv13);
    grid.sync();
    TICK_KERNEL_TIMER(bconv13);
    //========= Fc1 ============
    //Fc64Layer(bfc1);
    Fc64LayerBatched(bfc1);
    grid.sync();
    TICK_KERNEL_TIMER(bfc1);
    //========= Fc2 ============
    //Fc64Layer(bfc2);
    Fc64LayerBatched(bfc2);
    grid.sync();
    TICK_KERNEL_TIMER(bfc2);
    ////========== Output ===========
    //Out64Layer(bout);
    Out64LayerBatched(bout);
    TICK_KERNEL_TIMER(bout);
}
  
int main32()
{
    int dev = 5;
    cudaSetDevice(dev);

    const unsigned batch = 32;
    const unsigned output_size = 1000;
    const unsigned image_height = 224;
    const unsigned image_width = 224;
    const unsigned image_channel = 3;
    const unsigned filter_height = 3;
    const unsigned filter_width = 3;
    const unsigned n_hidden = 4096;

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
    read_ImageNet_normalized("./imagenet_files.txt", images, image_labels, batch);

    //================ Get Weight =================
    FILE* config_file = fopen("../pytorch/BinaryNet/vgg_imagenet.csv","r");

    //================ Set Network =================
    //Bconv1 Layer
    In32Conv32LayerParam* bconv1 = new In32Conv32LayerParam("Conv1", image_height, image_width, 
            filter_height, filter_width, 3, 64, batch); 
    In32Conv32LayerParam* bconv1_gpu = bconv1->initialize(images, config_file);
    //Bconv2 Layer
    Conv32LayerParam* bconv2 = new Conv32LayerParam("Conv2", bconv1->output_height, 
            bconv1->output_width, filter_height, filter_width, 64, 64, batch, 1, 1,
            true, 2, 2, false);
    Conv32LayerParam* bconv2_gpu = bconv2->initialize(config_file, bconv1->get_output_gpu());
    //Bconv3 Layer
    Conv32LayerParam* bconv3 = new Conv32LayerParam("Conv3", bconv2->output_height, 
            bconv2->output_width, filter_height, filter_width, 64, 128, batch);
    Conv32LayerParam* bconv3_gpu = bconv3->initialize(config_file, bconv2->get_output_gpu());
    //Bconv4 Layer
    Conv32LayerParam* bconv4 = new Conv32LayerParam("Conv4", bconv3->output_height, 
            bconv3->output_width, filter_height, filter_width, 128, 128, batch, 1, 1,
            true, 2, 2, false);
    Conv32LayerParam* bconv4_gpu = bconv4->initialize(config_file, bconv3->get_output_gpu());
    //Bconv5 Layer
    Conv32LayerParam* bconv5 = new Conv32LayerParam("Conv5", bconv4->output_height, 
            bconv4->output_width, filter_height, filter_width, 128, 256, batch);
    Conv32LayerParam* bconv5_gpu = bconv5->initialize(config_file, bconv4->get_output_gpu());
    //Bconv6 Layer
    Conv32LayerParam* bconv6 = new Conv32LayerParam("Conv6", bconv5->output_height, 
            bconv5->output_width, filter_height, filter_width, 256, 256, batch);
    Conv32LayerParam* bconv6_gpu = bconv6->initialize(config_file, bconv5->get_output_gpu());
    //Bconv7 Layer
    Conv32LayerParam* bconv7 = new Conv32LayerParam("Conv7", bconv6->output_height, 
            bconv6->output_width, filter_height, filter_width, 256, 256, batch, 1, 1,
            true, 2, 2, false);
    Conv32LayerParam* bconv7_gpu = bconv7->initialize(config_file, bconv6->get_output_gpu());
    //Bconv8 Layer
    Conv32LayerParam* bconv8 = new Conv32LayerParam("Conv8", bconv7->output_height, 
            bconv7->output_width, filter_height, filter_width, 256, 512, batch);
    Conv32LayerParam* bconv8_gpu = bconv8->initialize(config_file, bconv7->get_output_gpu());
    //Bconv9 Layer
    Conv32LayerParam* bconv9 = new Conv32LayerParam("Conv9", bconv8->output_height, 
            bconv8->output_width, filter_height, filter_width, 512, 512, batch);
    Conv32LayerParam* bconv9_gpu = bconv9->initialize(config_file, bconv8->get_output_gpu());
    //Bconv10 Layer
    Conv32LayerParam* bconv10 = new Conv32LayerParam("Conv10", bconv9->output_height, 
            bconv9->output_width, filter_height, filter_width, 512, 512, batch, 1, 1,
            true, 2, 2, false);
    Conv32LayerParam* bconv10_gpu = bconv10->initialize(config_file, bconv9->get_output_gpu());
    //Bconv11 Layer
    Conv32LayerParam* bconv11 = new Conv32LayerParam("Conv11", bconv10->output_height, 
            bconv10->output_width, filter_height, filter_width, 512, 512, batch);
    Conv32LayerParam* bconv11_gpu = bconv11->initialize(config_file, bconv10->get_output_gpu());
    //Bconv12 Layer
    Conv32LayerParam* bconv12 = new Conv32LayerParam("Conv12", bconv11->output_height, 
            bconv11->output_width, filter_height, filter_width, 512, 512, batch);
    Conv32LayerParam* bconv12_gpu = bconv12->initialize(config_file, bconv11->get_output_gpu());
    //Bconv13 Layer
    Conv32LayerParam* bconv13 = new Conv32LayerParam("Conv13", bconv12->output_height, 
            bconv12->output_width, filter_height, filter_width, 512, 512, batch, 1, 1,
            true, 2, 2, true);
    Conv32LayerParam* bconv13_gpu = bconv13->initialize(config_file, bconv12->get_output_gpu());
    //Fc1 Layer
    Fc32LayerParam* bfc1 = new Fc32LayerParam("Fc1", batch, (bconv13->output_height)
            *(bconv13->output_width)*512, n_hidden); 
    Fc32LayerParam* bfc1_gpu = bfc1->initialize(config_file, bconv13->get_output_gpu());
    //Fc2 Layer
    Fc32LayerParam* bfc2 = new Fc32LayerParam("Fc2", batch, n_hidden, n_hidden); 
    Fc32LayerParam* bfc2_gpu = bfc2->initialize(config_file, bfc1->get_output_gpu());
    //Out Layer
    Out32LayerParam* bout = new Out32LayerParam("Fout", batch, n_hidden, output_size, true);
    Out32LayerParam* bout_gpu = bout->initialize(config_file, bfc2->get_output_gpu());  

    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    int shared_memory = 512*sizeof(int)*32;
    cudaFuncSetAttribute(vggnet32, cudaFuncAttributeMaxDynamicSharedMemorySize,98304);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, vggnet32, numThreads, shared_memory);
    //cudaFuncSetAttribute(alexnet32, cudaFuncAttributePreferredSharedMemoryCarveout,0);

    void* args[] = {&bconv1_gpu, &bconv2_gpu, &bconv3_gpu, &bconv4_gpu, &bconv5_gpu, &bconv6_gpu,
        &bconv7_gpu, &bconv8_gpu, &bconv9_gpu, &bconv10_gpu, &bconv11_gpu, &bconv12_gpu, &bconv13_gpu,
        &bfc1_gpu, &bfc2_gpu, &bout_gpu};

    START_TIMER;

    cudaLaunchCooperativeKernel((void*)vggnet32, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);

    STOP_TIMER;

    //float* ss = bfc1->download_full_output();
    //int a = 20980;
    //int b = 21080;
    //int max_width = 4;
    //for (int i=a; i<b; i++)
    //{
        //printf("%*.0f ",max_width, ss[i]);
        //if ( (i-a+1)%18 == 0)
            //printf("\n");
    //}
    //printf("\n");

    //================ Output =================
    float* output = bout->download_output();
    validate_prediction(output, image_labels, output_size, batch);

    delete bconv1;
    delete bconv2;
    delete bconv3;
    delete bconv4;
    delete bconv5;
    delete bconv6;
    delete bconv7;
    delete bconv8;
    delete bconv9;
    delete bconv10;
    delete bconv11;
    delete bconv12;
    delete bconv13;
    delete bfc1;
    delete bfc2;
    delete bout;

    return 0;
}





     
int main64()
{
    int dev = 4;
    cudaSetDevice(dev);

    const unsigned batch = 32;
    const unsigned output_size = 1000;
    const unsigned image_height = 224;
    const unsigned image_width = 224;
    const unsigned image_channel = 3;
    const unsigned filter_height = 3;
    const unsigned filter_width = 3;
    const unsigned n_hidden = 4096;

    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
    read_ImageNet_normalized("./imagenet_files.txt", images, image_labels, batch);
    
    //================ Get Weight =================
    FILE* config_file = fopen("../pytorch/BinaryNet/vgg_imagenet.csv","r");
    //================ Set Network =================
    //Bconv1 Layer
    In32Conv64LayerParam* bconv1 = new In32Conv64LayerParam("Conv1", image_height, image_width, 
            filter_height, filter_width, 3, 64, batch); 
    In32Conv64LayerParam* bconv1_gpu = bconv1->initialize(images, config_file);
    //Bconv2 Layer
    Conv64LayerParam* bconv2 = new Conv64LayerParam("Conv2", bconv1->output_height, 
            bconv1->output_width, filter_height, filter_width, 64, 64, batch, 1, 1,
            true, 2, 2, false);
    Conv64LayerParam* bconv2_gpu = bconv2->initialize(config_file, bconv1->get_output_gpu());
    //Bconv3 Layer
    Conv64LayerParam* bconv3 = new Conv64LayerParam("Conv3", bconv2->output_height, 
            bconv2->output_width, filter_height, filter_width, 64, 128, batch);
    Conv64LayerParam* bconv3_gpu = bconv3->initialize(config_file, bconv2->get_output_gpu());
    //Bconv4 Layer
    Conv64LayerParam* bconv4 = new Conv64LayerParam("Conv4", bconv3->output_height, 
            bconv3->output_width, filter_height, filter_width, 128, 128, batch, 1, 1,
            true, 2, 2, false);
    Conv64LayerParam* bconv4_gpu = bconv4->initialize(config_file, bconv3->get_output_gpu());
    //Bconv5 Layer
    Conv64LayerParam* bconv5 = new Conv64LayerParam("Conv5", bconv4->output_height, 
            bconv4->output_width, filter_height, filter_width, 128, 256, batch);
    Conv64LayerParam* bconv5_gpu = bconv5->initialize(config_file, bconv4->get_output_gpu());
    //Bconv6 Layer
    Conv64LayerParam* bconv6 = new Conv64LayerParam("Conv6", bconv5->output_height, 
            bconv5->output_width, filter_height, filter_width, 256, 256, batch);
    Conv64LayerParam* bconv6_gpu = bconv6->initialize(config_file, bconv5->get_output_gpu());
    //Bconv7 Layer
    Conv64LayerParam* bconv7 = new Conv64LayerParam("Conv7", bconv6->output_height, 
            bconv6->output_width, filter_height, filter_width, 256, 256, batch, 1, 1,
            true, 2, 2, false);
    Conv64LayerParam* bconv7_gpu = bconv7->initialize(config_file, bconv6->get_output_gpu());
    //Bconv8 Layer
    Conv64LayerParam* bconv8 = new Conv64LayerParam("Conv8", bconv7->output_height, 
            bconv7->output_width, filter_height, filter_width, 256, 512, batch);
    Conv64LayerParam* bconv8_gpu = bconv8->initialize(config_file, bconv7->get_output_gpu());
    //Bconv9 Layer
    Conv64LayerParam* bconv9 = new Conv64LayerParam("Conv9", bconv8->output_height, 
            bconv8->output_width, filter_height, filter_width, 512, 512, batch);
    Conv64LayerParam* bconv9_gpu = bconv9->initialize(config_file, bconv8->get_output_gpu());
    //Bconv10 Layer
    Conv64LayerParam* bconv10 = new Conv64LayerParam("Conv10", bconv9->output_height, 
            bconv9->output_width, filter_height, filter_width, 512, 512, batch, 1, 1,
            true, 2, 2, false);
    Conv64LayerParam* bconv10_gpu = bconv10->initialize(config_file, bconv9->get_output_gpu());
    //Bconv11 Layer
    Conv64LayerParam* bconv11 = new Conv64LayerParam("Conv11", bconv10->output_height, 
            bconv10->output_width, filter_height, filter_width, 512, 512, batch);
    Conv64LayerParam* bconv11_gpu = bconv11->initialize(config_file, bconv10->get_output_gpu());
    //Bconv12 Layer
    Conv64LayerParam* bconv12 = new Conv64LayerParam("Conv12", bconv11->output_height, 
            bconv11->output_width, filter_height, filter_width, 512, 512, batch);
    Conv64LayerParam* bconv12_gpu = bconv12->initialize(config_file, bconv11->get_output_gpu());
    //Bconv13 Layer
    Conv64LayerParam* bconv13 = new Conv64LayerParam("Conv13", bconv12->output_height, 
            bconv12->output_width, filter_height, filter_width, 512, 512, batch, 1, 1,
            true, 2, 2, true);
    Conv64LayerParam* bconv13_gpu = bconv13->initialize(config_file, bconv12->get_output_gpu());
    //Fc1 Layer
    Fc64LayerParam* bfc1 = new Fc64LayerParam("Fc1", batch, (bconv13->output_height)
            *(bconv13->output_width)*512, n_hidden); 
    Fc64LayerParam* bfc1_gpu = bfc1->initialize(config_file, bconv13->get_output_gpu());
    //Fc2 Layer
    Fc64LayerParam* bfc2 = new Fc64LayerParam("Fc2", batch, n_hidden, n_hidden); 
    Fc64LayerParam* bfc2_gpu = bfc2->initialize(config_file, bfc1->get_output_gpu());
    //Out Layer
    Out64LayerParam* bout = new Out64LayerParam("Fout", batch, n_hidden, output_size, true);
    Out64LayerParam* bout_gpu = bout->initialize(config_file, bfc2->get_output_gpu());  


    //================ Setup Kernel =================
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    int shared_memory = 512*sizeof(int)*32;
    cudaFuncSetAttribute(vggnet64, cudaFuncAttributeMaxDynamicSharedMemorySize,98304);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, vggnet64, numThreads, shared_memory);
    //cudaFuncSetAttribute(alexnet64, cudaFuncAttributePreferredSharedMemoryCarveout,0);
    void* args[] = {&bconv1_gpu, &bconv2_gpu, &bconv3_gpu, &bconv4_gpu, &bconv5_gpu, &bconv6_gpu,
        &bconv7_gpu, &bconv8_gpu, &bconv9_gpu, &bconv10_gpu, &bconv11_gpu, &bconv12_gpu, &bconv13_gpu,
        &bfc1_gpu, &bfc2_gpu, &bout_gpu};

    START_TIMER;

    cudaLaunchCooperativeKernel((void*)vggnet64, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);

    //vggnet64<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads, shared_memory>>> (
    //bconv1_gpu, bconv2_gpu, bconv3_gpu, bconv4_gpu, bconv5_gpu, bfc1_gpu, bfc2_gpu, bout_gpu);

    STOP_TIMER;

    float* output = bout->download_output();
    validate_prediction(output, image_labels, output_size, batch);

    delete bconv1;
    delete bconv2;
    delete bconv3;
    delete bconv4;
    delete bconv5;
    delete bconv6;
    delete bconv7;
    delete bconv8;
    delete bconv9;
    delete bconv10;
    delete bconv11;
    delete bconv12;
    delete bconv13;
    delete bfc1;
    delete bfc2;
    delete bout;

    return 0;

}















