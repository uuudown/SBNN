/** @file sbnn32_param.h
 *  @brief Layer parameter definition 64bit implementation.
 *
 *  @author Ang Li (PNNL)
 *
*/
#ifndef SBNN32_PARAM_H
#define SBNN32_PARAM_H

#include <string.h>
#include "utility.h"

/** @brief Binarize and pack weight matrix into 32-bit unsigned matrix.
 *
 *  Binarization function to convert row-major 32-bit floating-point weight matrix into 
 *  bit column-major bit-matrix. This is for the preparation of the weight matrices for 
 *  FC layers.
 *
 *  @return Void.
 */
__global__ void PackFcWeight32(const float* __restrict__ A, unsigned* B, 
        const int A_height, const int A_width)
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y; 
    unsigned Bval=0;
    #pragma unroll
    for (int i=0; i<32; i++) 
    {
        float f0 = ((by*32+laneid<A_width)&&(bx*32+i<A_height))?A[(bx*32+i)*A_width+by*32+laneid]:-1.0f;
        Bval = (Bval<<1)|(f0>=0?1:0);
    }
    if (laneid < A_height*A_width)
        B[bx*gridDim.y*32+by*32+laneid] = Bval;
}

/** @brief Unpack 32-bit row-major unsigned activation matrix into floating-point.
 *
 *  Unpack compact 32-bit unsigned layer output activation matrix into floating-point for 
 *  validation purpose.
 *
 *  @return Void.
 */
__global__ void UnpackFcOutput32(const unsigned* __restrict__  A, float* B, 
        const int A_height, const int A_width)
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const unsigned by = blockIdx.y; 
    const unsigned bx = blockIdx.x;
    unsigned Aval = A[by*gridDim.x*32+bx*32+laneid];
    #pragma unroll
    for (int i=0; i<32; i++) 
    {
        unsigned r0 = __shfl_sync(0xFFFFFFFF, Aval, i); //from lane-i
        if ((32*bx+i)<A_height && by*32+laneid<A_width)
        {
            B[(32*bx+i)*A_width+by*32+laneid] =  2*(float)((r0>>(31-laneid)) & 0x1)-1;
        }
    }
}

/** @brief Binarize and pack filter tensor at output-channels into 32-bit unsigned tensor.
 *
 *  Binarization function to convert 32-bit floating-point filter tensor (HWIO) into 
 *  bit filter tensor (HWIO). This is for the preparation of the filter tensor for 
 *  Conv layers.
 *
 *  @return Void.
 */
__global__ void PackFiltersByOutChannels32(const float* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height) 
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over input_channels
    const int ots = CEIL(out_channels);//condense K:output_channel into 32bit-unsigned

    for (int k=0; k<ots; k++) //iter over K:output_channels
    {
        // From shape[filter_height, filter_width, in_channels, out_channels] 
        float f0 = ((k*32+laneid)<out_channels)? filter[bx*in_channels*out_channels 
            + by*out_channels + k*32 + laneid]:0;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0));
        // To shape[filter_height, filter_width, in_channels, out_channels/32]
        filter_binarized[bx*ots*in_channels+ by*ots + k] = r0;
    }
}

/** @brief Unpack condensed unsigned activation tensor into floating-point.
 *
 *  Unpack compact 32-bit unsigned layer output activation tensor into floating-point for 
 *  validation purpose.
 *
 *  @return Void.
 */
__global__ void UnpackConvOutput32(const unsigned* __restrict__ input_binarized, 
        float* input, const int input_height, const int input_width,
        const int input_channels, const int batch) 
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const int bx = blockIdx.x;//input_width
    const int by = blockIdx.y;//input_height
    const int bz = blockIdx.z;//batch
    const int ins = CEIL(input_channels);//condense C:in_channel into 32bit-unsigned
    for (int c=0; c<ins; c++) //iter over C:in_channels
    {
        // From shape[batch, input_height, input_width, in_channels/32] 
        unsigned r0 = input_binarized[bz*input_height*input_width*ins + by*input_width*ins + bx*ins + c];
        // To shape[batch, input_height, input_width, in_channels]
        if (c*32+laneid<input_channels)
        {
            input[bz*input_height*input_width*input_channels + by*input_width*input_channels
                + bx*input_channels + c*32 + laneid] = 2*(float)((r0>>(31-laneid)) & 0x1)-1;
        }
    }
}

/** @brief Binarize and pack filter tensor at input-channels into 32-bit unsigned tensor.
 *
 *  Binarization function to convert 32-bit floating-point filter tensor (HWIO) into 
 *  bit filter tensor (HWIO). This is for the preparation of the filter tensor for 
 *  Conv layers.
 *
 *  @return Void.
 */
__global__ void PackFiltersByInChannels32(const float* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height) 
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over out_channels
    const int ins = CEIL(in_channels);//condense C:in_channel into 32bit-unsigned
    for (int c=0; c<ins; c++) //iter over C:in_channels
    {
        // From shape[filter_height, filter_width, in_channels, out_channels] 
        float f0 = ((c*32+laneid)<in_channels)? filter[bx*in_channels*out_channels 
            + (c*32+laneid)*out_channels + by]:0;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0));
        //if (laneid == 0) //avoid warp conflict
        // To shape[filter_height, filter_width, in_channels/32, out_channels]
        filter_binarized[bx*ins*out_channels+ c*out_channels + by] = r0;
    }
}











//===============================================================
//1-Bit FC input layer (32-bit)
class In32LayerParam
{
    public:
        In32LayerParam(const char* name, unsigned input_height, unsigned input_width)
        {
            strncpy(this->name, name, 8);
            this->input_height = input_height;
            this->output_height = input_height;
            this->input_width = input_width;
            this->output_width = input_width;
            this->input = NULL;
            this->input_gpu = NULL;
            this->output = NULL;
            this->output_gpu = NULL;
            this->gpu = NULL;
        }
        In32LayerParam* initialize(float* input)
        {
            if (input == NULL)
            {
                fprintf(stderr, "Error: NULL input.\n");
                exit(1);
            }
            this->input = input;

            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->input_gpu), input_bytes()) );
            CUDA_SAFE_CALL( cudaMemcpy(input_gpu, input, input_bytes(), 
                        cudaMemcpyHostToDevice) );

            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->output_gpu), output_bit_bytes()) );
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            return this->ready();
        }
        int input_size() { return input_height * input_width; }
        int input_bytes() { return input_size() * sizeof(float);}
        int input_bit_size() { return input_height * input_width; }
        int input_bit_bytes() { return input_bit_size() * sizeof(float);}

        //column-major, ceil row to 32/64 bits
        int output_size() { return  output_height * output_width;}
        int output_bytes() { return output_size()*sizeof(unsigned);}
        int output_bit_size() { return  CEIL(output_height)*FEIL(output_width);}
        int output_bit_bytes() { return output_bit_size()*sizeof(unsigned);}

        In32LayerParam* ready()
        {
            if (input_gpu == NULL)
            {
                fprintf(stderr, "Input data has not been uploaded to GPU.\n");
                exit(1);
            }
            if (output_gpu == NULL)
            {
                fprintf(stderr, "Output on GPU has not been allocated.\n");
                exit(1);
            }
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->gpu), sizeof(In32LayerParam)) );
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(In32LayerParam), 
                        cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        void set_output_gpu(unsigned* _output_gpu)
        {
            this->output_gpu = _output_gpu; 
        }
        unsigned* get_output_gpu()
        {
            return this->output_gpu;
        }
        unsigned* download_output()
        {
            if (this->output == NULL) (this->output) = (unsigned*)malloc(output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, output_bit_bytes(), 
                        cudaMemcpyDeviceToHost) );
            return this->output;
        }
        float* download_full_output()
        {
            const int size = output_size()*sizeof(float);
            float* full_output = (float*)malloc(size);
            float* full_output_gpu = NULL;
            CUDA_SAFE_CALL( cudaMalloc((void**)&(full_output_gpu), size) );
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );
            UnpackFcOutput32<<<dim3( CEIL(output_height), CEIL(output_width) ),32>>>(
                    output_gpu, full_output_gpu, output_height, output_width);
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, size, 
                        cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_output_gpu) );
            return full_output;
        }
        void release() 
        {
            if (this->input_gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->input_gpu) ); 
                this->input_gpu = NULL; 
            }
            if (this->output != NULL) 
            { 
                free(this->output); 
                this->output = NULL; 
            }
            if (this->output_gpu != NULL) 
            {
                CUDA_SAFE_CALL( cudaFree(this->output_gpu) ); 
                this->output_gpu = NULL; 
            }
            if (this->gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->gpu) ); 
                this->gpu = NULL; 
            }
        }
        ~In32LayerParam() { release(); }
    public:
        //Input
        float* input;
        float* input_gpu;
        unsigned input_width;
        unsigned input_height;
        //Output
        unsigned* output;
        unsigned* output_gpu;
        unsigned output_width;
        unsigned output_height;
        //GPU Shadow
        In32LayerParam* gpu;
        char name[8];
};


class Fc32LayerParam
{
    public:
        Fc32LayerParam(const char* name, unsigned _input_height, unsigned _input_width, 
                unsigned _weight_width)
        {
            strncpy(this->name, name, 8);
            weight_height = input_width = _input_width;
            output_height = input_height = _input_height;
            bn_width = output_width = weight_width = _weight_width;
            this->weight = NULL;
            this->weight_gpu = NULL;
            this->bn = NULL;
            this->bn_gpu = NULL;
            this->output = NULL;
            this->output_gpu = NULL;
            this->input = NULL;
            this->input_gpu = NULL;
            this->gpu = NULL;
        }

        void set_input_gpu(unsigned* input_gpu)
        {
            this->input_gpu = input_gpu;
        }
        
        Fc32LayerParam* ready()
        {
            if (input_gpu == NULL)
            {
                fprintf(stderr, "Input data has not been uploaded to GPU.\n");
                exit(1);
            }
            if (output_gpu == NULL)
            {
                fprintf(stderr, "Output on GPU has not been allocated.\n");
                exit(1);
            }
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->gpu), sizeof(Fc32LayerParam)) );
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(Fc32LayerParam), 
                        cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        Fc32LayerParam* initialize(FILE* config_file, unsigned* prev_layer_gpu)
        {
            //Process weight
            this->weight = (float*)malloc(weight_bytes());
            launch_array(config_file, this->weight, weight_size());
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->weight_gpu), weight_bit_bytes()) );
            float* weight_float = NULL;
            CUDA_SAFE_CALL( cudaMalloc((void**)&(weight_float), weight_bytes()) );
            CUDA_SAFE_CALL( cudaMemcpy(weight_float, weight, weight_bytes(), 
                        cudaMemcpyHostToDevice) );
            //Binarize and compact weight
            PackFcWeight32<<<dim3( CEIL(weight_height), CEIL(weight_width) ), 32>>>(
                    weight_float, weight_gpu, weight_height, weight_width);
            CUDA_SAFE_CALL( cudaFree(weight_float) );
            
            //Process bn
            this->bn = (float*)malloc(bn_bytes());
            launch_array(config_file, this->bn, bn_size());
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->bn_gpu), bn_bytes()) );
            CUDA_SAFE_CALL( cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice) );

            //Allocate output
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->output_gpu), output_bit_bytes()) );
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );

            set_input_gpu(prev_layer_gpu);
            return this->ready();
        }

        //column-major, ceil row 
        int input_size() { return input_height*input_width;}
        int input_bytes() { return input_size()*sizeof(unsigned);}
        int input_bit_size() { return  FEIL(input_height)*CEIL(input_width);}
        int input_bit_bytes() { return input_bit_size()*sizeof(unsigned);}

        //row-major, ceil column to 32bit
        int weight_size() { return weight_height*weight_width;}
        int weight_bytes() { return weight_size()*sizeof(float);}
        int weight_bit_size() { return CEIL(weight_height)*FEIL(weight_width);}
        int weight_bit_bytes() { return weight_bit_size()*sizeof(unsigned);}

        //column-major, ceil row to 32bit
        int output_size() { return output_height*output_width;}
        int output_bytes() { return output_size()*sizeof(unsigned);}
        int output_bit_size() { return FEIL(output_height)*CEIL(output_width);}
        int output_bit_bytes() { return output_bit_size()*sizeof(unsigned);}

        //batch-norm
        int bn_size() { return bn_width;}
        int bn_bytes() { return bn_size()*sizeof(float);}

        unsigned* get_output_gpu()
        {
            return this->output_gpu;
        }
        unsigned* download_output()
        {
            if (output == NULL) output = (unsigned*)malloc(output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, output_bit_bytes(), 
                        cudaMemcpyDeviceToHost) );
            return this->output;
        }
        float* download_full_output()
        {
            const int size = FEIL(output_height)*FEIL(output_width)*sizeof(float);
            float* full_output = (float*)malloc(size);
            float* full_output_gpu = NULL;
            CUDA_SAFE_CALL( cudaMalloc((void**)&(full_output_gpu), size) );
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );
            UnpackFcOutput32<<<dim3( CEIL(output_height), CEIL(output_width) ), 32>>>(
                    output_gpu, full_output_gpu, output_height, output_width);
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, size, 
                        cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_output_gpu) );
            return full_output;
        }
        void release()
        {
            if (this->weight != NULL) { free(this->weight); this->weight = NULL;}
            if (this->bn != NULL) { free(this->bn); this->bn = NULL;}
            if (this->output != NULL) { free(this->output); this->output = NULL;}
            if (this->weight_gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(weight_gpu) ); 
                weight_gpu = NULL; 
            }
            if (this->bn_gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(bn_gpu) ); 
                bn_gpu = NULL; 
            }
            if (this->output_gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->output_gpu) ); 
                this->output_gpu = NULL; 
            }
            if (this->gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->gpu) ); 
                this->gpu = NULL; 
            }
        }
        ~Fc32LayerParam() { release(); }

    public:
        //Input
        unsigned* input;
        unsigned* input_gpu;
        unsigned input_width;
        unsigned input_height;
        //Weight
        float* weight;
        unsigned* weight_gpu;
        unsigned weight_width;
        unsigned weight_height;
        //Output
        unsigned* output;
        unsigned* output_gpu;
        unsigned output_width;
        unsigned output_height;
        //Batch normalization
        float* bn;
        float* bn_gpu;
        unsigned bn_width;
        //GPU shodow
        Fc32LayerParam* gpu;
        char name[8];
};

class Out32LayerParam
{
    public:
        Out32LayerParam(const char* name, unsigned _input_height, unsigned _input_width, 
                unsigned _weight_width, bool has_bn=true)
        {
            strncpy(this->name, name, 8);
            weight_height = input_width = _input_width;
            output_height = input_height = _input_height;
            output_width = weight_width = _weight_width;
            this->input = NULL;
            this->input_gpu = NULL;
            this->output = NULL;
            this->output_gpu = NULL;
            this->weight = NULL;
            this->weight_gpu = NULL;
            this->gpu = NULL;
            this->has_bn = has_bn;
            this->bn_scale = NULL;
            this->bn_scale_gpu = NULL;
            this->bn_bias = NULL;
            this->bn_bias_gpu = NULL;
        }

        Out32LayerParam* ready()
        {
            if (input_gpu == NULL)
            {
                fprintf(stderr, "Input data has not been uploaded to GPU.\n");
                exit(1);
            }
            if (output_gpu == NULL)
            {
                fprintf(stderr, "Output on GPU has not been allocated.\n");
                exit(1);
            }
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->gpu), sizeof(Out32LayerParam)) );
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(Out32LayerParam), 
                        cudaMemcpyHostToDevice) );
            return this->gpu;
        }
        void set_input_gpu(unsigned* _input_gpu)
        {
            this->input_gpu = _input_gpu;
        }
        Out32LayerParam* initialize(FILE* config_file, unsigned* prev_layer_gpu)
        {
            this->weight = (float*)malloc(weight_bytes());
            launch_array(config_file, this->weight, weight_size());
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->weight_gpu), weight_bit_bytes()) );
            float* weight_float = NULL;
            CUDA_SAFE_CALL( cudaMalloc((void**)&(weight_float), weight_bytes()) );
            CUDA_SAFE_CALL( cudaMemcpy(weight_float, weight, weight_bytes(), 
                        cudaMemcpyHostToDevice) );
            //Binarize and compact weight
            PackFcWeight32<<<dim3( CEIL(weight_height), CEIL(weight_width) ), 32>>>(
                    weight_float, weight_gpu, weight_height, weight_width);
            CUDA_SAFE_CALL( cudaFree(weight_float) );
            //BN
            if (this->has_bn)
            {
                this->bn_scale = (float*)malloc(bn_bytes());
                launch_array(config_file, this->bn_scale, bn_size());
                CUDA_SAFE_CALL( cudaMalloc((void**)&(this->bn_scale_gpu), bn_bytes()) );
                CUDA_SAFE_CALL( cudaMemcpy(bn_scale_gpu, bn_scale, 
                            bn_bytes(), cudaMemcpyHostToDevice) );
                this->bn_bias = (float*)malloc(bn_bytes());
                launch_array(config_file, this->bn_bias, bn_size());
                CUDA_SAFE_CALL( cudaMalloc((void**)&(this->bn_bias_gpu), bn_bytes()) );
                CUDA_SAFE_CALL( cudaMemcpy(bn_bias_gpu, bn_bias, bn_bytes(), 
                            cudaMemcpyHostToDevice) );
            }
            //Allocate output
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->output_gpu), output_bytes()) );
            set_input_gpu(prev_layer_gpu);
            return this->ready();

        }
        //column-major, ceil row to 32bit
        int input_size() { return  input_height*input_width;}
        int input_bytes() { return input_size()*sizeof(unsigned);}

        int input_bit_size() {return FEIL(input_height)*FEIL(input_width);}
        int input_bit_bytes() {return CEIL(input_height)*FEIL(input_width)*sizeof(unsigned);}

        //row-major, ceil column to 32bit
        int weight_size() { return weight_height*weight_width;}
        int weight_bytes() { return weight_size()*sizeof(unsigned);}

        int weight_bit_size() {return FEIL(weight_height)*FEIL(weight_width);}
        int weight_bit_bytes() {return FEIL(weight_height)*CEIL(weight_width)*sizeof(unsigned);}

        //row-major, float
        int output_size() { return output_height * output_width;}
        int output_bytes() { return output_size()*sizeof(float);}

        int output_bit_size() { return output_height * output_width;}
        int output_bit_bytes() { return output_bit_size()*sizeof(float);}

        int bn_size() { return output_width;}
        int bn_bytes() { return output_width*sizeof(float); }

        void allocate_input_gpu()
        {
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->input_gpu), input_bit_bytes()) );
        }

        float* download_output()
        {
            if (output == NULL) output = (float*)malloc(output_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, output_bytes(), 
                        cudaMemcpyDeviceToHost) );
            return this->output;
        }
        
        void release()
        {
            if (this->weight != NULL) { free(this->weight); this->weight = NULL;}
            if (this->output != NULL) { free(this->output); this->output = NULL;}
            if (this->weight_gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->weight_gpu) ); 
                this->weight_gpu = NULL; 
            }
            if (this->output_gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->output_gpu) ); 
                this->output_gpu = NULL; 
            }
            if (this->gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->gpu) ); 
                this->gpu = NULL; 
            }
            if (this->has_bn == true)
            {
                if (this->bn_scale != NULL) 
                {
                    free(this->bn_scale); 
                    this->bn_scale = NULL;
                }
                if (this->bn_scale_gpu != NULL) 
                {
                    CUDA_SAFE_CALL( cudaFree(this->bn_scale_gpu) ); 
                    this->bn_scale = NULL;
                }
                if (this->bn_bias != NULL) 
                {
                    free(this->bn_bias); 
                    this->bn_bias = NULL;
                }
                if (this->bn_bias_gpu != NULL) 
                { 
                    CUDA_SAFE_CALL( cudaFree(this->bn_bias_gpu) ); 
                    this->bn_bias_gpu = NULL;
                }
            }

        }
        ~Out32LayerParam() { release(); }

    public:
        //Input
        unsigned* input;
        unsigned* input_gpu;
        unsigned input_width;
        unsigned input_height;
        //Weight
        float* weight;
        unsigned* weight_gpu;
        unsigned weight_width;
        unsigned weight_height;
        //Output
        float* output;
        float* output_gpu;
        unsigned output_height;
        unsigned output_width;
        //Batch normalization
        bool has_bn;
        float* bn_scale;
        float* bn_scale_gpu;
        float* bn_bias;
        float* bn_bias_gpu;
        //GPU shadow
        Out32LayerParam* gpu;
        char name[8];
};

////======================== Convolution ==========================
class In32Conv32LayerParam
{
    public:
        In32Conv32LayerParam(const char* name, 
                unsigned input_height, 
                unsigned input_width, 
                unsigned filter_height, 
                unsigned filter_width, 
                unsigned input_channels, 
                unsigned output_channels, 
                unsigned batch, 
                unsigned stride_vertical=1, 
                unsigned stride_horizontal=1, 
                bool same_padding=true, 
                unsigned pool_height=0, 
                unsigned pool_width=0, 
                bool output_transpose=false,
                bool save_residual=false)
        {
            strncpy(this->name, name, 8);
            this->input_height = input_height;
            this->input_width = input_width;
            this->filter_height = filter_height;
            this->filter_width = filter_width;
            this->input_channels = input_channels;
            this->output_channels = output_channels;
            this->batch = batch;
            this->stride_vertical = stride_vertical;
            this->stride_horizontal = stride_horizontal;
            this->pool_height = pool_height;
            this->pool_width = pool_width;
            this->output_transpose = output_transpose;
            this->save_residual = save_residual;
            this->pad_h = same_padding?((( (input_height+stride_vertical-(input_height%stride_vertical))
                            /stride_vertical-1)*stride_vertical+filter_height-input_height)>>1):0;
            this->pad_w = same_padding?((( (input_width+stride_horizontal-(input_width%stride_horizontal))
                                /stride_horizontal-1)*stride_horizontal+filter_width-input_width)>>1):0; 
            if (pool_height == 0)
            {
                output_height = same_padding?(input_height+stride_vertical-1)/stride_vertical
                    :((input_height-filter_height)/stride_vertical+1);
                this->buf_height = 0;
            }
            else
            {
                buf_height = same_padding?(input_height+stride_vertical-1)/stride_vertical
                    :((input_height-filter_height)/stride_vertical+1);
                output_height = (buf_height+pool_height-1)/pool_height;//pooling height
            }
            if (pool_width == 0)
            {
                output_width = same_padding?(input_width+stride_horizontal-1)/stride_horizontal
                    :((input_width-filter_width)/stride_horizontal+1);
                this->buf_width = 0;
            }
            else
            {
                buf_width = same_padding?(input_width+stride_horizontal-1)/stride_horizontal
                    :((input_width-filter_width)/stride_horizontal+1);
                output_width = (buf_width+pool_width-1)/pool_width; //pooling width
            }
            this->bn = NULL;
            this->filter = NULL;
            this->output = NULL;
            this->output_gpu = NULL;
            this->input = NULL;
            this->input_gpu = NULL;
            this->gpu = NULL;
            this->save_residual_gpu = NULL;
        }
        In32Conv32LayerParam* ready()
        {
            if (input_gpu == NULL)
            {
                fprintf(stderr, "Input data has not been uploaded to GPU.\n");
                exit(1);
            }
            if (output_gpu == NULL)
            {
                fprintf(stderr, "Output on GPU has not been allocated.\n");
                exit(1);
            }
            if (save_residual && save_residual_gpu == NULL)
            {
                fprintf(stderr, "Residual for saving on GPU has not been allocated.\n");
                exit(1);
            }
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->gpu), sizeof(In32Conv32LayerParam)) );
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, 
                        sizeof(In32Conv32LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        In32Conv32LayerParam* initialize(float* input, FILE* config_file)
        {
            //Process input
            if (input == NULL)
            {
                fprintf(stderr, "Error: NULL image input.\n");
                exit(1);
            }
            this->input = input;
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->input_gpu), input_bytes()) );
            CUDA_SAFE_CALL( cudaMemcpy(input_gpu, input, 
                        input_bytes(), cudaMemcpyHostToDevice) );
            //Process weight
            this->filter = (float*)malloc(filter_bytes());
            launch_array(config_file, this->filter, filter_size());
            CUDA_SAFE_CALL (cudaMalloc((void**)&(this->filter_gpu), filter_bit_bytes()) );


            float* filter_float = NULL;
            CUDA_SAFE_CALL( cudaMalloc((void**)&(filter_float), filter_bytes()) );
            CUDA_SAFE_CALL( cudaMemcpy(filter_float, filter, 
                        filter_bytes(), cudaMemcpyHostToDevice) );
            //Binarize Filter
            PackFiltersByOutChannels32<<<dim3(filter_height*filter_width, input_channels), 32>>>(
                filter_float, filter_gpu, input_channels, output_channels, filter_width, filter_height);
            CUDA_SAFE_CALL( cudaFree(filter_float) );
            //Process bn
            this->bn = (float*)malloc(bn_bytes());
            launch_array(config_file, this->bn, bn_size());
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->bn_gpu), bn_bytes()) );
            CUDA_SAFE_CALL( cudaMemcpy(this->bn_gpu, this->bn, 
                        bn_bytes(), cudaMemcpyHostToDevice) );

            //Allocate output
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->output_gpu), output_bit_bytes()) );
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );

            //Allocate residual for saving
            if (save_residual)
            {
                CUDA_SAFE_CALL( cudaMalloc((void**)&(this->save_residual_gpu), output_bytes()) );
                CUDA_SAFE_CALL( cudaMemset(this->save_residual_gpu, 0, output_bytes()) );
            }

            return this->ready();
        }
        int input_size() { return input_channels*input_height*input_width*batch;}
        int input_bytes() { return input_size()*sizeof(float);}
        int input_bit_size() { return  input_channels*input_height*input_width*batch;}
        int input_bit_bytes() {return input_bit_size()*sizeof(float);}
        
        //Size has problem, should be a factor of 32, also applied to other convParam
        int filter_size() { return output_channels*input_channels*filter_height*filter_width;}
        int filter_bytes() { return filter_size()*sizeof(float);}
        int filter_bit_size() {return FEIL(output_channels)*input_channels*filter_height*filter_width;}
        int filter_bit_bytes() { return CEIL(output_channels)*input_channels
            *filter_height*filter_width*sizeof(unsigned);}

        //In In32Conv32 we binarize output_channels
        int output_size() { return output_channels*output_height*output_width*batch;}
        int output_bytes() { return output_size()*sizeof(unsigned);}
        int output_bit_size() 
        { 
            return output_transpose?FEIL(output_channels)*output_height*output_width*FEIL(batch):
                FEIL(output_channels)*output_height*output_width*batch;
        }
        int output_bit_bytes() 
        { 
            return output_transpose?CEIL(output_channels)*output_height*output_width*
                FEIL(batch)*sizeof(unsigned): CEIL(output_channels)*output_height*
                output_width*batch*sizeof(unsigned);
        }
        //batch-norm
        int bn_size() { return output_channels;}
        int bn_bytes() { return bn_size()*sizeof(float);}

        unsigned* get_output_gpu()
        {
            return this->output_gpu;
        }

        int* get_residual_gpu()
        {
            return this->save_residual_gpu;
        }

        unsigned* download_output()
        {
            if (output == NULL) output = (unsigned*)malloc(output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }

        float* download_full_output()
        {
            float* full_output = (float*)malloc(output_bytes());
            float* full_output_gpu = NULL;
            CUDA_SAFE_CALL( cudaMalloc((void**)&(full_output_gpu), output_bytes()) );
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, output_bytes()) );
            UnpackConvOutput32<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch);
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, 
                        output_bytes(), cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_output_gpu) );
            return full_output;
        }
        void release()
        {
            if (this->filter!=NULL) {free(this->filter); this->filter=NULL;}
            if (this->bn!=NULL) {free(this->bn); this->bn=NULL;}
            if (this->output!=NULL) {free(this->output); this->output=NULL;}
            if (this->input_gpu!=NULL) 
            {
                CUDA_SAFE_CALL( cudaFree(this->input_gpu) ); 
                this->input_gpu=NULL;
            }
            if (this->output_gpu!=NULL) 
            {
                CUDA_SAFE_CALL( cudaFree(this->output_gpu) ); 
                this->output_gpu=NULL;
            }
            if (this->filter_gpu != NULL) 
            {
                CUDA_SAFE_CALL( cudaFree(this->filter_gpu) ); 
                this->filter_gpu = NULL; 
            }
            if (this->bn_gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->bn_gpu) ); 
                this->bn_gpu = NULL; 
            }
            if (this->gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->gpu) ); 
                this->gpu = NULL; 
            }

            if (this->save_residual && this->save_residual_gpu != NULL)
            {
                CUDA_SAFE_CALL( cudaFree(this->save_residual_gpu) );
                this->save_residual_gpu = NULL;
            }
        }
        ~In32Conv32LayerParam() { release(); }

    public:
        //Input
        float* input;
        float* input_gpu;
        unsigned input_width;
        unsigned input_height;
        unsigned input_channels;
        //Weight
        float* filter;
        unsigned* filter_gpu;
        unsigned filter_width;
        unsigned filter_height;
        //Output
        unsigned* output;
        unsigned* output_gpu;
        unsigned output_width;
        unsigned output_height;
        unsigned output_channels;
        bool output_transpose;
        //Batch normalization
        float* bn;
        float* bn_gpu;
        //Others
        unsigned batch;
        unsigned stride_vertical;
        unsigned stride_horizontal;
        unsigned pad_h;
        unsigned pad_w;
        //Pooling
        unsigned pool_width;
        unsigned pool_height;
        unsigned buf_width;
        unsigned buf_height;
        //GPU shadow
        In32Conv32LayerParam* gpu;
        char name[8];
        //Residual
        bool save_residual;
        int* save_residual_gpu;

};

class Conv32LayerParam
{
    public:
        Conv32LayerParam(const char* name,
                unsigned input_height, 
                unsigned input_width, 
                unsigned filter_height, 
                unsigned filter_width, 
                unsigned input_channels, 
                unsigned output_channels, 
                unsigned batch, 
                unsigned stride_vertical=1, 
                unsigned stride_horizontal=1, 
                bool same_padding=true, 
                unsigned pool_height=0, 
                unsigned pool_width=0, 
                bool output_transpose=false,
                bool save_residual=false,
                bool inject_residual=false,
                unsigned residual_channels=0,
                bool residual_pool=false)
                
        {
            strncpy(this->name, name, 8);
            this->input_height = input_height;
            this->input_width = input_width;
            this->filter_height = filter_height;
            this->filter_width = filter_width;
            this->input_channels = input_channels;
            this->output_channels = output_channels;
            this->batch = batch;
            this->stride_vertical = stride_vertical;
            this->stride_horizontal = stride_horizontal;
            this->pool_height = pool_height;
            this->pool_width = pool_width;
            this->output_transpose = output_transpose;
            this->save_residual = save_residual;
            this->inject_residual = inject_residual;
            this->residual_channels = residual_channels;
            this->residual_pool = residual_pool;

            this->pad_h = same_padding?((( (input_height+stride_vertical-(input_height%stride_vertical))
                            /stride_vertical-1)*stride_vertical+filter_height-input_height)>>1):0;
            this->pad_w = same_padding?((( (input_width+stride_horizontal-(input_width%stride_horizontal))
                                /stride_horizontal-1)*stride_horizontal+filter_width-input_width)>>1):0; 

            if (pool_height == 0)
            {
                output_height = same_padding?(input_height+stride_vertical-1)/stride_vertical
                    :((input_height-filter_height)/stride_vertical+1);
                this->buf_height = 0;
            }
            else
            {
                buf_height = same_padding?(input_height+stride_vertical-1)/stride_vertical
                    :((input_height-filter_height)/stride_vertical+1);
                output_height = (buf_height+pool_height-1)/pool_height;//pooling height
            }

            if (pool_width == 0)
            {
                output_width = same_padding?(input_width+stride_horizontal-1)/stride_horizontal
                    :((input_width-filter_width)/stride_horizontal+1);
                this->buf_width = 0;
            }
            else
            {
                buf_width = same_padding?(input_width+stride_horizontal-1)/stride_horizontal
                    :((input_width-filter_width)/stride_horizontal+1);
                output_width = (buf_width+pool_width-1)/pool_width; //pooling width
            }
            this->bn = NULL;
            this->bn_gpu = NULL;
            this->filter = NULL;
            this->filter_gpu = NULL;
            this->output = NULL;
            this->output_gpu = NULL;
            this->input = NULL;
            this->input_gpu = NULL;
            this->gpu = NULL;
            this->save_residual_gpu = NULL;
            this->inject_residual_gpu = NULL;
        }
        Conv32LayerParam* ready()
        {
            if (input_gpu == NULL)
            {
                fprintf(stderr, "Input data has not been uploaded to GPU.\n");
                exit(1);
            }
            if (output_gpu == NULL)
            {
                fprintf(stderr, "Output on GPU has not been allocated.\n");
                exit(1);
            }
            if (save_residual && save_residual_gpu == NULL)
            {
                fprintf(stderr, "Residual for saving on GPU has not been allocated.\n");
                exit(1);
            }
            if (inject_residual && inject_residual_gpu == NULL)
            {
                /*fprintf(stderr, this->name);*/
                
                fprintf(stderr, "Residual for injecting on GPU has not been allocated.\n");
                exit(1);
            }
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->gpu), sizeof(Conv32LayerParam)) );
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, 
                        sizeof(Conv32LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }
        void set_input_gpu(unsigned* input_gpu)
        {
            this->input_gpu = input_gpu;
        }
        void set_inject_residual_gpu(int* inject_residual_gpu)
        {
            this->inject_residual_gpu = inject_residual_gpu;
        }

        Conv32LayerParam* initialize(FILE* config_file, unsigned* prev_layer_gpu,
                int* inject_residual_gpu = NULL)
        {
            //Process weight
            this->filter = (float*)malloc(filter_bytes());
            launch_array(config_file, this->filter, filter_size());
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->filter_gpu), filter_bit_bytes()) );


            float* filter_float = NULL;
            CUDA_SAFE_CALL( cudaMalloc((void**)&(filter_float), filter_bytes()) );
            CUDA_SAFE_CALL( cudaMemcpy(filter_float, filter, 
                        filter_bytes(), cudaMemcpyHostToDevice) );
            //Binarize Filter
            PackFiltersByInChannels32<<<dim3(filter_height*filter_width, output_channels), 32>>>(
                filter_float, filter_gpu, input_channels, output_channels, 
                filter_width, filter_height);
            CUDA_SAFE_CALL( cudaFree(filter_float) );
            //Process bn
            this->bn = (float*)malloc(bn_bytes());
            launch_array(config_file, this->bn, bn_size());
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->bn_gpu), bn_bytes()) );
            CUDA_SAFE_CALL( cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice) );

            //Allocate output gpu
            CUDA_SAFE_CALL( cudaMalloc((void**)&(this->output_gpu), output_bit_bytes()) );
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            
            set_input_gpu(prev_layer_gpu);

            //Allocate residual for saving
            if (save_residual)
            {
                CUDA_SAFE_CALL( cudaMalloc((void**)&(this->save_residual_gpu), output_bytes()) );
                CUDA_SAFE_CALL( cudaMemset(this->save_residual_gpu, 0, output_bytes()) );
            }

            //inject residual
            if (inject_residual) set_inject_residual_gpu(inject_residual_gpu);

            return this->ready();
        }
        
        int input_size() { return  input_channels*input_height*input_width*batch;}
        int input_bytes() { return input_size()*sizeof(unsigned);}
        int input_bit_size() { return  CEIL(input_channels)*input_height*input_width*batch;}
        int input_bit_bytes() { return input_bit_size()*sizeof(unsigned);}

        int filter_size() { return output_channels*input_channels*filter_height*filter_width;}
        int filter_bytes() { return filter_size()*sizeof(float);}
        int filter_bit_size() {return output_channels*FEIL(input_channels)*filter_height*filter_width;}
        int filter_bit_bytes() { return output_channels*CEIL(input_channels)
            *filter_height*filter_width*sizeof(unsigned);}

        int output_size() { return output_channels*output_height*output_width*batch;}
        int output_bytes() { return output_size()*sizeof(unsigned);}
        int output_bit_size() 
        { 
            return output_transpose?FEIL(output_channels)*output_height*output_width*FEIL(batch):
                FEIL(output_channels)*output_height*output_width*batch;
        }
        int output_bit_bytes() 
        { 
            return output_transpose?CEIL(output_channels)*output_height*output_width*
                FEIL(batch)*sizeof(unsigned): CEIL(output_channels)*output_height*
                output_width*batch*sizeof(unsigned);
        }

        int bn_size() { return output_channels;}
        int bn_bytes() { return bn_size()*sizeof(float);}

        unsigned* get_output_gpu()
        {
            return this->output_gpu;
        }
        int* get_residual_gpu()
        {
            return this->save_residual_gpu;
        }

        unsigned* download_output()
        {
            if (output == NULL) output = (unsigned*)malloc(output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }
        float* download_full_output()
        {
            const int size = output_size()*sizeof(float);
            float* full_output = (float*)malloc(size);
            float* full_output_gpu = NULL;
            CUDA_SAFE_CALL( cudaMalloc((void**)&(full_output_gpu), size) );
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );
            UnpackConvOutput32<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch);
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, 
                        size, cudaMemcpyDeviceToHost) );
            CUDA_SAFE_CALL( cudaFree(full_output_gpu) );
            return full_output;
        }
        void release()
        {
            if (this->filter!=NULL) {free(this->filter); this->filter=NULL;}
            if (this->bn!=NULL) {free(this->bn); this->bn=NULL;}
            if (this->output!=NULL) {free(this->output); this->output=NULL;}
            if (this->output_gpu!=NULL) 
            {
                CUDA_SAFE_CALL( cudaFree(this->output_gpu) ); 
                this->output_gpu=NULL;
            }
            if (this->filter_gpu != NULL) 
            {
                CUDA_SAFE_CALL( cudaFree(this->filter_gpu) ); 
                this->filter_gpu = NULL; 
            }
            if (this->bn_gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->bn_gpu) ); 
                this->bn_gpu = NULL; 
            }
            if (this->gpu != NULL) 
            { 
                CUDA_SAFE_CALL( cudaFree(this->gpu) ); 
                this->gpu = NULL; 
            }
            if (this->save_residual && this->save_residual_gpu != NULL) 
            {
                CUDA_SAFE_CALL( cudaFree(this->save_residual_gpu) ); 
                this->save_residual_gpu=NULL; 
            }
        }
        ~Conv32LayerParam() { release(); }

    public:
        //Input
        unsigned* input;
        unsigned* input_gpu;
        unsigned input_width;
        unsigned input_height;
        unsigned input_channels;
        //Weight
        float* filter;
        unsigned* filter_gpu;
        unsigned filter_width;
        unsigned filter_height;
        //Output
        unsigned* output;
        unsigned* output_gpu;
        unsigned output_width;
        unsigned output_height;
        unsigned output_channels;
        bool output_transpose;
        //Batch normalization
        float* bn;
        float* bn_gpu;
        //Others
        unsigned batch;
        unsigned stride_vertical;
        unsigned stride_horizontal;
        unsigned pad_h;
        unsigned pad_w;
        //Pooling
        unsigned pool_width;
        unsigned pool_height;
        unsigned buf_width;
        unsigned buf_height;
        //GPU shadow
        Conv32LayerParam* gpu;
        char name[8];
        //Residual
        bool save_residual;
        int* save_residual_gpu;
        bool inject_residual;
        int* inject_residual_gpu; 
        unsigned residual_channels;
        bool residual_pool;

};






#endif
