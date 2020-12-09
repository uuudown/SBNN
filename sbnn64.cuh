/** @file sbnn64.cuh
 *  @brief Layer functions for 64-bit SBNN.
 *
 *  Layer functions including input, convolution, fully-connect and output functions
 *
 *  @author Ang Li (PNNL)
 *
*/

#ifndef SBNN64_CUH
#define SBNN64_CUH

/** @brief MLP input layer binarization.
 *
 *  Binarization function for the input layer of a MLP network (e.g., MLP for MNIST).
 *
 *  @param In64LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void In64Layer(In64LayerParam* p)
{
    GET_LANEID;
    const int gdx = (2*CEIL64(p->input_height));
    const int gdy = (CEIL64(p->input_width));
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        int bx = bid / gdy;
        int by = bid % gdy;
        ullong val;
        for (int i=0; i<32; i++)
        {
            float f0 = ( (by*64+laneid<(p->input_width)) && (bx*32+i<(p->input_height)) )?
                p->input_gpu[(bx*32+i)*(p->input_width)+by*64 +laneid]:-1.0f;
            float f1 = ( (by*64+32+laneid<(p->input_width)) && (bx*32+i<(p->input_height)) )?
                p->input_gpu[(bx*32+i)*(p->input_width)+by*64 + 32 + laneid]:-1.0f;
            unsigned r0 = __ballot_sync(0xFFFFFFFF, f0>=0?1:0);
            unsigned r1 = __ballot_sync(0xFFFFFFFF, f1>=0?1:0);
            ullong l0;
            asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1));//lo,hi
            if (laneid == i) val = __brevll(l0);
        }
        if (laneid < (p->input_height)*(p->input_width))
            p->output_gpu[by*gdx*32+bx*32+laneid] = val;
    }
}

/** @brief MLP input layer batched binarization.
 *
 *  Binarization function for the input layer of a MLP network (e.g., MLP for MNIST). This
 *  is the batched version for good load balancing when batch size is small.
 *
 *  @param In64LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void In64LayerBatched(In64LayerParam* p)
{
    GET_LANEID;
    const int gdx = (2*CEIL64(p->input_height));
    const int gdy = (CEIL64(p->input_width));
    const int gw = 32;
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy*gw; bid+=gridDim.x*32)
    {
        const int wid = (bid % (32 * gdy)) % 32;
        const int by = (bid % (32 * gdy)) / 32;
        const int bx = bid / (32 * gdy);
        ullong val;
        float f0 = ( (by*64+laneid<(p->input_width)) && (bx*32+wid<(p->input_height)) )?
            p->input_gpu[(bx*32+wid)*(p->input_width)+by*64 +laneid]:-1.0f;
        float f1 = ( (by*64+32+laneid<(p->input_width)) && (bx*32+wid<(p->input_height)) )?
            p->input_gpu[(bx*32+wid)*(p->input_width)+by*64 + 32 + laneid]:-1.0f;
        unsigned r0 = __ballot_sync(0xFFFFFFFF, f0>=0?1:0);
        unsigned r1 = __ballot_sync(0xFFFFFFFF, f1>=0?1:0);
        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1));//lo,hi
        if (laneid==0)
            p->output_gpu[by*gdx*32+bx*32+wid] = __brevll(l0);
    }
}

/** @brief 64-bit fully-connect SBNN layer function.
 *
 *  64-bit FC layer function: BMM=>BN=>Bin
 *
 *  @param Fc64LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void Fc64Layer(Fc64LayerParam* p)
{
    GET_LANEID;
    const int gdx = CEIL64(p->input_height); //vertical
    const int gdy = CEIL64(p->weight_width); //horizontal
    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        int bx = bid / gdy;
        int by = bid % gdy;
        const ullong* input_sub = &(p->input_gpu[bx*64]);
        const ullong* weight_sub = &(p->weight_gpu[by*64]);
        ullong* output_sub = &(p->output_gpu[by*gdx*64+bx*64]);
        register unsigned Cm[64] = {0};
        for (int i=0; (i*64) < (p->input_width); i++)
        {
            ullong a0 = input_sub[i*64*gdx+laneid];
            ullong a1 = input_sub[i*64*gdx+32+laneid];
            ullong b0 = weight_sub[i*64*gdy+laneid];
            ullong b1 = weight_sub[i*64*gdy+32+laneid];
            for (int j=0; j<32; j++)
            {
                ullong l0 = __shfl_sync(0xFFFFFFFF, b0, j);
                ullong l1 = __shfl_sync(0xFFFFFFFF, b1, j);
                Cm[j] += (__popcll(a0^l0)<<16) + __popcll(a1^l0);
                Cm[32+j] += (__popcll(a0^l1)<<16) + __popcll(a1^l1);
            }
        }
        ullong C0 = 0;
        ullong C1 = 0;
        for (int i=0; i<64; i++)
        {
            //if (by*64+i<(p->weight_width)) //required when matrix size cannot divide 64
            {
                short t0,t1;
                asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[i])); //lo, hi
                //if (bx*64+laneid<(p->input_height))
                {
                    C0 |= ((((((float)p->input_width)-2*(float)t0)
                                    <(p->bn_gpu[by*64+i]))?(ullong)0:(ullong)1)<<(63-i));
                }
                //if (bx*64+32+laneid<(p->input_height))
                {
                    C1 |= ((((((float)p->input_width)-2*(float)t1)
                                    <(p->bn_gpu[by*64+i]))?(ullong)0:(ullong)1)<<(63-i));
                }
            }
        }
        output_sub[laneid] = C0;
        output_sub[laneid+32] = C1;
    }
}

/** @brief 64-bit fully-connect SBNN batched layer function.
 *
 *  64-bit FC layer batched function: BMM=>BN=>Bin
 *
 *  @param Fc64LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void Fc64LayerBatched(Fc64LayerParam* p)
{
    GET_LANEID;
    const int gdx = CEIL64(p->input_height); //vertical
    const int gdy = CEIL64(p->weight_width); //horizontal
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy*32; bid+=gridDim.x*32)
    {
        const int bx = bid / (32 * gdy);
        const int by = (bid % (32 * gdy)) / 32;
        const int wid = (bid % (32 * gdy)) % 32;
        const ullong* input_sub = &(p->input_gpu[bx*64]);
        const ullong* weight_sub = &(p->weight_gpu[by*64]);
        ullong* output_sub = &(p->output_gpu[by*gdx*64+bx*64]);
        register unsigned c0=0, c1=0, c2=0, c3=0;
        for (int i=0; (i*64) < (p->input_width); i++)
        {
            ullong a0 = input_sub[i*64*gdx+wid];
            ullong a1 = input_sub[i*64*gdx+32+wid];
            ullong b0 = weight_sub[i*64*gdy+laneid];
            ullong b1 = weight_sub[i*64*gdy+32+laneid];
            c0 += __popcll(a0 ^ b0);
            c1 += __popcll(a1 ^ b0);
            c2 += __popcll(a0 ^ b1);
            c3 += __popcll(a1 ^ b1);
        }
        unsigned r0 = __ballot_sync(0xFFFFFFFF, ((((float)p->input_width)-2*(float)c0)<(p->bn_gpu[by*64+laneid]))?0:1);
        unsigned r1 = __ballot_sync(0xFFFFFFFF, ((((float)p->input_width)-2*(float)c2)<(p->bn_gpu[by*64+32+laneid]))?0:1);
        unsigned r2 = __ballot_sync(0xFFFFFFFF, ((((float)p->input_width)-2*(float)c1)<(p->bn_gpu[by*64+laneid]))?0:1);
        unsigned r3 = __ballot_sync(0xFFFFFFFF, ((((float)p->input_width)-2*(float)c3)<(p->bn_gpu[by*64+32+laneid]))?0:1);
        ullong l0,l1;
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1));//lo,hi
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l1):"r"(r2),"r"(r3));//lo,hi
        output_sub[wid] = __brevll(l0);
        output_sub[wid+32] = __brevll(l1);
    }
}


/** @brief 64-bit fully-connect SBNN output layer function.
 *
 *  64-bit FC output layer function: BMM
 *
 *  @param Out64LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void Out64Layer(Out64LayerParam* p)
{
    GET_LANEID;
    const int gdx = (CEIL64(p->input_height));
    const int gdy = (CEIL64(p->weight_width));
    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        int bx = bid / gdy;
        int by = bid % gdy;
        const ullong* input_sub = &(p->input_gpu[bx*64]);
        const ullong* weight_sub = &(p->weight_gpu[by*64]);
        float* output_sub = &(p->output_gpu[bx*(p->weight_width)*64+by*64]);
        register int Cm[64] = {0};
        for (int i=0; (i*64)<(p->input_width); i++)
        {
            ullong a0 = input_sub[i*64*gdx+laneid];
            ullong a1 = input_sub[i*64*gdx+32+laneid];
            ullong b0 = weight_sub[i*64*gdy+laneid];
            ullong b1 = weight_sub[i*64*gdy+32+laneid];
            #pragma unroll
            for (int j=0; j<32; j++)
            {
                ullong l0 = __shfl_sync(0xFFFFFFFF,b0,j);
                ullong l1 = __shfl_sync(0xFFFFFFFF,b1,j);
                Cm[j] += (__popcll(a0^l0)<<16) + __popcll(a1^l0);
                Cm[32+j] += (__popcll(a0^l1)<<16) + __popcll(a1^l1);
            }
        }
        for (int i=0; i<64; i++)
            if (by*64+i<(p->weight_width))
            {
                if (p->has_bn) 
                {
                    short t0,t1;
                    asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[i]));//lo,hi
                    if ((bx*64+laneid)<(p->input_height))
                    {
                        output_sub[laneid*(p->weight_width)+i] = 
                            ((float)(p->input_width) - (float)t0*2)*(p->bn_scale_gpu[by*64+i]) 
                            + (p->bn_bias_gpu[by*64+i]);
                    }
                    if ((bx*64+32+laneid)<(p->input_height))
                    {
                        output_sub[(laneid+32)*(p->weight_width)+i] = 
                            ((float)(p->input_width) - (float)t1*2)*(p->bn_scale_gpu[by*64+i]) 
                            + (p->bn_bias_gpu[by*64+i]);
                    }
                }
                else
                {
                    short t0,t1;
                    asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[i]));//lo,hi
                    if ((bx*64+laneid)<(p->input_height))
                        output_sub[laneid*(p->weight_width)+i] = ((float)(p->input_width) - (float)t0*2);
                    if ((bx*64+32+laneid)<(p->input_height))
                        output_sub[(laneid+32)*(p->weight_width)+i] = ((float)(p->input_width) - (float)t1*2);
                }

            }
    }
}


/** @brief 64-bit fully-connect SBNN batched output layer function.
 *
 *  64-bit FC output layer function: BMM
 *
 *  @param Out64LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void Out64LayerBatched(Out64LayerParam* p)
{
    GET_LANEID;
    const int gdx = CEIL64(p->input_height); //vertical
    const int gdy = CEIL64(p->weight_width); //horizontal
    const int gw = 32;
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy*gw; bid+=gridDim.x*32)
    {
        const int wid = (bid % (32 * gdy)) % 32;
        const int by = (bid % (32 * gdy)) / 32;
        const int bx = bid / (32 * gdy);
        const ullong* input_sub = &(p->input_gpu[bx*64]);
        const ullong* weight_sub = &(p->weight_gpu[by*64]);
        float* output_sub = &(p->output_gpu[bx*(p->weight_width)*64+by*64]);
        register unsigned c0=0, c1=0, c2=0, c3=0;
        for (int i=0; (i*64) < (p->input_width); i++)
        {
            ullong a0 = input_sub[i*64*gdx+wid];
            ullong a1 = input_sub[i*64*gdx+32+wid];
            ullong b0 = weight_sub[i*64*gdy+laneid];
            ullong b1 = weight_sub[i*64*gdy+32+laneid];
            c0 += __popcll(a0 ^ b0);
            c1 += __popcll(a1 ^ b0);
            c2 += __popcll(a0 ^ b1);
            c3 += __popcll(a1 ^ b1);
        }

        if (p->has_bn)
        {
            if (bx*64+wid<(p->input_height))
            {
                if ((by*64+laneid)<(p->weight_width))
                    output_sub[wid*(p->weight_width)+laneid] = 
                        ((float)(p->input_width) - (float)c0*2)*(p->bn_scale_gpu[by*64+laneid])
                        + (p->bn_bias_gpu[by*64+laneid]);

                if ((by*64+32+laneid)<(p->weight_width))
                    output_sub[wid*(p->weight_width)+laneid+32] = 
                        ((float)(p->input_width) - (float)c2*2)*(p->bn_scale_gpu[by*64+laneid+32])
                        + (p->bn_bias_gpu[by*64+laneid+32]);
            }
            if (bx*64+32+wid<(p->input_height))
            {
                if ((by*64+laneid)<(p->weight_width))
                    output_sub[(wid+32)*(p->weight_width)+laneid] 
                        = ((float)(p->input_width)-(float)c1*2)*(p->bn_scale_gpu[by*64+laneid])
                        + (p->bn_bias_gpu[by*64+laneid]);

                if ((by*64+32+laneid)<(p->weight_width))
                    output_sub[(wid+32)*(p->weight_width)+laneid+32]=
                        ((float)(p->input_width)-(float)c3*2)*(p->bn_scale_gpu[by*64+laneid+32])
                        + (p->bn_bias_gpu[by*64+laneid+32]);
            }
        }
        else
        {
            if (bx*64+wid<(p->input_height))
            {
                if ((by*64+laneid)<(p->weight_width))
                    output_sub[wid*(p->weight_width)+laneid] = ((float)(p->input_width) - (float)c0*2);
                if ((by*64+32+laneid)<(p->weight_width))
                    output_sub[wid*(p->weight_width)+laneid+32] = ((float)(p->input_width) - (float)c2*2);
            }
            if (bx*64+32+wid<(p->input_height))
            {
                if ((by*64+laneid)<(p->weight_width))
                    output_sub[(wid+32)*(p->weight_width)+laneid] = ((float)(p->input_width)-(float)c1*2);
                if ((by*64+32+laneid)<(p->weight_width))
                    output_sub[(wid+32)*(p->weight_width)+laneid+32]=((float)(p->input_width)-(float)c3*2);
            }
        }
    }
}


////======================== Convolution ==========================

/** @brief CNN input layer without input binarization.
 *
 *  The first layer of a binarized CNN. The input image is not binarized
 *  to avoid lossing too much information, so essentially it is a BW layer.
 *  The input image has 3 channels (R,G,B) with normalized floating-point 
 *  values around 0 via preprocessing. 
 *  
 *  @param In32Conv64LayerParam Parameter with FP32 input image and 3 channels.
 *  @return Void
 */

__device__ __inline__ void In32Conv64Layer(In32Conv64LayerParam* p)
{
    GET_LANEID;
    extern __shared__ int Cs[];
    const int ots = CEIL64(p->output_channels); //number of steps in K: output_channels
    volatile float* Csub = (float*)&Cs[warpid*(p->output_channels)];
    volatile ullong* sfilter = (ullong*)&Cs[32*(p->output_channels)]; 
    for (int i=threadIdx.x; i<(p->filter_height)*(p->filter_width)* (p->input_channels)*ots; i+=32*32)
        sfilter[i] = p->filter_gpu[i];
    __syncthreads();
    for (int bid = blockIdx.x*32+warpid; bid < (p->output_height) * (p->output_width) * (p->batch); 
            bid += gridDim.x*32)
    {
        const int bz = bid / (p->output_width * p->output_height); //over N:batch
        const int by = (bid % (p->output_width * p->output_height)) / (p->output_width);//over P:out_height
        const int bx = (bid % (p->output_width * p->output_height)) % (p->output_width);//over Q:out_width 
        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_horizontal)-(p->pad_w);
        const int ay0 = by*(p->stride_vertical)-(p->pad_h);
        for (int i=laneid; i<(p->output_channels); i+=32) 
            Csub[i] = 0;
        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            if ((ay>=0) && (ay<(p->input_height)))
            {
                for (int s=0; s<(p->filter_width); s++)
                {
                    const int ax = ax0 + s; //x-coord in Input
                    //within Input frame
                    if ((ax>=0) && (ax<(p->input_width)) )
                    {
                        float f0 = p->input_gpu[(bz*3+0)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//R
                        float f1 = p->input_gpu[(bz*3+1)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//G
                        float f2 = p->input_gpu[(bz*3+2)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//B

                        for (int k=0; k<ots; k++)
                        {
                            ullong l0 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 0*ots+k];
                            ullong l1 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 1*ots+k];
                            ullong l2 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 2*ots+k];
                            Csub[64*k+laneid] += (((l0>>(63-laneid))&0x1)?f0:-f0)
                                + (((l1>>(63-laneid))&0x1)?f1:-f1)
                                + (((l2>>(63-laneid))&0x1)?f2:-f2);
                            Csub[64*k+32+laneid] += (((l0>>(31-laneid))&0x1)?f0:-f0)
                                + (((l1>>(31-laneid))&0x1)?f1:-f1)
                                + (((l2>>(31-laneid))&0x1)?f2:-f2);
                        }
                    }
                }
            }
        }
        for (int k=0; k<ots; k++)
        {
            // save shape[batch, output_height, output_width, out_channels/64]
            bool b0 = (Csub[k*64+laneid])<(p->bn_gpu)[k*64+laneid]?0:1;
            bool b1 = (Csub[k*64+32+laneid])<(p->bn_gpu)[k*64+32+laneid]?0:1;
            unsigned r0 = __ballot_sync(0xFFFFFFFF, b0);
            unsigned r1 = __ballot_sync(0xFFFFFFFF, b1);
            ullong l0;
            asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //(low,high)
            ullong C = __brevll(l0);
            if (p->output_transpose) //If FC layer follows, store in column-major
            {
                p->output_gpu[(((by*p->output_width)+bx)*ots+k)*FEIL64(p->batch)+bz] = C;
            }
            else //Otherwise, store in row-major
            {
                p->output_gpu[(bz*(p->output_height)*(p->output_width)*ots) //N
                    + (by*(p->output_width)*ots)//P 
                    + (bx*ots) + k] //Q
                    = C;
            }
            if (p->save_residual)
            {
                p->save_residual_gpu[(bz*(p->output_height)*(p->output_width) *(p->output_channels))
                    + (by*(p->output_width)*(p->output_channels))
                    + (bx*(p->output_channels)) + k*64 + laneid] = (Csub[k*64+laneid]);

                p->save_residual_gpu[(bz*(p->output_height)*(p->output_width) *(p->output_channels))
                    + (by*(p->output_width)*(p->output_channels))
                    + (bx*(p->output_channels)) + k*64 + 32 + laneid] = (Csub[k*64+32+laneid]);
            }
        }
    }
}

/** @brief CNN input layer with 2x2 max pooling and without input binarization.
 *
 *  The first layer of a binarized CNN with 2x2 max pooling. The input image is not 
 *  binarized to avoid lossing too much information, so essentially it is a BW layer.
 *  The input image has 3 channels (R,G,B) with normalized floating-point 
 *  values around 0 via preprocessing. 
 *  
 *  @param In32Conv64LayerParam Parameter with FP32 input image and 3 channels.
 *  @return Void
 */
__device__ __inline__ void In32ConvPool64Layer(In32Conv64LayerParam* p)
{
    GET_LANEID;
    extern __shared__ int Cs[];
    const int ots = CEIL64(p->output_channels); //number of steps in K: output_channels
    volatile float* Csub = (float*)&Cs[warpid*(p->output_channels)];
    volatile ullong* sfilter = (ullong*)&Cs[32*(p->output_channels)]; 
    for (int i=threadIdx.x; i<(p->filter_height)*(p->filter_width)* (p->input_channels)*ots; i+=32*32)
        sfilter[i] = p->filter_gpu[i];
    __syncthreads();
    for (int bid = blockIdx.x*32+warpid; bid < 4*(p->output_height) * (p->output_width) * (p->batch); 
            bid += gridDim.x*32)
    {
        const int bz = bid / (4*p->output_width * p->output_height); //over N:batch
        const int by = (bid % (4*p->output_width * p->output_height)) / (2*p->output_width);//over P:out_height
        const int bx = (bid % (4*p->output_width * p->output_height)) % (2*p->output_width);//over Q:out_width 
        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_horizontal)-(p->pad_w);
        const int ay0 = by*(p->stride_vertical)-(p->pad_h);
        for (int i=laneid; i<(p->output_channels); i+=32) 
            Csub[i] = 0;
        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            if ((ay>=0) && (ay<(p->input_height))) 
            {
                for (int s=0; s<(p->filter_width); s++)
                {
                    const int ax = ax0 + s; //x-coord in Input
                    //within Input frame
                    if ((ax>=0) && (ax<(p->input_width)) )
                    {
                        float f0 = p->input_gpu[(bz*3+0)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//R
                        float f1 = p->input_gpu[(bz*3+1)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//G
                        float f2 = p->input_gpu[(bz*3+2)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//B

                        for (int k=0; k<ots; k++)
                        {
                            ullong l0 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 0*ots+k];
                            ullong l1 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 1*ots+k];
                            ullong l2 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 2*ots+k];

                            Csub[64*k+laneid] += (((l0>>(63-laneid))&0x1)?f0:-f0)
                                + (((l1>>(63-laneid))&0x1)?f1:-f1)
                                + (((l2>>(63-laneid))&0x1)?f2:-f2);
                            Csub[64*k+32+laneid] += (((l0>>(31-laneid))&0x1)?f0:-f0)
                                + (((l1>>(31-laneid))&0x1)?f1:-f1)
                                + (((l2>>(31-laneid))&0x1)?f2:-f2);
                        }
                    }
                }
            }
        }

        for (int k=0; k<ots; k++)
        {
            // save shape[batch, output_height, output_width, out_channels/64]
            bool b0 = (Csub[k*64+laneid])<(p->bn_gpu)[k*64+laneid]?0:1;
            bool b1 = (Csub[k*64+32+laneid])<(p->bn_gpu)[k*64+32+laneid]?0:1;
            unsigned r0 = __ballot_sync(0xFFFFFFFF, b0);
            unsigned r1 = __ballot_sync(0xFFFFFFFF, b1);
            ullong l0;
            asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //(low,high)
            ullong C = __brevll(l0);
            if (p->output_transpose) //If FC layer follows, store in column-major
            {
                if (laneid==0)
                    atomicOr(&p->output_gpu[((((by/2)*p->output_width)+(bx/2))*ots+k)
                            *FEIL64(p->batch)+bz],C);
            }
            else //Otherwise, store in row-major
            {
                if (laneid==0)
                    atomicOr(&p->output_gpu[(bz*(p->output_height)*(p->output_width)*ots) //N
                            + ((by/2)*(p->output_width)*ots)//P 
                            + ((bx/2)*ots) + k], C); //Q

            }
        }
    }
}

/** @brief CNN convolution layer.
 *
 *  The normal convoution layer of a binarized CNN. Input channel is 
 *  assumed to be a factor of 64 which is the general case.
 *  
 *  @param Conv64LayerParam Parameter object.
 *  @return Void
 */
__device__ __inline__ void Conv64Layer(Conv64LayerParam* p)
{
    GET_LANEID;
    extern __shared__ int Cs[];
    volatile int* Csub = (int*)&Cs[warpid*(p->output_channels)];

    const int ins = CEIL64(p->input_channels); //number of steps in C: input_channels
    const int ots = CEIL64(p->output_channels); //number of steps in K: output_channels
    for (int bid = blockIdx.x*32+warpid; bid < (p->output_height) * (p->output_width) * (p->batch); 
            bid += gridDim.x*32)
    {
        const int bz = bid / (p->output_width * p->output_height); //over N:batch
        const int by = (bid % (p->output_width * p->output_height)) / (p->output_width);//over P:out_height
        const int bx = (bid % (p->output_width * p->output_height)) % (p->output_width);//over Q:out_width 
        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_horizontal)-(p->pad_w);
        const int ay0 = by*(p->stride_vertical)-(p->pad_h);
        //track the number of filter entries that are masked off
        int exclude = 0;
        for (int i=laneid; i<(p->output_channels); i+=32) Csub[i] = 0;
        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            for (int s=0; s<(p->filter_width); s++)
            {
                const int ax = ax0 + s; //x-coord in Input
                if ( (ay>=0) && (ay<(p->input_height)) && (ax>=0) && (ax<(p->input_width)) ) //within Input frame
                {
                    for (int c=0; c<ins; c++)
                    {
                        ullong l0 = (p->input_gpu)[bz*(p->input_width)*(p->input_height)*ins
                            +(ay*(p->input_width)+ax)*ins+c]; //coalesced access
                        for (int i=laneid; i<(p->output_channels); i+=32)
                        {
                            //ullong l1; BYPASS_ULL(l1,(p->filter_gpu)[(r*(p->filter_width)+s)*ins*(p->output_channels) +c*(p->output_channels)+i]);

                            ullong l1 = (p->filter_gpu)[(r*(p->filter_width)+s)*ins*(p->output_channels) + c*(p->output_channels)+i];
                            Csub[i] +=  __popcll(l0 ^ l1);
                        }
                    }
                }
                else //not in frame
                {
                    exclude++; //accumulate
                }
            }
        }
        for (int k=0; k<ots; k++)
        {
            int a0 = (p->input_channels)*(p->filter_width)*(p->filter_height) 
                - exclude*(p->input_channels) - (2*Csub[k*64+laneid]);
            int a1 = (p->input_channels)*(p->filter_width)*(p->filter_height) 
                - exclude*(p->input_channels) - (2*Csub[k*64+32+laneid]);

            //take residual into considration
            if (p->inject_residual && ((k*64+laneid)<(p->residual_channels)))
            {
                if (p->residual_pool)
                {
                    int pl0 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*64 + laneid];
                    int pl1 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*64 + laneid];
                    int pl2 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*64 + laneid];
                    int pl3 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*64 + laneid];
                    a0 += max(pl3,max(pl2,max(pl0,pl1)));
                }
                else
                {
                    a0 += p->inject_residual_gpu[(bz*(p->output_height)*(p->output_width)
                            *(p->output_channels))
                        + (by*(p->output_width)*(p->output_channels))
                        + (bx*(p->output_channels)) + k*64 + laneid];
                }
            }

            if (p->inject_residual && (k*64+32+laneid<(p->residual_channels)))
            {
                if (p->residual_pool)
                {
                    int pl0 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*64 + 32 + laneid];
                    int pl1 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*64 + 32 + laneid];
                    int pl2 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*64 + 32 + laneid];
                    int pl3 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*64 + 32 + laneid];


                    a1 += max(pl3,max(pl2,max(pl0,pl1)));
                }
                else
                {
                    a1 += p->inject_residual_gpu[(bz*(p->output_height)*(p->output_width)
                            *(p->output_channels))
                        + (by*(p->output_width)*(p->output_channels))
                        + (bx*(p->output_channels)) + k*64 + 32 + laneid];
                }

            }

            unsigned r0 = __ballot_sync(0xFFFFFFFF, ((float)a0<(p->bn_gpu)[k*64+laneid])?0:1);
            unsigned r1 = __ballot_sync(0xFFFFFFFF, ((float)a1<(p->bn_gpu)[k*64+32+laneid])?0:1);

            //////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ////bool b0 = ((k*64+laneid>=(p->output_channels)) |  
                ////(float)((p->input_channels)*(p->filter_width)*(p->filter_height)
                ////- exclude*(p->input_channels)-(2*Csub[k*64+laneid]))<(p->bn_gpu)[k*64+laneid])?0:1;
            ////bool b1 = ((k*64+32+laneid>=(p->output_channels)) |  
                ////(float)((p->input_channels)*(p->filter_width)*(p->filter_height)
                ////- exclude*(p->input_channels)-(2*Csub[k*64+32+laneid]))<(p->bn_gpu)[k*64+32+laneid])?0:1;
                //unsigned r0 = __ballot(b0);
                //unsigned r1 = __ballot(b1);
            ullong l0;
            asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //(low,high)
            ullong C = __brevll(l0);

            if (p->output_transpose) //If FC layer follows, store in column-major
            {
                if (laneid==0)
                    p->output_gpu[(((by*p->output_width)+bx)*ots+k)*FEIL64(p->batch)+bz] = C;
            }
            else //Otherwise, store in row-major
            {
                if (laneid==0)
                    p->output_gpu[(bz*(p->output_height)*(p->output_width)*ots) //N
                        + (by*(p->output_width)*ots)//P 
                        + (bx*ots) + k] //Q
                        = C;
            }

            //save residual
            if (p->save_residual)
            {
                p->save_residual_gpu[(bz*(p->output_height)*(p->output_width) *(p->output_channels))
                    + (by*(p->output_width)*(p->output_channels))
                    + (bx*(p->output_channels)) + k*64 + laneid] = a0;

                p->save_residual_gpu[(bz*(p->output_height)*(p->output_width) *(p->output_channels))
                    + (by*(p->output_width)*(p->output_channels))
                    + (bx*(p->output_channels)) + k*64 + 32 + laneid] = a1;
            }

        }
    }
}

/** @brief CNN convolution layer with 2x2 max pooling.
 *
 *  The normal convoution layer of a binarized CNN. Input channel is 
 *  assumed to be a factor of 32 which is the general case.
 *  
 *  @param Conv32LayerParam Parameter object.
 *  @return Void
 */
__device__ __inline__ void ConvPool64Layer(Conv64LayerParam* p)
{
    GET_LANEID;
    extern __shared__ int Cs[];
    volatile int* Csub = (int*)&Cs[warpid*(p->output_channels)];
    const int ins = CEIL64(p->input_channels); //number of steps in C: input_channels
    const int ots = CEIL64(p->output_channels); //number of steps in K: output_channels
    for (int bid = blockIdx.x*32+warpid; bid < 4*(p->output_height) 
            * (p->output_width) * (p->batch); bid += gridDim.x*32)
    {   
        const int bz = bid / (4*p->output_width * p->output_height); //over N:batch
        const int by = (bid % (4*p->output_width * p->output_height)) / (2*p->output_width); //over P:out_height
        const int bx = (bid % (4*p->output_width * p->output_height)) % (2*p->output_width); //over Q:out_width 
        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_horizontal)-(p->pad_w);
        const int ay0 = by*(p->stride_vertical)-(p->pad_h);
        int exclude = 0; //track the number of filter entries that are masked off
        for (int i=laneid; i<(p->output_channels); i+=32) Csub[i] = 0;
        for (int r=0; r<(p->filter_height); r++) //load a window of data from Input
        {
            const int ay = ay0 + r; //y-coord in Input
            for (int s=0; s<(p->filter_width); s++)
            {
                const int ax = ax0 + s; //x-coord in Input
                if ( (ay>=0) && (ay<(p->input_height)) && (ax>=0) && (ax<(p->input_width)) ) //within Input frame
                {
                    for (int c=0; c<ins; c++)
                    {
                        ullong l0 = (p->input_gpu)[bz*(p->input_width)*(p->input_height)*ins
                            +(ay*(p->input_width)+ax)*ins+c]; //coalesced access
                        for (int i=laneid; i<(p->output_channels); i+=32)
                        {
                            //ullong l1;
                            //BYPASS_ULL(l1,(p->filter_gpu)[(r*(p->filter_width)+s)*ins*(p->output_channels)
                            //+c*(p->output_channels)+i]);


                            ullong l1 = (p->filter_gpu)[(r*(p->filter_width)+s)*ins*(p->output_channels)
                                +c*(p->output_channels)+i];
                            Csub[i] +=  __popcll(l0 ^ l1);
                        }
                    }
                }
                else //not in frame
                {
                    exclude++; //accumulate
                }
            }
        }
        for (int k=0; k<ots; k++)
        {
            int a0 = (p->input_channels)*(p->filter_width)*(p->filter_height) 
                - exclude*(p->input_channels) - (2*Csub[k*64+laneid]);
            int a1 = (p->input_channels)*(p->filter_width)*(p->filter_height) 
                - exclude*(p->input_channels) - (2*Csub[k*64+32+laneid]);

            //take residual into considration
            if (p->inject_residual && ((k*64+laneid)<(p->residual_channels)))
            {
                if (p->residual_pool)
                {
                    int pl0 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*64 + laneid];
                    int pl1 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*64 + laneid];
                    int pl2 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*64 + laneid];
                    int pl3 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*64 + laneid];
                    a0 += max(pl3,max(pl2,max(pl0,pl1)));
                }
                else
                {
                    a0 += p->inject_residual_gpu[(bz*(p->output_height)*(p->output_width)
                            *(p->output_channels))
                        + (by*(p->output_width)*(p->output_channels))
                        + (bx*(p->output_channels)) + k*64 + laneid];
                }
            }

            if (p->inject_residual && (k*64+32+laneid<(p->residual_channels)))
            {
                if (p->residual_pool)
                {
                    int pl0 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*64 + 32 + laneid];
                    int pl1 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*64 + 32 + laneid];
                    int pl2 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*64 + 32 + laneid];
                    int pl3 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*64 + 32 + laneid];


                    a1 += max(pl3,max(pl2,max(pl0,pl1)));
                }
                else
                {
                    a1 += p->inject_residual_gpu[(bz*(p->output_height)*(p->output_width)
                            *(p->output_channels))
                        + (by*(p->output_width)*(p->output_channels))
                        + (bx*(p->output_channels)) + k*64 + 32 + laneid];
                }

            }

            unsigned r0 = __ballot_sync(0xFFFFFFFF, ((float)a0<(p->bn_gpu)[k*64+laneid])?0:1);
            unsigned r1 = __ballot_sync(0xFFFFFFFF, ((float)a1<(p->bn_gpu)[k*64+32+laneid])?0:1);
            ullong l0;
            asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //(low,high)
            ullong C = __brevll(l0);
            if (p->output_transpose) //If FC layer follows, store in column-major
            {
                if (laneid==0)
                atomicOr(&p->output_gpu[((((by/2)*p->output_width)
                        +(bx/2))*ots+k)*FEIL64(p->batch)+bz],C);
            }
            else //Otherwise, store in row-major
            {
                if (laneid==0)
                    atomicOr(&p->output_gpu[(bz*(p->output_height)*(p->output_width)*ots) //N
                            + ((by/2)*(p->output_width)*ots)//P 
                            + ((bx/2)*ots) + k],C);
            }

            //save residual
            if (p->save_residual)
            {
                p->save_residual_gpu[(bz*(p->output_height)*(p->output_width) *(p->output_channels))
                    + (by*(p->output_width)*(p->output_channels))
                    + (bx*(p->output_channels)) + k*64 + laneid] = a0;

                p->save_residual_gpu[(bz*(p->output_height)*(p->output_width) *(p->output_channels))
                    + (by*(p->output_width)*(p->output_channels))
                    + (bx*(p->output_channels)) + k*64 + 32 + laneid] = a1;
            }




        }
    }
}


#endif
