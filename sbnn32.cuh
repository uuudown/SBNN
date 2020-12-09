/** @file sbnn32.cuh
 *  @brief Layer functions for 32-bit SBNN.
 *
 *  Layer functions including input, convolution, fully-connect and output functions
 *
 *  @author Ang Li (PNNL)
 *
*/

#ifndef SBNN32_CUH
#define SBNN32_CUH

/** @brief MLP input layer binarization.
 *
 *  Binarization function for the input layer of a MLP network (e.g., MLP for MNIST).
 *
 *  @param In32LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void In32Layer(In32LayerParam* p)
{
    GET_LANEID;
    const int gdx = (CEIL(p->input_height));
    const int gdy = (CEIL(p->input_width));
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        unsigned bx = bid / gdy;
        unsigned by = bid % gdy;
        unsigned val;
        #pragma unroll
        for (int i=0; i<32; i++)
        {
            float f0 = ( (by*32+laneid<(p->input_width)) && (bx*32+i<(p->input_height)) )?
                p->input_gpu[(bx*32+i)*(p->input_width)+by*32 +laneid]:-1.0f;
            unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0?1:0));
            if (laneid == i) val = r0;
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
 *  @param In32LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void In32LayerBatched(In32LayerParam* p)
{
    GET_LANEID;
    const int gdx = (CEIL(p->input_height));
    const int gdy = (CEIL(p->input_width));
    const int gw = 32;
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy*gw; bid+=gridDim.x*32)
    {
        const unsigned wid = (bid % (32 * gdy)) % 32;
        const unsigned by = (bid % (32 * gdy)) / 32;
        const unsigned bx = bid / (32 * gdy);
        float f0 = ( (by*32+laneid<(p->input_width)) && (bx*32+wid<(p->input_height)) )?
            p->input_gpu[(bx*32+wid)*(p->input_width)+by*32 +laneid]:-1.0f;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0?1:0));
        if (laneid==0)
            p->output_gpu[by*gdx*32+bx*32+wid] = r0;
    }
}

/** @brief 32-bit fully-connect SBNN layer function.
 *
 *  32-bit FC layer function: BMM=>BN=>Bin
 *
 *  @param Fc32LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void Fc32Layer(Fc32LayerParam* p)
{
    GET_LANEID;
    const int gdx = CEIL(p->input_height); //vertical
    const int gdy = CEIL(p->weight_width); //horizontal
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        unsigned bx = bid / gdy;
        unsigned by = bid % gdy;
        const unsigned* input_sub = &(p->input_gpu[bx*32]);
        const unsigned* weight_sub = &(p->weight_gpu[by*32]);
        unsigned* output_sub = &(p->output_gpu[by*gdx*32+bx*32]);
        register int Cm[32] = {0};
        for (int i=0; (i*32) < (p->input_width); i++)
        {
            unsigned r0 = input_sub[i*32*gdx+laneid];
            unsigned r1 = weight_sub[i*32*gdy+laneid];
            #pragma unroll
            for (int j=0; j<32; j++)
            {
                unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, j); //from lane-j, r1 of weight matrix
                Cm[j] += __popc(r0 ^ r2);
            }
        }
        unsigned C = 0;
        if (bx*32+laneid<(p->input_height))
        {
            for (int i=0; i<32; i++)
            {
                C = C+C;
                if (by*32+i<(p->weight_width))
                {
                    float t = ((float)p->input_width)-2*(float)Cm[i];
                    C += ((t<(p->bn_gpu[by*32+i]))?0:1);
                }
            }
        }
        output_sub[laneid] = C;
    }
}

/** @brief 32-bit fully-connect SBNN batched layer function.
 *
 *  32-bit FC layer batched function: BMM=>BN=>Bin
 *
 *  @param Fc32LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void Fc32LayerBatched(Fc32LayerParam* p)
{
    GET_LANEID;
    const int gdx = CEIL(p->input_height); //vertical
    const int gdy = CEIL(p->weight_width); //horizontal
    const int gw = 32;
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy*gw; bid+=gridDim.x*32)
    {
        const int bx = bid / (32 * gdy);
        const int by = (bid % (32 * gdy)) / 32;
        const int wid = (bid % (32 * gdy)) % 32;
        const unsigned* input_sub = &(p->input_gpu[bx*32]);
        const unsigned* weight_sub = &(p->weight_gpu[by*32]);
        unsigned* output_sub = &(p->output_gpu[by*gdx*32+bx*32]);
        register int C = 0;
        for (int i=0; (i*32) < (p->input_width); i++)
        {
            unsigned r0 = input_sub[i*32*gdx+wid];
            unsigned r1 = weight_sub[i*32*gdy+laneid];
            C += __popc(r0 ^ r1);
        }
        unsigned r2 = __ballot_sync(0xFFFFFFFF, (((float)p->input_width)-2*(float)C <(p->bn_gpu[by*32+laneid]))?0:1);
        output_sub[wid] = __brev(r2);
    }
}


/** @brief 32-bit fully-connect SBNN output layer function.
 *
 *  32-bit FC output layer function: BMM
 *
 *  @param Out32LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void Out32Layer(Out32LayerParam* p)
{
    GET_LANEID;
    const int gdx = (CEIL(p->input_height));
    const int gdy = (CEIL(p->weight_width));
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        unsigned bx = bid / gdy;
        unsigned by = bid % gdy;
        const unsigned* input_sub = &(p->input_gpu[bx*32]);
        const unsigned* weight_sub = &(p->weight_gpu[by*32]);
        float* output_sub = &(p->output_gpu[bx*(p->weight_width)*32+by*32]);
        register int Cm[32] = {0};
        for (int i=0; (i*32)<(p->input_width); i++)
        {
            unsigned r0 = input_sub[i*32*gdx+laneid];
            unsigned r1 = weight_sub[i*32*gdy+laneid];
            #pragma unroll
            for (int j=0; j<32; j++)
            {
                unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, j); //from lane-j, r1 of weight matrix
                Cm[j] += __popc(r0 ^ r2);
            }
        }
        if ((bx*32+laneid)<(p->input_height))
        {
            for (int i=0; i<32; i++)
            {
                if (by*32+i<(p->weight_width))
                {
                    if (p->has_bn) 
                    {
                        output_sub[laneid*(p->weight_width)+i] = ((float)(p->input_width) 
                            - (float)Cm[i]*2.0f)*(p->bn_scale_gpu[by*32+i]) 
                            + (p->bn_bias_gpu[by*32+i]);
                    }
                    else
                        output_sub[laneid*(p->weight_width)+i] = ((float)(p->input_width)
                            - (float)Cm[i]*2);
                }
            }
        }
    }
}

/** @brief 32-bit fully-connect SBNN batched output layer function.
 *
 *  32-bit FC output layer function: BMM
 *
 *  @param Out32LayerParam The layer input parameter object.
 *  @return Void
 */
__device__ __inline__ void Out32LayerBatched(Out32LayerParam* p)
{
    GET_LANEID;
    const int gdx = CEIL(p->input_height); //vertical
    const int gdy = CEIL(p->weight_width); //horizontal
    const int gw = 32;
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy*gw; bid+=gridDim.x*32)
    {
        const int bx = bid / (32 * gdy);
        const int by = (bid % (32 * gdy)) / 32;
        const int wid = (bid % (32 * gdy)) % 32;
        const unsigned* input_sub = &(p->input_gpu[bx*32]);
        const unsigned* weight_sub = &(p->weight_gpu[by*32]);
        float* output_sub = &(p->output_gpu[bx*(p->weight_width)*32+by*32]);
        register int C = 0;
        for (int i=0; (i*32) < (p->input_width); i++)
        {
            unsigned r0 = input_sub[i*32*gdx+wid];
            unsigned r1 = weight_sub[i*32*gdy+laneid];
            C += __popc(r0 ^ r1);
        }
        if (((bx*32+wid)<(p->input_height)) && (by*32+laneid<(p->weight_width)))
        {
            if (p->has_bn)
                output_sub[wid*(p->weight_width)+laneid] = ((float)(p->input_width) 
                        - (float)C*2) * (p->bn_scale_gpu[by*32+laneid]) + (p->bn_bias_gpu[by*32+laneid]);
            else
                output_sub[wid*(p->weight_width)+laneid] = ((float)(p->input_width) - (float)C*2);
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
 *  @param In32Conv32LayerParam Parameter with FP32 input image and 3 channels.
 *  @return Void
 */

__device__ __inline__ void In32Conv32Layer(In32Conv32LayerParam* p)
{
    GET_LANEID;
    extern __shared__ int Cs[];
    const int ots = CEIL(p->output_channels); //number of steps in K: output_channels
    volatile float* Csub = (float*)&Cs[warpid*(p->output_channels)];
    volatile unsigned* sfilter = (unsigned*)&Cs[32*(p->output_channels)]; 
    for (int i=threadIdx.x; i<(p->filter_height)*(p->filter_width)* (p->input_channels)*ots; i+=32*32)
        sfilter[i] = p->filter_gpu[i];
    __syncthreads();
    for (int bid = blockIdx.x*32+warpid; bid < (p->output_height) * (p->output_width) 
            * (p->batch); bid += gridDim.x*32)
    {
        const int bz = bid / (p->output_width * p->output_height); //over N:batch
        const int by = (bid % (p->output_width * p->output_height)) / (p->output_width);//over P:out_height
        const int bx = (bid % (p->output_width * p->output_height)) % (p->output_width);//over Q:out_width 
        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_horizontal)-(p->pad_w);
        const int ay0 = by*(p->stride_vertical)-(p->pad_h);
        for (int i=laneid; i<(p->output_channels); i+=32) Csub[i] = 0; 
        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            if ( (ay>=0) && (ay<(p->input_height)) )
            {
                for (int s=0; s<(p->filter_width); s++)
                {
                    const int ax = ax0 + s; //x-coord in Input
                    //within Input frame
                    if ( (ax>=0) && (ax<(p->input_width)) )
                    {
                        float f0 = p->input_gpu[(bz*3+0)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//R
                        float f1 = p->input_gpu[(bz*3+1)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//G
                        float f2 = p->input_gpu[(bz*3+2)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//B

                        for (int k=0; k<ots; k++)
                        {
                            unsigned l0 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 0*ots+k];
                            unsigned l1 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 1*ots+k];
                            unsigned l2 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 2*ots+k];

                            Csub[32*k+laneid] += (((l0>>(31-laneid))&0x1)?f0:-f0)
                                + (((l1>>(31-laneid))&0x1)?f1:-f1)
                                + (((l2>>(31-laneid))&0x1)?f2:-f2);
                        }
                    }
                }
            }
        }
        for (int k=0; k<ots; k++)
        {
            // save shape[batch, output_height, output_width, out_channels/32]
            bool bin = (Csub[k*32+laneid])<(p->bn_gpu)[k*32+laneid]?0:1;
            unsigned C = __brev(__ballot_sync(0xFFFFFFFF,bin));
            //If FC layer follows, store in column-major
            if (p->output_transpose)
            {
                p->output_gpu[(((by*p->output_width)+bx)*ots+k)*FEIL(p->batch)+bz] = C;
            }
            //Otherwise, store in row-major
            else
            {
                p->output_gpu[(bz*(p->output_height)*(p->output_width)*ots) //N
                    + (by*(p->output_width)*ots)//P 
                    + (bx*ots) + k] //Q
                    = C;
            }
            if (p->save_residual)
            {
                p->save_residual_gpu[(bz*(p->output_height)*(p->output_width)
                        *(p->output_channels))
                    + (by*(p->output_width)*(p->output_channels))
                    + (bx*(p->output_channels)) + k*32 + laneid]=Csub[k*32+laneid];
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
 *  @param In32Conv32LayerParam Parameter with FP32 input image and 3 channels.
 *  @return Void
 */
__device__ __inline__ void In32ConvPool32Layer(In32Conv32LayerParam* p)
{
    GET_LANEID;
    extern __shared__ int Cs[];
    const int ots = CEIL(p->output_channels); //number of steps in K: output_channels
    volatile float* Csub = (float*)&Cs[warpid*(p->output_channels)];
    volatile unsigned* sfilter = (unsigned*)&Cs[32*(p->output_channels)]; 
    for (int i=threadIdx.x; i<(p->filter_height)*(p->filter_width)*(p->input_channels)*ots; i+=32*32)
        sfilter[i] = p->filter_gpu[i];
    __syncthreads();
    for (int bid = blockIdx.x*32+warpid; bid < 4*(p->output_height)*(p->output_width) 
            * (p->batch); bid += gridDim.x*32)
    {
        const int bz = bid/(4*p->output_width*p->output_height); //N:batch
        const int by = (bid%(4*p->output_width*p->output_height))/(2*p->output_width);//P:out_height
        const int bx = (bid%(4*p->output_width*p->output_height))%(2*p->output_width);//Q:out_width 
        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_horizontal)-(p->pad_w);
        const int ay0 = by*(p->stride_vertical)-(p->pad_h);

        for (int i=laneid; i<(p->output_channels); i+=32) Csub[i] = 0; 
        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            if ( (ay>=0) && (ay<(p->input_height)) )
            {
                for (int s=0; s<(p->filter_width); s++)
                {
                    const int ax = ax0 + s; //x-coord in Input
                    //within Input frame
                    if ( (ax>=0) && (ax<(p->input_width)) )
                    {
                        float f0 = p->input_gpu[(bz*3+0)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//R
                        float f1 = p->input_gpu[(bz*3+1)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//G
                        float f2 = p->input_gpu[(bz*3+2)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//B
                        for (int k=0; k<ots; k++)
                        {
                            unsigned l0 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 0*ots+k];
                            unsigned l1 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 1*ots+k];
                            unsigned l2 = sfilter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 2*ots+k];

                            Csub[32*k+laneid] += (((l0>>(31-laneid))&0x1)?f0:-f0)
                                + (((l1>>(31-laneid))&0x1)?f1:-f1)
                                + (((l2>>(31-laneid))&0x1)?f2:-f2);
                        }
                    }
                }
            }
        }
        for (int k=0; k<ots; k++)
        {
            // save shape[batch, output_height, output_width, out_channels/32]
            bool bin = ((float)(Csub[k*32+laneid]))<(p->bn_gpu)[k*32+laneid]?0:1;
            unsigned C = __brev(__ballot_sync(0xFFFFFFFF, bin));
            //If FC layer follows, store in column-major
            if (p->output_transpose)
            {
                if (laneid==0)
                    atomicOr(&p->output_gpu[((((by/2)*p->output_width)+(bx/2))*ots+k)
                            *FEIL(p->batch)+bz],C);
            }
            //Otherwise, store in row-major
            else
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
 *  assumed to be a factor of 32 which is the general case.
 *  
 *  @param Conv32LayerParam Parameter object.
 *  @return Void
 */
__device__ __inline__ void Conv32Layer(Conv32LayerParam* p)
{
    GET_LANEID;
    extern __shared__ int Cs[];
    volatile int* Csub = (int*)&Cs[warpid*(p->output_channels)];
    const int ins = CEIL(p->input_channels); //number of steps in C: input_channels
    const int ots = CEIL(p->output_channels); //number of steps in K: output_channels

    for (int bid = blockIdx.x*32+warpid; bid < (p->output_height) * 
            (p->output_width) * (p->batch); bid += gridDim.x*32)
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
                //within Input frame
                if ( (ay>=0) && (ay<(p->input_height)) && (ax>=0) && (ax<(p->input_width)) )
                {
                    for (int c=0; c<ins; c++)
                    {
                        unsigned r0 = (p->input_gpu)[bz*(p->input_width)*(p->input_height)*ins
                            +(ay*(p->input_width)+ax)*ins+c]; //coalesced access

                        for (int i=laneid; i<(p->output_channels); i+=32)
                        {
                            unsigned r1 = (p->filter_gpu)[(r*(p->filter_width)+s)*ins*(p->output_channels)
                                +c*(p->output_channels)+i];
                            Csub[i] +=  __popc(r0 ^ r1);
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
            int res = (p->input_channels)*(p->filter_width)*(p->filter_height) //C*R*S
                - exclude*(p->input_channels) //eliminate padding distoration 
                - (2*Csub[k*32+laneid]);//n-2acc(a^b) for 0/1 to simulate acc(a*b) for +1/-1

            //take residual into considration
            if (p->inject_residual && ((k*32+laneid)<(p->residual_channels)))
            {
                if (p->residual_pool)
                {
                    int pl0 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*32 + laneid];

                    int pl1 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*32 + laneid];

                    int pl2 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*32 + laneid];

                    int pl3 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*32 + laneid];
                    res += max(pl3,max(pl2,max(pl0,pl1)));
                }
                else
                {
                    res += p->inject_residual_gpu[(bz*(p->output_height)*(p->output_width)
                            *(p->residual_channels))
                        + (by*(p->output_width)*(p->residual_channels))
                        + (bx*(p->residual_channels)) + k*32 + laneid];
                }

            }
             
            unsigned C = __brev(__ballot_sync(0xFFFFFFFF, (float)res<(p->bn_gpu)[k*32+laneid]?0:1));
            if (p->output_transpose) //If FC layer follows, store in column-major
            {
                p->output_gpu[(((by*p->output_width)+bx)*ots+k)*FEIL(p->batch)+bz] = C;
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
                p->save_residual_gpu[(bz*(p->output_height)*(p->output_width)
                        *(p->output_channels))
                    + (by*(p->output_width)*(p->output_channels))
                    + (bx*(p->output_channels)) + k*32 + laneid] = res;
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

__device__ __inline__ void ConvPool32Layer(Conv32LayerParam* p)
{
    GET_LANEID;
    extern __shared__ int Cs[];
    volatile int* Csub = (int*)&Cs[warpid*(p->output_channels)];
    const int ins = CEIL(p->input_channels); //number of steps in C: input_channels
    const int ots = CEIL(p->output_channels); //number of steps in K: output_channels
    for (int bid = blockIdx.x*32+warpid; bid < 4*(p->output_height) * 
            (p->output_width) * (p->batch); bid += gridDim.x*32)
    {
        const int bz = bid / (4*p->output_width * p->output_height); //over N:batch
        const int by = (bid % (4*p->output_width * p->output_height)) / (2*p->output_width);//over P:out_height
        const int bx = (bid % (4*p->output_width * p->output_height)) % (2*p->output_width);//over Q:out_width 
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
                        unsigned r0 = (p->input_gpu)[bz*(p->input_width)*(p->input_height)*ins
                            +(ay*(p->input_width)+ax)*ins+c]; //coalesced access

                        for (int i=laneid; i<(p->output_channels); i+=32)
                        {
                            unsigned r1;
                            BYPASS_US(r1, (p->filter_gpu)[(r*(p->filter_width)+s)
                                    *ins*(p->output_channels) +c*(p->output_channels)+i] );

                            Csub[i] +=  __popc(r0 ^ r1);
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
            float res = (float)(p->input_channels)*(p->filter_width)*(p->filter_height) //C*R*S
                - (float)exclude*(p->input_channels) //eliminate padding distoration 
                - (float)(2*Csub[k*32+laneid]);//n-2acc(a^b) for 0/1 to simulate acc(a*b) for +1/-1

            if (p->inject_residual && ((k*32+laneid)<(p->residual_channels)))
            {
                if (p->residual_pool)
                {
                    int pl0 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*32 + laneid];

                    int pl1 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*32 + laneid];

                    int pl2 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx)*(p->residual_channels)) + k*32 + laneid];

                    int pl3 = p->inject_residual_gpu[(bz*(2*p->output_height)
                            *(2*p->output_width)*(p->residual_channels))
                        + ((2*by+1)*(2*p->output_width)*(p->residual_channels))
                        + ((2*bx+1)*(p->residual_channels)) + k*32 + laneid];
                    res += max(pl3,max(pl2,max(pl0,pl1)));
                }
                else
                {
                    res += p->inject_residual_gpu[(bz*(p->output_height)*(p->output_width)
                            *(p->residual_channels))
                        + (by*(p->output_width)*(p->residual_channels))
                        + (bx*(p->residual_channels)) + k*32 + laneid];
                }
            }

            unsigned C = __brev(__ballot_sync(0xFFFFFFFF, (float)res< ((p->bn_gpu)[k*32+laneid])?0:1));
            if (p->output_transpose) //If FC layer follows, store in column-major
            {
                //For Tensorflow (HWCN)
                if (laneid==0) atomicOr(&p->output_gpu[((((by/2)*p->output_width)+(bx/2))*ots+k)*FEIL(p->batch)+bz],C);
            }
            else //Otherwise, store in row-major
            {
                if (laneid==0) //For normal convolution layer BHWC
                    atomicOr(&p->output_gpu[(bz*(p->output_height)*(p->output_width)*ots) //N
                    + ((by/2)*(p->output_width)*ots)//P 
                    + ((bx/2)*ots) + k],C);
            }

            if (p->save_residual)
            {
                p->save_residual_gpu[(bz*(p->output_height)*(p->output_width)
                        *(p->output_channels))
                    + (by*(p->output_width)*(p->output_channels))
                    + (bx*(p->output_channels)) + k*32 + laneid] = res;
            }
        }
    }
}



#endif
