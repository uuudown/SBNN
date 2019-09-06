# SBNN: Singular Binarized Neural Network based on GPU Bit Operations

This is the code repository for our SuperComputig (SC-19) paper. Please see my homepage:www.angliphd.com. The code is developed by **Ang Li** from the HPC group of Pacific Northwest National Laboratory (PNNL). 

*Ang Li, Tong Geng, Tianqi Wang, Martin Herbordt, Shuaiwen Leon Song, and Kevin Barker.2019.BSTC: A Novel Binarized-Soft-Tensor-Core Design for Accelerating Bit-Based Approximated Neural Nets. The International Conference for High Performance Computing, Networking, Storage, and Analysis (SC'19), November 17-22, 2019, Denver, CO, USA. ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3295500.3356169*

**Abstract** Binarized neural networks (or BNNs) promise tremendous performance improvement over traditional DNNs through simplified bitlevel
computation and significantly reduced memory access/storage cost. In addition, it has advantages of low-cost, low-energy, and high-robustness, showing great potential in resources-constrained, volatile, and latency-critical applications, which are critical for future HPC, cloud, and edge applications. However, the promised significant performance gain of BNN inference has never been fully demonstrated on general-purpose processors, particularly on GPUs, due to: (i) the challenge of extracting and leveraging sufficient finegrained bit-level-parallelism to saturate GPU cores when the batch size is small; (ii) the fundamental design conflict between bit-based BNN algorithm and word-based architecture; and (iii) architecture & performance unfriendly to BNN network design. To address (i) and (ii), we propose a binarized-soft-tensor-core as a software-hardware codesign approach to construct bit-manipulation capability for modern GPUs and thereby effectively harvest bit-level-parallelism (BLP).
To tackle (iii), we propose intra- and inter-layer fusion techniques so
that the entire BNN inference execution can be packed into a single
GPU kernel, and so avoid the high-cost of frequent launching and
releasing. 

**Data Input** We tested SBNN on MNIST (http://yann.lecun.com/exdb/mnist/), CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html) and ImageNet (http://www.image-net.org/). We provide C++ based image reading and preprocessing (e.g., for imageNet, we do RGB channel normalization, rescaling smaller edge to 256 and central-crop to 224x224) functions for reading images from the three datasets. We validated the correctness by comparing the RGB normalized floating-point values with PyTorch. See data.h and data.cpp for details. We provide "process_one_image.py" to show the results for ImageNet image reading to verify the data reading interface. We provide "process\_imagenet\_file.py" to adjust the path for loading the images.

**BNN Models** We tested a 4-layer MLP on MNIST. We tested a VGG-like and ResNet-14 networks on CIFAR-10. We tested an AlexNet and ResNet-18 on ImageNet. Please see our paper for the detailed network configuration. The classification results are validated with the results from PyTorch and Tensorflow. Please each .cu file. We provide trained BNN network configuration (i.e., binary weights & thresholds) in pytroch_training directory. We will provide the PyTorch training scripts in the future.

**Bit-Matrix-Multiplication and Bit-Convolution** We provide Bit-Matrix-Multiplication and Bit-Convolution functions. The convolution function follows the definition of conv2d() of Tensorflow (with stride, padding, pooling, transpose, residual).

**SBNN-32 and SBNN-64** We have two implementations SBNN32 and SBNN64. The difference is not data type but the granularity of workload processed by each lane of a warp -- for SBNN-32, each lane processes a 32-bit unsigned int, for SBNN-64, each lane processes a 64-bit unsigned long long. For BMM, 32bits is better than 64bits, but for BCONV, BCONV64 is better than BCONV-32, see our paper for details.

**Compile** Normally, you just need to update the NVCC\_FLAG in Makefile according to your system environment and "make". The architecture flag "sm\_xx" needs to match your GPU. 

**Run** Just execute the binary generated. To switch between SBNN32 and SBNN64, modify the corresponding source file and use "main32()" or "main64" in the main function. You may also need to set the GPU device number, the data file path, the model configuration file path in the source file for proper execution. We use cudaEvent to measure the time. We report Top-1 and Top-5 inference accuracy for the current batch. We tested the code on NVIDIA P100 DGX-1 and V100 DGX-1 systems.

**Acknowledgement** 
This research is mainly supported by PNNL's DeepScience-HPC LDRD project. I would specially thank Dr. Courtney Corley and Dr. Nathan Hodas for their support on this work. This code is also partially supported by PNNL's DMC-CFA LDRD project, DOE ASCR CENATE - Center for Advanced Architecture Evaluation project, and PNNL's High Performance Data Analytics (HPDA) program.

