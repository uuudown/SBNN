NVCC = nvcc
NVCC_FLAG = -std=c++11 -O3 -w -arch=sm_70 -maxrregcount=64 -rdc=true  # -Xptxas -v
LIBS = -ljpeg

# For debug
#NVCC_FLAG = -std=c++11 -w -O0 -g -G -arch=sm_70 -maxrregcount=64 -rdc=true -Xptxas -v


all: cifar10_resnet imagenet_resnet alexnet vggnet mnist_mlp
#all: imagenet_resnet


cifar10_resnet: cifar10_resnet.cu sbnn32_param.h sbnn64_param.h sbnn32.cuh sbnn64.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ cifar10_resnet.cu data.cpp $(LIBS)

imagenet_resnet: imagenet_resnet.cu sbnn32_param.h sbnn64_param.h sbnn32.cuh sbnn64.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ imagenet_resnet.cu data.cpp $(LIBS)

alexnet: alexnet.cu sbnn32_param.h sbnn64_param.h sbnn32.cuh sbnn64.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ alexnet.cu data.cpp $(LIBS)

imagenet_vgg: imagenet_vgg.cu sbnn32_param.h sbnn64_param.h sbnn32.cuh sbnn64.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ imagenet_vgg.cu data.cpp $(LIBS)

vggnet: cifar10_vgg.cu sbnn32_param.h sbnn64_param.h sbnn32.cuh sbnn64.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ cifar10_vgg.cu data.cpp $(LIBS)

mnist_mlp: mnist_mlp.cu sbnn32_param.h sbnn64_param.h sbnn32.cuh sbnn64.cuh data.h data.cpp utility.h
	$(NVCC) $(NVCC_FLAG) -o $@ mnist_mlp.cu data.cpp $(LIBS)

clean:
	rm cifar10_resnet imagenet_resnet alexnet vggnet mnist_mlp



