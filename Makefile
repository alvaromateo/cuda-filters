CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DSIZE=32


EXEFILTERS  = filters.exe

OBJFILTERS  = filters.o

default: $(EXEFILTERS)

filters.o: filters.cu
	$(NVCC) -c -o $@ filters.cu $(NVCC_FLAGS) $(PROG_FLAGS)


$(EXEFILTERS): $(OBJFILTERS)
	$(NVCC) $(OBJFILTERS) -o $(EXEFILTERS) $(LD_FLAGS)


all:	$(EXEFILTERS) 

clean:
	rm -rf *.o *.exe

