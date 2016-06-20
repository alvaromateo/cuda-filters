CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DSIZE=32

VPATH = src/
BUILDDIR = ./build/


EXEFILTERS  = filters.exe
OBJFILTERS  = mainFilters.o tools.o test.o image.o
OBJINCLUDES = $(addprefix $(BUILDDIR),$(OBJFILTERS))


default: $(EXEFILTERS)

$(BUILDDIR)%.o: %.cpp
	$(NVCC) $(NVCC_FLAGS) $(PROG_FLAGS) -c -o $@ $<

./build/kernel.o: kernel.cu
	$(NVCC) $(NVCC_FLAGS) $(PROG_FLAGS) -c -o $@ $<

$(EXEFILTERS): $(OBJINCLUDES) ./build/kernel.o
	$(NVCC) $^ -o $(EXEFILTERS) $(LD_FLAGS) -g


all:	$(EXEFILTERS) 

clean:
	rm -rf build/*.o *.exe

clean-output:
	rm -rf cuda-filters.* .test

