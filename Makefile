CUDA_HOME   = /Soft/cuda/7.5.18

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc -rdc=true
LD_FLAGS    = -lcudadevrt -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DSIZE=32 -lm

VPATH = src/
BUILDDIR = ./build/
FILTER = filter-
DEPS = stb_image.h stb_image_write.h readCommandLine.h


EXEFILTERS  = sequential singleCardAsyn singleCardSyn multiCard
OBJFILTERS  = sequential.o singleCardAsyn.o singleCardSyn.o multiCard.o
OBJINCLUDES = $(addprefix $(BUILDDIR),$(OBJFILTERS))
EXES = $(addsuffix .exe, $(EXEFILTERS))


all: $(EXES)

$(BUILDDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(NVCC_FLAGS) $(PROG_FLAGS) -c -o $@ $< -g

./build/readCommandLine.o: readCommandLine.c
	$(NVCC) $(NVCC_FLAGS) $(PROG_FLAGS) -c -o $@ $< -g

$(EXES): %.exe: $(BUILDDIR)%.o ./build/readCommandLine.o
	$(NVCC) -o $@ $< build/readCommandLine.o $(LD_FLAGS) -g

clean:
	rm -rf build/*.o *.exe

clean-output:
	rm -rf cuda-filters.* *.test

$(BUILDDIR):
	mkdir $(BUILDDIR)

