
NVCC = nvcc
CC = g++

BASE_ADR = $(shell pwd)

CUDA_INCLUDEPATH = $(CUDA_INSTALL_PATH)/include

NVCC_OPTS =  -g -lcudart -lssl -lcrypto -Xcompiler -m64 -gencode=arch="compute_30, code=compute_30" -I $(BASE_ADR)/..

OBJ = cuda_neoscrypt.o

all: ${OBJ}

cuda_neoscrypt.o: cuda_neoscrypt.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

clean:
	rm $(BASE_ADR)/*.o