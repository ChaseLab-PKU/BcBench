
NVCC = nvcc
CC = g++

BASE_ADR = $(shell pwd)

CUDA_INCLUDEPATH = $(CUDA_INSTALL_PATH)/include

NVCC_OPTS = -lcudart -lssl -lcrypto -Xcompiler -m64  -gencode arch=compute_75,code=compute_75 -I $(BASE_ADR)/..
CPPFLAGS = -lcudart -lssl -g -I $(CUDA_INCLUDEPATH) -I $(BASE_ADR)/../ -I $(BASE_ADR)

OBJ =	cryptonight.o cryptonight-extra.o cryptonight-core.o \
	cryptonight-cpu.o c_keccak.o oaes_lib.o c_keccak.o aesb.o

all: ${OBJ}

c_keccak.o: cpu/c_keccak.cpp
	$(CC) $(CPPFLAGS) -o c_keccak.o -c $(BASE_ADR)/cpu/c_keccak.cpp

cryptonight.o: cryptonight.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cryptonight-core.o: cryptonight-core.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cryptonight-extra.o: cryptonight-extra.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

clean:
	rm *.o