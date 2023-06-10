NVCC = nvcc
CC = g++

BASE_ADR = $(shell pwd)

CUDA_INCLUDEPATH = $(CUDA_INSTALL_PATH)/include

NVCC_OPTS = -lcudart -lssl -lcrypto -Xcompiler -m64 -gencode=arch="compute_50, code=compute_50" -Xptxas='-w' -I $(BASE_ADR)/../

OBJ = cuda_jha_keccak512.o jha.o

ALL: $(OBJ)

#cuda_jha_compactionTest.o: cuda_jha_compactionTest.cu
#	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_jha_keccak512.o: cuda_jha_keccak512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

#jackpotcoin.o: jackpotcoin.cu
#	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

jha.o: jha.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

clean:
	rm *.o