NVCC = nvcc
CC = g++

BASE_ADR = $(shell pwd)

CUDA_INCLUDEPATH = $(CUDA_INSTALL_PATH)/include

NVCC_OPTS = -lcudart -lssl -lcrypto -Xcompiler -m64 -gencode=arch="compute_50, code=compute_50" -Xptxas='-w' -I $(BASE_ADR)/../

OBJ = cuda_bmw512.o cuda_jh512.o cuda_quark_blake512.o cuda_quark_groestl512.o cuda_quark_keccak512.o \
	cuda_skein512.o nist5.o 

ALL: $(OBJ)

cuda_bmw512.o: cuda_bmw512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_jh512.o: cuda_jh512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_quark_blake512.o: cuda_quark_blake512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_quark_groestl512.o: cuda_quark_groestl512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_quark_keccak512.o: cuda_quark_keccak512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

#cuda_quark_compactionTest.o: cuda_quark_compactionTest.cu
#	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_skein512.o: cuda_skein512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

nist5.o: nist5.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

#quarkcoin.o: quarkcoin.cu
#	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

clean:
	rm *.o