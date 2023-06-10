
NVCC = nvcc
CC = g++

BASE_ADR = $(shell pwd)

CUDA_INCLUDEPATH = $(CUDA_INSTALL_PATH)/include

NVCC_OPTS = -lcudart -lssl -lcrypto -Xcompiler -m64 -gencode=arch="compute_75, code=compute_75" -I $(BASE_ADR)/..

OBJ = x11.o cuda_x11_simd512.o cuda_x11_echo.o cuda_x11_luffa512_Cubehash.o \
	cuda_x11_shavite512.o cuda_x11_cubehash512.o cuda_x11_luffa512.o

all: ${OBJ}

x11.o: x11.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_x11_simd512.o: cuda_x11_simd512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_x11_cubehash512.o: cuda_x11_cubehash512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_x11_luffa512.o: cuda_x11_luffa512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_x11_luffa512_Cubehash.o: cuda_x11_luffa512_Cubehash.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_x11_shavite512.o: cuda_x11_shavite512.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_x11_echo.o: cuda_x11_echo.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

clean:
	rm $(BASE_ADR)/*.o