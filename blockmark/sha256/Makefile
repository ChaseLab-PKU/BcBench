NVCC = nvcc
CC = g++

NVCC_OPTS = -lcudart -lssl -lcrypto -Xcompiler -m64 -gencode=arch="compute_50, code=compute_50" -Xptxas='-w' -I ../
GCC_OPTS = -lcudart -lssl -g -I ../

OBJ = cuda_sha256t.o cuda_sha256d.o sha256t.o sha256d.o

ALL: $(OBJ)

cuda_sha256t.o: cuda_sha256t.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

sha256t.o: sha256t.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

cuda_sha256d.o: cuda_sha256d.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

sha256d.o: sha256d.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<


clean:
	rm $(OBJ)