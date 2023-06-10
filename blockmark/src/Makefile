
NVCC = nvcc
CC = g++

BASE_ADR = $(shell pwd)

CUDA_INCLUDEPATH = $(CUDA_INSTALL_PATH)/include
CPPFLAGS = -DINTENSITY=16 -lcudart -lssl -g -I $(CUDA_INCLUDEPATH) -I $(BASE_ADR)/../ -I $(BASE_ADR)/../x11
NVCC_OPTS = -lcudart -lssl -lcrypto -Xcompiler -m64 -gencode arch=compute_30,code=compute_30 -Xptxas='-w' -I $(BASE_ADR)/../
OBJ = util.o cuda_checkhash.o


ALL_HASH = blake_test bmw_test grs_test skein_test jh_test luffa_test cube_test keccak_test shavite_test simd_test echo_test

.PHONY: neoscrypt_test cryptonight_test $(ALL_HASH)

cryptonight_test: $(OBJ) cryptonight_test.o	

neoscrypt_test: $(OBJ) neoscrypt_test.o

blake_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DBLAKE -c hash_test.cpp -o $@.o;

bmw_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DBMW -c hash_test.cpp -o $@.o;

grs_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DGRS -c hash_test.cpp -o $@.o;

skein_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DSKEIN -c hash_test.cpp -o $@.o;

jh_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DJH -c hash_test.cpp -o $@.o;

luffa_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DLUFFA -c hash_test.cpp -o $@.o;

cube_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DCUBE -c hash_test.cpp -o $@.o;

keccak_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DKECCAK -c hash_test.cpp -o $@.o;

shavite_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DSHAVITE -c hash_test.cpp -o $@.o;

simd_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DSIMD -c hash_test.cpp -o $@.o;

echo_test: $(OBJ)
	$(CC) $(CPPFLAGS) -DECHO -c hash_test.cpp -o $@.o;

cuda_checkhash.o: cuda_checkhash.cu
	$(NVCC) $(NVCC_OPTS) -o $@ -c $<

clean:
	rm -f $(BASE_ADR)/*.o