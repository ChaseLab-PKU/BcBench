NVCC = nvcc
CC = g++

BASE_ADR = $(shell pwd)

CPPFLAGS = -lcudart -lssl -g -fpermissive -I  ../

# md_helper can't compile standalone

OBJ = aes_helper.o blake.o blake2b.o blake2s.o bmw.o cubehash.o echo.o fugue.o \
	groestl.o hamsi_helper.o haval.o jh.o keccak.o luffa.o ripemd.o \
	sha2.o sha2big.o shabal.o shavite.o simd.o skein.o streebog.o whirlpool.o

ALL: $(OBJ)

clean:
	rm $(BASE_ADR)/*.o