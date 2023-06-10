NVCC = nvcc
CC = g++

BASE_ADR = $(shell pwd)
BIN_CRYPTONIGHT = $(BASE_ADR)/bin_cryptonight
BIN_HASH = $(BASE_ADR)/bin_hash
BIN_NEOSCRYPT = $(BASE_ADR)/bin_neoscrypt
BIN_ADR = $(BASE_ADR)/bin2
BIN_TEST = $(BASE_ADR)/test_bin

CUDA_INCLUDEPATH = $(CUDA_INSTALL_PATH)/include

NVCC_OPTS = -lcudart -lssl -lcrypto -Xcompiler -m64 -gencode arch=compute_75,code=compute_75 -Xptxas='-w' -I ./
CPPFLAGS = -lcudart -lssl -g -I $(CUDA_INCLUDEPATH)

SRC = main.cpp
EXE = blockbench

define test_hash
	@echo "make $(1)"
	cd $(BIN_HASH); rm -f ./*
	cd x11; make; cp $(BASE_ADR)/x11/*.o $(BIN_HASH)
	cd sph; make; cp $(BASE_ADR)/sph/*.o $(BIN_HASH)
	cd JHA; make; cp $(BASE_ADR)/JHA/*.o $(BIN_HASH)
	cd quark; make; cp $(BASE_ADR)/quark/*.o $(BIN_HASH)
	cd src; make clean; make $(1); cp $(BASE_ADR)/src/*.o $(BIN_HASH)
	cd $(BIN_HASH);  $(NVCC) $(NVCC_OPTS) $(BIN_HASH)/*.o -o $(2).o; cp $(2).o $(BIN_TEST); rm ./*
endef

test_cryptonight:
	cd crypto; make; cp $(BASE_ADR)/crypto/*.o $(BIN_CRYPTONIGHT)
	cd sph; make; cp $(BASE_ADR)/sph/*.o $(BIN_CRYPTONIGHT)
	cd src; make clean; make cryptonight_test; cp $(BASE_ADR)/src/*.o $(BIN_CRYPTONIGHT)
	cd $(BIN_CRYPTONIGHT); $(NVCC) $(NVCC_OPTS) $(BIN_CRYPTONIGHT)/*.o -o 	test_cryptonight

test_neocrypt:
	cd neoscrypt; make; cp $(BASE_ADR)/neoscrypt/*.o $(BIN_NEOSCRYPT)
	cd src; make clean; make neoscrypt_test; cp $(BASE_ADR)/src/*.o $(BIN_NEOSCRYPT)
	cd $(BIN_NEOSCRYPT); $(NVCC) $(NVCC_OPTS) $(BIN_NEOSCRYPT)/*.o -o test_neocrypt

test_blake:
	$(call test_hash,  blake_test, $@)

test_bmw:
	$(call test_hash,  bmw_test, $@)

test_grs:
	$(call test_hash,  grs_test, $@)

test_skein:
	$(call test_hash,  skein_test, $@)

test_jh:
	$(call test_hash,  jh_test, $@)

test_luffa:
	$(call test_hash,  luffa_test, $@)

test_cube:
	$(call test_hash,  cube_test, $@)

test_keccak:
	$(call test_hash,  keccak_test, $@)

test_shavite:
	$(call test_hash,  shavite_test, $@)

test_simd:
	$(call test_hash,  simd_test, $@)

test_echo:
	$(call test_hash,  echo_test, $@)

test_all_hash: test_blake test_bmw test_grs test_skein test_jh  test_luffa test_cube test_keccak test_shavite test_simd test_echo

blockbench:
	cd x11; make; cp $(BASE_ADR)/x11/*.o $(BIN_ADR)
	cd crypto; make; cp $(BASE_ADR)/crypto/*.o $(BIN_ADR)
	cd sph; make; cp $(BASE_ADR)/sph/*.o $(BIN_ADR)
	cd JHA; make; cp $(BASE_ADR)/JHA/*.o $(BIN_ADR)
	cd quark; make; cp $(BASE_ADR)/quark/*.o $(BIN_ADR)
	cd sha256; make; cp $(BASE_ADR)/sha256/*.o $(BIN_ADR)
	cd src; make; cp $(BASE_ADR)/src/*.o $(BIN_ADR)
	cd $(BIN_ADR); $(NVCC) $(NVCC_OPTS) $(BIN_ADR)/*.o -o $(EXE)

clean:
	cd $(BASE_ADR)/x11; make clean
	cd $(BASE_ADR)/crypto; make clean
	cd $(BASE_ADR)/sph; make clean
	cd $(BASE_ADR)/JHA; make clean
	cd $(BASE_ADR)/quark; make clean
	cd $(BASE_ADR)/sha256; make clean
	cd $(BASE_ADR)/src; make clean
	cd $(BIN_ADR); rm -f *.o $(EXE)

clean_gpgpusim:
	cd $(BIN_ADR); rm _cuobjdump* _app_cuda_* *.ptx *.ptxas