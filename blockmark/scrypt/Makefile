NVCC = nvcc
CC = g++

BASE_ADR = $(shell pwd)

CUDA_INCLUDEPATH = $(CUDA_INSTALL_PATH)/include

NVCC_OPTS = -lcudart -lssl -lcrypto -Xcompiler -m64 -gencode=arch="compute_50, code=compute_50" -I $(BASE_ADR)/..
CPPFLAGS = -lcudart -lssl -g -I $(CUDA_INCLUDEPATH) -I $(BASE_ADR)/../ -I $(BASE_ADR)