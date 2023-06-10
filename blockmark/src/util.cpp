#include <stdarg.h>
#include <cstring>
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "cuda_debug.cuh"
#include "miner.h"

long device_sm;
short device_mpcount;
int device_bfactor;
int cryptonight_fork;

void init_device_gpgpusim() {
	cudaDeviceProp props;

	cudaGetDeviceProperties(&props, 0);

	// device_sm = (props.major * 100 + props.minor * 10);
	device_sm = 500;
	device_mpcount = (short) props.multiProcessorCount;

	cryptonight_fork = 1;
	device_bfactor = 0;
}

bool fulltest(const uint32_t *hash, const uint32_t *target) {
	int i;
	bool rc = true;
	
	for (i = 7; i >= 0; i--) {
		if (hash[i] > target[i]) {
			rc = false;
			break;
		}
		if (hash[i] < target[i]) {
			rc = true;
			break;
		}
	}

	return rc;
}

void applog(int prio, const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	int len = 40 + (int) strlen(fmt) + 2;
	char * f = (char*) alloca(len);
	sprintf(f, "%s%s\n", fmt);
	vfprintf(stdout, f, ap);	/* atomic write to stdout */	
	va_end(ap);
}

void gpulog(int prio, const char *fmt, ...)
{
	char pfmt[128];
	char line[256];
	int len;
	va_list ap;

	len = snprintf(pfmt, 128, "GPU: %s", fmt);
	pfmt[sizeof(pfmt)-1]='\0';

	va_start(ap, fmt);

	if (len && vsnprintf(line, sizeof(line), pfmt, ap)) {
		line[sizeof(line)-1]='\0';
		applog(prio, "%s", line);
	} else {
		fprintf(stderr, "%s OOM!\n", __func__);
	}

	va_end(ap);
}

int cuda_available_memory()
{
	size_t mtotal = 0, mfree = 0;
	cudaDeviceSynchronize();
	cudaMemGetInfo(&mfree, &mtotal);
	return (int) (mfree / (1024 * 1024));
}

void test_scanhash(struct work* work,
		void (*scanhash)(uint32_t threads, uint32_t startNounce, uint32_t *d_hash),
		void (*init)(uint32_t threads)) {
	#ifndef INTENSITY
		exit(-1);	
	#endif // !INTENSITY
	uint32_t throughput = 1 << INTENSITY;
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *d_hash;
	init(throughput);
	CUDA_SAFE_CALL(cudaMalloc(&d_hash, (size_t) 64 * throughput));
	scanhash(throughput, pdata[19], d_hash);
}

void test_scanhash(struct work* work,
		void (*scanhash)(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order),
		void (*init)(uint32_t threads)) {
	#ifndef INTENSITY
		exit(-1);	
	#endif // !INTENSITY
	uint32_t throughput = 1 << INTENSITY;
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *d_hash;
	init(throughput);
	CUDA_SAFE_CALL(cudaMalloc(&d_hash, (size_t) 64 * throughput));
	scanhash(throughput, pdata[19], NULL, d_hash, 0);
}

void test_scanhash(struct work* work,
		void (*scanhash)(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order),
		int (*init)(uint32_t threads)) {
	#ifndef INTENSITY
		exit(-1);	
	#endif // !INTENSITY
	uint32_t throughput = 1 << INTENSITY;
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *d_hash;
	if(init(throughput)) {
		exit(-1);
	}
	CUDA_SAFE_CALL(cudaMalloc(&d_hash, (size_t) 64 * throughput));
	scanhash(throughput, pdata[19], NULL, d_hash, 0);
}