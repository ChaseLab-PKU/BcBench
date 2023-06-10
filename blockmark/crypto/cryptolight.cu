
#include "cryptolight.h"

extern char *device_config; // -l 32x16

static __thread uint32_t cn_blocks  = 32;
static __thread uint32_t cn_threads = 16;

static uint32_t *d_long_state;
static uint32_t *d_ctx_state;
static uint32_t *d_ctx_key1;
static uint32_t *d_ctx_key2;
static uint32_t *d_ctx_text;
static uint64_t *d_ctx_tweak;
static uint32_t *d_ctx_a;
static uint32_t *d_ctx_b;

static bool init_flag = false;

extern "C" int scanhash_cryptolight(struct work* work, uint32_t max_nonce, unsigned long *hashes_done, int variant)
{
	int res = 0;
	uint32_t throughput = 0;

	uint32_t *ptarget = work->target;
	uint8_t *pdata = (uint8_t*) work->data;
	uint32_t *nonceptr = (uint32_t*) (&pdata[39]);
	const uint32_t first_nonce = *nonceptr;
	uint32_t nonce = first_nonce;
	// int dev_id = device_map[thr_id];

	ptarget[7] = 0x00ff;

	if(!init_flag)
	{
		/*
		if (!device_config[thr_id] && strcmp(device_name[dev_id], "TITAN V") == 0) {
			device_config[thr_id] = strdup("80x32");
		}

		if (device_config[thr_id]) {
			sscanf(device_config[thr_id], "%ux%u", &cn_blocks, &cn_threads);
			throughput = cuda_default_throughput(thr_id, cn_blocks*cn_threads);
			gpulog(LOG_INFO, thr_id, "Using %u x %u kernel launch config, %u threads",
				cn_blocks, cn_threads, throughput);
		} else {
			throughput = cuda_default_throughput(thr_id, cn_blocks*cn_threads);
			if (throughput != cn_blocks*cn_threads && cn_threads) {
				cn_blocks = throughput / cn_threads;
				throughput = cn_threads * cn_blocks;
			}
			gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u threads (%ux%u)",
				throughput2intensity(throughput), throughput, cn_blocks, cn_threads);
		}

		if(sizeof(size_t) == 4 && throughput > UINT32_MAX / MEMORY) {
			gpulog(LOG_ERR, thr_id, "THE 32bit VERSION CAN'T ALLOCATE MORE THAN 4GB OF MEMORY!");
			gpulog(LOG_ERR, thr_id, "PLEASE REDUCE THE NUMBER OF THREADS OR BLOCKS");
			exit(1);
		}

		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		*/

		throughput = cn_blocks * cn_threads;
		const size_t alloc = MEMORY * throughput;
		cryptonight_extra_init();

		cudaMalloc(&d_long_state, alloc);
		// exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_state, 25 * sizeof(uint64_t) * throughput);
		// exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_key1, 40 * sizeof(uint32_t) * throughput);
		// exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_key2, 40 * sizeof(uint32_t) * throughput);
		// exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_text, 32 * sizeof(uint32_t) * throughput);
		// exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_a, 4 * sizeof(uint32_t) * throughput);
		// exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_b, 4 * sizeof(uint32_t) * throughput);
		// exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_tweak, sizeof(uint64_t) * throughput);

		init_flag = true;
	}

	throughput = cn_blocks*cn_threads;

	do
	{
		const uint32_t Htarg = ptarget[7];
		uint32_t resNonces[2] = { UINT32_MAX, UINT32_MAX };

		cryptonight_extra_setData(pdata, ptarget);
		cryptonight_extra_prepare(throughput, nonce, d_ctx_state, d_ctx_a, d_ctx_b, d_ctx_key1, d_ctx_key2, variant, d_ctx_tweak);
		cryptolight_core_hash(cn_blocks, cn_threads, d_long_state, d_ctx_state, d_ctx_a, d_ctx_b, d_ctx_key1, d_ctx_key2, variant, d_ctx_tweak);
		cryptonight_extra_final(throughput, nonce, resNonces, d_ctx_state);

		*hashes_done = nonce - first_nonce + throughput;

		if(resNonces[0] != UINT32_MAX)
		{
			uint32_t vhash[8];
			uint32_t tempdata[19];
			uint32_t *tempnonceptr = (uint32_t*)(((char*)tempdata) + 39);
			memcpy(tempdata, pdata, 76);
			*tempnonceptr = resNonces[0];
			cryptolight_hash_variant(vhash, tempdata, 76, variant);
			if(vhash[7] <= Htarg && fulltest(vhash, ptarget))
			{
				res = 1;
				work->nonces[0] = resNonces[0];
				//work_set_target_ratio(work, vhash);
				// second nonce
				if(resNonces[1] != UINT32_MAX)
				{
					*tempnonceptr = resNonces[1];
					cryptolight_hash_variant(vhash, tempdata, 76, variant);
					if(vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						res++;
						work->nonces[1] = resNonces[1];
					}
				}
				goto done;
			} 
			/*
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for nonce %08x does not validate on CPU!", resNonces[0]);
			}
			*/
		}

		if ((uint64_t) throughput + nonce >= max_nonce - 127) {
			nonce = max_nonce;
			break;
		}

		nonce += throughput;
		// gpulog(LOG_DEBUG, thr_id, "nonce %08x", nonce);

	} while (max_nonce > (uint64_t)throughput + nonce);

done:
	// gpulog(LOG_DEBUG, thr_id, "nonce %08x exit", nonce);
	work->valid_nonces = res;
	*nonceptr = nonce;
	return res;
}

void free_cryptolight()
{
	if (!init_flag)
		return;

	cudaFree(d_long_state);
	cudaFree(d_ctx_state);
	cudaFree(d_ctx_key1);
	cudaFree(d_ctx_key2);
	cudaFree(d_ctx_text);
	cudaFree(d_ctx_tweak);
	cudaFree(d_ctx_a);
	cudaFree(d_ctx_b);

	cryptonight_extra_free();

	cudaDeviceSynchronize();

	init_flag = false;
}
