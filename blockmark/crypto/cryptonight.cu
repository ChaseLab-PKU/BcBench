
#include "cryptonight.h"

static __thread uint32_t cn_blocks;
static __thread uint32_t cn_threads;

// used for gpu intensity on algo init
static __thread bool gpu_init_shown = false;
#define gpulog_init(p,thr,fmt, ...) if (!gpu_init_shown) \
	gpulog(p, thr, fmt, ##__VA_ARGS__)

static uint64_t *d_long_state;
static uint32_t *d_ctx_state;
static uint32_t *d_ctx_key1;
static uint32_t *d_ctx_key2;
static uint32_t *d_ctx_text;
static uint64_t *d_ctx_tweak;
static uint32_t *d_ctx_a;
static uint32_t *d_ctx_b;

static bool init_flag = false;

extern "C" int scanhash_cryptonight(struct work* work, uint32_t max_nonce, unsigned long *hashes_done, int variant)
{
	int res = 0;
	uint32_t throughput = 0;
	uint64_t* h_memory;
	size_t alloc;

	uint32_t *ptarget = work->target;
	uint8_t *pdata = (uint8_t*) work->data;
	uint32_t *nonceptr = (uint32_t*) (&pdata[39]);
	const uint32_t first_nonce = *nonceptr;
	uint32_t nonce = first_nonce;

	printf("%u\n", nonce);
	ptarget[7] = 0x0000ffff;
	ptarget[6] = 0xffffffff; // cryptonight hsa different check method

	uint32_t file_num = 0;
	uint32_t len;
	FILE* fp;
	char buffer[100];

	if(!init_flag)
	{
		int mem = cuda_available_memory();
		int mul = device_sm >= 300 ? 4 : 1; // see cryptonight-core.cu
		cn_threads = device_sm >= 600 ? 16 : 8; // real TPB is x4 on SM3+
		cn_blocks = device_mpcount * 4;
		// if (cn_blocks*cn_threads*2.2 > mem) 
			cn_blocks = device_mpcount * 2;

		printf("%d MB available, %hd SMX\n", mem, device_mpcount);

		/*
		if (!device_config[thr_id]) {
			if(strcmp(device_name[dev_id], "TITAN V") == 0)
				device_config[thr_id] = strdup("80x24");
			if(strstr(device_name[dev_id], "V100"))
				device_config[thr_id] = strdup("80x24");
		}

		if (device_config[thr_id]) {
			int res = sscanf(device_config[thr_id], "%ux%u", &cn_blocks, &cn_threads);
			throughput = cuda_default_throughput(thr_id, cn_blocks*cn_threads);
			gpulog_init(LOG_INFO, thr_id, "Using %ux%u(x%d) kernel launch config, %u threads",
				cn_blocks, cn_threads, mul, throughput);
		} else {
		*/
			throughput = cn_blocks*cn_threads;
			if (throughput != cn_blocks*cn_threads && cn_threads) {
				cn_blocks = throughput / cn_threads;
				throughput = cn_threads * cn_blocks;
			}
			printf("%u threads with %u blocks\n", throughput, cn_blocks);//, cn_threads, mul);
		// }

		if(sizeof(size_t) == 4 && throughput > UINT32_MAX / MEMORY) {
			printf("THE 32bit VERSION CAN'T ALLOCATE MORE THAN 4GB OF MEMORY!\n");
			printf("PLEASE REDUCE THE NUMBER OF THREADS OR BLOCKS\n");
			exit(1);
		}

		/*
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		*/

		alloc = MEMORY * size_t(throughput);
		printf("Need %d MBs", throughput * 2);
		cryptonight_extra_init();

		cudaMalloc(&d_long_state, alloc);
		exit_if_cudaerror(__FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_state, 50 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(__FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_key1, 40 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(__FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_key2, 40 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(__FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_text, 32 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(__FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_a, 4 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(__FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_b, 4 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(__FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_tweak, sizeof(uint64_t) * throughput);
		exit_if_cudaerror(__FILE__, __LINE__);

		gpu_init_shown = true;
		init_flag = true;
	}

	throughput = cn_blocks*cn_threads;

	do
	{
		const uint32_t Htarg = ptarget[7];
		uint32_t resNonces[2] = { UINT32_MAX, UINT32_MAX };

		cryptonight_extra_setData(pdata, ptarget);
		cryptonight_extra_prepare(throughput, nonce, d_ctx_state, d_ctx_a, d_ctx_b, d_ctx_key1, d_ctx_key2, variant, d_ctx_tweak);

		FILE* fp;
		fp = fopen("../../bin_cryptonight/data.bin", "rb");
		
		h_memory = (uint64_t*)malloc(alloc);
		fread(h_memory, 1, alloc, fp);
		cudaMemcpy(d_long_state, h_memory, alloc, cudaMemcpyHostToDevice);

		fclose(fp);
		cryptonight_core_cuda(cn_blocks, cn_threads, d_long_state, d_ctx_state, d_ctx_a, d_ctx_b, d_ctx_key1, d_ctx_key2, variant, d_ctx_tweak);

		/*
		uint32_t len = 50 * sizeof(uint32_t) * throughput;
		h_debug = (uint32_t*)malloc(len);
		cudaMemcpy(h_debug, d_ctx_state, len, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_memory, d_debug, sizeof(uint32_t) * 100, cudaMemcpyDeviceToHost);

		for(int i = 0; i < 8; i++)  {
			printf("%x\n", h_memory[i]);
		}

		sprintf(buffer, "debug_file%d", file_num++);
		fp = fopen(buffer, "w+");
		for(int i = 0; i < len/(sizeof(uint64_t)); i++) {
			fprintf(fp, "%x\n", h_debug[i]);
		}
		*/

		// cryptonight_extra_final(throughput, nonce, resNonces, d_ctx_state);

		*hashes_done = nonce - first_nonce + throughput;
		exit(0);



		if(resNonces[0] != UINT32_MAX)
		{
			uint32_t vhash[8];
			uint32_t tempdata[19];
			uint32_t *tempnonceptr = (uint32_t*)(((char*)tempdata) + 39);
			memcpy(tempdata, pdata, 76);
			*tempnonceptr = resNonces[0];
			cryptonight_hash_variant(vhash, tempdata, 76, variant);
			if(vhash[7] <= Htarg && fulltest(vhash, ptarget))
			{
				res = 1;
				work->nonces[0] = resNonces[0];
				// work_set_target_ratio(work, vhash);
				// second nonce
				if(resNonces[1] != UINT32_MAX)
				{
					*tempnonceptr = resNonces[1];
					cryptonight_hash_variant(vhash, tempdata, 76, variant);
					if(vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						res++;
						work->nonces[1] = resNonces[1];
					} else {
						// gpu_increment_reject(thr_id);
					}
				}
				goto done;
			} else if (vhash[7] > Htarg) {
				//gpu_increment_reject(thr_id);
				printf("result for nonce %08x does not validate on CPU!\n", resNonces[0]);
			}
		}

		if ((uint64_t) throughput + nonce >= max_nonce - 127) {
			nonce = max_nonce;
			break;
		}

		nonce += throughput;
		printf("nonce %08x\n", nonce);

	} while (max_nonce > (uint64_t)throughput + nonce);

done:
	printf("nonce %08x exit\n", nonce);
	work->valid_nonces = res;
	*nonceptr = nonce;
	return res;
}

void free_cryptonight()
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
