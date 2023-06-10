/**
 * JHA v8 algorithm - compatible implementation
 * @author tpruvot@github 05-2017
 */

extern "C" {
#include "sph/sph_keccak.h"
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_jh.h"
#include "sph/sph_skein.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "quark/cuda_quark.h"
#include "cuda_debug.cuh"

static uint32_t *d_hash = nullptr;
static uint32_t *d_hash_br2;
static uint32_t *d_tempBranch;

extern void jackpot_keccak512_cpu_init(uint32_t threads);
extern void jackpot_keccak512_cpu_setBlock(void *pdata, size_t inlen);
extern void jackpot_keccak512_cpu_hash(uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);

// CPU HASH
extern "C" void jha_hash(void *output, const void *input)
{
	uint32_t hash[16];

	sph_blake512_context     ctx_blake;
	sph_groestl512_context   ctx_groestl;
	sph_jh512_context        ctx_jh;
	sph_keccak512_context    ctx_keccak;
	sph_skein512_context     ctx_skein;

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, input, 80);
	sph_keccak512_close(&ctx_keccak, hash);

	for (int rnd = 0; rnd < 3; rnd++)
	{
		if (hash[0] & 0x01) {
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512 (&ctx_groestl, (&hash), 64);
			sph_groestl512_close(&ctx_groestl, (&hash));
		}
		else {
			sph_skein512_init(&ctx_skein);
			sph_skein512 (&ctx_skein, (&hash), 64);
			sph_skein512_close(&ctx_skein, (&hash));
		}

		if (hash[0] & 0x01) {
			sph_blake512_init(&ctx_blake);
			sph_blake512 (&ctx_blake, (&hash), 64);
			sph_blake512_close(&ctx_blake, (&hash));
		}
		else {
			sph_jh512_init(&ctx_jh);
			sph_jh512 (&ctx_jh, (&hash), 64);
			sph_jh512_close(&ctx_jh, (&hash));
		}
	}
	memcpy(output, hash, 32);
}

__global__ __launch_bounds__(128, 8)
void jha_filter_gpu(const uint32_t threads, const uint32_t* d_hash, uint32_t* d_branch2, uint32_t* d_NonceBranch)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t offset = thread * 16U; // 64U / sizeof(uint32_t);
		uint4 *psrc = (uint4*) (&d_hash[offset]);
		d_NonceBranch[thread] = ((uint8_t*)psrc)[0] & 0x01;
		if (d_NonceBranch[thread]) return;
		// uint4 = 4x uint32_t = 16 bytes
		uint4 *pdst = (uint4*) (&d_branch2[offset]);
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}

__global__ __launch_bounds__(128, 8)
void jha_merge_gpu(const uint32_t threads, uint32_t* d_hash, uint32_t* d_branch2, uint32_t* const d_NonceBranch)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads && !d_NonceBranch[thread])
	{
		const uint32_t offset = thread * 16U;
		uint4 *pdst = (uint4*) (&d_hash[offset]);
		uint4 *psrc = (uint4*) (&d_branch2[offset]);
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}

__host__
uint32_t jha_filter_cpu(const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_branch2)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	// extract algo permution hashes to a second branch buffer
	jha_filter_gpu <<<grid, block>>> (threads, inpHashes, d_branch2, d_tempBranch);
	return threads;
}

__host__
void jha_merge_cpu(const uint32_t threads, uint32_t *outpHashes, uint32_t* d_branch2)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	// put back second branch hashes to the common buffer d_hash
	jha_merge_gpu <<<grid, block>>> (threads, outpHashes, d_branch2, d_tempBranch);
}

static bool init_flag = false;

extern "C" int scanhash_jha(struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t __align__(64) endiandata[22];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	int dev_id = 0;

	uint32_t throughput =  1U << 10;
	if (init_flag) throughput = min(throughput, max_nonce - first_nonce);

	// if (opt_benchmark)
	ptarget[7] = 0x000f;

	if (!init_flag)
	{
		cudaSetDevice(dev_id);
		/*
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		cuda_get_arch(thr_id);
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		*/
		CUDA_SAFE_CALL(cudaMalloc(&d_hash, (size_t) 64 * throughput));
		cudaMemset(d_hash, 0, (size_t) 64 * throughput);
		CUDA_SAFE_CALL(cudaMalloc(&d_hash_br2, (size_t) 64 * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_tempBranch, sizeof(uint32_t) * throughput));

		jackpot_keccak512_cpu_init(throughput);
		quark_blake512_cpu_init(throughput);
		quark_groestl512_cpu_init(throughput);
		quark_jh512_cpu_init(throughput);
		quark_skein512_cpu_init(throughput);

		cuda_check_cpu_init(throughput);

		init_flag = true;
	}

	for (int k=0; k < 22; k++)
		be32enc(&endiandata[k], pdata[k]);

	jackpot_keccak512_cpu_setBlock((void*)endiandata, 80);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		jackpot_keccak512_cpu_hash(throughput, pdata[19], d_hash, order++);

		/*
		cudaMemcpy(h_hash, d_hash, (size_t) 64 * throughput, cudaMemcpyDeviceToHost);
		sprintf(buffer, "d_hash%d", file_num++);
		fp = fopen(buffer, "w+");
		for(int i = 0; i < (64 * throughput)/(sizeof(uint32_t)); i++) {
			fprintf(fp, "%u, ", h_hash[i]);
		}
		*/

		for (int rnd = 0; rnd < 3; rnd++)
		{
			jha_filter_cpu(throughput, d_hash, d_hash_br2);
			quark_groestl512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
			quark_skein512_cpu_hash_64(throughput, pdata[19], NULL, d_hash_br2, order++);
			jha_merge_cpu(throughput, d_hash, d_hash_br2);

			jha_filter_cpu(throughput, d_hash, d_hash_br2);
			quark_blake512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
			quark_jh512_cpu_hash_64(throughput, pdata[19], NULL, d_hash_br2, order++);
			jha_merge_cpu(throughput, d_hash, d_hash_br2);
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		// CUDA_LOG_ERROR()		
		work->nonces[0] = cuda_check_hash(throughput, pdata[19], d_hash);

		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t __align__(64) vhash[8];

			be32enc(&endiandata[19], work->nonces[0]);
			jha_hash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				// work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(throughput, pdata[19], d_hash, 1);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					jha_hash(vhash, endiandata);
					// bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				/*
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				*/
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (true);

	*hashes_done = pdata[19] - first_nonce;

	// CUDA_LOG_ERROR();

	return 0;
}

// cleanup
extern "C" void free_jha()
{
	if (!init_flag)
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash);
	cudaFree(d_hash_br2);
	cudaFree(d_tempBranch);

	quark_blake512_cpu_free();
	quark_groestl512_cpu_free();

	cuda_check_cpu_free();
	// CUDA_LOG_ERROR();

	cudaDeviceSynchronize();
	init_flag = false;
}
