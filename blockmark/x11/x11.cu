extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x11.h"

#include <stdio.h>
#include <memory.h>

static uint32_t *d_hash;

// X11 CPU Hash
extern "C" void x11hash(void *output, const void *input)
{
	unsigned char hash[128] = { 0 };

	// blake1-bmw2-grs3-skein4-jh5-keccak6-luffa7-cubehash8-shavite9-simd10-echo11

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;

	sph_blake512_init(&ctx_blake);
	sph_blake512 (&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, (void*) hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*) hash);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*) hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512 (&ctx_skein, (const void*) hash, 64);
	sph_skein512_close(&ctx_skein, (void*) hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512 (&ctx_jh, (const void*) hash, 64);
	sph_jh512_close(&ctx_jh, (void*) hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) hash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, (const void*) hash, 64);
	sph_luffa512_close (&ctx_luffa, (void*) hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512 (&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512 (&ctx_shavite, (const void*) hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512 (&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512 (&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(output, hash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "x11"
#include "cuda_debug.cuh"

static bool init_flag = false;

extern "C" int scanhash_x11(struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	uint32_t* h_debug_memory;
	uint32_t* h_hash;
	uint32_t file_num;
	FILE* fp;
	char buffer[100];

	int intensity = 12;
	//int intensity = (device_sm[device_map[thr_id]] >= 500 && !is_windows()) ? 20 : 19;
	uint32_t throughput = 1U << intensity;

	// uint32_t throughput = 1U << intensity; // 19=256*256*8;

	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	ptarget[7] = 0x5;

	if (!init_flag)
	{

		quark_blake512_cpu_init(throughput);
		quark_bmw512_cpu_init(throughput);
	 	quark_groestl512_cpu_init(throughput);
		quark_skein512_cpu_init(throughput);
		quark_keccak512_cpu_init(throughput);
		quark_jh512_cpu_init(throughput);
		x11_luffaCubehash512_cpu_init(throughput);
		x11_shavite512_cpu_init(throughput);
		x11_echo512_cpu_init(throughput);
		if (x11_simd512_cpu_init(throughput) != 0) {
			return 0;
		}
		CUDA_SAFE_CALL(cudaMalloc(&d_hash, (size_t) 64 * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&global_memory, (size_t) 64 * throughput));

		h_debug_memory = (uint32_t*) malloc((size_t) 64 * throughput);
		h_hash = (uint32_t*) malloc((size_t) 64 * throughput);

		cuda_check_cpu_init(throughput);

		init_flag = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++){
		be32enc(&endiandata[k], pdata[k]);
		printf("%u\n", pdata[k]);
	}

	quark_blake512_cpu_setBlock_80(endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// Hash with CUDA
		quark_blake512_cpu_hash_80(throughput, pdata[19], d_hash); order++;
		TRACE("blake  :");
		quark_bmw512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		TRACE("bmw    :");
		quark_groestl512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		TRACE("groestl:");
		quark_skein512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		TRACE("skein  :");
		quark_jh512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		TRACE("jh512  :");
		quark_keccak512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		TRACE("keccak :");
		x11_luffaCubehash512_cpu_hash_64(throughput, d_hash, order++);
		TRACE("luffa+c:");
		x11_shavite512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		TRACE("shavite:");
		x11_simd512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		TRACE("simd   :");
		x11_echo512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		TRACE("echo => ");

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(throughput, pdata[19], d_hash);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			x11hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				// work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(throughput, pdata[19], d_hash, 1);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					x11hash(vhash, endiandata);
					// bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			/* else {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			} */
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (true);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_x11()
{
	if (!init_flag)
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash);

	quark_blake512_cpu_free();
	quark_groestl512_cpu_free();
	x11_simd512_cpu_free();

	cuda_check_cpu_free();
	init_flag = false;

	cudaDeviceSynchronize();
}
