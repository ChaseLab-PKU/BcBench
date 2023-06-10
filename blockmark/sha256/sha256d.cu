/**
 * SHA256d
 * by tpruvot@github - 2017
 */

#include <openssl/sha.h>

#include "miner.h"
#include "cuda_helper.h"
// CPU Check
extern "C" void sha256d_hash(void *output, const void *input)
{
	unsigned char hash[64];
	SHA256_CTX sha256;

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, (unsigned char *)input, 80);
	SHA256_Final(hash, &sha256);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, hash, 32);
	SHA256_Final((unsigned char *)output, &sha256);
}

static bool init_flag = false;
extern void sha256d_init();
extern void sha256d_free();
extern void sha256d_setBlock_80(uint32_t *pdata, uint32_t *ptarget);
extern void sha256d_hash_80(uint32_t threads, uint32_t startNonce, uint32_t *resNonces);

extern "C" int scanhash_sha256d(struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t /*_ALIGN(64)*/ endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = 1U << 10;

	((uint32_t*)ptarget)[7] = 0x03;

	if (!init_flag)
	{
		sha256d_init();
		init_flag = true;
	}

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	sha256d_setBlock_80(endiandata, ptarget);

	do {
		// Hash with CUDA
		*hashes_done = pdata[19] - first_nonce + throughput;

		sha256d_hash_80(throughput, pdata[19], work->nonces);

		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t /*_ALIGN(64)*/ vhash[8];

			endiandata[19] = swab32(work->nonces[0]);
			sha256d_hash(vhash, endiandata);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				// work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					endiandata[19] = swab32(work->nonces[1]);
					sha256d_hash(vhash, endiandata);
					if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
						work->valid_nonces++;
						// bn_set_target_ratio(work, vhash, 1);
					}
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1;
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > ptarget[7]) {
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
	} while (*hashes_done + throughput < max_nonce);
	// work_restart? while (!work_restart[thr_id].restart);
	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

// cleanup
extern "C" void free_sha256d()
{
	if (!init_flag)
		return;

	cudaThreadSynchronize();

	sha256d_free();

	init_flag = false;

	cudaDeviceSynchronize();
}
