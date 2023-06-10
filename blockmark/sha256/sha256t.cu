/**
 * SHA256 3x
 * by tpruvot@github - 2017
 */

#include <miner.h>
#include <cuda_helper.h>
#include <openssl/sha.h>

// CPU Check
extern "C" void sha256t_hash(void *output, const void *input)
{
	unsigned char hash[64];
	SHA256_CTX sha256;

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, (unsigned char *)input, 80);
	SHA256_Final(hash, &sha256);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, hash, 32);
	SHA256_Final(hash, &sha256);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, hash, 32);
	SHA256_Final((unsigned char *)output, &sha256);
}

static bool init_flag = false;
extern void sha256t_init();
extern void sha256t_free();
extern void sha256t_setBlock_80(uint32_t *pdata, uint32_t *ptarget);
extern void sha256t_hash_80(uint32_t threads, uint32_t startNonce, uint32_t *resNonces);

extern "C" int scanhash_sha256t(struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = 1U << 10;
	if (init_flag) throughput = min(throughput, (max_nonce - first_nonce));

	((uint32_t*)ptarget)[7] = 0x03;

	printf("enter sha256t, init_flag: %d\n", init_flag);
	if (!init_flag)
	{
		sha256t_init();

		init_flag = true;
	}

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	sha256t_setBlock_80(endiandata, ptarget);

	do {
		// Hash with CUDA
		*hashes_done = pdata[19] - first_nonce + throughput;

		sha256t_hash_80(throughput, pdata[19], work->nonces);
		printf("sha256t: start nonce: %d,  done: %d\n", pdata[19], *hashes_done);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t vhash[8];

			endiandata[19] = swab32(work->nonces[0]);
			sha256t_hash(vhash, endiandata);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				// work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					endiandata[19] = swab32(work->nonces[1]);
					sha256t_hash(vhash, endiandata);
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
			/*
			else if (vhash[7] > ptarget[7]) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
			*/
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
extern "C" void free_sha256t()
{
	if (!init_flag)
		return;

	cudaThreadSynchronize();

	sha256t_free();

	init_flag = false;

	cudaDeviceSynchronize();
}
