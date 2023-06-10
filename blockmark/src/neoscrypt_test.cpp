#include <cstring>
#include "miner.h"
#include "cuda_runtime.h"

extern void neoscrypt_setBlockTarget(uint32_t* const data, uint32_t* const ptarget);
extern void neoscrypt_init(uint32_t threads);
extern void neoscrypt_free();
extern void neoscrypt_hash_k4(uint32_t threads, uint32_t startNounce, uint32_t *resNonces, bool stratum);

int main() {
  struct work work;
  uint32_t answer_data[20] = {0x5b7cac2f, 0x33719156, 0x69a318fe, 0x78f68477, 0x70c770aa, 0x3818b042, 0x64cd22ff, 0x76030494, 0x49ca86a1, 0x33e5d576, 0x47abdc, 0x6f77f55f, 0x2f9b619e, 0x3c128b72, 0x48cfa162, 0x10b3668c, 0x40c3c013, 0x4013a137, 0x659c4827, 0};
  uint32_t answer_target[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000003};

#ifndef RANDOM_GEN
  for(int i = 0; i < 20; i++)  {
    work.data[i] = answer_data[i];
  }
#endif // !RANDOM_GEN
  
  for(int i = 0; i < 8; i++) {
    work.target[i] = answer_target[i];
  }


#ifdef RANDOM_GEN
  srand(time(nullptr));
  for(int i = 0; i < 20; i ++) {
    work.data[i] = rand();
  }
#endif // RANDOM_GEN
  work.nonces[0] = UINT32_MAX;
  work.valid_nonces = 0;
  work.data[19] = 0x3feec9;

	uint32_t __align__(64) endiandata[20];
	uint32_t *pdata = work.data;
	uint32_t *ptarget = work.target;
	const uint32_t first_nonce = pdata[19];

	uint32_t throughput = 1U << INTENSITY;
	throughput = throughput / 32; /* set for max intensity ~= 20 */
  ptarget[7] = 0x00ff;
  neoscrypt_init(throughput);

  bool have_stratum = false;
  for (int k = 0; k < 20; k++)
    endiandata[k] = pdata[k];

	neoscrypt_setBlockTarget(endiandata,ptarget);
  memset(work.nonces, 0xff, sizeof(work.nonces));
  neoscrypt_hash_k4(throughput, pdata[19], work.nonces, have_stratum);
  for(int i = 0; i < 20; i++) printf("%x ", work.data[i]);
  printf("\n");
  printf("%x\n", work.nonces[0]);
  neoscrypt_free();
}