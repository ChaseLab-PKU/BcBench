#include "miner.h"
#include "cuda_x11.h"



// enum ALGORITHM_TYPE {
//   BLAKE,
//   BMW,
//   GRS,
//   SKEIN,
//   JH,
//   KECCAK,
//   LUFFA,
//   CUBE,
//   SHAVITE,
//   SIMD,
//   ECHO,
// };


int main() {
  struct work work;
  uint32_t answer_data[20] = {0x0b169d41, 0x4999ee67, 0x66c59228, 0x28764abe, 0x12d795a5, 0x4d14e541, 0x2e571ba1, 0x2e84f33a, 0x40351210, 0x5b4f2f11, 0x7275f310, 0x2b80233d, 0x691254d4, 0x4569a4f2, 0x2b5bceaf, 0x73e85d38, 0x1c2ba71d, 0x44c3680c, 0x535c206f, 0x0780477c};
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

#ifdef BLAKE
    test_scanhash(&work, quark_blake512_cpu_hash_80, quark_blake512_cpu_init);
#endif
#ifdef BMW
    test_scanhash(&work, quark_bmw512_cpu_hash_64, quark_bmw512_cpu_init);
#endif
#ifdef GRS
    test_scanhash(&work, quark_groestl512_cpu_hash_64, quark_groestl512_cpu_init);
#endif
#ifdef SKEIN
    test_scanhash(&work, quark_skein512_cpu_hash_64, quark_skein512_cpu_init);
#endif
#ifdef JH
    test_scanhash(&work, quark_jh512_cpu_hash_64, quark_jh512_cpu_init);
#endif
#ifdef KECCAK
    test_scanhash(&work, quark_keccak512_cpu_hash_64, quark_keccak512_cpu_init);
#endif
#ifdef LUFFA
    test_scanhash(&work, x11_luffa512_cpu_hash_64, x11_luffa512_cpu_init);
#endif
#ifdef CUBE
    test_scanhash(&work, x11_cubehash512_cpu_hash_64, x11_cubehash512_cpu_init);
#endif
#ifdef SHAVITE
    test_scanhash(&work, x11_shavite512_cpu_hash_64, x11_shavite512_cpu_init);
#endif
#ifdef SIMD
    test_scanhash(&work, x11_simd512_cpu_hash_64, x11_simd512_cpu_init);
#endif
#ifdef ECHO
    test_scanhash(&work, x11_echo512_cpu_hash_64, x11_echo512_cpu_init);
#endif
}

/*
		quark_blake512_cpu_hash_80(throughput, pdata[19], d_hash); order++;
		quark_bmw512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		quark_groestl512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		quark_skein512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		quark_jh512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		quark_keccak512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		x11_luffaCubehash512_cpu_hash_64(throughput, d_hash, order++);
		x11_shavite512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		x11_simd512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
		x11_echo512_cpu_hash_64(throughput, pdata[19], NULL, d_hash, order++);
*/