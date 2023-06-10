/*
 * @Descripttion: 
 * @version: 1.0.0
 * @Author: CYKS
 * @Date: 2022-03-09 14:39:39
 * @LastEditors: CYKS
 * @LastEditTime: 2022-03-23 11:37:42
 */
#include <iostream>

#include <cstdlib>

#include "miner.h"
#include "cuda_helper.h"


int main() {
  init_device_gpgpusim();
  struct work work;

  opt_n_threads = 0;
  opt_quite = 0;

  unsigned long hash_done = 0;
    

  // a computed sha256 example
  // uint32_t answer_data[19] = {0x0879a41a, 0x505ab21a, 0x054b3490, 0x33c1f7c7, 0x2e5d181c, 0x6092030a, 0x2b7bd054, 0x641c442a, 0x43e7ce6b, 0x30a66196, 0x43d5c8f3, 0x60e5a25d, 0x20a1e4ba, 0x2f8b120c, 0x076a999f, 0x4a385e08, 0x5dc5934f, 0x12bd40fc, 0x52fbb314};
  // a computed jha exmaple
  // uint32_t answer_data[22] = {0x6b8b4567, 0x327b23c6, 0x643c9869, 0x66334873, 0x74b0dc51, 0x19495cff, 0x2ae8944a, 0x625558ec, 0x238e1f29, 0x46e87ccd, 0x3d1b58ba, 0x507ed7ab, 0x2eb141f2, 0x41b71efb, 0x79e2a9e3, 0x7545e146, 0x515f007c, 0x5bd062c2, 0x12200854, 0x0bff9312, 0x0216231b, 0x1f16e9e8};
  // uint32_t answer_data[22] = {0x6b8b4567, 0x327b23c6, 0x643c9869, 0x66334873, 0x74b0dc51, 0x19495cff, 0x2ae8944a, 0x625558ec, 0x238e1f29, 0x46e87ccd, 0x3d1b58ba, 0x507ed7ab, 0x2eb141f2, 0x41b71efb, 0x79e2a9e3, 0x7545e146, 0x515f007c, 0x5bd062c2, 0x12200854, 0x03600000, 0x0216231b, 0x1f16e9e8};
  // uint32_t answer_target[8] = {0x00000000, 0x00000000, 0x586af6c8, 0x00007f80, 0x00000006, 0x00000000, 0x13a1f300, 0x00000003};
  // uint32_t answer_data[20] = {0x382179ce, 0x171ca5b2, 0x3d878c11, 0x7faa1556, 0x6cf00906, 0x0ad92526, 0x773913f0, 0x1b24f60f, 0x01df222c, 0x23058f6f, 0x7d6b0a3c, 0x3bf4e153, 0x2f7ebe5d, 0x647f5cde, 0x4b68646d, 0x4ef3356d, 0x192ef16f, 0x31afa1f9, 0x16171a81, 0x019b3b78};
  // X11
  uint32_t answer_data[20] = {0x0b169d41, 0x4999ee67, 0x66c59228, 0x28764abe, 0x12d795a5, 0x4d14e541, 0x2e571ba1, 0x2e84f33a, 0x40351210, 0x5b4f2f11, 0x7275f310, 0x2b80233d, 0x691254d4, 0x4569a4f2, 0x2b5bceaf, 0x73e85d38, 0x1c2ba71d, 0x44c3680c, 0x535c206f, 0x0780477c};
  // Crypto
  // uint32_t answer_data[20] = {0x1872b86e, 0x45a10014, 0x235ce531, 0x137dcf38, 0x772d75a1, 0x3de17ed0, 0x51ab5ba5, 0x4c0e866e, 0x7a2d2716, 0xf5b834c6, 0x500023c, 0x5513360, 0x3c7c4bb1, 0xf5bfb5d, 0x7a473e0a, 0x584e10a1, 0x62124aa5, 0x668291b3, 0x118e4718, 0x6f0a35e4};
  uint32_t answer_target[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000003};

  std::cout << device_sm << " " << device_mpcount << std::endl;

#ifndef RANDOM_GEN
  for(int i = 0; i < 20; i++)  {
    work.data[i] = answer_data[i];
    printf("%x %x\n", work.data[i], answer_data[i]);
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
  // initial work struct
  // *(uint32_t*)(((uint8_t*)work.data) + 39) = 0x00023cf5;

  for(int i = 0; i < 20; i++) {
    printf("%x\n", work.data[i]);
  }
  uint32_t max_nonce = 1U << 28;
  work.nonces[0] = UINT32_MAX;
  work.valid_nonces = 0;

  int cn_variant = 0;
  if(cryptonight_fork > 1 && ((unsigned char*)work.data)[0] >= cryptonight_fork)
    cn_variant = ((unsigned char*)work.data)[0] - cryptonight_fork + 1;

  int result;
  uint64_t total_hash_done = 0;
  uint32_t total_num = 0;  

  uint32_t temp = work.data[19];
  work.data[19] = 0x0780477b;


  do {
    result = scanhash_cryptonight(&work, max_nonce, &hash_done);
    if(result == 0) {
      for(int i = 0; i < 20; i ++) {
        work.data[i] = rand();
      }
      work.data[19] = 0;
      total_hash_done += hash_done;
      total_num += 1;
      hash_done = 0;
    } else {
      total_hash_done += hash_done;
      std::cout << "computing yes!: " << result << " nonce: ";
      printf("0x%08x\n", work.nonces[0]);
      std::cout << "total hash done: " << total_hash_done << " " << "total retry num" << total_num << std::endl;
      for(int i = 0; i < 20; i++) {
        printf("0x%08x, ", work.data[i]);
      }
    }
    free_x11();
  } while(!result);
}