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
    
  // Crypto
  uint32_t answer_data[20] = {0x1872b86e, 0x45a10014, 0x235ce531, 0x137dcf38, 0x772d75a1, 0x3de17ed0, 0x51ab5ba5, 0x4c0e866e, 0x7a2d2716, 0xf5b834c6, 0x500023c, 0x5513360, 0x3c7c4bb1, 0xf5bfb5d, 0x7a473e0a, 0x584e10a1, 0x62124aa5, 0x668291b3, 0x118e4718, 0x6f0a35e4};
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


  do {
    result = scanhash_cryptonight(&work, max_nonce, &hash_done, cn_variant);
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
    free_cryptonight();
  } while(!result);
}