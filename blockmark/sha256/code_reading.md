### 改造的关键

`thr_id` must taken out. There is no reason preserve it. However, 
There are many ways use it.

### 程序流程
  进入Device的代码只有`sha256_gpu_hash_shared`，即host端调用的第一个代码。该Kernel传入的参数为
`threads`、`startNonce`、`resNonces`。threads表示该次调用计算的上限，startNonce表示起始Nonce。
也就是说，调用一次kernel的结果为计算从`startNonce`开始的`threads`个nonce. `resNonces`保存的是运行的结果. resNonces数组的大小为2，`resNonces[0]`保存最新的结果，`resNonces[1]`维护次新的结果

  需要注意的是`resNonces`存储在work->nonces中


