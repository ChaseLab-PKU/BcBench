# BcBench

BcBench is for evaluating blockchain workloads on throughput processors(like GPGPU) at micro-architecture level. BcBench contains a collection of blockchain-based cryptocurrency apps with different characteristics. BcBench mainly focuses on CUDA platform and can be run on top of real hardware as well as simulators (like GPGPU-Sim).

BcBench makes several modifcations to support blockchain workloads with GPGPU-Sim. The modified GPGPU-Sim is in `accel-sim-framework-1.2.0.tar.gz`. You can build GPGPU-Sim following the GitHub page of [accel-sim-framework](https://github.com/accel-sim/accel-sim-framework).

The BcBench workload repo contains two folders of workloads, namely `blockmark` and `equihash`. `blockmark` conatins workloads NeoScrypt, CryptoNight, X11, and `equihash` contains workload `equihash`.

## Build
We recommend to put `blockmark` and `equihash` in the /root folder, or in a docker container environment.

To build the workloads of BcBench, please execute the following commands with corresponding workload:

```
./blockmark/make test_cryptonight
./blockmark/make test_neocrypt
./blockmark/make test_all_hash
./equihash/make eqcudah
```



## Run
To run the workloads of Bcbench inside GPGPU-Sim, please execute the following commands:

```
./blockmark/test_neoscrypt.sh
./blockmark/test_crypto.sh
./blockmark/test_hash.sh
./equihash/eqcudah
```

You may need to set the correct path of `GPGPUSIM_ARCH_PATH` as well as modify the corresponding parts of the test script.

### Data Prepration
For fast evaluation on simulators, we pre-computed some data of CryptoNight. The binary data file can be downloaded from [Google Drive](https://drive.google.com/file/d/1EHyb3LLh32BTqXe9szSYYRCeLsMq7TEe/view?usp=sharing) and then put in the path `BcBench/blockmark/data.bin`.

## Publication
The BibTex is shown below:
```
@inproceedings{10.1145/3555776.3577701,
author = {Pan, Xiurui and Chen, Yue and Yi, Shushu and Zhang, Jie},
title = {BcBench: Exploring Throughput Processor Designs Based on Blockchain Benchmarking},
year = {2023},
doi = {10.1145/3555776.3577701},
booktitle = {Proceedings of the 38th ACM/SIGAPP Symposium on Applied Computing},
pages = {88–97},
numpages = {10},
location = {Tallinn, Estonia},
series = {SAC '23}
}
```

## Contact
Feel free to contact panxiurui at outlook dot com if you have any questions.