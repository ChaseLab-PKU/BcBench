export PTX_SIM_MODE_FUNC=0

TEST_BIN=$(pwd)/bin_neoscrypt/test_neocrypt
COLLECT_DIR=$(pwd)/collect

export GPGPUSIM_ARCH_PATH=/root/accel-sim-framework-1.2.0/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3070

cd $COLLECT_DIR
# export GPGPUSIM_ARCH_PATH=/root/custom/SM75_RTX2060_shf2/
mkdir -p  rtx3070_neoscrypt
cd rtx3070_neoscrypt
rm -r ./*
screen -d -m -L -Logfile neoscrypt.log $TEST_BIN