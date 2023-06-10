export PTX_SIM_MODE_FUNC=0
currentTime=`date "+%a-%b-%d-%H-%M-%S-%Y"`

TEST_BIN=$(pwd)/bin_cryptonight/test_cryptonight
COLLECT_DIR=$(pwd)/collect_test
mkdir -p $COLLECT_DIR

cd $COLLECT_DIR
export GPGPUSIM_ARCH_PATH=/root/custom_arch/SM75_RTX2060_shmbank16/
mkdir -p  SM75_RTX2060_bank16
cd SM75_RTX2060_bank16
screen -d -m -L -Logfile cryptonight-$currentTime $TEST_BIN

cd $COLLECT_DIR
export GPGPUSIM_ARCH_PATH=/root/custom_arch/SM75_RTX2060_shmbank32/
mkdir -p  SM75_RTX2060_bank32
cd SM75_RTX2060_bank32
screen -d -m -L -Logfile cryptonight-$currentTime $TEST_BIN

cd $COLLECT_DIR
export GPGPUSIM_ARCH_PATH=/root/custom_arch/SM75_RTX2060_shmbank64/
mkdir -p  SM75_RTX2060_bank64
cd SM75_RTX2060_bank64
screen -d -m -L -Logfile cryptonight-$currentTime $TEST_BIN

cd $COLLECT_DIR
export GPGPUSIM_ARCH_PATH=/root/custom_arch/SM75_RTX2060_shmbank128/
mkdir -p  SM75_RTX2060_bank128
cd SM75_RTX2060_bank128
screen -d -m -L -Logfile cryptonight-$currentTime $TEST_BIN