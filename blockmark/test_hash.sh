export PTX_SIM_MODE_FUNC=0
export GPGPUSIM_ARCH_PATH=/root/custom_arch/SM75_RTX2060_base/

TEST_BIN=$(pwd)/test_bin
COLLECT_DIR=$(pwd)/collect_X11
mkdir -p $COLLECT_DIR

files=$(ls $TEST_BIN)
echo $files

for bin_name in $files
do
  bin_dir=$(echo $bin_name | cut -d '.' -f1)
  bin_dir="hash_"$bin_dir
  echo "start run $bin_name"
  cd $COLLECT_DIR
  rm -rf $COLLECT_DIR/$bin_dir
  mkdir -p  $bin_dir
  cd $bin_dir
  screen -d -m -L -Logfile $bin_dir".log" $TEST_BIN/$bin_name
done