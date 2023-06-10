TEST_BIN=$(pwd)/test_bin
COLLECT_DIR=$(pwd)/collect

files=$(ls $TEST_BIN)
echo $files

for bin_name in $files
do
  bin_dir=$(echo $bin_name | cut -d '.' -f1)
  bin_dir="hash_"$bin_dir
  cd $COLLECT_DIR/$bin_dir
  cat $(pwd)/$bin_dir".log" | grep -P "gpu_tot_ipc"
done