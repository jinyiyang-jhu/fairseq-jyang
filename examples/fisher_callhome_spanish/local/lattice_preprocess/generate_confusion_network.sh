#!/bin/bash

# This script read lattices as input, and generate confusion networks as output.

cmd=run.pl

[ -f ./path.sh ] && . ./path.sh;
. cmd.sh
. parse_options.sh || exit 1;

# Use Kaldi tools to process lattice
if [ $# -ne 2 ]; then
    echo "Usage : $0 <input-lat-dir> <output-sausage-dir>"
    echo "E.g.: $0 data/kaldi_prepared/unselected/train_lats_pruned_acwt_0.1_lmwt_1.0.es data/kaldi_prepared/unselected/train_sausage_pruned_acwt_0.1_lmwt_1.0.es"
    exit 1
fi

lat_dir=$1
sausage_dir=$2

nj=$(ls $lat_dir/lat.*.gz | wc -l)

$cmd JOB=1:$nj $sausage_dir/log/decode_mbr.JOB.log \
    lattice-mbr-decode ark:"gunzip -c $lat_dir/lat.JOB.gz|" \
        ark:/dev/null ark:/dev/null ark,t:$sausage_dir/JOB.sau || { echo "MBR decoding failed" && exit 1; }

echo "Successfully generated sausages in $sausage_dir"
