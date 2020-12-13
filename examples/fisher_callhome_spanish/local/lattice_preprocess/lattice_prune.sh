#!/bin/bash

stage=-1
cmd=run.pl
acwt=0.1
beam=4
depth_thres=90

[ -f ./path.sh ] && . ./path.sh;
. cmd.sh
. parse_options.sh || exit 1;

# Use Kaldi tools to process lattice
if [ $# -ne 2 ]; then
    echo "Usage : $0 <old-lat-dir> <new-lat-dir>"
    echo "E.g.: $0 data/kaldi/train_lattice data/kaldi/train_lattice_pruned"
    exit 1
fi

old_lat_dir=$1
new_lat_dir=$2

nj=$(ls $old_lat_dir/lat.*.gz | wc -l)
if [ $stage -le 0 ]; then
    echo "Pruning lattice"
    $cmd JOB=1:$nj $new_lat_dir/log/lat_prune.JOB.log \
        lattice-prune --acoustic-scale=$acwt --beam=$beam ark:"gunzip -c $old_lat_dir/lat.JOB.gz|" \
        "ark:|gzip -c > $new_lat_dir/lat.JOB.gz" || exit 1;
fi

if [ $stage -le 1 ]; then
    mkdir -p $new_lat_dir/analyze || exit 1;
    $cmd JOB=1:$nj $new_lat_dir/log/lat_depth.JOB.log \
        lattice-depth ark:"gunzip -c $new_lat_dir/lat.JOB.gz|" ark,t:$new_lat_dir/analyze/lat.JOB.depth || exit 1;
fi

if [ $stage -le 2 ]; then
    filtered_lat=$new_lat_dir/analyze/lattice_below_thres.depth
    skipped_lat=$new_lat_dir/analyze/lattice_above_thres.depth
    [ ! -f $filtered_lat ] || rm $filtered_lat || exit 1;
    [ ! -f $skipped_lat ] || rm $skipped_lat || exit 1;

    for n in $(seq 1 $nj); do
        awk -v thres=$depth_thres '$2<=thres' $new_lat_dir/analyze/lat.$n.depth >> $filtered_lat
        awk -v thres=$depth_thres '$2>thres' $new_lat_dir/analyze/lat.$n.depth >> $skipped_lat
    done
fi


