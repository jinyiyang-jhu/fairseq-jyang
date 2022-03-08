#!/bin/bash

# This script splits the scp files into N splits by demand.

numsplit=$1
data=$2
split_fname=$3

splits=$(for n in `seq $numsplit`; do echo $data/split${numsplit}/$n/${split_fname}; done)

directories=$(for n in `seq $numsplit`; do echo $data/split${numsplit}/$n; done)

# if this mkdir fails due to argument-list being too long, iterate.
if ! mkdir -p $directories >&/dev/null; then
  for n in `seq $numsplit`; do
    mkdir -p $data/split${numsplit}/$n
  done
fi

perl local/split_scp.pl $data/${split_fname} $splits || exit 1