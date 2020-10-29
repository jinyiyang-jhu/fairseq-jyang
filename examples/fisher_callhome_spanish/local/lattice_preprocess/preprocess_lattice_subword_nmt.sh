#!/bin/bash

# This script extracts probabilistic masks, position infos from lattice 
# and encode lattice; it also binarize the lattices into Fairseq idx,bin files.

stage=-1
source_lang="es"
target_lang="en"
case=".lc.rm"
preprocess_num_workers=40

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
    echo "Usage: $0 <bpe-dir> <exp-dir>"
    echo "E.g.: $0 data/gold_mt/bpe_subword_nmt exp/gold_mt_subword_nmt"
    exit 1
fi

bpe_dir=$1
exp_dir=$2

if [ $stage -le 1 ]; then
    echo "$(date) Fairseq preprocess for target text files."
    preprocess_dir=$exp_dir/bpe_bin
    [ -d $preprocess_dir ] || mkdir -p $preprocess_dir || exit 1
    fairseq-preprocess \
        --only-target \
        --source-lang $source_lang --target-lang $target_lang \
        --trainpref $bpe_dir/train --validpref $bpe_dir/train_dev \
        --testpref $bpe_dir/fisher_dev,$bpe_dir/fisher_dev2,$bpe_dir/fisher_test,$bpe_dir/callhome_devtest,$bpe_dir/callhome_evltest \
        --destdir $preprocess_dir --joined-dictionary --append-eos-src --append-eos-tgt \
        --workers $preprocess_num_workers
fi
