#!/bin/bash

# This script perform BPE tokenization for train/dev/eval data.
# The datasets are provided from ESPNET recipe.

stage=-1
source_lang="es"
target_lang="en"
case=".lc.rm"
preprocess_num_workers=40

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
    echo "Usage: $0 <bpe-code-dir> <input-data-dir> <output-data-dir> <exp-dir>"
    echo "E.g.: $0 exp/bpe_es_en_lc_subword_nmt data/espnet_prepared data/gold_mt exp/gold_mt"
    exit 1
fi

bpe_code_dir=$1
idata_dir=$2 # contains "train train_dev fisher_dev fisher_dev2 fisher_test callhome_devtest callhome_evltest"
bpe_dir=$3
exp_dir=$4

if [ $stage -le 0 ]; then
    [ -d $bpe_dir ] || mkdir -p $bpe_dir || exit 1
    for d in train train_dev fisher_dev fisher_dev2 fisher_test callhome_devtest callhome_evltest; do
        for lan in $source_lang $target_lang; do
            if [ -d $idata_dir/$d.$lan ]; then
                echo "$(date) Processing BPE tokenization for dataset $d.$lan"
                input_file=$idata_dir/$d.$lan/text${case}
                cut -f 2- -d " " $input_file | sed -e 's/\&apos\;/ \&apos\; /g' |\
                    subword-nmt apply-bpe -c $bpe_code_dir/code.txt \
                        --vocabulary $bpe_code_dir/vocab.all.txt \
                        --glossaries "$(cat ${bpe_code_dir}/glossaries.txt)" \
                        --vocabulary-threshold 1 > $bpe_dir/$d.$lan || exit 1
            else
                echo "$(date) Warning: escaping $d since $idata_dir/$d.$lan does not exist !"
            fi
        done
    done
fi

if [ $stage -le 1 ]; then
    echo "$(date) Fairseq preprocess for dataset"
    preprocess_dir=$exp_dir/bpe_bin
    [ -d $preprocess_dir ] || mkdir -p $preprocess_dir || exit 1
    fairseq-preprocess --source-lang $source_lang --target-lang $target_lang \
        --trainpref $bpe_dir/train --validpref $bpe_dir/train_dev \
        --testpref $bpe_dir/fisher_dev,$bpe_dir/fisher_dev2,$bpe_dir/fisher_test,$bpe_dir/callhome_devtest,$bpe_dir/callhome_evltest \
        --destdir $preprocess_dir --joined-dictionary --append-eos-src --append-eos-tgt \
        --workers $preprocess_num_workers
fi
