#!/bin/bash

# This script perform BPE tokenization for train/dev/eval data.
# This BPE model and data are provided from ESPNET recipe.
stage=-1
source_lang="es"
target_lang="en"
case="lc.rm"
preprocess_num_workers=40

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
    echo "Usage: $0 <bpe-model-dir> <input-data-dir> <output-data-dir> <exp-dir>"
    echo "E.g.: $0 exp/bpe_es_en_lc/bpe_1000.lc.model data/espnet_prepared data/gold_mt exp/gold_mt"
    exit 1
fi

bpe_model_dir=$1
idata_dir=$2 # contains "train train_dev fisher_dev fisher_dev2 fisher_test callhome_devtest callhome_evltest"
odata_dir=$3
exp_dir=$4

bpedir=$odata_dir/bpe

if [ $stage -le 0 ]; then
    bpemodel=$bpe_model_dir/bpe_1000_${case}.model
    [ ! -f $bpemodel ] && echo "$bpemodel does not exists in $bpe_model_dir !" && exit 1;
    [ -d $bpedir ] || mkdir $bpedir || exit 1
    for d in train train_dev fisher_dev fisher_dev2 fisher_test callhome_devtest callhome_evltest; do
        if [ -d $idata_dir/$d.$lan ]; then
            echo "$(date) Processing BPE tokenization for dataset $d"
            for lan in $source_lang $target_lang; do
                input_file=$idata_dir/$d.$lan/text.${case}
                cut -f 2- -d " " $input_file |\
                    spm_encode --model=$bpemodel --output_format=piece |\
                    cut -f 2- -d " " > $bpedir/$d.$lan || exit 1;
            done
        else
            echo "$(date -u) Warning: escaping $d since $idata_dir/$d.$lan does not exist !"
        fi
    done
fi

if [ $stage -le 1 ]; then
    echo "$(date) Fairseq preprocess for dataset"
    preprocess_dir=$exp_dir/bpe_bin
    [ -d $preprocess_dir ] || mkdir -p $preprocess_dir || exit 1
    fairseq-preprocess --source-lang $source_lang --target-lang $target_lang \
        --trainpref $bpedir/train --validpref $bpedir/train_dev \
        --testpref $bpedir/fisher_dev,$bpedir/fisher_dev2,$bpedir/fisher_test,$bpedir/callhome_devtest,$bpedir/callhome_evltest \
        --destdir $preprocess_dir --joined-dictionary --append-eos-src --append-eos-tgt \
        --workers $preprocess_num_workers
fi
