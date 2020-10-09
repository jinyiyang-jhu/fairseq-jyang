#!/bin/bash

# This script perform BPE tokenization for train/dev/eval data.
# This BPE model and data are provided from ESPNET recipe.
stage=-1
espnet_data=data/espnet_prepared
bpemodel=exp/espnet_bpe_model/train_sp.en_bpe1000_lc.rm.model
datadir=data
source_lang="es"
target_lang="en"
case=".lc.rm"
preprocess_num_workers=40

exp_dir=exp/gold_mt
bpedir=$datadir/bpe

if [ $stage -le 0 ]; then
    [ -d $bpedir ] || mkdir $bpedir || exit 1
    for d in train train_dev fisher_dev fisher_dev2 fisher_test callhome_devtest callhome_evltest; do
        echo "$(date -u) Processing BPE tokenization for dataset $d"
        for lan in $source_lang $target_lang; do
            input_file=$espnet_data/$d.$lan/text${case}
            bash local/preprocess_bpe.sh $input_file $bpemodel $bpedir/$d.$lan || exit 1
        done
    done
fi

if [ $stage -le 1 ]; then
    echo "$(date -u) Fairseq preprocess for dataset"
    preprocess_dir=$exp_dir/bpe_bin
    [ -d $preprocess_dir ] || mkdir -p $preprocess_dir || exit 1
    fairseq-preprocess --source-lang $source_lang --target-lang $target_lang \
        --trainpref $bpedir/train --validpref $bpedir/train_dev \
        --testpref $bpedir/fisher_dev,$bpedir/fisher_dev2,$bpedir/fisher_test,$bpedir/callhome_devtest,$bpedir/callhome_evltest \
        --destdir $preprocess_dir --joined-dictionary --append-eos-src --append-eos-tgt \
        --workers $preprocess_num_workers
fi
