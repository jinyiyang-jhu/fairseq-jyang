#!/bin/bash
stage=-1
src="zh"
tgt="en"
lang="$src-$tgt"
nprocessor=4
bpe_src=30000
bpe_tgt=30000

origin_dir="data"
processed_dir="data/preprocessed-subwordnmt"
sets=("train" "dev" "test" "sharedeval-bn" "sharedeval-bc")
tok_dir=$processed_dir/tokenized
bpe_dir=$processed_dir/bpe

. path.sh
. parse_options.sh

if [ $stage -le 0 ]; then
    echo "$(date) Preprocessing ===> Step 0: Clean (and segmentation)"
    mkdir -p $tok_dir || exit 1;
    for dset in "${sets[@]}"; do
        mkdir -p $tok_dir/$dset || exit 1;
        for t in $src $tgt; do
            cat $origin_dir/$dset/text.$t | python local/clean_text.py \
                --lan $t --output $tok_dir/$dset/text.clean.$t || exit 1;
        done
    done
fi

if [ $stage -le 1 ]; then
    echo "$(date) Preprocessing ===> Step 1: Tokenization"
    # Use sacremoses for tokenization
    mkdir -p $tok_dir || exit 1;
    for dset in "${sets[@]}"; do
        mkdir -p $tok_dir/$dset || exit 1;
        for t in $src $tgt; do
            sacremoses -l $src -j $nprocessor tokenize \
                < $tok_dir/$dset/text.clean.$t \
                > $tok_dir/$dset/text.tok.$t || exit 1;
        done
    done
fi

if [ $stage -le 2 ]; then
    echo "$(date) Preprocessing ===> Step 2: BPE processing"
    # Use subword-nmt for BPE
    mkdir -p $bpe_dir || exit 1;
    [ ! -f $bpe_dir/bpe.code.$src ] && echo "No existing bpe $src files, learning BPE from $tok_dir/train/text.tok.$src"
    subword-nmt learn-joint-bpe-and-vocab \
        -s $bpe_src \
        --input $tok_dir/train/text.tok.$src \
        -o $bpe_dir/bpe.code.$src \
        --write-vocabulary $bpe_dir/bpe.vocab.$src || exit 1;
    echo "Learning bpe for $src is done"
    
    [ ! -f $bpe_dir/bpe.code.$tgt ] && echo "No existing bpe $tgt files, learning BPE from $tok_dir/train/text.tok.$tgt"
    subword-nmt learn-joint-bpe-and-vocab \
        -s $bpe_tgt \
        --input $tok_dir/train/text.tok.$tgt \
        -o $bpe_dir/bpe.code.$tgt \
        --write-vocabulary $bpe_dir/bpe.vocab.$tgt || exit 1;
    echo "Learning bpe for $tgt is done"

    for dset in "${sets[@]}"; do
        for t in $src $tgt; do
            echo "Applying BPE for $dset: $t"
            subword-nmt apply-bpe -c $bpe_dir/bpe.code.$t \
                --vocabulary $bpe_dir/bpe.vocab.$t
                < $tok_dir/$dset/text.tok.$t \
                > $bpe_dir/$dset.bpe.$t || exit 1;
        done
    done
fi

bash train.sh conf_zh_en_subwordnmt.sh


