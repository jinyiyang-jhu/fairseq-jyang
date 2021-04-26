#!/bin/bash
stage=-1
src="zh"
tgt="en"
lang="$src-$tgt"
nprocessor=4
bpe_src=30000
bpe_tgt=30000

origin_dir="data"
processed_dir="data/preprocessed"
sets=("train" "dev" "test" "sharedeval-bn" "sharedeval-bc")
bpe_dir=$processed_dir/bpe

. path.sh
. parse_options.sh

if [ $stage -le 1 ]; then
    echo "$(date) Preprocessing ===> Step 1: BPE processing"
    # Use sentencepiece for BPE from raw input text
    mkdir -p $bpe_dir || exit 1;
    if [ ! -f $bpe_dir/bpe.$src.model ]; then
        echo "No existing bpe $src model, learning BPE from training data ..."
        spm_train.py --input=$origin_dir/train/text.$src \
            --model_prefix=$bpe_dir/bpe.$src \
            --vocab_size=$bpe_src \
            --character_coverage=1.0 \
            --model_type=bpe || exit 1;
    fi

    if [ ! -f $bpe_dir/bpe.$tgt.model ]; then
        echo "No existing bpe $tgt model, learning BPE from training data ..."
        spm_train.py --input=$origin_dir/train/text.$tgt \
            --model_prefix=$bpe_dir/bpe.$tgt \
            --vocab_size=$bpe_tgt \
            --character_coverage=1.0 \
            --model_type=bpe || exit 1;
    fi

    for dset in "${sets[@]}"; do
        for t in $src $tgt; do
            spm_encode.py --model=$bpe_dir/bpe.$t.model \
                --output_format=piece \
                < $origin_dir/$dset/text.$t \
                > $bpe_dir/$dset.bpe.$t || exit 1;
        done
    done
fi



