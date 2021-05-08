#!/bin/bash
stage=-1

. path.sh
. parse_options.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 <train-configuration>"
    exit 1;
fi

conf=$1
source $conf

lang="$src-$tgt"

if [ $stage -le 1 ]; then
    echo "$(date) Preprocessing ===> Step 1: BPE processing"
    # Use sentencepiece for BPE from raw input text
    mkdir -p $bpe_dir || exit 1;
    if [ ! -f $bpe_dir/bpe.$src.model ]; then
        echo "No existing bpe $src model, learning BPE from training data ..."
        spm_train.py --input=$data_dir/train/text.$src \
            --model_prefix=$bpe_dir/bpe.$src \
            --vocab_size=$bpe_src \
            --character_coverage=1.0 \
            --input_sentence_size=$spm_input_sentence_size \
            --shuffle_input_sentence=true \
            --model_type=bpe || exit 1;
        echo "Learning BPE done: $src"
    fi

    if [ ! -f $bpe_dir/bpe.$tgt.model ]; then
        echo "No existing bpe $tgt model, learning BPE from training data ..."
        spm_train.py --input=$data_dir/train/text.$tgt \
            --model_prefix=$bpe_dir/bpe.$tgt \
            --vocab_size=$bpe_tgt \
            --character_coverage=1.0 \
            --input_sentence_size=$spm_input_sentence_size \
            --shuffle_input_sentence=true \
            --model_type=bpe || exit 1;
        echo "Learning BPE done: $tgt"
    fi

    for dset in "${sets[@]}"; do
        for t in $src $tgt; do
            spm_encode.py --model=$bpe_dir/bpe.$t.model \
                --output_format=piece \
                < $data_dir/$dset/text.$t \
                > $bpe_dir/$dset.bpe.$t || exit 1;
        done
    done
fi



