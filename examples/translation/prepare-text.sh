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

if [ $stage -le 0 ]; then
    echo "$(date) Preprocessing ===> Step 1: BPE processing"
    # Use sentencepiece for BPE from raw input text
    mkdir -p $bpe_dir || exit 1;
    if [ ! -f $bpe_dir/bpe.$src.model ]; then
        echo "$(date) No existing bpe $src model, learning BPE from training data ..."
        spm_train.py --input=$data_dir/train/text.$src \
            --model_prefix=$bpe_dir/bpe.$src \
            --vocab_size=$bpe_src \
            --character_coverage=1.0 \
            --input_sentence_size=$spm_input_sentence_size \
            --shuffle_input_sentence=true \
            --model_type=bpe || exit 1;
        echo "$(date) Learning BPE done: $src"
    fi

    if [ ! -f $bpe_dir/bpe.$tgt.model ]; then
        echo "$(date) No existing bpe $tgt model, learning BPE from training data ..."
        spm_train.py --input=$data_dir/train/text.$tgt \
            --model_prefix=$bpe_dir/bpe.$tgt \
            --vocab_size=$bpe_tgt \
            --character_coverage=1.0 \
            --input_sentence_size=$spm_input_sentence_size \
            --shuffle_input_sentence=true \
            --model_type=bpe || exit 1;
        echo "$(date) Learning BPE done: $tgt"
    fi
fi

if [ $stage -le 1 ]; then
    for dset in "${sets[@]}"; do
        for t in $src $tgt; do
        echo "$(date) Applying BPE to $src: $dset"
            awk 'BEGIN{i=1}{print i" "$0;++i}' $data_dir/$dset/text.$t |\
                spm_encode.py --model=$bpe_dir/bpe.$t.model \
                --output_format=piece \
                > $bpe_dir/$dset.bpe.$t.tmp || exit 1;
        echo "$(date) Applying BPE to $src: $dset ===> Done !"
        done
        num_src=$(wc -l $bpe_dir/$dset.bpe.$src.tmp | cut -d " " -f1)
        num_tgt=$(wc -l $bpe_dir/$dset.bpe.$tgt.tmp | cut -d " " -f1)

        if [ $num_src -ne $num_tgt ]; then
            echo "Numbef of sentence diffs after BPE for $dset: $src($num_src) v.s. $tgt($num_tgt)"
            comm -12 <(cut -d " " -f1 test1|sort) <(cut -d " " -f1 test2|sort) > $bpe_dir/$dset.uttid.comm
            for t in $src $tgt; do
                awk 'NR==FNR{a[$1];next} $1 in a{print $0}' $bpe_dir/$dset.uttid.comm $bpe_dir/$dset.bpe.$t.tmp |\
                cut -d " " -f2- > $$bpe_dir/$dset.bpe.$t || exit 1
            done
        else
            for t in $src $tgt; do
                cut -d " " -f2- $bpe_dir/$dset.bpe.$t.tmp > $bpe_dir/$dset.bpe.$t || exit 1
            done
        fi
    done
fi



