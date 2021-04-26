#!/bin/bash


conf=$1
infile=$2
decode_mdl="checkpoint_best"
generate_bsz=32
bpe_type="sentencepiece"
decode_num_workers=8

. $conf
bin_dir=$exp_dir/bpe_bin

cat $infile | \
    fairseq-interactive $bin_dir \
    --no-progress-bar \
    --source-lang $src \
    --target-lang $tgt \
    --task $task \
    --path $exp_dir/checkpoints/${decode_mdl}.pt \
    --buffer-size 100 \
    --batch-size $generate_bsz \
    --beam 5 \
    --remove-bpe "$bpe_type" \
    --scoring sacrebleu \
    --num-workers $decode_num_workers || exit 1;