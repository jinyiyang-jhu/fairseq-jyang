#!/bin/bash

n=$1
nj=8

tgt_lan="en" # Original target language for this dataset
src_lan="ar" # source input language
decode_tgt_lan="ta" # Decoded target language, same as the decoding model's target language
tgt_case="tc.rm"
src_case="tc"

skip_split="True"
skip_decode="False"

. path.sh
. cmd.sh

dset="train"
decode_dir=exp_msa-ta_bpe2000_msa2ta_msa_translated_ta_true/decode_spm2000_msa2en_manual_clean_${dset}
path_to_eval_data=data/msa-en_processed/spm2000
dictdir=exp_msa-ta_bpe2000_msa2ta_msa_translated_ta_true/bin_${src_lan}2${decode_tgt_lan}
path_to_mdl=exp_msa-ta_bpe2000_msa2ta_msa_translated_ta_true/checkpoints/checkpoint_best.pt

echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding ${src_lan}-${tgt_lan}:${src_lan}; split : ${n}"
${decode_cmd} --gpu 1 ${decode_dir}/split${n}/decode.${n}.log \
    cat ${path_to_eval_data}/split${nj}/${n}/${dset}.bpe.${src_lan}-${tgt_lan}.${src_lan} \| \
        fairseq-interactive ${dictdir} \
            --source-lang "${src_lan}" --target-lang "${decode_tgt_lan}" \
            --task translation \
            --path ${path_to_mdl}\
            --batch-size 64 \
            --fix-batches-to-gpus \
            --beam 5 \
            --buffer-size 4000 \
            --skip-invalid-size-inputs-valid-test \
            --remove-bpe=sentencepiece || exit 1

grep ^D $decode_dir/split${n}/decode.${n}.log | cut -f3 | \
    detokenizer.perl -q -no-escape > ${decode_dir}/split${n}/hyp.${decode_tgt_lan}.txt || exit 1
grep ^S $decode_dir/split${n}/decode.${n}.log | cut -f2- | \
    detokenizer.perl -q -no-escape > ${decode_dir}/split${n}/src.${src_lan}.txt || exit 1 