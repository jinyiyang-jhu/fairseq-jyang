#!/bin/bash

stage=-1

src_lan="ta"
tgt_lan="en"
decode_tgt_lan="en"

skip_split="False"
skip_decode="False"
hyp_is_scp="True"

dset="blind_eval"
path_to_eval_data="amir_asr_row16/${dset}"
mdl_dir=exp_ta-en_ta_translated_en_true_tune_with_true_ta-en
decode_dir=${mdl_dir}/decode_amir_row16_${dset}_interactive

path_to_bpe_mdl=data/msa-en_processed/spm2000/ar_bpe_spm2000/bpe.model
path_to_dict_dir=exp_msa-en_bpe2000/bin_ar2en

path_to_eval_src=${path_to_eval_data}/text
path_to_mdl=${mdl_dir}/checkpoints/checkpoint_best.pt
#decode_dir=${mdl_dir}/decode_gold_transcript_${dset}_interactive

. path.sh
. cmd.sh
. parse_options.sh

mkdir -p ${decode_dir}/log || exit 1;

if [ ${skip_decode} != "True" ]; then
    cat ${path_to_eval_src} | cut -d" " -f2- | tokenizer.perl -q -no-escape > ${decode_dir}/${src_lan}.txt
    cat ${path_to_eval_src} | cut -d" " -f1 > ${decode_dir}/${src_lan}.uttids

    if [ $stage -le 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') Apply BPE to source data"
        spm_encode \
        --model=${path_to_bpe_mdl} \
        --output_format=piece \
        < ${decode_dir}/${src_lan}.txt \
        > ${decode_dir}/${src_lan}.bpe.txt || exit 1;
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding ${src_lan}-${tgt_lan}:${src_lan}"

    ${decode_cmd} --gpu 1 ${decode_dir}/decode.log \
        cat ${decode_dir}/${src_lan}.bpe.txt \| \
        fairseq-interactive ${path_to_dict_dir} \
            --source-lang "${src_lan}" --target-lang "${tgt_lan}" \
            --task translation \
            --path ${path_to_mdl}\
            --batch-size 256 \
            --fix-batches-to-gpus \
            --beam 5 \
            --buffer-size 2000 \
            --remove-bpe=sentencepiece || exit 1
    
    grep ^D $decode_dir/decode.log | cut -f3 | \
        detokenizer.perl -q -no-escape > ${decode_dir}/hyp.${decode_tgt_lan}.txt || exit 1
    grep ^S $decode_dir/decode.log | cut -f2- | \
        detokenizer.perl -q -no-escape > ${decode_dir}/src.${src_lan}.txt || exit 1
    paste -d " " ${decode_dir}/${src_lan}.uttids ${decode_dir}/hyp.${decode_tgt_lan}.txt \
        > ${decode_dir}/hyp.${decode_tgt_lan}.text
fi

#sacrebleu ${decode_dir}/ref.${tgt_lan}.txt -i ${decode_dir}/hyp.${decode_tgt_lan}.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
echo "$(date '+%Y-%m-%d %H:%M:%S') Evaluation done !"

    