#!/bin/bash

stage=-1
nj=8 # no greater than qquota

# src_lan="ta"
# tgt_lan="en"
# src_case="tc.rm"
# tgt_case="tc"

tgt_lan="ta"
src_lan="en"
tgt_case="tc.rm"
src_case="tc"

dset="train"
path_to_eval_data=data/ta-en_clean/${dset}
path_to_eval_src=${path_to_eval_data}/text.${src_case}.${src_lan}
path_to_eval_tgt=${path_to_eval_data}/text.${tgt_case}.${tgt_lan}

path_to_bpe_mdl=data/msa-en_processed/spm2000/en_bpe_spm2000/bpe.model
path_to_dict_dir=exp_en-msa_bpe2000/bin_en2ar
path_to_mdl=exp_en-msa_bpe2000/checkpoints/checkpoint_best.pt

decode_dir=exp_en-msa_bpe2000/decode_spm2000_en2ta_manual_clean_${dset}_interactive

num_src_lines=$(wc -l ${path_to_eval_src} | cut -d" " -f1)
num_tgt_lines=$(wc -l ${path_to_eval_tgt} | cut -d" " -f1)

if [ ${num_src_lines} -ne ${num_tgt_lines} ]; then
    echo "Line mismatch: src ($num_src_lines) vs tgt ($num_tgt_lines)"
    exit 1;
fi

. path.sh
. cmd.sh
. parse_options.sh

mkdir -p $decode_dir/logs || exit 1

local/split_scp.sh $nj ${path_to_eval_data} "text.${src_case}.${src_lan}" || exit 1
local/split_scp.sh $nj ${path_to_eval_data} "text.${tgt_case}.${tgt_lan}" || exit 1

for n in $(seq $nj); do
    eval_src=

    cat ${path_to_eval_src} | cut -d" " -f2- | tokenizer.perl -q -no-escape > ${decode_dir}/${src_lan}.txt
    cat ${path_to_eval_tgt} | cut -d" " -f2- | detokenizer.perl -q -no-escape > ${decode_dir}/${tgt_lan}.txt
    cat ${path_to_eval_src} | cut -d" " -f1 > ${decode_dir}/${src_lan}.uttids
    cat ${path_to_eval_tgt} | cut -d" " -f1 > ${decode_dir}/${tgt_lan}.uttids

    if [ $stage -le 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') Apply BPE to source data"
        spm_encode \
        --model=${path_to_bpe_mdl} \
        --output_format=piece \
        < ${decode_dir}/${src_lan}.txt \
        > ${decode_dir}/${src_lan}.bpe.txt
    fi


    echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding ${src_lan}-${tgt_lan}:${src_lan}"

    [ -f ${decode_dir}/logs/decode.log ] && rm ${decode_dir}/logs/decode.log
    ${decode_cmd} --gpu 1 ${decode_dir}/logs/decode.log \
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
    grep ^D $decode_dir/logs/decode.log | cut -f3 | detokenizer.perl -q > ${decode_dir}/hyp.txt || exit 1

done
sacrebleu ${decode_dir}/${tgt_lan}.txt -i ${decode_dir}/hyp.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding done !"

    