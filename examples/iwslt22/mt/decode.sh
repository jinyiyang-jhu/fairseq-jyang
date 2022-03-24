#!/bin/bash

stage=-1
nj=1 # no greater than qquota

src_lan="ta"
tgt_lan="en"
src_case="tc.rm"
tgt_case="tc"
decode_tgt_lan="en"

skip_decode="False"
hyp_is_scp="True"

sets=("dev" "test1" "blind_eval")
path_to_eval_data="amir_asr/tuned_models_results/row15"
path_to_ref_data="data/ta-en_clean"
set_name="row15"
nbpe=4000

path_to_bpe_mdl=/home/hltcoe/jyang1/tools/espnet/egs2/iwslt22_dialect/st1/data_clean/spm_bpe_4000/${tgt_lan}_bpe_spm4000/bpe.model
mdl_dir=/exp/jyang1/exp/IWSLT22/TA-EN/MT/fairseq/exp_clean_spm4000
path_to_dict_dir=${mdl_dir}/bin_${src_lan}2${tgt_lan}
path_to_mdl=${mdl_dir}/checkpoints/checkpoint_best.pt

. path.sh
. cmd.sh
. parse_options.sh

if [ ${skip_decode} != "True" ]; then
    for dset in ${sets[@]}; do
    (
        if [ $dset != "blind_eval" ]; then
            path_to_eval_src=${path_to_eval_data}/${dset}/text.${src_case}.${src_lan}
            path_to_eval_tgt=${path_to_eval_data}/${dset}/text.${tgt_case}.${tgt_lan}

            num_src_lines=$(wc -l ${path_to_eval_src} | cut -d" " -f1)
            num_tgt_lines=$(wc -l ${path_to_eval_tgt} | cut -d" " -f1)
            if [ ${num_src_lines} -ne ${num_tgt_lines} ]; then
                echo "Line mismatch: src ($num_src_lines) vs tgt ($num_tgt_lines)"
                exit 1;
            fi
        fi
        decode_dir=${mdl_dir}/decode_${set_name}_${dset}_interactive
        mkdir -p ${decode_dir} || exit 1

        if [ ${hyp_is_scp} == "True" ]; then
            path_to_eval_src_sorted=${path_to_eval_data}/${dset}/text.${tgt_case}.${tgt_lan}.sorted
            python local/sort_src_to_tgt_order.py \
                --src_text ${path_to_eval_tgt} \
                --hyp_in_text ${path_to_eval_src} \
                --hyp_out_text ${path_to_eval_src_sorted}
            path_to_eval_src=${path_to_eval_src_sorted}
        fi

        cat ${path_to_eval_src} | cut -d" " -f2- | tokenizer.perl -q -no-escape > ${decode_dir}/${src_lan}.txt
        cat ${path_to_eval_src} | cut -d" " -f1 > ${decode_dir}/${src_lan}.uttids

        if [ $stage -le 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') Apply BPE to source data: ${dset}-${src_lan}"
            spm_encode \
            --model=${path_to_bpe_mdl} \
            --output_format=piece \
            < ${decode_dir}/${src_lan}.txt \
            > ${decode_dir}/${src_lan}.bpe.txt || exit 1;
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding ${src_lan}-${tgt_lan}:${src_lan}: ${dset}"

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

        if [ $dset != "blind_eval" ]; then
            cat ${path_to_eval_tgt} | cut -d" " -f2- | detokenizer.perl -q -no-escape > ${decode_dir}/ref.${tgt_lan}.txt
            cat ${path_to_eval_tgt} | cut -d" " -f1 > ${decode_dir}/${tgt_lan}.uttids
            diff_utts=$(diff "${decode_dir}/${src_lan}.uttids" "${decode_dir}/${tgt_lan}.uttids")
            if [ ! -z "${diff_utts}" ]; then
                echo "Utt order mismatch: src [ ${decode_dir}/${src_lan}.uttids ] vs tgt [ ${decode_dir}/${tgt_lan}.uttids ]"
                exit 1
            fi

            sacrebleu ${decode_dir}/ref.${tgt_lan}.txt -i ${decode_dir}/hyp.${decode_tgt_lan}.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
            cp ${decode_dir}/hyp.${decode_tgt_lan}.txt ${path_to_eval_data}
            cp ${decode_dir}/results.txt ${path_to_eval_data}/bleus.txt
        fi
        echo "$(date '+%Y-%m-%d %H:%M:%S') Evaluation done !"
    ) &
    done
fi


    