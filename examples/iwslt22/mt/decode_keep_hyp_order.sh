#!/bin/bash

stage=-1
nj=1 # no greater than qquota

############ Change #############
#src_case=".tc.rm"
#set_name="gold_transcript"
#path_to_eval_data="data/ta-en_clean"


src_case=""
set_name="row20_beam40"
path_to_eval_data="amir_asr/beam40/row20"
############ Change #############
path_to_bpe_mdl=/home/hltcoe/jyang1/tools/espnet/egs2/iwslt22_dialect/st1/data_clean/spm_bpe_4000/ta_bpe_spm4000/bpe.model
mdl_name="row27"
mdl_dir=/exp/jyang1/exp/IWSLT22/TA-EN/MT/fairseq/exp_clean_spm4000


src_lan="ta"
tgt_lan="en"
tgt_case="tc"
decode_tgt_lan="en"

skip_decode="False"
hyp_is_scp="True"

sets=("dev" "test1" "blind_eval")
path_to_ref_data="data/ta-en_clean"
nbpe=4000

path_to_dict_dir=${mdl_dir}/bin_${src_lan}2${tgt_lan}
path_to_mdl=${mdl_dir}/checkpoints/checkpoint_best.pt

. path.sh
. cmd.sh
. parse_options.sh

if [ ${skip_decode} != "True" ]; then
    for dset in ${sets[@]}; do
        path_to_eval_src=${path_to_eval_data}/${dset}/text${src_case}.${src_lan}
        path_to_eval_tgt=${path_to_ref_data}/${dset}/text.${tgt_case}.${tgt_lan}
        if [ $dset != "blind_eval" ]; then
            num_src_lines=$(wc -l ${path_to_eval_src} | cut -d" " -f1)
            num_tgt_lines=$(wc -l ${path_to_eval_tgt} | cut -d" " -f1)
            if [ ${num_src_lines} -ne ${num_tgt_lines} ]; then
                echo "Line mismatch: src ($num_src_lines) vs tgt ($num_tgt_lines)"
                exit 1;
            fi
        fi

        decode_dir=${mdl_dir}/decode_${set_name}_${dset}_interactive
        mkdir -p ${decode_dir} || exit 1

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

        #${decode_cmd} --gpu 1 ${decode_dir}/decode.log \
            # cat ${decode_dir}/${src_lan}.bpe.txt \| \
        cat ${decode_dir}/${src_lan}.bpe.txt | \
            fairseq-interactive ${path_to_dict_dir} \
                --source-lang "${src_lan}" --target-lang "${tgt_lan}" \
                --task translation \
                --path ${path_to_mdl}\
                --batch-size 256 \
                --fix-batches-to-gpus \
                --beam 5 \
                --buffer-size 2000 \
                --remove-bpe=sentencepiece > ${decode_dir}/decode.log
        
        grep ^D $decode_dir/decode.log | cut -f3 | \
            detokenizer.perl -q -no-escape > ${decode_dir}/hyp.${decode_tgt_lan}.txt || exit 1
        grep ^S $decode_dir/decode.log | cut -f2- | \
            detokenizer.perl -q -no-escape > ${decode_dir}/src.${src_lan}.txt || exit 1  

        paste -d " " ${decode_dir}/${src_lan}.uttids ${decode_dir}/hyp.${decode_tgt_lan}.txt \
            > ${decode_dir}/hyp.${decode_tgt_lan}.text

        if [ $dset != "blind_eval" ]; then

            path_to_eval_tgt_sorted=${path_to_eval_data}/${dset}/text.${tgt_case}.${tgt_lan}.sorted
            python local/sort_tgt_to_ref_order.py \
                --ref_text ${decode_dir}/hyp.${decode_tgt_lan}.text \
                --tgt_in_text ${path_to_eval_tgt} \
                --tgt_out_text ${path_to_eval_tgt_sorted} || exit 1;

            cat ${path_to_eval_tgt_sorted} | cut -d" " -f2- | detokenizer.perl -q -no-escape > ${decode_dir}/ref.${tgt_lan}.sorted.txt
            cat ${path_to_eval_tgt_sorted} | cut -d" " -f1 > ${decode_dir}/${tgt_lan}.sorted.uttids
            diff_utts=$(diff "${decode_dir}/${src_lan}.uttids" "${decode_dir}/${tgt_lan}.sorted.uttids")
            if [ ! -z "${diff_utts}" ]; then
                echo "Utt order mismatch: src [ ${decode_dir}/${src_lan}.uttids ] vs tgt [ ${decode_dir}/${tgt_lan}.sorted.uttids ]"
                exit 1
            fi

            sacrebleu ${decode_dir}/ref.${tgt_lan}.sorted.txt -i ${decode_dir}/hyp.${decode_tgt_lan}.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
            cp ${decode_dir}/hyp.${decode_tgt_lan}.text ${path_to_eval_data}/${dset}/hyp.${mdl_name}.text
            cp ${decode_dir}/results.txt ${path_to_eval_data}/${dset}/bleus.${mdl_name}.txt
        else
            cp ${decode_dir}/hyp.${decode_tgt_lan}.text ${path_to_eval_data}/${dset}/hyp.${mdl_name}.text
        fi
        echo "$(date '+%Y-%m-%d %H:%M:%S') Evaluation done !"
    done
fi


    