#!/bin/bash

stage=-1
nj=1 # no greater than qquota

src_lan="ta"
tgt_lan="en"
src_case="tc.rm"
tgt_case="tc"
decode_tgt_lan="en"

# tgt_lan="ta"
# src_lan="en"
# tgt_case="tc.rm"
# src_case="tc"
# decode_tgt_lan="ta"

skip_split="False"
skip_decode="False"
hyp_is_scp="True"

dset="test1"
path_to_eval_data="amir_asr_cleaned_BPE1000/${dset}"
#path_to_eval_data=data/ta-en_clean/${dset}
#path_to_eval_data=amir_asr_mgb2_finetuning_transformer/${dset}
# bpe and dic
#path_to_bpe_mdl=data/ta-en_clean/spm_bpe/${src_lan}_bpe_spm1000/bpe.model
#path_to_dict_dir=exp_clean/bin_ta2en
#path_to_mdl=exp_clean/checkpoints/checkpoint_best.pt
path_to_bpe_mdl=data/msa-en_processed/spm2000/ar_bpe_spm2000/bpe.model
path_to_dict_dir=exp_msa-en_bpe2000/bin_ar2en
mdl_dir=exp_ta-en_ta_translated_en_true_tune_with_true_ta-en
path_to_mdl=${mdl_dir}/checkpoints/checkpoint_best.pt
decode_dir=${mdl_dir}/decode_amir_cleaned_BPE1000_${dset}_interactive
#decode_dir=${mdl_dir}/decode_gold_transcript_${dset}_interactive

. path.sh
. cmd.sh
. parse_options.sh

path_to_eval_src=${path_to_eval_data}/text.${src_case}.${src_lan}
path_to_eval_tgt=${path_to_eval_data}/text.${tgt_case}.${tgt_lan}

num_src_lines=$(wc -l ${path_to_eval_src} | cut -d" " -f1)
num_tgt_lines=$(wc -l ${path_to_eval_tgt} | cut -d" " -f1)

if [ ${skip_split} != "True" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')  Splitting the dir: ${path_to_eval_data}"
    bash local/split_scp.sh $nj ${path_to_eval_data} "text.${src_case}.${src_lan}" || exit 1
    bash local/split_scp.sh $nj ${path_to_eval_data} "text.${tgt_case}.${tgt_lan}" || exit 1
fi

if [ ${skip_decode} != "True" ]; then
    for f in "${decode_dir}/hyp.${decode_tgt_lan}.txt" "${decode_dir}/${src_lan}.uttids" "${decode_dir}/split${n}/src.${src_lan}.txt"; do
        [ -f $f ] && rm $f 
    done
    for n in $(seq $nj); do
    (
        mkdir -p ${decode_dir}/split${n} || exit 1
        if [ ${num_src_lines} -ne ${num_tgt_lines} ]; then
            echo "Line mismatch: src ($num_src_lines) vs tgt ($num_tgt_lines)"
            exit 1;
        fi
        path_to_eval_src=${path_to_eval_data}/split${nj}/${n}/text.${src_case}.${src_lan}
        path_to_eval_tgt=${path_to_eval_data}/split${nj}/${n}/text.${tgt_case}.${tgt_lan}

        if [ ${hyp_is_scp} == "True" ]; then
            path_to_eval_src_sorted=${path_to_eval_data}/split${nj}/${n}/text.${tgt_case}.${tgt_lan}.sorted
            python local/sort_tgt_to_ref_order.py \
                --ref_text ${path_to_eval_tgt} \
                --tgt_in_text ${path_to_eval_src} \
                --tgt_out_text ${path_to_eval_src_sorted}
            path_to_eval_src=${path_to_eval_src_sorted}
        fi

        cat ${path_to_eval_src} | cut -d" " -f2- | tokenizer.perl -q -no-escape > ${decode_dir}/split${n}/${src_lan}.txt
        cat ${path_to_eval_src} | cut -d" " -f1 > ${decode_dir}/split${n}/${src_lan}.uttids

        if [ $stage -le 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') Apply BPE to source data"
            spm_encode \
            --model=${path_to_bpe_mdl} \
            --output_format=piece \
            < ${decode_dir}/split${n}/${src_lan}.txt \
            > ${decode_dir}/split${n}/${src_lan}.bpe.txt || exit 1;
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding ${src_lan}-${tgt_lan}:${src_lan}; split : ${n}"

        ${decode_cmd} --gpu 1 ${decode_dir}/split${n}/decode.${n}.log \
            cat ${decode_dir}/split${n}/${src_lan}.bpe.txt \| \
            fairseq-interactive ${path_to_dict_dir} \
                --source-lang "${src_lan}" --target-lang "${tgt_lan}" \
                --task translation \
                --path ${path_to_mdl}\
                --batch-size 256 \
                --fix-batches-to-gpus \
                --beam 5 \
                --buffer-size 2000 \
                --remove-bpe=sentencepiece || exit 1
        
        grep ^D $decode_dir/split${n}/decode.${n}.log | cut -f3 | \
            detokenizer.perl -q -no-escape > ${decode_dir}/split${n}/hyp.${decode_tgt_lan}.txt || exit 1
        grep ^S $decode_dir/split${n}/decode.${n}.log | cut -f2- | \
            detokenizer.perl -q -no-escape > ${decode_dir}/split${n}/src.${src_lan}.txt || exit 1  
    ) &
    done
    wait
    echo "$(date '+%Y-%m-%d %H:%M:%S') All ${nj} decoding done !"
fi

for n in $(seq $nj); do
    cat ${decode_dir}/split${n}/hyp.${decode_tgt_lan}.txt >> ${decode_dir}/hyp.${decode_tgt_lan}.txt
    cat ${decode_dir}/split${n}/${src_lan}.uttids >> ${decode_dir}/${src_lan}.uttids
done

cat ${path_to_eval_tgt} | cut -d" " -f2- | detokenizer.perl -q -no-escape > ${decode_dir}/ref.${tgt_lan}.txt
cat ${path_to_eval_tgt} | cut -d" " -f1 > ${decode_dir}/${tgt_lan}.uttids
diff_utts=$(diff "${decode_dir}/${src_lan}.uttids" "${decode_dir}/${tgt_lan}.uttids")
if [ ! -z "${diff_utts}" ]; then
    echo "Utt order mismatch: src [ ${decode_dir}/${src_lan}.uttids ] vs tgt [ ${decode_dir}/${tgt_lan}.uttids ]"
    exit 1
fi

sacrebleu ${decode_dir}/ref.${tgt_lan}.txt -i ${decode_dir}/hyp.${decode_tgt_lan}.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
echo "$(date '+%Y-%m-%d %H:%M:%S') Evaluation done !"

    