#!/bin/bash

stage=-1
nj=8 # no greater than qquota

src_lan="ta"
tgt_lan="en"
src_case="tc.rm"
tgt_case="tc"

# tgt_lan="ta"
# src_lan="en"
# tgt_case="tc.rm"
# src_case="tc"

skip_split="True"
skip_decode="False"

dset="dev"
path_to_eval_data=data/ta-en_clean/${dset}

path_to_bpe_mdl=data/msa-en_processed/spm2000/ar_bpe_spm2000/bpe.model
path_to_dict_dir=exp_msa-en_bpe2000_tune_with_ta/bin_ta2en
path_to_mdl=exp_msa-en_bpe2000_tune_with_ta/checkpoints/checkpoint_best.pt

decode_dir=exp_msa-en_bpe2000_tune_with_ta/decode_spm2000_ta2en_manual_clean_${dset}_interactive

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
    for f in "${decode_dir}/hyp.txt" "${decode_dir}/${src_lan}.uttids"; do
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
        grep ^D $decode_dir/split${n}/decode.${n}.log | cut -f3 | detokenizer.perl -q > ${decode_dir}/split${n}/hyp.txt || exit 1
    ) &
    done
    wait
    echo "$(date '+%Y-%m-%d %H:%M:%S') All ${nj} decoding done !"
fi

for n in $(seq $nj); do
    cat ${decode_dir}/split${n}/hyp.txt >> ${decode_dir}/hyp.txt
    cat ${decode_dir}/split${n}/${src_lan}.uttids >> ${decode_dir}/${src_lan}.uttids
done

cat ${path_to_eval_tgt} | cut -d" " -f2- | detokenizer.perl -q -no-escape > ${decode_dir}/${tgt_lan}.txt
cat ${path_to_eval_tgt} | cut -d" " -f1 > ${decode_dir}/${tgt_lan}.uttids
diff_utts=$(diff "${decode_dir}/${src_lan}.uttids" "${decode_dir}/${tgt_lan}.uttids")
if [ ! -z "${diff_utts}" ]; then
    echo "Utt order mismatch: src [ ${decode_dir}/${src_lan}.uttids ] vs tgt [ ${decode_dir}/${tgt_lan}.uttids ]"
    exit 1
fi

sacrebleu ${decode_dir}/${tgt_lan}.txt -i ${decode_dir}/hyp.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
echo "$(date '+%Y-%m-%d %H:%M:%S') Evaluation done !"

    