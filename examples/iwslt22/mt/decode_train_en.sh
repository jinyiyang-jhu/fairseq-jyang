#!/bin/bash

stage=1
nj=8 # no greater than qquota

src_lan="en" # source input language
tgt_lan="ar"
decode_tgt_lan="ta"

skip_split="True"
skip_decode="False"

# datadir
dset="train"
path_to_eval_data=data/msa-en_processed/spm4000

# bpe and dict
dictdir=exp_clean_en2ta_ta_true_en_42m_spm4k/bin_en2ta
path_to_mdl=exp_clean_en2ta_ta_true_en_42m_spm4k/checkpoints/checkpoint_best.pt
srcdict=${dictdir}/dict.${src_lan}.txt
tgtdict=${dictdir}/dict.${decode_tgt_lan}.txt

decode_dir=exp_clean_en2ta_ta_true_en_42m_spm4k/decode_spm4k_en2ta_manual_clean_${dset}

. path.sh
. cmd.sh
. parse_options.sh

path_to_eval_src=${path_to_eval_data}/${dset}.bpe.${src_lan}-${tgt_lan}.${src_lan}

num_src_lines=$(wc -l ${path_to_eval_src} | cut -d" " -f1)

if [ ${skip_split} != "True" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')  Splitting the dir: ${path_to_eval_data}"
    bash local/split_scp.sh $nj ${path_to_eval_data} "${dset}.bpe.${src_lan}-${tgt_lan}.${src_lan}" || exit 1
fi

if [ ${skip_decode} != "True" ]; then
    for f in "${decode_dir}/hyp.${decode_tgt_lan}.txt" "${decode_dir}/src.${src_lan}.txt" "${decode_dir}/ref.${tgt_lan}.txt"; do
        [ -f $f ] && rm $f 
    done
    for n in $(seq $nj); do
    (
        mkdir -p ${decode_dir}/split${n} || exit 1
        trainpref="${path_to_eval_data}/split${nj}/${n}/${dset}.bpe.${src_lan}-${tgt_lan}"
        bindir=${path_to_eval_data}/split${nj}/${n}/bin_${src_lan}2${decode_tgt_lan}
        cp $tgtdict $bindir || exit 1;

        if [ $stage -le 0 ]; then
            mkdir -p $bindir || exit 1
            echo "$(date '+%Y-%m-%d %H:%M:%S') fairseq-preprocess for split : $n"
                fairseq-preprocess \
                    --source-lang ${src_lan} --target-lang ${decode_tgt_lan} \
                    --only-source \
                    --srcdict ${srcdict} --tgtdict ${tgtdict} \
                    --trainpref $trainpref \
                    --destdir ${bindir} --thresholdtgt 0 --thresholdsrc 0 \
                    --workers 8 || exit 1;
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding ${src_lan}-${tgt_lan}:${src_lan}; split : ${n}"

        ${decode_cmd} --gpu 1 ${decode_dir}/split${n}/decode.${n}.log \
            fairseq-generate ${bindir} \
                --source-lang "${src_lan}" --target-lang "${decode_tgt_lan}" \
                --task translation \
                --tokenizer moses \
                --path ${path_to_mdl} \
                --batch-size 256 \
                --beam 5 \
                --gen-subset "train" \
                --skip-invalid-size-inputs-valid-test \
                --remove-bpe=sentencepiece || exit 1
        grep ^D ${decode_dir}/split${n}/decode.${n}.log | cut -f3 > $decode_dir/split${n}/hyp.${decode_tgt_lan}.txt || exit 1
        grep ^S ${decode_dir}/split${n}/decode.${n}.log | cut -f2- > $decode_dir/split${n}/src.${src_lan}.txt || exit 1

        # ${decode_cmd} --gpu 1 ${decode_dir}/split${n}/decode.${n}.log \
        #     cat ${path_to_eval_data}/split${nj}/${n}/${dset}.bpe.${src_lan}-${tgt_lan}.${src_lan} \| \
        #         fairseq-interactive ${dictdir} \
        #             --source-lang "${src_lan}" --target-lang "${decode_tgt_lan}" \
        #             --task translation \
        #             --path ${path_to_mdl}\
        #             --batch-size 128 \
        #             --fix-batches-to-gpus \
        #             --beam 5 \
        #             --buffer-size 4000 \
        #             --remove-bpe=sentencepiece || exit 1

        # grep ^D $decode_dir/split${n}/decode.${n}.log | cut -f3 | \
        #     detokenizer.perl -q -no-escape > ${decode_dir}/split${n}/hyp.${decode_tgt_lan}.txt || exit 1
        # grep ^S $decode_dir/split${n}/decode.${n}.log | cut -f2- | \
        #     detokenizer.perl -q -no-escape > ${decode_dir}/split${n}/src.${src_lan}.txt || exit 1  
    ) &
    done
    wait
    echo "$(date '+%Y-%m-%d %H:%M:%S') All ${nj} decoding done !"
fi
exit 0

for n in $(seq $nj); do
    cat ${decode_dir}/split${n}/hyp.${decode_tgt_lan}.txt  >> ${decode_dir}/hyp.${decode_tgt_lan}.txt 
    cat ${decode_dir}/split${n}/src.${src_lan}.txt >> ${decode_dir}/src.${src_lan}.txt
    # cat ${decode_dir}/split${n}/ref.${tgt_lan}.txt >> ${decode_dir}/ref.${tgt_lan}.txt 
done

cat ${path_to_eval_tgt} | detokenizer.perl -q -no-escape > ${decode_dir}/ref.${tgt_lan}.txt

#sacrebleu ${decode_dir}/${tgt_lan}.txt -i ${decode_dir}/hyp.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
echo "$(date '+%Y-%m-%d %H:%M:%S') Evaluation done !"

    