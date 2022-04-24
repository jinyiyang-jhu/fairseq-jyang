#!/bin/bash

stage=-1

src_lan="ta"
tgt_lan="en"

#dev_test_dir="data/ta-en_clean/spm_ta_true_en_joint_4000/${src_lan}2${tgt_lan}"
dev_test_dir="data/ta-en_clean/spm_bpe/en_bpe_spm1000"
decode_dir="exp_clean_en2ta_ta_true_en_42m_spm4k/decode_spm4k_en2ta_manual_clean_train"

srcdict="exp_clean_en2ta_ta_true_en_42m_spm4k/bin_en2ta/dict.${src_lan}.txt"
tgtdict="exp_clean_en2ta_ta_true_en_42m_spm4k/bin_en2ta/dict.${tgt_lan}.txt"

#bpemdl_dir="data/ta-en_clean/spm_ta_true_en_joint_4000/"
bpemdl_dir="data/ta-en_clean/spm_ta_true_4k_en_joint_32000/"
bpedir="data/ta-en_ta_translated_en_true_translated_from_en2ta_mdl/spm_ta_condA_4k_en_32k_joint/${src_lan}2${tgt_lan}" # To store the encoded BPEs

bindir="exp_ta2en_ta_translated_from_en2ta_en_true_ta_4k_en_32k/bin${src_lan}2${tgt_lan}"

src_len=0
tgt_len=0

. path.sh
. parse_options.sh

if [ $stage -le 0 ]; then
    echo "$(date) Stage 1: Encoding src/tgt text files"
    for lan in ${src_lan} ${tgt_lan}; do

        bpe_mdl=${bpemdl_dir}/${lan}/bpe.model
        if [ -f "${bpedir}/train.bpe.${src_lan}-${tgt_lan}.${lan}" ]; then
            rm  "${bpedir}/train.bpe.${src_lan}-${tgt_lan}.${lan}"
        fi

        for i in $(seq 8); do
        (
            echo "$(date) Processing split ${i}: ${lan}"
            outdir=${bpedir}/split${i}
            mkdir -p $outdir || exit 1

            logfile="${decode_dir}/split${i}/decode.${i}.log"
            [ ! -f ${logfile} ] && echo "No such file: $logfile" && exit 1

            if [ ${lan} == ${tgt_lan} ]; then # The backtranslation was generated from a ${tgt_lan}2${src_lan} model
                grep ^S ${logfile} | cut -f2- > ${outdir}/train.text.${lan}
            elif [ ${lan} == ${src_lan} ]; then
                grep ^D ${logfile} | cut -f3 > ${outdir}/train.text.${lan}
            fi
            spm_encode \
                --model="${bpe_mdl}" \
                --output_format=piece \
                < ${outdir}/train.text.${lan} \
                > ${outdir}/train.bpe.${src_lan}-${tgt_lan}.${lan} || exit 1;   
        ) &
        done
        wait

        for i in $(seq 8); do
            outdir=${bpedir}/split${i}
            cat "${outdir}/train.bpe.${src_lan}-${tgt_lan}.${lan}" >> "${bpedir}/train.bpe.${src_lan}-${tgt_lan}.${lan}"
        done
    done
fi

if [ $stage -le 1 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Encoding dev/test data"
    for lan in ${src_lan} ${tgt_lan}; do
        bpe_mdl=${bpemdl_dir}/${lan}/bpe.model
        for dset in dev test1; do
            text=${dev_test_dir}/${dset}.txt
            spm_encode \
                --model="${bpe_mdl}" \
                --output_format=piece \
                < ${text} \
                > ${bpedir}/${dset}.bpe.${src_lang}-${tgt_lang}.${lan}
        done
    done
fi

if [ $stage -le 2 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Binarizing data"
    trainpref="${bpedir}/train.bpe.${src_lan}-${tgt_lan}"
    validpref="${dev_test_dir}/dev.bpe.${src_lan}-${tgt_lan}"
    testpref="${dev_test_dir}/test1.bpe.${src_lan}-${tgt_lan}"

    fairseq-preprocess \
        --source-lang ${src_lan} --target-lang ${tgt_lan} \
        --srcdict ${srcdict} --tgtdict ${tgtdict} \
        --trainpref $trainpref --validpref $validpref --testpref $testpref \
        --destdir ${bindir} --thresholdtgt 0 --thresholdsrc 0 \
        --workers 16 || exit 1;
fi


