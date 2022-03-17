#!/bin/bash

stage=-1

src_lan="ar" # Not used for training, just for sanity check
tgt_lan="en"
hyp_lan="ta"

src_case="tc.rm"
tgt_case="tc"
hyp_case="tc.rm"
trainset="train"
num_bpe=2000

ori_tgt_dir="data/msa-en_processed/spm2000/split8"
decode_dir="exp_msa-ta_bpe2000_msa2ta_msa_translated_ta_true/decode_spm2000_msa2en_manual_clean_train"
datadir="data/ta-en_ta_translated_en_true"
bindir="exp_ta-en_ta_translated_en_true/bin_${hyp_lan}2${tgt_lan}"
hyp_lan_bpe_mdl="data/msa-en_processed/spm2000/${src_lan}_bpe_spm2000/bpe.model"
tgt_lan_bpe_mdl="data/msa-en_processed/spm2000/${tgt_lan}_bpe_spm2000/bpe.model"

srcdict="exp_msa-ta_bpe2000_msa2ta_msa_translated_ta_true/bin_ar2ta/dict.ta.txt"
tgtdict="exp_msa-en_bpe2000/bin_ar2en/dict.en.txt"

# dev
dev_test_text_dir="data/ta-en_clean"
# dev_hyp_lan_text="${dev_test_text_dir}/dev/text.${hyp_case}.${hyp_lan}"
# dev_tgt_lan_text="${dev_test_text_dir}/dev/text.${tgt_case}.${tgt_lan}"

# test
# test_hyp_lan_text="data/ta-en_clean/test1/text.${hyp_case}.${hyp_lan}"
# test_tgt_lan_text="data/ta-en_clean/test1/text.${tgt_case}.${tgt_lan}"

. path.sh
. parse_options.sh

if [ $stage -le 0 ]; then
    echo "$(date) Stage 1: Filtering src/hyp/tgt text files"
    for i in $(seq 8); do
    (
        echo "$(date) Processing split ${i}"
        grep ^D ${decode_dir}/split${i}/decode.${i}.log | grep -v "fairseq.logging.progress_bar" | awk -F"\t" '{print $1"\t"$3}' \
            > $decode_dir/split${i}/hyp.${hyp_lan}.undetok.text || exit 1
        
        src_path=${decode_dir}/split${i}/src.${src_lan}.text
        hyp_path=${decode_dir}/split${i}/hyp.${hyp_lan}.undetok.text
        tgt_path=${ori_tgt_dir}/${i}/${trainset}.bpe.${src_lan}-${tgt_lan}.${tgt_lan}
        outdir=${datadir}/${trainset}/split${i}
        mkdir -p ${outdir} || exit 1
        echo "$(date) Filtering hyp and target: split ${i}"
        for f in ${src_path} ${hyp_path} ${tgt_path}; do
            if [ ! -f $f ]; then
                echo "No such file: $f"
                exit 1
            fi
        done

        python local/sort_fairseq_gen_hyp_to_ref.py \
            --src_text $src_path \
            --hyp_text $hyp_path \
            --tgt_txt $tgt_path \
            --out_src ${outdir}/text.${src_case}.${src_lan} \
            --out_hyp ${outdir}/text.${hyp_case}.${hyp_lan} \
            --out_tgt ${outdir}/text.bpe.${tgt_case}.${tgt_lan}

        echo "Detokenize : split ${i}"
        text="${outdir}/text.${hyp_case}.${hyp_lan}"
        detok="${outdir}/text.${hyp_case}.${hyp_lan}.detok"
        tok="${outdir}/text.${hyp_case}.${hyp_lan}.tok"
        bpe="${outdir}/train.bpe.${hyp_lan}-${tgt_lan}.${hyp_lan}"

        cut -d " " -f2- ${text} | detokenizer.perl -q -no-escape  > ${detok} || exit 1;
        tokenizer.perl -q -no-escape -threads 2 < ${detok} > ${tok} || exit 1;

        spm_encode \
            --model="${hyp_lan_bpe_mdl}" \
            --output_format=piece \
            < ${tok} \
            > ${bpe} || exit 1;

    ) &
    done
    wait

    for lan in ${hyp_lan} ${tgt_lan}; do
        if [ -f "${datadir}/spm${num_bpe}/${trainset}.bpe.${hyp_lan}-${tgt_lan}.${lan}" ]; then
            rm  "${datadir}/spm${num_bpe}/${trainset}.bpe.${hyp_lan}-${tgt_lan}.${lan}"
        fi
    done
    mkdir -p ${datadir}/spm${num_bpe} || exit 1;
    for n in $(seq 8); do
        outdir=${datadir}/${trainset}/split${n}
        cut -d " " -f2- "${outdir}/text.bpe.${tgt_case}.${tgt_lan}"  >> "${datadir}/spm${num_bpe}/${trainset}.bpe.${hyp_lan}-${tgt_lan}.${tgt_lan}"
        cat "${outdir}/train.bpe.${hyp_lan}-${tgt_lan}.${hyp_lan}" >> "${datadir}/spm${num_bpe}/${trainset}.bpe.${hyp_lan}-${tgt_lan}.${hyp_lan}"
    done
fi

if [ $stage -le 1 ]; then
    echo "$(date) Preprocess dev/test: normalization, BPE convertion, and binarization"
    for s in "dev" "test1"; do
        for lan in ${hyp_lan} ${tgt_lan}; do
            if [ "${lan}" == "${hyp_lan}" ]; then
                lan_case=${src_case}
                bpe_mdl=${hyp_lan_bpe_mdl}
            else
                lan_case=${tgt_case}
                bpe_mdl=${tgt_lan_bpe_mdl}
            fi
            echo "$(date) Preprocess: ${s}:${lan}"
            mkdir -p ${datadir}/${s} || exit 1
            text="${dev_test_text_dir}/${s}/text.${lan_case}.${lan}"
            tok="${datadir}/${s}/text.${lan_case}.${lan}.tok"
            bpe="${datadir}/spm${num_bpe}/${s}.bpe.${hyp_lan}-${tgt_lan}.${lan}"

            cut -d " " -f2- ${text} | tokenizer.perl -q -no-escape > ${tok} || exit 1;

            spm_encode \
                --model="${bpe_mdl}" \
                --output_format=piece \
                < ${tok} \
                > ${bpe} || exit 1;
        done
    done
fi

if [ $stage -le 2 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Binarizing data"
    trainpref="${datadir}/spm${num_bpe}/${trainset}.bpe.${hyp_lan}-${tgt_lan}"
    validpref="${datadir}/spm${num_bpe}/dev.bpe.${hyp_lan}-${tgt_lan}"
    testpref="${datadir}/spm${num_bpe}/test1.bpe.${hyp_lan}-${tgt_lan}"

    fairseq-preprocess \
        --source-lang ${hyp_lan} --target-lang ${tgt_lan} \
        --srcdict ${srcdict} --tgtdict ${tgtdict} \
        --trainpref $trainpref --validpref $validpref --testpref $testpref \
        --destdir ${bindir} --thresholdtgt 0 --thresholdsrc 0 \
        --workers 16 || exit 1;
fi
