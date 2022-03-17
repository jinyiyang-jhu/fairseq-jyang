#!/bin/bash

stage=-1
ori_tgt_dir="data/msa-en_processed/spm2000/split8"
decode_dir="exp_msa-ta_bpe2000_msa2ta_msa_translated_ta_true/decode_spm2000_msa2en_manual_clean_train"
outdir="data/ta-en_ta_translated_en_true"
spm_mdl=

src_lan="ar" # Not used for training, just for sanity check
tgt_lan="en"
hyp_lan="ta"

src_case="rm"
tgt_case="lc.rm"
hyp_case="rm"
set="train"

# src_path=exp_msa-ta_bpe2000_msa2ta_msa_translated_ta_true/decode_spm2000_msa2en_manual_clean_train/split1/src.ar.text
# hyp_path=exp_msa-ta_bpe2000_msa2ta_msa_translated_ta_true/decode_spm2000_msa2en_manual_clean_train/split1/hyp.ta.text
# tgt_path=data/msa-en_processed/spm2000/split8/1/train.bpe.ar-en.en
# outdir=data/ta-en_ta_translated_en_true

mkdir -p $outdir || exit 1

if [ $stage -le 0 ]; then
    echo "$(date) Stage 1: Filtering src/hyp/tgt text files"
    for i in $(seq 1 8); do
    (
        echo "$(date) Processing split ${i}"
        src_path=${decode_dir}/split${i}/src.${src_lan}.text
        hyp_path=${decode_dir}/decode_spm2000_msa2en_manual_clean_${set}/split${i}/hyp.${hyp_lan}.text
        tgt_path=${ori_tgt_dir}/${i}/${set}.bpe.${src_lan}-${tgt_lan}.${tgt_lan}

        python local/sort_fairseq_gen_hyp_to_ref.py \
            --src_text $src_path \
            --hyp_text $hyp_path \
            --tgt_txt $tgt_path \
            --out_src "${outdir}/${set}.bpe.${src_lan}-${tgt_lan}.${tgt_lan}" \
            --out_hyp "${outdir}/${set}.${hyp_lan}-${tgt_lan}.${hyp_lan}.${hyp_case}" \
            --out_tgt "${outdir}/${set}.${hyp_lan}-${tgt_lan}.${tgt_lan}.${tgt_case}" \
    ) &
    done
    wait
fi

