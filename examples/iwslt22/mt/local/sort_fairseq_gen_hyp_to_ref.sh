#!/bin/bash


src_path=exp_msa-ta_bpe2000_msa2ta_msa_translated_ta_true/decode_spm2000_msa2en_manual_clean_train/split1/src.ar.text
hyp_path=exp_msa-ta_bpe2000_msa2ta_msa_translated_ta_true/decode_spm2000_msa2en_manual_clean_train/split1/hyp.ta.text
tgt_path=data/msa-en_processed/spm2000/split8/1/train.bpe.ar-en.en
outdir=data/ta-en_ta_translated_en_true/split1

mkdir -p $outdir || exit 1

python local/sort_fairseq_gen_hyp_to_ref.py \
    --src_text $src_path \
    --hyp_text $hyp_path \
    --tgt_txt $tgt_path \
    --out_dir $outdir