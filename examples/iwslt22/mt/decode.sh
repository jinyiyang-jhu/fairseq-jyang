#!/bin/bash

stage=-1
nj_preprocess=4
ngpus=4

src_lan="ta"
tgt_lan="en"

# path_to_eval_src=$1 # scp file with words: <uttid> <token1> <token2> ...
# path_to_eval_tgt=$2 # scp file with words: <uttid> <token1> <token2> ...
# path_to_dict_dir=$3
# path_to_bpe_mdl=$4 # BPE model for source data
# path_to_mdl=$5
# decode_dir=$6

# if [ $# -ne 6 ]; then
#     echo "Usage: $0 <path-to-src-bpe-text> <path-to-tgt-word-text> <path-to-dict-dir> <path-to-bpe-mdl> <path-to-trained-mdl> <path-to-output-dir>"
#     exit 0
# fi
dset="dev"
path_to_eval_src=data/ta-en_clean/${dset}/text.tc.rm.ta
#path_to_eval_src=amir_asr_cleaned_BPE1000/${dset}.txt
path_to_eval_tgt=data/ta-en_clean/${dset}/text.tc.en
#path_to_bpe_mdl=~/tools/espnet/egs2/iwslt22_dialect/mt/data_clean/spm_bpe/ta_bpe_spm1000/bpe.model
path_to_bpe_mdl=data/msa-en_processed/spm2000/ar_bpe_spm2000/bpe.model

#path_to_dict_dir=exp_clean/bin_ta2en
#path_to_dict_dir=exp_msa-en_bpe2000/bin_ar2en
path_to_dict_dir=exp_msa-en_bpe2000_tune_with_ta/bin_ta2en
#path_to_mdl=exp_clean/checkpoints/checkpoint_best.pt
#path_to_mdl=exp_msa-en_bpe2000/checkpoints/checkpoint_best.pt
path_to_mdl=exp_msa-en_bpe2000_tune_with_ta/checkpoints/checkpoint_best.pt
#decode_dir=exp_clean/decode_asr_cleaned_BPE1000_${dset}_interactive
#decode_dir=exp_msa-en_bpe2000/decode_spm2000_ta2en_manual_${dset}_interactive
decode_dir=exp_msa-en_bpe2000_tune_with_ta/decode_spm2000_ta2en_manual_${dset}_interactive

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

sort ${path_to_eval_src} | cut -d" " -f2- | tokenizer.perl -q > ${decode_dir}/${src_lan}.txt
sort ${path_to_eval_tgt} | cut -d" " -f2- | detokenizer.perl -q > ${decode_dir}/${tgt_lan}.txt
sort ${path_to_eval_src} | cut -d" " -f1 > ${decode_dir}/${src_lan}.uttids
sort ${path_to_eval_tgt} | cut -d" " -f1 > ${decode_dir}/${tgt_lan}.uttids

if [ $stage -le 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Apply BPE to source data"
    spm_encode \
    --model=${path_to_bpe_mdl} \
    --output_format=piece \
    < ${decode_dir}/${src_lan}.txt \
    > ${decode_dir}/${src_lan}.bpe.txt
    
    mkdir decode_debug/logs -p
    mkdir -p decode_debug/bin_ta2en
    fairseq-preprocess \
        --source-lang ${src_lan} --target-lang ${tgt_lan} \
        --srcdict exp_msa-en_bpe2000_tune_with_ta/bin_ta2en/dict.ta.txt \
        --tgtdict exp_msa-en_bpe2000_tune_with_ta/bin_ta2en/dict.en.txt \
        --validpref decode_debug/valid.ta-en\
        --testpref decode_debug/test.ta-en\
        --destdir decode_debug/bin_ta2en --thresholdtgt 0 --thresholdsrc 0 \
        --workers 4 || exit 1;
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
sacrebleu ${decode_dir}/${tgt_lan}.txt -i ${decode_dir}/hyp.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding done !"

    