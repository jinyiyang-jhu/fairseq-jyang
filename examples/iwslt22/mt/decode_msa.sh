#!/bin/bash

stage=-1
nj_preprocess=4
ngpus=4

src_lan="ar"
tgt_lan="en"
src_case="rm"
tgt_case="lc.rm"

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
#path_to_eval_src=~/tools/espnet/egs2/iwslt22_dialect/mt/data/${dset}/text.tc.rm.ta
path_to_eval_src=data/msa-en_processed/${dset}.${src_lan}-${tgt_lan}.${src_lan}.${src_case}
path_to_eval_tgt=data/msa-en_processed/${dset}.${src_lan}-${tgt_lan}.${tgt_lan}.${tgt_case}
path_to_bpe_mdl=data/msa-en_processed/spm2000/ar_bpe_spm2000/bpe.model 

path_to_dict_dir=exp_msa-en_bpe2000/bin_ar2en
path_to_mdl=exp_msa-en_bpe2000/checkpoints/checkpoint_best.pt
decode_dir=exp_msa-en_bpe2000/decode_spm2000_${dset}_interactive

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

cat ${path_to_eval_src} | tokenizer.perl -q > ${decode_dir}/${src_lan}.txt
cat ${path_to_eval_tgt} | detokenizer.perl -q > ${decode_dir}/${tgt_lan}.txt

if [ $stage -le 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Apply BPE to source data"
    spm_encode \
    --model=${path_to_bpe_mdl} \
    --output_format=piece \
    < ${decode_dir}/${src_lan}.txt \
    > ${decode_dir}/${src_lan}.bpe.txt
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding ${src_lan}-${tgt_lan}:${src_lan}"

[ -f ${decode_dir}/logs/decode.log ] && rm ${decode_dir}/logs/decode.log
cat ${decode_dir}/${src_lan}.bpe.txt | \
    fairseq-interactive ${path_to_dict_dir} \
        --source-lang "${src_lan}" --target-lang "${tgt_lan}" \
        --task translation \
        --path ${path_to_mdl}\
        --batch-size 256 \
        --buffer-size 2000 \
        --fix-batches-to-gpus \
        --beam 5 \
        --skip-invalid-size-inputs-valid-test \
        --remove-bpe=sentencepiece > $decode_dir/logs/decode.log || exit 1
grep ^D $decode_dir/logs/decode.log | cut -f3 | detokenizer.perl -q > ${decode_dir}/hyp.txt || exit 1
grep ^S $decode_dir/logs/decode.log | cut -f2 | detokenizer.perl -q > ${decode_dir}/ref.txt 
sacrebleu ${decode_dir}/${tgt_lan}.txt -i ${decode_dir}/hyp.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding done !"

    