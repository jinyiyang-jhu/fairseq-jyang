#!/bin/bash
# This script performs BPE tokenization of given input text, with given BPE model.
# Input file has format:
# <uttid> whom aim i speaking
# Output file has format:
# _wh om _am _i _speak ing

nbpe=1000
case="lc.rm" # lower case

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
    echo "Usage: $0 <train-text> <non-lang-syms-file> <bpe-model-dir> "
    exit 1
fi

train_text=$1
user_defined_syms_file=$2
bpe_model_dir=$3

bpemodel=$bpe_model_dir/bpe_${nbpe}_${case}
if [ ! -f ${bpemodel}.model ]; then
    [ -d $bpe_model_dir ] || mkdir -p $bpe_model_dir || exit 1; 
    for f in $train_text $user_defined_symbols; do
        [ ! -f $f ] && (echo "No such file: $f" && exit 1)
    done
    spm_train --user_defined_symbols='$(tr "\n" "," < ${user_defined_syms_file})' \
    --input=$train_text \
    --vocab_size=${nbpe} --model_type=bpe \
    --model_prefix=${bpemodel} \
    --input_sentence_size=100000000 \
    --character_coverage=1.0 || exit 1;
fi

echo "$(date -u) Successfully convert the $inputfile to BPE tokens"
