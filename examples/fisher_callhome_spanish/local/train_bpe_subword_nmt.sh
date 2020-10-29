#!/bin/bash
# This script performs BPE tokenization of given input text, with given BPE model.
# The BPE model is trained with subword-nmt toolkit.
# Input file has format:
# <uttid> whom aim i speaking
# Output file has format:
# _wh om _am _i _speak ing

nbpe=1000
case="lc.rm" # lower case

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
    echo "Usage: $0 <train-text> <bpe-model-dir> <glossaries>"
    exit 1
fi

train_text=$1
bpe_model_dir=$2
glossaries=$3

[ -d $bpe_model_dir ] || mkdir -p $bpe_model_dir || exit 1; 
[ ! -f $train_text ] && (echo "No such file: $train_text" && exit 1)

cp $glossaries $bpe_model_dir/glossaries.txt
#sed -e "s@\&apos\;@\'@g" $train_text | sacremoses -l en -j 4 tokenize > $bpe_model_dir/clean_text.txt
sed -e 's/\&apos\;/ \&apos\; /g' $train_text > $bpe_model_dir/clean_text.txt
if [ ! -f $bpe_model_dir/code.txt ]; then
    subword-nmt learn-joint-bpe-and-vocab --input $bpe_model_dir/clean_text.txt \
        -s $nbpe -o $bpe_model_dir/code.txt \
        --write-vocabulary $bpe_model_dir/vocab.all.txt || exit 1
fi

