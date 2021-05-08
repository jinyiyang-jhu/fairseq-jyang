#!/bin/bash
# This script performs specified pre-precessing
# (a) convert all text to lower case
# (b) MOSES tokenization
# (c) train and apply BPE

if [ $# -ne 1 ]; then
    echo "Usage: $0 <hyperparams.txt>"
    exit 1;
fi

. path.sh || exit 1;
source $1

lans=($src $trg)

for lan in "${lans[@]}"; do
    for d in "${dsets[@]}"; do
        ifile=$datadir/generaldomain."$d".raw."$lan"
        cleanfile=$datadir/generaldomain."$d".cln."$lan"
        tknfile=$datadir/generaldomain."$d".tkn."$lan"

        echo "$(date '+%Y-%m-%d %H:%M:%S') lan: $lan, dset: $d, step 1: clean and segmentation"
        cat $ifile | python local/clean_text.py --lan $lan --output $cleanfile
        
        # Tokenization
        echo "$(date '+%Y-%m-%d %H:%M:%S') lan: $lan, dset: $d, step 2: tokenization"
        $MOSES_PATH/tokenizer/tokenizer.perl -l $lan -q < $cleanfile > $tknfile
        num_lines=$(wc -l $tknfile | cut -d " " -f1)
        echo "$(date '+%Y-%m-%d %H:%M:%S') after tokenization: $d - $lan: $num_lines lines"
    done
done
mkdir -p $bpe_dir || exit 1;
$SOCKEYE_PATH/preprocess-bpe.sh $1
