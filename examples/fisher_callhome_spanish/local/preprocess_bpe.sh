#!/bin/bash
# This script performs BPE tokenization of given input text, with given BPE model.
# Input file has format:
# <uttid> whom aim i speaking
# Output file has format:
# _wh om _am _i _speak ing

if [ $# -ne 3 ]; then
    echo "Usage: $0 <data-dir> <bpe-model> <output-dir>"
    exit 1
fi

inputfile=$1
bpemodel=$2
outputfile=$3

for f in $inputfile $bpemodel; do
    [ ! -f $f ] && (echo "No such file: $f" && exit 1)
done

# Perform BPE conversion on given text
cut -f 2- -d " " $inputfile | spm_encode --model=${bpemodel} --output_format=piece |
    cut -f 2- -d " " > $outputfile
