#!/bin/bash

# This script filters out the empty plf files, and the files not in the input uttid list

plf_file=$1
utt_list=$2
plf_filtered_file=$3

for f in $plf_file $utt_list; do
    [ ! -f $f ] && echo "No such file: $f !" && exit 1;
done

#cat $plf_file | awk 'NF!=1' | awk 'NR==FNR {a[$1];next} $1 in a{print $0}' $utt_list - \
#    > $plf_filtered_file
cat $plf_file | awk 'NR==FNR {a[$1];next} $1 in a{if (NF !=1 ) {print $0} else {print $1" [noise]"}}' $utt_list - \
    > ${plf_filtered_file}.tmp

num_src=$(wc -l $utt_list | cut -d " " -f1)
num_current=$(wc -l ${plf_filtered_file}.tmp | cut -d " "  -f1)
if [ $num_src -ne $num_current ]; then
    echo "Number of filtered utterances differs from given utt list !" && exit 1;
fi

# Reorder the output as the same order as input file
awk 'FNR == NR { lineno[$1] = NR; next} {print lineno[$1], $0;}' $utt_list \
    ${plf_filtered_file}.tmp | sort -k 1,1n | cut -d' ' -f2-  > $plf_filtered_file

num_prev=$(wc -l $plf_file | cut -d " " -f1)
echo "Reducing the input file from $num_prev to $num_current" > /dev/stderr
rm ${plf_filtered_file}.tmp