#!/bin/bash

# This script filters and sorts the second input file, by the order of the first input file

filter=$1
input=$2
output=$3

for f in $filter $input; do
    [ ! -f $f ] && echo "No such file: $f !" && exit 1;
done

#cat $plf_file | awk 'NF!=1' | awk 'NR==FNR {a[$1];next} $1 in a{print $0}' $utt_list - \
#    > $plf_filtered_file
cat $input | awk 'NR==FNR {a[$1];next} $1 in a{if (NF !=1 ) {print $0} else {print $1" [oov]"}}' $filter - \
    > ${output}.tmp

num_src=$(wc -l $filter | cut -d " " -f1)
num_current=$(wc -l ${output}.tmp | cut -d " "  -f1)
if [ $num_src -ne $num_current ]; then
    echo "Number of filtered utterances differs from given utt list: $num_src vs $num_current !"
fi

# Reorder the output as the same order as input file
awk 'FNR == NR { lineno[$1] = NR; next} {print lineno[$1], $0;}' $filter \
    ${output}.tmp | sort -k 1,1n | cut -d' ' -f2-  > $output

num_prev=$(wc -l $input | cut -d " " -f1)
if [ $num_prev -ne $num_current ]; then
    echo "Reducing the input file from $num_prev to $num_current" > /dev/stderr
fi
rm ${output}.tmp
