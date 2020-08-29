#!/bin/bash
 
# Remove the transcripts that has number of words less than given threshold.
# Original transcript.txt file has format like:
# <file-id>, alexa stop
# Note that the field separator for file id and content is space (" ")
 
if [ $# -ne 4 ]; then
 echo "$0: <origin-utt-list> <original-reference-txt> <output-dir> <word-count-threshold>"
 exit 0
fi
 
origin_utt_list=$1 # The current utterance list used for filtering utterances.
trans=$2 # Raw reference.txt provided by Yi
out_dir=$3
word_count_threshold=$4
echo "Word count threshold is $word_count_threshold"
 
field_count_threshold=$((word_count_threshold+1)) # 1st field is file id, so the actual number of fileds needs to add 1
 
[ -d $out_dir ] || mkdir -p $out_dir || exit "Making dir: $out_dir failed !"
 
awk -v thres=$field_count_threshold '{if (NF > thres) {gsub(",","",$1);print $1}}' $trans | sort > $out_dir/trans_longer_than_${word_count_threshold}_uttid.index
 
sort $origin_utt_list > $out_dir/sorted_comm.index
 
comm -12 $out_dir/trans_longer_than_${word_count_threshold}_uttid.index $out_dir/sorted_comm.index > $out_dir/comm_filtered_short_utt.index
 
selected_utt_count=`wc -l $out_dir/comm_filtered_short_utt.index | cut -d " " -f1`
 
rm $out_dir/sorted_comm.index
 
if [ $selected_utt_count == 0 ]; then
 echo "Selection failed: the final number of selected sentences is 0 ! "
 exit 1
fi
 

