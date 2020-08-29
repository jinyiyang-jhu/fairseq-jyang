#!/bin/bash
 
# Select dataset based on the given lattice file list in PLF processed dir.
 
stage=-1
if [ $# -ne 4 ]; then
 echo "Usage: $0 <lat-list> <transcript-csv> <1best-hyp-csv> <target-dir> "
 exit 0
fi
 
lat_list=$1
ref_file=$2
best_hyp_file=$3
new_dir=$4
 
comm_list=$new_dir/selected_utt.index
[ -d $new_dir ] || mkdir -p $new_dir || exit 1
 
echo "Creating selected_utt.index for the dataset"
cat $lat_list | sed 's/\(.*\)lattice\///; s/\(.*\)\-/\1\//; s/.lat.ark//g' > $comm_list
 
 
if [ $stage -le 0 ]; then
# Select transcript. Remove utterances.
 echo "Selecting transcription"
 tran_dir=$new_dir/transcript
 [ -d $tran_dir ] || mkdir -p $tran_dir || exit 1
 cp $comm_list $tran_dir
 awk -F "," 'NR==FNR{a[$1]=NR;next}$1 in a{print a[$1], $0}' $comm_list $ref_file \
   | sort -k1 -n | cut -d " " -f2- > $tran_dir/transcript.csv
 cut -d "," -f 2- $tran_dir/transcript.csv > $tran_dir/transcript.txt
fi
 
if [ $stage -le 1 ]; then
 echo "Selecting ASR 1best hypotheses"
 best_hyp_dir=$new_dir/asr_1best
 [ -d $best_hyp_dir ] || mkdir -p $best_hyp_dir || exit 1
 cp $comm_list $best_hyp_dir
 awk -F "," 'NR==FNR{a[$1]=NR;next}$1 in a{print a[$1], $0}' $comm_list $best_hyp_file \
   | sort -k1 -n | cut -d " " -f2- > $best_hyp_dir/asr_1best.csv
 cut -d "," -f 2- $best_hyp_dir/asr_1best.csv > $best_hyp_dir/asr_1best.txt
fi
 

