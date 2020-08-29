#!/bin/bash
 
# This script select comman sentences across provided transcript, 1best, oracle and lattice files.
# and filter out the transcripts that contain word counts lower than a threshold.
# This script will generate a new folder, containing the (1) transcript (2) asr_1best (3) asr_oracle and (4) lattice file list
# Please note that since the utterance names in the four files may differ, change the lat_remove_anchor, lat_extension and lat_prefix
# accordingly. See local/data_prepare/filter_uttid.sh for more details
# <transcript-file>, <1best-hyp-file> <asr-oracle> all have format like:
# <uttid>, alexa stop
# Note that filed separator for uttid and content is space (" ")
 
stage=-1
if [ $# -ne 5 ]; then
 echo "Usage: $0 <transcript-file> <1best-hyp-file> <asr-oracle> <lat-file-list> <target-dir>"
 exit 0
fi
 
lat_remove_anchor="lattice-dir/"
lat_extension=".lattice"
lat_prefix="s3://bluetrain-resources/yilegu/for_jinyi/feb_2019/part_0/lattice-dir/"
 
trans_file="$1"
best_hyp_file="$2"
asr_oracle_file="$3"
lat_file_list="$4"
tgt_dir="$5"
 
word_count_threshold=1
 
if [ $stage -le 0 ]; then
 echo "`date` Stage 0 ===> Get common sentence uttids in $tgt_dir/utt_counts/comm_uttids.index"
 mkdir -p $tgt_dir/utt_counts || echo "Mkdir $tgt_dir/utt_counts failed !"
 bash $(dirname $0)/filter_uttid.sh $trans_file $best_hyp_file $asr_oracle_file \
   $lat_file_list $tgt_dir/utt_counts $lat_remove_anchor $lat_extension || exit 1
fi
 
if [ $stage -le 1 ]; then
 echo "`date` Stage 1 ===> Filter out sentences with word counts less than threshold."
 # Final selected utt ids are in $tgt_dir/utt_counts/selected_utt.index
 [ ! -f $tgt_dir/utt_counts/comm_trans_1best_oracle_lat.index ] && echo "No such file $tgt_dir/utt_counts/comm_trans_1best_oracle_lat.index!" && exit 1
 bash $(dirname $0)/filter_short_utterance.sh $tgt_dir/utt_counts/comm_trans_1best_oracle_lat.index $trans_file \
   $tgt_dir/utt_counts/ $word_count_threshold || exit 1
fi
 
if [ $stage -le 2 ]; then
 echo "`date` Stage 2 ===> Selecting sentences"
 # Create a new directory containing selected (1) transcript (2) asr_1best (3) asr_oracle_file (4) lat_file_list
 bash $(dirname $0)/filter_data.sh $tgt_dir/utt_counts/comm_filtered_short_utt.index \
   $trans_file $best_hyp_file $asr_oracle_file $tgt_dir $lat_file_list $lat_remove_anchor $lat_extension || exit 1
fi

