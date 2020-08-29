#!/usr/bin/bash
 
# This script selects common sentences across transcript, asr_1best, asr_oracle, and lattice
# The trans_file, best_hyp_file, asr_oracle_file contain lines like:
# A1RABVCI4QCIKC:1.0/2019/07/03/09/G090XG0692732FW7/04:31::TNIH_2V.e191fbb7-0a78-4594-b08b-981af9131e6fZXV/0, alexa
# the lat_file_list contains lines like:
# s3://bluetrain-resources/yilegu/for_jinyi/feb_2019/part_0/lattice-dir/A1RABVCI4QCIKC:1.0/2019/07/01/02/G090XG06927117TH/30:42::TNIH_2V.5f53b3e3-adb3-4201-b12f-16973026476bZXV-0.lattice
# CAUTION !!!
# if your any of your input file contains differnt field separators, or lat_file_list contains file name with "@",
# you should customize the patterns in `cut` and `sed` commands in step 0. See details below.
 
stage=-1
 
if [ $# -ne 7 ]; then
 echo "Usage: $0 <transcript-file> <1best-hyp-file> <asr-oracle> <lat-file-list> <target-dir>"
 exit 1
fi
 
trans_file="$1"
best_hyp_file="$2"
asr_oracle_file="$3"
lat_file_list="$4"
count_dir="$5"
lat_remove_anchor="$6"
lat_extension="$7"
 
if [ $stage -le 0 ]; then
 # Create uttid index files from the four input files
 
 [ -d $count_dir ] || mkdir -p $count_dir || exit 1
 
 # Sort transcription utt list
 # the first field (seperated by comma) is the identical utterance id
 cut -d "," -f1 $trans_file | sort > $count_dir/trans_uttid.index
 
 # Sort 1best utt list
 # the first field (seperated by comma) is the identical utterance id
 cut -d "," -f1 $best_hyp_file  | sort > $count_dir/asr_1best_uttid.index
 
 # Sort asr oracle utt list
 # the first field (seperated by comma) is the identical utterance id
 cut -d "," -f1 $asr_oracle_file  | sort > $count_dir/asr_oracle_uttid.index
 
 # Sort lat list
 # We want to convert:
 # s3://bluetrain-resources/yilegu/for_jinyi/feb_2019/part_0/lattice-dir/A1RABVCI4QCIKC:1.0/2019/07/01/02/G090XG06927117TH/30:42::TNIH_2V.5f53b3e3-adb3-4201-b12f-16973026476bZXV-0.lattice
 # to "A1RABVCI4QCIKC:1.0/2019/07/01/02/G090XG06927117TH/30:42::TNIH_2V.5f53b3e3-adb3-4201-b12f-16973026476bZXV/0"
 # The command below will (1) remove absolute path till $lat_remove_anchor (included) (2) remove extension name $lat_extension (3) convert the last "-"" to "/"
 # Change the sed separator "@" $lat_file_list (1) contains "@"
 cat $lat_file_list | sed "s@^.*$lat_remove_anchor@@g; s@$lat_extension@@g; s@\(.*\)-@\1\/@g" | sort > $count_dir/lat_uttid.index
 #cat $lat_file_list | sed 's/^.*lattice-dir\///g; s/.lattice//g; s/\(.*\)-/\1\//g' | sort > $count_dir/lat_uttid.index
fi
 
num_of_trans=`wc -l $count_dir/trans_uttid.index | cut -d " " -f1`
comm -12 $count_dir/trans_uttid.index $count_dir/asr_1best_uttid.index > $count_dir/comm_trans_1best.index
 
comm -12 $count_dir/comm_trans_1best.index $count_dir/asr_oracle_uttid.index > $count_dir/comm_trans_1best_oracle.index
 
comm -12 $count_dir/comm_trans_1best_oracle.index $count_dir/lat_uttid.index > $count_dir/comm_trans_1best_oracle_lat.index
 
num_of_all_comm=`wc -l $count_dir/comm_trans_1best_oracle_lat.index | cut -d " " -f1`
 
if [ $num_of_all_comm == 0 ]; then
 echo "Selection failed: the final number of common sentences is 0 ! Check if the $count_dir/comm_*.index files for details"
 exit 1
fi
 
if [ $num_of_trans -ne $num_of_all_comm ]; then
 echo "Removing $(($num_of_trans-$num_of_all_comm)) lines from transcription"
 diff $count_dir/trans_uttid.index $count_dir/comm_trans_1best_oracle_lat.index > $count_dir/removed_trans_uttid.index
fi
echo "Keeping $num_of_all_comm utterances"

