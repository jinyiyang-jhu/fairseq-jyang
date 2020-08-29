#!/usr/bin/bash
 
# Select the common files in transcripts, 1best hypotheses asr oracles and lattice file list into a new directory.
# The comm_list contains lines like:
# A1RABVCI4QCIKC:1.0/2019/07/03/09/G090XG0692732FW7/04:31::TNIH_2V.e191fbb7-0a78-4594-b08b-981af9131e6fZXV/0
# the trans_file, best_hyp_file and oracle_file contain lines like:
# A1RABVCI4QCIKC:1.0/2019/07/03/09/G090XG0692732FW7/04:31::TNIH_2V.e191fbb7-0a78-4594-b08b-981af9131e6fZXV/0, alexa
# the old_lat_list contains lines like:
# s3://bluetrain-resources/yilegu/for_jinyi/feb_2019/part_0/lattice-dir/A1RABVCI4QCIKC:1.0/2019/07/01/02/G090XG06927117TH/30:42::TNIH_2V.5f53b3e3-adb3-4201-b12f-16973026476bZXV-0.lattice
 
# CAUTION !!!
# if your any of your input file contains differnt absolute path or file extension names,
# you should customize the patterns in `awk` and `sed` commands in step 0, 1, 2, 3. See details below.
 
stage=-1
filter_lat="False"
if [ $# -lt 5 ]; then
 echo "Usage: $0 <utt-list> <transcript-csv> <1best-hyp-csv> <oracle-csv> <target-dir> [<lat-file-list>] [lat-remove-anchor] [lat-extension]"
 exit 0
fi
 
comm_list=$1
trans_file=$2
best_hyp_file=$3
oracle_file=$4
new_dir=$5
 
if [ $# == 8 ]; then
 echo "Fiter lattice is required."
 filter_lat="True"
 old_lat_list=$6
 lat_remove_anchor=$7
 lat_extension=$8
fi
 
[ -d $new_dir ] || mkdir -p $new_dir || exit 1
 
if [ $filter_lat == "True" ]; then
 echo "Selecting ASR lattices"
 lat_dir=$new_dir/lattice
 [ -d $lat_dir ] || mkdir -p $lat_dir || exit 1
 cp $comm_list $lat_dir
 # This command will (1) append prefix path $lat_prefix (2) add lattice extension $lat_extension (3) convert last "/" to "-"
 # in order to match the old_lat_list.
 # Change the sed separator "@" if it's in your lat_prefix
 awk 'NR==FNR{a=gensub(/(.*)\//,"\\1-","",$0); arr[a];next}{b=gensub(/(.*)lattice-dir\//, "", $0);gsub(".lattice", "",b);if (b in arr) print $0}' \
   $comm_list $old_lat_list > $lat_dir/lat_file.index
 cat $lat_dir/lat_file.index | sed "s@^.*$lat_remove_anchor@@g; s@$lat_extension@@g; s@\(.*\)-@\1\/@g"  > $lat_dir/selected_utt.index
 comm_list=$lat_dir/selected_utt.index
fi
 
 
if [ $stage -le 0 ]; then
# Select transcript. Remove utterances.
 echo "Selecting transcriptions"
 tran_dir=$new_dir/transcript
 [ -d $tran_dir ] || mkdir -p $tran_dir || exit 1
 cp $comm_list $tran_dir
  # The first field (seperated by comma) of $trans_file is the identical utterance id
 awk -F "," 'NR==FNR{a[$1]=NR;next}$1 in a{print a[$1], $0}' $comm_list $trans_file \
   | sort -k1 -n | cut -d " " -f2- > $tran_dir/transcript.csv
 cut -d "," -f 2- $tran_dir/transcript.csv > $tran_dir/transcript.txt
fi
 
if [ $stage -le 1 ]; then
 echo "Selecting ASR 1best hypotheses"
 best_hyp_dir=$new_dir/asr_1best
 [ -d $best_hyp_dir ] || mkdir -p $best_hyp_dir || exit 1
 cp $comm_list $best_hyp_dir
  # The first field (seperated by comma) of $best_hyp_file is the identical utterance id
 awk -F "," 'NR==FNR{a[$1]=NR;next}$1 in a{print a[$1], $0}' $comm_list $best_hyp_file \
   | sort -k1 -n | cut -d " " -f2- > $best_hyp_dir/asr_1best.csv
 cut -d "," -f 2- $best_hyp_dir/asr_1best.csv > $best_hyp_dir/asr_1best.txt
fi
 
if [ $stage -le 2 ]; then
 echo "Selecting ASR oracle paths"
 oracle_dir=$new_dir/asr_oracle
 [ -d $oracle_dir ] || mkdir -p $oracle_dir || exit 1
 cp $comm_list $oracle_dir
 
 # The first field (seperated by comma) of $oracle_file is the identical utterance id
 awk -F "," 'NR==FNR{a[$1]=NR;next}$1 in a{print a[$1], $0}' $comm_list $oracle_file \
   | sort -k1 -n | cut -d " " -f2- > $oracle_dir/asr_oracle.csv
 cut -d "," -f 2- $oracle_dir/asr_oracle.csv > $oracle_dir/asr_oracle.txt
fi
 
echo "Selecting data done"
 

