#!/bin/bash

# This script extract 1best path from lattices.

lat_dir="kaldi_data/kaldi_lat_dir"
uttid_dir="data/uttids"
output_dir="data"
src=".es"

dsets=("fisher_dev" "fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest" "train")
fairseq_dsets=("valid" "test" "test1" "test2" "test3" "test4" "train")

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

for idx in $(seq 0 $((${#dsets[@]}-1))); do
    dset=${dsets[$idx]}
    if [ -f $lat_dir/${dset}${src}/scoring_kaldi/best_wer ]; then
        opts=$(cat  $lat_dir/${dset}${src}/scoring_kaldi/best_wer | rev | cut -d "/" -f 1 | rev)
        lmwt=$(echo $opts | cut -d "_" -f2)
        penalty=$(echo $opts | cut -d "_" -f3)
        one_best=$lat_dir/${dset}${src}/scoring_kaldi/penalty_${penalty}/$lmwt.txt
        cat $one_best | sed -e "s/<unk>//g; s/\&apos\;/ \&apos\; /g; s/'/ \&apos\; /g;" \
            > $lat_dir/${dset}${src}/scoring_kaldi/1best.txt
        
        local/filter_scp.sh $uttid_dir/$dset.uttids $lat_dir/${dset}${src}/scoring_kaldi/1best.txt \
            $output_dir/${dset}${src}/text.1best
    else
        echo "Generating 1best path from lattice: $lat_dir/${dset}${src}"
        # TBD
        exit 1;
    fi 
done
