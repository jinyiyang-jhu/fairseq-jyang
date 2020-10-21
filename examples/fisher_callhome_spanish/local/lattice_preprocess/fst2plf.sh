#!/bin/bash

systab=$1
sos=$2
eos=$3

# This script read from STDIN (output from lattice-to-fst), and write to STDOUT
{
    uttid=""
    while read -r; do
      if [[ $REPLY =~ ^$ ]]; then
        plf=$(echo "$block" |\
          fstcompile --arc_type=log |\
          fstpush --push_weights --remove_total_weight |\
          fstconcat $sos - | fstconcat - $eos |\
          fstrmepsilon |\
          fstminimize |\
          fsttopsort |\
          fstprint --isymbols=$systab --osymbols=$systab |\
          perl local/lattice_preprocess/txt2plf.pl)
        echo -e "$uttid\t$plf"
        block=""
        uttid=""
      elif [ "$uttid" == "" ]; then
        uttid="$REPLY"
      else
        block+=$REPLY$'\n'
      fi
    done
} < /dev/stdin
