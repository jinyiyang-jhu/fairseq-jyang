#!/bin/bash

prep_stage=-1
train_stage=-1
skip_decode=false

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 <train-configuration>"
    exit 1;
fi

conf=$1
. $conf

prepare-text.sh --stage $prep_stage $conf || exit 1;
train.sh --stage $train_stage $conf || exit 1;
decode.sh --skip-decode $skip_decode $conf || exit 1;