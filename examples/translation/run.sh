#!/bin/bash

train_stage=-1
skip_decode=false

if [ $# -ne 1 ]; then
    echo "Usage: $0 <train-configuration>"
    exit 1;
fi

conf=$1
. $conf

prepare-text.sh $conf || exit 1;
train.sh --stage $train_stage $conf || exit 1;
decode.sh --skip-decode $skip_decode $conf || exit 1;