#!/bin/bash

train_stage=-1
skip_decode=false
if [ $# -ne 1 ]; then
    echo "Usage: $0 <configuration>"
    echo "E.g., $0 conf_zh_en.sh"
    exit 1
fi

conf=$1

prepare-kevin-man.sh $conf || exit 1;
train.sh --stage $train_stage $conf || exit 1;
decode.sh --skip-decode $skip_decode $conf || exit 1;