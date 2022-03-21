#!/bin/bash

nbpe=32000

train_text_dir="data/ta-en_ta_translated_en_true/train"
bpedir="data/ta-en_ta_translated_en_true/spm_${nbpe}_ta_en_both_42M"

src_lan="ta"
tgt_lan="en"
src_case="tc.rm"
tgt_case="tc"

oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
bos="<s>" # start of sentence symbol
eos="</s>"

for lan in $src_lan $tgt_lan; do
    if [ $lan == ${src_lan} ]; then
        case=$src_case
        bpe_nlsyms="--user_defined_symbols="""
    else
        case=$tgt_case
        bpe_nlsyms="--user_defined_symbols=\"&apos;\",\"&amp;\",\"&quot;\""
    fi
    spm_dir=${bpedir}/$lan
    mkdir -p $spm_dir || exit 1;

    train_text=${train_text_dir}/text.${case}.${lan}.tok
    echo "$(date) Training the spm model for lan: ${lan}, train text is ${train_text}"
    spm_train \
    --input="${train_text}" \
    --vocab_size="${nbpe}" \
    --model_type="bpe" \
    --model_prefix="${spm_dir}"/bpe \
    --character_coverage=1.0 \
    --input_sentence_size=100000000 \
    "${bpe_nlsyms}" || exit 1;
    {
        echo "${blank}"
        echo "${oov}"
        <"${spm_dir}"/bpe.vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
        echo "${bos}"
        echo "${eos}"
    } > "${spm_dir}"/train.tokens.txt
done

