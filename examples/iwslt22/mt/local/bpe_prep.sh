#!/bin/bash
# This script train and apply BPE for src and tgt language separately.

src_lang="ta"
tgt_lang="en"
src_case="tc.rm"
tgt_case="tc"

data_feats="data_clean"
token_listdir="data_clean/spm_bpe"
train_set="train"
dev_set="dev"
test_set="test1"
bpemode="bpe"

# src bpe
src_nbpe=1000
src_skip_train="False" # train or encode

# tgt bpe
tgt_nbpe=1000
tgt_skip_train="False" # train or encode

oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
bos="<s>" # start of sentence symbol
eos="</s>"

for lan in "${src_lang}" "${tgt_lang}"; do
    if [ "${lan}" == "${src_lang}" ]; then
        lan_case=$src_case
        nbpe=$src_nbpe
        bpe_nlsyms="--user_defined_symbols="""
        skip_train=${src_skip_train}
    else
        lan_case=$tgt_case
        nbpe=$tgt_nbpe
        bpe_nlsyms="--user_defined_symbols=\"&apos;\",\"&amp;\",\"&quot;\""
        mode=$tgt_mode
        skip_train=${tgt_skip_train}
    fi
    echo "bpe_nlsyms is ${bpe_nlsyms}"
    bpe_train_text="${data_feats}/${train_set}/text.${lan_case}.${lan}"
    bpedir="${token_listdir}/${lan}_bpe_spm${nbpe}"
    bpetoken_list="${bpedir}"/train.tokens.txt
    bpemodel="${bpedir}"/bpe.model

    echo "$(date) Training spm model for language : ${lan}"
    if [ -n "${bpe_nlsyms}" ]; then
        _opts_spm="${bpe_nlsyms}"
    else
        _opts_spm=""
    fi

    if [ ${skip_train} != "True" ]; then
        mkdir -p $bpedir || exit 1;
        text=$"${bpedir}"/train.txt 
        echo $bpe_train_text
        if [ ! -f ${text} ] || [ ! -s ${text} ]; then
            cat ${bpe_train_text} | cut -f 2- -d" "  > ${text} || exit 1;
        fi
        spm_train \
            --input="${bpedir}"/train.txt \
            --vocab_size="${nbpe}" \
            --model_type="${bpemode}" \
            --model_prefix="${bpedir}"/bpe \
            --character_coverage=1.0 \
            --input_sentence_size=100000000 \
            ${_opts_spm} || exit 1;

        {
        echo "${blank}"
        echo "${oov}"
        <"${bpedir}"/bpe.vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
        echo "${bos}"
        echo "${eos}"
        } > "${bpetoken_list}"
    fi
    for dset in "${train_set}" "${dev_set}" "${test_set}"; do
        echo "$(date) Encoding with spm model for language : ${lan} , set is ${dset}"
        dtext=${data_feats}/${dset}/text.${lan_case}.${lan}
        text=${bpedir}/${dset}.txt
        if [ ! -f ${text} ] || [ ! -s ${text} ]; then
            cat ${dtext} | cut -f 2- -d" "  > ${text} || exit 1;
        fi
        cut -d" " -f1 ${dtext} > ${token_listdir}/${dset}.uttids
            spm_encode \
            --model="${bpemodel}" \
            --output_format=piece \
            < ${bpedir}/${dset}.txt \
            > ${token_listdir}/${dset}.bpe.${src_lang}-${tgt_lang}.${lan} 
    done
done

echo "$(date) Succeed !"
