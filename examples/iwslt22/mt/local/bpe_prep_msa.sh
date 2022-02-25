#!/bin/bash
# This script train and apply BPE for src and tgt language separately.

src_lang="ar"
tgt_lang="en"
src_case="rm"
tgt_case="lc.rm"

data_feats="data/msa-en_processed/"
token_listdir="data/msa-en_processed/spm"

ta_en_ta_train_text="/home/hltcoe/jyang1/tools/espnet/egs2/iwslt22_dialect/st1/data_clean/spm_bpe/ta_bpe_spm1000/train.txt"
ta_en_en_train_text="/home/hltcoe/jyang1/tools/espnet/egs2/iwslt22_dialect/st1/data_clean/spm_bpe/en_bpe_spm1000/train.txt"

train_set="train"
dev_set="dev"
test_set="test"
bpemode="bpe"

# src bpe
src_nbpe=2000
src_skip_train="False" # train or encode

# tgt bpe
tgt_nbpe=2000
tgt_skip_train="False" # train or encode

oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
bos="<s>" # start of sentence symbol
eos="</s>"

mkdir -p $token_listdir || exit 1;

cat $ta_en_ta_train_text "${data_feats}/train.${src_lang}-${tgt_lang}.ar.${src_case}" \
    > $token_listdir/bpe_train.ar.txt

cat $ta_en_en_train_text "${data_feats}/train.${src_lang}-${tgt_lang}.en.${src_case}" \
    > $token_listdir/bpe_train.en.txt

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
    bpe_train_text=$token_listdir/bpe_train.${lan}.txt
    #bpe_train_text="${data_feats}/train.${src_lang}-${tgt_lang}.${lan}.${lan_case}"
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

        spm_train \
            --input="${bpe_train_text}" \
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
        dtext="${data_feats}/${dset}.${src_lang}-${tgt_lang}.${lan}.${lan_case}"
        
        spm_encode \
        --model="${bpemodel}" \
        --output_format=piece \
        < ${dtext} \
        > ${token_listdir}/${dset}.bpe.${src_lang}-${tgt_lang}.${lan} 
    done
done

echo "$(date) Succeed !"
