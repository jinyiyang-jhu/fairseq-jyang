FAIRSEQ_PATH=$PWD/../../fairseq
KALDI_PATH=/export/b07/jyang/kaldi-jyang/kaldi/src
OPENFST_PATH=/export/b07/jyang/kaldi-jyang/kaldi/tools/openfst/bin
export PATH=$FAIRSEQ_PATH:$FAIRSEQ_PATH/scripts:$FAIRSEQ_PATH/utils:$FAIRSEQ_PATH/utils/parallel:$KALDI_PATH/latbin:$KALDI_PATH/bin:$OPENFST_PATH:$PWD:$PATH
unset PYTHONPATH
export PYTHONPATH=$FAIRSEQ_PATH/../scripts:$PYTHONPATH
