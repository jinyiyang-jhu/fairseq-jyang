
FAIRSEQ_PATH=/home/hltcoe/jyang1/fairseq/fairseq
SOCKEYE_PATH=/home/hltcoe/jyang1/sockeye-recipes/scripts
MOSES_PATH=/home/hltcoe/jyang1/mosesdecoder/scripts
export PATH=$FAIRSEQ_PATH:$FAIRSEQ_PATH/utils:$FAIRSEQ_PATH/utils/parallel:$FAIRSEQ_PATH/scripts:$SOCKEYE_PATH:$MOSES_PATH:$PATH
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH
