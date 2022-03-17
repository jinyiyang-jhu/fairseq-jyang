"""
This script will sort the output from fairseq-generate,
according to the original source file utterance order
"""

import argparse
from fairseq.data import Dictionary, data_utils, indexed_dataset
import os

def get_parser():
    parser = argparse.ArgumentParser(
        description="writes text from binarized file to stdout"
    )
    parser.add_argument('--src_text', help='src (ar) text file (each row: S-uttid\t word1 word2 ...', required=True)
    parser.add_argument('--hyp_text', help='hyp (ta) text file (each row: S-uttid\t word1 word2 ...', required=True)
    parser.add_argument('--tgt_txt', help='target (en) text file (each row: word1 word2 ...', required=True)
    parser.add_argument('--out_dir', help='output directory', required=True)
    parser.add_argument('--src_lan', help='source language', default="ar")
    parser.add_argument('--hyp_lan', help='source language', default="ta")
    parser.add_argument('--tgt_lan', help='target language', default="en")
    parser.add_argument('--setname', help='name of the set', default="train")
    parser.add_argument('--tok_len_limit', help="Keep the utterances with token length lower than limit",
                        type=int, default=1024)
    return parser

def read_text(textfile: str, use_uttid=False, tok_len_limit=None):
    """Read the text file
    Args:
    textfile (str): file path to the text
    use_uttid (bool): if True, the first field of the textfile is utterance id
    # filter_empty (bool): if True, filter out the utterances which are empty (or only contains <200f>)
    tok_len_limit (int): if True, filter out the utterances which contain more tokens than the limit
    Return:
    texts (list): 
    """
    dict_scp = {}
    idx = 0
    with open(textfile, 'r', encoding="utf-8") as ifh:
        for line in ifh:
            if use_uttid: # First field is utt idx
                tokens = line.strip().split('\t')
                if len(tokens) == 1: # only utt idx and text is empty
                    continue
                uttid = int(tokens[0].split('-')[1])
                text = tokens[1].split()
                if tok_len_limit is None or len(text) <= tok_len_limit:
                    dict_scp[uttid] = ' '.join(text)

            else: # Text starts from first field
                uttid = idx
                text = line.strip().split()
                idx += 1
                if len(text) == 0: # empty text
                    continue
                if tok_len_limit is None or len(text) <= tok_len_limit:
                    dict_scp[uttid] = ' '.join(text)

    return dict_scp

def write_filtered_src_hyp(dict_src, dict_hyp, dict_tgt, uttlists, src_ofile, hyp_ofile, tgt_ofile):
    sofh = open(src_ofile, 'w', encoding='utf-8')
    hofh = open(hyp_ofile, 'w', encoding='utf-8')
    tofh = open(tgt_ofile, 'w', encoding='utf-8')
    
    for utt in uttlists:
        print(str(utt) + " " + dict_src[utt], file=sofh)
        print(str(utt) + " " + dict_hyp[utt], file=hofh)
        print(str(utt) + " " + dict_tgt[utt], file=tofh)
    sofh.close()
    hofh.close()
    tofh.close()

# def read_binarized(bin_prefix, dict_path):
#         dictionary = Dictionary.load(dict_path) if path is not None else None
#         dataset = data_utils.load_indexed_dataset(
#             bin_prefix,
#             dictionary,
#             dataset_impl='mmap',
#             default="lazy")

#     for tensor_line in dataset:
#         if dictionary is None:
#             line = " ".join([str(int(x)) for x in tensor_line])
#         else:
#             line = dictionary.string(tensor_line)
#         print(line)
           
def main():
    parser = get_parser()
    args = parser.parse_args()

    dict_hyp = read_text(args.hyp_text, use_uttid=True, tok_len_limit=1024) # TA (words)
    hyp_utt_list = list(dict_hyp.keys())
    dict_tgt = read_text(args.tgt_txt, use_uttid=False, tok_len_limit=None) # EN (BPE)
    tgt_utt_list = list(dict_tgt.keys())
    dict_src = read_text(args.src_text, use_uttid=True, tok_len_limit=1024) # TA (words)
    src_utt_list = list(dict_src.keys())

    intersec_utts = set(src_utt_list) & set(tgt_utt_list)
    intersec_utts = sorted(list(intersec_utts & set(hyp_utt_list)))

    src_ofile_path = os.path.join(args.out_dir, args.setname+"."+args.src_lan+"-"+args.tgt_lan+"."+args.src_lan) # AR-EN.AR
    hyp_ofile_path = os.path.join(args.out_dir, args.setname+"."+args.hyp_lan+"-"+args.tgt_lan+"."+args.hyp_lan) # TA-EN.TA
    tgt_ofile_path = os.path.join(args.out_dir, args.setname+"."+args.hyp_lan+"-"+args.tgt_lan+"."+args.tgt_lan) # TA-EN.EN
    write_filtered_src_hyp(dict_src, dict_hyp, dict_tgt,
        intersec_utts, src_ofile_path, hyp_ofile_path, tgt_ofile_path)

if __name__ == "__main__":
    main()