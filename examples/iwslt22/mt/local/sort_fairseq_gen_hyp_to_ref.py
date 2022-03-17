"""
This script will sort the output from fairseq-generate,
according to the original source file utterance order
"""

import argparse
from fairseq.data import Dictionary, data_utils, indexed_dataset

def get_parser():
    parser = argparse.ArgumentParser(
        description="writes text from binarized file to stdout"
    )
    parser.add_argument('--src_text', help='src (ar) text file (each row: S-uttid\t word1 word2 ...')
    parser.add_argument('--hyp_text', help='hyp (ta) text file (each row: S-uttid\t word1 word2 ...')
    parser.add_argument('--tgt_txt', help='target (en) text file (each row: word1 word2 ...')
    parser.add_argument('--out_dir', help='output directory')
    parser.add_argument('--hyp_lan', help='source language', default="ta")
    parser.add_argument('--tgt_lan', help='target language', default="en")
    return parser

def read_text(textfile: str, use_uttid=False, filter_empty=True, filter_len_limit=1024):
    """Read the text file
    Args:
    textfile (str): file path to the text
    use_uttid (bool): if True, the first field of the textfile is utterance id
    filter_empty (bool): if True, filter out the utterances which are empty (or only contains <200f>)
    filter_len_limit (int): if True, filter out the utterances which contain more tokens than the limit
    Return:
    texts (list): 
    """
    with open(textfile, 'r', encoding="utf-8") as ifh:
        for line in ifh:
            tokens = line.strip().split()

def read_binarized(bin_prefix, dict_path):
        dictionary = Dictionary.load(dict_path) if path is not None else None
        dataset = data_utils.load_indexed_dataset(
            bin_prefix,
            dictionary,
            dataset_impl='mmap',
            default="lazy",
        )

    for tensor_line in dataset:
        if dictionary is None:
            line = " ".join([str(int(x)) for x in tensor_line])
        else:
            line = dictionary.string(tensor_line)
        print(line)
           
def main():
    parser = get_parser()
    args = parser.parse_args()


if __name__ == "__main__":
    main()