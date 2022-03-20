'''
This script sort the hyp file utternace order, according to the given src file.
The src and hyp has following format:
<uttid> word1 word2 ...
'''
import argparse
import logging

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',)

def read_scp_to_dict(ifile):
    dict_scp = {}
    with open(ifile, 'r', encoding="utf-8") as ifh:
        for line in ifh:
            tokens = line.strip().split()
            uttid = tokens.pop(0)
            dict_scp[uttid] = " ".join(tokens)
    return dict_scp

def sort_hyp(src_file, hyp_dict, hyp_out_file):
    hyp_ofhd = open(hyp_out_file, 'w', encoding="utf-8")
    with open(src_file, 'r', encoding="utf-8") as ifh:
        for line in ifh:
            uttid = line.strip().split().pop(0)
            if uttid in hyp_dict:
                print(uttid + " " + hyp_dict[uttid], file=hyp_ofhd)
            else:
                logging.warning(f'Missing src utterance in hyp file: {uttid}')
    hyp_ofhd.close()

def main():
    parser = argparse.ArgumentParser(
        description="Sort the hyp file utterance order accoring to the src file"
    )
    parser.add_argument('--src_text', help='src scp file (each row: S-uttid word1 word2 ...', required=True)
    parser.add_argument('--hyp_in_text', help='hyp input scp file (each row: S-uttid word1 word2 ...', required=True)
    parser.add_argument('--hyp_out_text', help='hyp output scp file (each row: S-uttid word1 word2 ...', required=True)
    args = parser.parse_args()

    dict_hyp = read_scp_to_dict(args.hyp_in_text)
    sort_hyp(args.src_text, dict_hyp, args.hyp_out_text)

if __name__ == "__main__":
    main()