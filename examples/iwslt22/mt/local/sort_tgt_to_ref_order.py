'''
This script sort the tgt file utternace order, according to the given ref file.
The ref and tgt has following format:
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

def sort_files(ref_file, tgt_dict, tgt_out_file):
    tgt_ofhd = open(tgt_out_file, 'w', encoding="utf-8")
    with open(ref_file, 'r', encoding="utf-8") as ifh:
        for line in ifh:
            uttid = line.strip().split().pop(0)
            if uttid in tgt_dict:
                print(uttid + " " + tgt_dict[uttid], file=tgt_ofhd)
            else:
                logging.warning(f'Missing ref utterance in tgt file: {uttid}')
    tgt_ofhd.close()

def main():
    parser = argparse.ArgumentParser(
        description="Sort the tgt file utterance order accoring to the ref file"
    )
    parser.add_argument('--ref_text', help='ref scp file (each row: S-uttid word1 word2 ...', required=True)
    parser.add_argument('--tgt_in_text', help='tgt input scp file (each row: S-uttid word1 word2 ...', required=True)
    parser.add_argument('--tgt_out_text', help='tgt output scp file (each row: S-uttid word1 word2 ...', required=True)
    args = parser.parse_args()

    dict_tgt = read_scp_to_dict(args.tgt_in_text)
    sort_files(args.ref_text, dict_tgt, args.tgt_out_text)

if __name__ == "__main__":
    main()
